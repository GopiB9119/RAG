"""Tests for the RAG pipeline.

Run with:  python -m unittest test_rag -v

Chunking and ID tests are pure unit tests. Retrieval tests run only when
the vector database exists (build it first with ingest_sources.py).
"""

from __future__ import annotations

import unittest

from ingest_sources import make_record_id, split_into_chunks, title_from_source
from rag_core import (
    DEFAULT_COLLECTION,
    DEFAULT_DATABASE,
    MODEL_NAME,
    extract_keywords,
    retrieve,
)


class SplitIntoChunksTests(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(split_into_chunks(""), [])

    def test_short_text_returns_single_chunk(self):
        self.assertEqual(split_into_chunks("Hello world."), ["Hello world."])

    def test_whitespace_is_normalized(self):
        self.assertEqual(split_into_chunks("Hello   \n  world."), ["Hello world."])

    def test_long_text_is_split_into_chunks(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(200))
        chunks = split_into_chunks(text, chunk_size=300)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 350 for chunk in chunks))

    def test_all_sentences_are_kept(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(200))
        chunks = split_into_chunks(text, chunk_size=300)
        joined = " ".join(chunks)
        for i in range(200):
            self.assertIn(f"Sentence number {i} is here.", joined)


class RecordIdTests(unittest.TestCase):
    def test_same_input_produces_same_id(self):
        metadata = {"source": "a.pdf", "page": 1, "chunk": 0}
        self.assertEqual(
            make_record_id(metadata, "some text"),
            make_record_id(metadata, "some text"),
        )

    def test_different_text_produces_different_id(self):
        metadata = {"source": "a.pdf", "page": 1, "chunk": 0}
        self.assertNotEqual(
            make_record_id(metadata, "text one"),
            make_record_id(metadata, "text two"),
        )


class ExtractKeywordsTests(unittest.TestCase):
    def test_extracts_codes_with_digits(self):
        self.assertIn("Q4", extract_keywords("What was revenue in Q4 FY26?"))
        self.assertIn("FY26", extract_keywords("What was revenue in Q4 FY26?"))

    def test_extracts_years_and_skips_short_numbers(self):
        keywords = extract_keywords("What happened on May 20, 2026 in Bengaluru?")
        self.assertIn("2026", keywords)
        self.assertIn("May", keywords)
        self.assertIn("Bengaluru", keywords)
        self.assertNotIn("20", keywords)

    def test_skips_common_words(self):
        self.assertEqual(extract_keywords("What is the revenue for the year?"), [])


class TitleFromSourceTests(unittest.TestCase):
    def test_extracts_file_name_from_url(self):
        self.assertEqual(
            title_from_source("https://example.com/reports/annual.pdf?download=1"),
            "annual.pdf",
        )

    def test_extracts_file_name_from_windows_path(self):
        self.assertEqual(title_from_source("C:\\docs\\report.pdf"), "report.pdf")


class RetrievalTests(unittest.TestCase):
    """Integration tests against the real index; skipped if it is not built."""

    @classmethod
    def setUpClass(cls):
        from pathlib import Path

        if not Path(DEFAULT_DATABASE).exists():
            raise unittest.SkipTest(
                f"'{DEFAULT_DATABASE}' not found. Run ingest_sources.py first."
            )
        import chromadb
        from sentence_transformers import SentenceTransformer

        client = chromadb.PersistentClient(path=DEFAULT_DATABASE)
        cls.collection = client.get_or_create_collection(name=DEFAULT_COLLECTION)
        if not cls.collection.count():
            raise unittest.SkipTest(
                f"Collection '{DEFAULT_COLLECTION}' is empty. Run ingest_sources.py first."
            )
        cls.model = SentenceTransformer(MODEL_NAME)

    def test_collection_has_chunks(self):
        self.assertGreater(self.collection.count(), 0)

    def test_retrieve_returns_a_list(self):
        retrieved = retrieve(
            "company revenue",
            self.collection,
            self.model,
            self.collection.count(),
        )
        self.assertIsInstance(retrieved, list)


class FakeVectors(list):
    def tolist(self):
        return list(self)


class FakeModel:
    def __init__(self, name):
        assert name == MODEL_NAME
        self.batches = []

    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        self.batches.append(texts)
        return FakeVectors([[1.0, 0.0] for text in texts])


class FakeCollection:
    def __init__(self):
        from rag_core import COLLECTION_METADATA

        self.metadata = dict(COLLECTION_METADATA)
        self.records = {}
        self.fail_upsert = False

    def count(self):
        return len(self.records)

    def upsert(self, ids, documents, embeddings, metadatas):
        assert len(ids) == len(set(ids))
        assert len(ids) == len(documents) == len(embeddings) == len(metadatas)
        if self.fail_upsert:
            raise RuntimeError("Simulated vector write failure")
        self.records.update({record_id: (text, metadata, vector)
                             for record_id, text, metadata, vector
                             in zip(ids, documents, metadatas, embeddings)})

    def get(self, where, include):
        return {"ids": [record_id for record_id, (_, metadata, _) in self.records.items()
                        if metadata["source"] == where["source"]]}

    def delete(self, ids):
        for record_id in ids:
            del self.records[record_id]

    def query(self, query_embeddings, n_results, include, where_document=None):
        assert query_embeddings == [[1.0, 0.0]]
        rows = list(self.records.values())[:n_results]
        return {"documents": [[row[0] for row in rows]],
                "metadatas": [[row[1] for row in rows]],
                "distances": [[0.1 for row in rows]]}


def install_index_fakes(monkeypatch, collection):
    import sys
    from types import ModuleType, SimpleNamespace

    model = FakeModel(MODEL_NAME)
    chroma = ModuleType("chromadb")
    chroma.PersistentClient = lambda **kwargs: SimpleNamespace(
        get_or_create_collection=lambda **kwargs: collection,
    )
    errors = ModuleType("chromadb.errors")
    errors.NotFoundError = type("NotFoundError", (Exception,), {})
    transformers = ModuleType("sentence_transformers")
    transformers.SentenceTransformer = lambda name: model
    monkeypatch.setitem(sys.modules, "chromadb", chroma)
    monkeypatch.setitem(sys.modules, "chromadb.errors", errors)
    monkeypatch.setitem(sys.modules, "sentence_transformers", transformers)
    return model


def test_index_deduplicates_batches_and_removes_stale_source_chunks(monkeypatch):
    import ingest_sources

    collection = FakeCollection()
    model = install_index_fakes(monkeypatch, collection)
    monkeypatch.setattr(ingest_sources, "UPSERT_BATCH_SIZE", 1)
    old = {"text": "Old revenue.", "metadata": {"source": "report.pdf", "page": 1, "chunk": 0}}
    other = {"text": "Other document.", "metadata": {"source": "other.pdf", "page": 1, "chunk": 0}}
    ingest_sources.build_index([old, other], "unused", "test", False)
    new = {**old, "text": "Updated revenue."}
    assert ingest_sources.build_index([new, new], "unused", "test", False) == 1
    assert {record[0] for record in collection.records.values()} == {"Updated revenue.", "Other document."}
    assert all(len(batch) == 1 for batch in model.batches)


def test_index_failure_keeps_old_source_chunks(monkeypatch):
    import pytest
    import ingest_sources

    collection = FakeCollection()
    install_index_fakes(monkeypatch, collection)
    chunk = {"text": "Original text.", "metadata": {"source": "report.pdf", "page": 1, "chunk": 0}}
    ingest_sources.build_index([chunk], "unused", "test", False)
    before = dict(collection.records)
    collection.fail_upsert = True
    with pytest.raises(RuntimeError, match="vector write failure"):
        ingest_sources.build_index([{**chunk, "text": "Updated text."}], "unused", "test", False)
    assert collection.records == before


def test_index_rejects_incompatible_embedding_model(monkeypatch):
    import pytest
    import ingest_sources

    collection = FakeCollection()
    collection.metadata["embedding_model"] = "different-model"
    install_index_fakes(monkeypatch, collection)
    chunk = {"text": "Text.", "metadata": {"source": "report.pdf", "page": 1, "chunk": 0}}
    with pytest.raises(ValueError, match="embedding configuration"):
        ingest_sources.build_index([chunk], "unused", "test", False)
    assert collection.count() == 0


def test_retrieval_preserves_distinct_long_chunks_and_limits_results(monkeypatch):
    import rag_core
    from types import SimpleNamespace

    prefix = "Shared prefix " * 20
    text = [prefix + "First fact.", prefix + "Second fact.", "Third fact."]
    responses = iter([
        {"documents": [[text[0], text[1]]],
         "metadatas": [[{"source": "a", "page": 1, "chunk": number} for number in range(2)]],
         "distances": [[0.1, 0.2]]},
        {"documents": [[text[2]]], "metadatas": [[{"source": "b", "page": 1, "chunk": 0}]],
         "distances": [[0.3]]},
    ])
    monkeypatch.setenv("RAG_TOP_K", "2")
    monkeypatch.setenv("RAG_MAX_DISTANCE", "1.6")
    collection = SimpleNamespace(query=lambda **kwargs: next(responses))
    results = rag_core.retrieve("Q4 revenue?", collection, FakeModel(MODEL_NAME), 3)
    assert [row[0] for row in results] == text[:2]


def test_empty_evidence_never_calls_answer_service(monkeypatch):
    import rag_core

    def unexpected_client():
        raise AssertionError("No API request should be made without evidence")

    monkeypatch.setattr(rag_core, "_create_client", unexpected_client)
    assert rag_core.generate_answer("Question?", []) == rag_core.NOT_FOUND_ANSWER


def test_connected_pdf_to_answer_contract(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    import ingest_sources
    import rag_core
    from pdf_pipeline.collector import collect_results
    from pdf_pipeline.models import PageResult

    pipeline = ModuleType("pdf_pipeline.main")
    fact = "The ORBIT42 project budget is 731 credits."

    def fake_pipeline(**kwargs):
        return collect_results(kwargs["document_id"], [
            PageResult("page:1", kwargs["document_id"], 0, fact, True),
        ], kwargs["output_root"], write_outputs=kwargs["write_outputs"])

    pipeline.run_pipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "pdf_pipeline.main", pipeline)
    collection = FakeCollection()
    model = install_index_fakes(monkeypatch, collection)
    source = tmp_path / "budget.pdf"
    records = ingest_sources.load_pdf(source)
    chunks = ingest_sources.chunk_records(records)
    assert ingest_sources.build_index(chunks, str(tmp_path / "index"), "test", False) == 1
    monkeypatch.setenv("RAG_TOP_K", "8")
    monkeypatch.setenv("RAG_MAX_DISTANCE", "1.6")
    question = "What is the ORBIT42 project budget?"
    evidence = rag_core.retrieve(question, collection, model, collection.count())
    assert len(evidence) == 1
    assert evidence[0][0] == fact
    assert evidence[0][1]["source"] == str(source.resolve())
    assert evidence[0][1]["page"] == 1

    def fake_response(**kwargs):
        assert fact in kwargs["input"]
        assert question in kwargs["input"]
        assert f"{source.resolve()}, page 1" in kwargs["input"]
        assert "untrusted evidence" in kwargs["instructions"]
        assert kwargs["model"] == "test-deployment"
        return SimpleNamespace(output_text="## RAG Answer\n\n731 credits.\n\n**Source:** budget.pdf, page 1.")

    for setting in rag_core.REQUIRED_AZURE_SETTINGS:
        monkeypatch.setenv(setting, "test-deployment")
    monkeypatch.setattr(rag_core, "_create_client", lambda: SimpleNamespace(
        responses=SimpleNamespace(create=fake_response),
    ))
    answer = rag_core.generate_answer(question, evidence)
    assert "731 credits" in answer
    assert "budget.pdf, page 1" in answer


def test_live_azure_answer_with_synthetic_evidence():
    import os
    import pytest
    from rag_core import generate_answer

    if os.environ.get("RUN_RAG_AZURE") != "1":
        pytest.skip("Set RUN_RAG_AZURE=1 to send synthetic evidence to the configured Azure deployment")
    evidence = [("The ORBIT42 project budget is 731 credits.",
                 {"source": "synthetic-budget.pdf", "page": 1, "chunk": 0}, 0.1)]
    try:
        answer = generate_answer("What is the ORBIT42 project budget?", evidence)
    except Exception as error:
        pytest.fail(f"Azure live request failed ({type(error).__name__}); credentials and response details hidden", pytrace=False)
    assert "731" in answer
    assert "Source:" in answer
    assert "synthetic-budget.pdf" in answer


def test_chat_refreshes_index_count_for_each_question(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    import chat_rag

    counts = iter([1, 4, 8])
    collection = SimpleNamespace(count=lambda: next(counts))
    questions = iter(["first question", "second question", "exit"])
    seen = []
    transformers = ModuleType("sentence_transformers")
    transformers.SentenceTransformer = lambda name: object()
    monkeypatch.setitem(sys.modules, "sentence_transformers", transformers)
    monkeypatch.setattr(chat_rag, "parse_args", lambda: SimpleNamespace(database="unused", collection="unused"))
    monkeypatch.setattr(chat_rag, "load_collection", lambda *args: collection)
    monkeypatch.setattr("builtins.input", lambda prompt: next(questions))
    monkeypatch.setattr(chat_rag, "answer_question", lambda question, collection, model, count: seen.append(count))
    chat_rag.main()
    assert seen == [4, 8]


if __name__ == "__main__":
    unittest.main()
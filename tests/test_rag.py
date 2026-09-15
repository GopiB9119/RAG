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
        from rag_core import open_published_collection

        cls.collection = open_published_collection(
            client.get_collection(name=DEFAULT_COLLECTION), DEFAULT_DATABASE, DEFAULT_COLLECTION,
        )
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


class FakeTokenizer:
    model_max_length = 256

    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is True and truncation is False
        return [0, *text.split(), 1]


class FakeModel:
    def __init__(self, name):
        assert name == MODEL_NAME
        self.batches = []
        self.tokenizer = FakeTokenizer()
        self.max_seq_length = 256

    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        self.batches.append(texts)
        return FakeVectors([[1.0, 0.0] for text in texts])


class FakeCollection:
    def __init__(self):
        from rag_core import COLLECTION_METADATA

        self.metadata = dict(COLLECTION_METADATA)
        self.id = "test-collection-id"
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
                        if all(metadata.get(key) == value for key, value in where.items())]}

    def delete(self, ids):
        for record_id in ids:
            del self.records[record_id]

    def query(self, query_embeddings, n_results, include, where_document=None, where=None):
        assert query_embeddings == [[1.0, 0.0]]
        rows = list(self.records.values())
        if where:
            rows = [row for row in rows if row[1].get("_rag_revision") in where["_rag_revision"]["$in"]]
        rows = rows[:n_results]
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


def test_index_deduplicates_batches_and_hides_stale_source_chunks(monkeypatch, tmp_path):
    import ingest_sources
    from rag_core import open_published_collection

    collection = FakeCollection()
    model = install_index_fakes(monkeypatch, collection)
    monkeypatch.setattr(ingest_sources, "UPSERT_BATCH_SIZE", 1)
    old = {"text": "Old revenue.", "metadata": {"source": "report.pdf", "page": 1, "chunk": 0}}
    other = {"text": "Other document.", "metadata": {"source": "other.pdf", "page": 1, "chunk": 0}}
    database = str(tmp_path / "index")
    ingest_sources.build_index([old, other], database, "test", False)
    new = {**old, "text": "Updated revenue."}
    assert ingest_sources.build_index([new, new], database, "test", False) == 1
    view = open_published_collection(collection, database, "test")
    assert view.count() == 2
    assert set(view.query(query_embeddings=[[1.0, 0.0]], n_results=8, include=[])["documents"][0]) == {"Updated revenue.", "Other document."}
    assert collection.count() == 3
    assert all(len(batch) == 1 for batch in model.batches)


def test_index_failure_keeps_old_source_chunks(monkeypatch, tmp_path):
    import pytest
    import ingest_sources

    collection = FakeCollection()
    install_index_fakes(monkeypatch, collection)
    chunk = {"text": "Original text.", "metadata": {"source": "report.pdf", "page": 1, "chunk": 0}}
    database = str(tmp_path / "index")
    ingest_sources.build_index([chunk], database, "test", False)
    before = dict(collection.records)
    collection.fail_upsert = True
    with pytest.raises(RuntimeError, match="vector write failure"):
        ingest_sources.build_index([{**chunk, "text": "Updated text."}], database, "test", False)
    assert collection.records == before


def test_index_rejects_incompatible_embedding_model(monkeypatch, tmp_path):
    import pytest
    import ingest_sources

    collection = FakeCollection()
    collection.metadata["embedding_model"] = "different-model"
    install_index_fakes(monkeypatch, collection)
    chunk = {"text": "Text.", "metadata": {"source": "report.pdf", "page": 1, "chunk": 0}}
    with pytest.raises(ValueError, match="embedding configuration"):
        ingest_sources.build_index([chunk], str(tmp_path / "index"), "test", False)
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
    view = rag_core.open_published_collection(collection, str(tmp_path / "index"), "test")
    evidence = rag_core.retrieve(question, view, model, view.count())
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


def test_failed_second_batch_never_publishes_partial_document(tmp_path, monkeypatch):
    import pytest
    import ingest_sources
    from rag_core import open_published_collection

    collection = FakeCollection()
    install_index_fakes(monkeypatch, collection)
    database = str(tmp_path / "index")
    old = {"text": "Old complete content.", "metadata": {"source": "report.pdf", "page": 1, "chunk": 0}}
    ingest_sources.build_index([old], database, "documents", False)
    view = open_published_collection(collection, database, "documents")
    pinned = view.snapshot()
    original_upsert = collection.upsert
    batches = []

    def interrupted_upsert(**kwargs):
        batches.append(kwargs)
        assert view.query(query_embeddings=[[1.0, 0.0]], n_results=8, include=[])["documents"][0] == [old["text"]]
        if len(batches) == 2:
            raise RuntimeError("Second batch failed")
        original_upsert(**kwargs)

    monkeypatch.setattr(ingest_sources, "UPSERT_BATCH_SIZE", 1)
    monkeypatch.setattr(collection, "upsert", interrupted_upsert)
    updated = [{"text": f"New chunk {number}.", "metadata": {**old["metadata"], "chunk": number}} for number in range(2)]
    with pytest.raises(RuntimeError, match="Second batch"):
        ingest_sources.build_index(updated, database, "documents", False)
    assert view.count() == 1
    assert collection.count() == 2
    monkeypatch.setattr(collection, "upsert", original_upsert)
    ingest_sources.build_index(updated, database, "documents", False)
    assert view.count() == 2
    assert set(view.query(query_embeddings=[[1.0, 0.0]], n_results=8, include=[])["documents"][0]) == {chunk["text"] for chunk in updated}
    assert pinned.query(query_embeddings=[[1.0, 0.0]], n_results=8, include=[])["documents"][0] == [old["text"]]


def test_first_publication_failure_can_retry_without_reset(tmp_path, monkeypatch):
    import pytest
    import ingest_sources
    from rag_core import open_published_collection

    collection = FakeCollection()
    install_index_fakes(monkeypatch, collection)
    database = str(tmp_path / "index")
    chunks = [{"text": f"Page {number}.", "metadata": {"source": "report.pdf", "page": number, "chunk": 0}}
              for number in (1, 2)]
    original_upsert = collection.upsert
    calls = []

    def interrupted(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("Interrupted")
        original_upsert(**kwargs)

    monkeypatch.setattr(ingest_sources, "UPSERT_BATCH_SIZE", 1)
    monkeypatch.setattr(collection, "upsert", interrupted)
    with pytest.raises(RuntimeError):
        ingest_sources.build_index(chunks, database, "documents", False)
    view = open_published_collection(collection, database, "documents")
    assert view.count() == 0
    assert view.query(query_embeddings=[[1.0, 0.0]], n_results=8, include=[])["documents"] == [[]]
    monkeypatch.setattr(collection, "upsert", original_upsert)
    assert ingest_sources.build_index(chunks, database, "documents", False) == 2
    assert view.count() == 2


def test_missing_staged_vector_prevents_publication(tmp_path, monkeypatch):
    import pytest
    import ingest_sources
    from rag_core import open_published_collection

    collection = FakeCollection()
    install_index_fakes(monkeypatch, collection)
    monkeypatch.setattr(collection, "upsert", lambda **kwargs: None)
    chunk = {"text": "Text.", "metadata": {"source": "report.pdf", "page": 1, "chunk": 0}}
    database = str(tmp_path / "index")
    with pytest.raises(RuntimeError, match="every expected chunk"):
        ingest_sources.build_index([chunk], database, "documents", False)
    assert open_published_collection(collection, database, "documents").count() == 0


def test_retrieval_pins_both_passes_during_publication(tmp_path, monkeypatch):
    import ingest_sources
    import rag_core

    collection = FakeCollection()
    model = install_index_fakes(monkeypatch, collection)
    database = str(tmp_path / "index")
    old = {"text": "Q4 old facts.", "metadata": {"source": "report.pdf", "page": 1, "chunk": 0}}
    new = {**old, "text": "Q4 new facts."}
    ingest_sources.build_index([old], database, "documents", False)
    view = rag_core.open_published_collection(collection, database, "documents")
    original_query = collection.query
    filters = []

    def query(**kwargs):
        filters.append(kwargs["where"])
        response = original_query(**kwargs)
        if len(filters) == 1:
            ingest_sources.build_index([new], database, "documents", False)
        return response

    monkeypatch.setenv("RAG_TOP_K", "8")
    monkeypatch.setenv("RAG_MAX_DISTANCE", "1.6")
    monkeypatch.setattr(collection, "query", query)
    evidence = rag_core.retrieve("Q4 facts?", view, model, view.count())
    assert len(filters) == 2 and filters[0] == filters[1]
    assert [row[0] for row in evidence] == [old["text"]]
    assert [row[0] for row in rag_core.retrieve("Q4 facts?", view, model, view.count())] == [new["text"]]


def test_raw_versioned_collection_cannot_bypass_filtering():
    import pytest

    with pytest.raises(ValueError, match="open_published_collection"):
        retrieve("Question?", FakeCollection(), FakeModel(MODEL_NAME), 1)


def test_index_embeds_all_token_parts_without_truncation(tmp_path, monkeypatch):
    import ingest_sources
    import rag_core

    collection = FakeCollection()
    model = install_index_fakes(monkeypatch, collection)
    model.max_seq_length = 6
    original_encode = model.encode

    def checked_encode(texts, **kwargs):
        assert all(len(model.tokenizer.encode(text, add_special_tokens=True, truncation=False)) <= 6 for text in texts)
        return original_encode(texts, **kwargs)

    model.encode = checked_encode
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    chunk = {"text": text, "metadata": {"source": "report.pdf", "page": 9, "chunk": 2}}
    database = str(tmp_path / "index")
    count = ingest_sources.build_index([chunk], database, "documents", False)
    assert count > 1
    stored = sorted(collection.records.values(), key=lambda record: record[1]["token_part"])
    assert "".join(record[0] for record in stored) == text
    assert all(record[1]["page"] == 9 and record[1]["chunk"] == 2 for record in stored)
    view = rag_core.open_published_collection(collection, database, "documents")
    monkeypatch.setenv("RAG_TOP_K", "20")
    monkeypatch.setenv("RAG_MAX_DISTANCE", "1.6")
    results = rag_core.retrieve("budget?", view, model, count)
    assert len(results) == count


def test_impossible_embedding_budget_does_not_modify_index(tmp_path, monkeypatch):
    import pytest
    import ingest_sources

    collection = FakeCollection()
    model = install_index_fakes(monkeypatch, collection)
    model.max_seq_length = 1
    with pytest.raises(ValueError, match="special tokens"):
        ingest_sources.build_index([{"text": "text", "metadata": {"source": "report.pdf", "page": 1, "chunk": 0}}],
                                   str(tmp_path / "index"), "documents", True)
    assert collection.count() == 0
    assert not (tmp_path / "index").exists()


def test_publication_receipt_is_atomic_and_replay_does_not_reindex(tmp_path, monkeypatch):
    import ingest_sources

    collection = FakeCollection()
    model = install_index_fakes(monkeypatch, collection)
    chunk = {"text": "Original fact.", "metadata": {"source": "report.pdf", "page": 1, "chunk": 0}}
    database = str(tmp_path / "index")
    generation = "a" * 32
    receipt = ingest_sources.build_index([chunk], database, "documents", False, publication_generation=generation)
    assert receipt["generation"] == generation
    assert receipt["collection_identity"] == collection.id
    assert receipt["source"] == "report.pdf" and receipt["chunk_count"] == 1
    encoded = len(model.batches)
    repeated = ingest_sources.build_index([chunk], database, "documents", False, publication_generation=generation)
    assert repeated == receipt and len(model.batches) == encoded
    assert collection.count() == 1


def test_receipt_replay_cannot_roll_back_newer_publication(tmp_path, monkeypatch):
    import pytest
    import ingest_sources
    from rag_core import open_published_collection

    collection = FakeCollection()
    install_index_fakes(monkeypatch, collection)
    database = str(tmp_path / "index")
    original = {"text": "Original fact.", "metadata": {"source": "report.pdf", "page": 1, "chunk": 0}}
    newer = {**original, "text": "New fact."}
    first = ingest_sources.build_index([original], database, "documents", False, publication_generation="a" * 32)
    ingest_sources.build_index([newer], database, "documents", False, publication_generation="b" * 32)
    assert ingest_sources.build_index([original], database, "documents", False, publication_generation="a" * 32) == first
    view = open_published_collection(collection, database, "documents")
    assert view.query(query_embeddings=[[1.0, 0.0]], n_results=2, include=[])["documents"] == [["New fact."]]
    with pytest.raises(ValueError, match="reused"):
        ingest_sources.build_index([newer], database, "documents", False, publication_generation="a" * 32)


def test_receipt_is_not_saved_when_vector_write_fails(tmp_path, monkeypatch):
    import pytest
    import ingest_sources
    from index_publication import PublicationStore

    collection = FakeCollection()
    install_index_fakes(monkeypatch, collection)
    collection.fail_upsert = True
    database = str(tmp_path / "index")
    with pytest.raises(RuntimeError):
        ingest_sources.build_index([{"text": "fact", "metadata": {"source": "r.pdf", "page": 1, "chunk": 0}}],
                                   database, "documents", False, publication_generation="a" * 32)
    with PublicationStore(database).connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM publication_receipts").fetchone()[0] == 0


def test_receipt_insert_failure_rolls_back_active_revision(tmp_path, monkeypatch):
    import pytest
    import ingest_sources
    from index_publication import PublicationStore
    from rag_core import open_published_collection

    collection = FakeCollection()
    install_index_fakes(monkeypatch, collection)
    database = str(tmp_path / "index")
    chunk = {"text": "Original fact.", "metadata": {"source": "report.pdf", "page": 1, "chunk": 0}}
    ingest_sources.build_index([chunk], database, "documents", False)

    def failed_receipt(*args, **kwargs):
        raise RuntimeError("Receipt transaction failed")

    monkeypatch.setattr(PublicationStore, "record_receipt", failed_receipt)
    with pytest.raises(RuntimeError, match="Receipt transaction"):
        ingest_sources.build_index([{**chunk, "text": "New fact."}], database, "documents", False,
                                   publication_generation="a" * 32)
    view = open_published_collection(collection, database, "documents")
    assert view.query(query_embeddings=[[1.0, 0.0]], n_results=2, include=[])["documents"] == [["Original fact."]]
    with PublicationStore(database).connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM publication_receipts").fetchone()[0] == 0


def test_shared_document_publisher_reuses_model_and_committed_receipt(tmp_path, monkeypatch):
    from ingest_sources import DocumentPublisher

    collection = FakeCollection()
    model = install_index_fakes(monkeypatch, collection)
    publisher = DocumentPublisher(str(tmp_path / "index"), "documents")
    records = [{"text": "Page text.", "metadata": {"source": "report.pdf", "page": 1}}]
    first = publisher.publish(records, "a" * 32)
    assert first["source"] == "report.pdf" and first["generation"] == "a" * 32
    assert publisher.model is model
    encoded_batches = len(model.batches)
    assert publisher.publish(records, "a" * 32) == first
    assert len(model.batches) == encoded_batches
    assert collection.count() == 1


def test_shared_publisher_rejects_mixed_sources_before_loading_model(tmp_path):
    import pytest
    from ingest_sources import DocumentPublisher

    publisher = DocumentPublisher(str(tmp_path / "index"), "documents")
    records = [{"text": "Text", "metadata": {"source": source, "page": 1}} for source in ("a.pdf", "b.pdf")]
    with pytest.raises(ValueError, match="exactly one"):
        publisher.publish(records, "a" * 32)
    assert publisher.model is None and not (tmp_path / "index").exists()


if __name__ == "__main__":
    unittest.main()
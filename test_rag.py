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


if __name__ == "__main__":
    unittest.main()
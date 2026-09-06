"""Interactive chat over the indexed collection."""

from __future__ import annotations

import argparse
import logging

import chromadb
from sentence_transformers import SentenceTransformer

from rag_core import DEFAULT_COLLECTION, DEFAULT_DATABASE, MODEL_NAME, answer_question

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the indexed PDF and web collection.")
    parser.add_argument("--database", default=DEFAULT_DATABASE, help="Chroma database folder")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name")
    return parser.parse_args()


def load_collection(database: str, name: str):
    """Open an existing, non-empty collection or exit with a helpful message."""
    client = chromadb.PersistentClient(path=database)
    try:
        collection = client.get_collection(name=name)
    except Exception as error:
        raise SystemExit(
            f"Collection '{name}' was not found. Run ingest_sources.py first."
        ) from error
    if not collection.count():
        raise SystemExit("The collection is empty. Run ingest_sources.py first.")
    return collection


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    collection = load_collection(args.database, args.collection)
    chunk_count = collection.count()
    print(f"Loaded {chunk_count} searchable chunks from '{args.collection}'.")
    print("Ask questions about the indexed sources. Type 'exit' to stop.")

    model = SentenceTransformer(MODEL_NAME)
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if question:
            answer_question(question, collection, model, chunk_count)


if __name__ == "__main__":
    main()
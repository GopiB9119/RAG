"""Interactive chat over the indexed collection."""

from __future__ import annotations

import argparse
import logging

from rag_core import DEFAULT_COLLECTION, DEFAULT_DATABASE, MODEL_NAME, answer_question, open_published_collection

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the indexed PDF and web collection.")
    parser.add_argument("--database", default=DEFAULT_DATABASE, help="Chroma database folder")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name")
    return parser.parse_args()


def load_collection(database: str, name: str):
    """Open an existing, non-empty collection or exit with a helpful message."""
    import chromadb

    client = chromadb.PersistentClient(path=database)
    try:
        collection = client.get_collection(name=name)
    except Exception as error:
        raise SystemExit(
            f"Collection '{name}' was not found. Run ingest_sources.py first."
        ) from error
    collection = open_published_collection(collection, database, name)
    if not collection.count():
        raise SystemExit("The collection is empty. Run ingest_sources.py first.")
    # The view counts and searches only committed revisions, not staging vectors.
    return collection


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    collection = load_collection(args.database, args.collection)
    from sentence_transformers import SentenceTransformer

    chunk_count = collection.count()
    print(f"Loaded {chunk_count} searchable chunks from '{args.collection}'.")
    print("Ask questions about the indexed sources. Type 'exit' to stop.")

    # Load once per chat process, not once per question. This is the same embedding
    # model used at indexing time; Azure is the separate answer-generation model.
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
            # Each question is currently independent: this CLI does not pass chat
            # history or resolve follow-ups such as "what about the previous one?".
            # The watcher can change the index after chat starts. Refresh the count
            # rather than limiting later searches to the startup snapshot count.
            answer_question(question, collection, model, collection.count())


if __name__ == "__main__":
    main()
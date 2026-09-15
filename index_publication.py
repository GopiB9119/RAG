"""Local atomic publication pointers over immutable Chroma document revisions."""

from __future__ import annotations

import sqlite3
import json
import uuid
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any


REVISION_FIELD = "_rag_revision"
PUBLICATION_SCHEMA = 2


class ReaderLease:
    """No automatic expiry: an abandoned lease retains data rather than risking deletion."""

    def __init__(self, store, token: str):
        self.store = store
        self.token = token
        self.closed = False
        self.finalizer = weakref.finalize(self, self._release, store, token)

    @staticmethod
    def _release(store, token):
        try:
            with store.reader_registry() as connection:
                connection.execute("DELETE FROM readers WHERE token=?", (token,))
        except sqlite3.Error:
            # Failed best-effort release leaves a protective lease, never a gap.
            pass

    def close(self):
        if not self.closed:
            self.closed = True
            self.finalizer()


class PublicationStore:
    def __init__(self, database: str):
        self.path = Path(database).resolve() / "rag_publications.sqlite3"

    @contextmanager
    def reader_registry(self):
        # Independent DB: readers can register while a vector writer holds the
        # publication transaction. This lock serializes registration with cleanup.
        if not self.path.exists():
            raise ValueError("Publication state is missing")
        connection = sqlite3.connect(self.path.with_name("rag_readers.sqlite3"), timeout=10)
        try:
            connection.row_factory = sqlite3.Row
            with connection:
                connection.execute("""CREATE TABLE IF NOT EXISTS readers (
                    token TEXT PRIMARY KEY, collection TEXT NOT NULL,
                    identity TEXT NOT NULL, revisions TEXT NOT NULL
                )""")
                connection.execute("BEGIN IMMEDIATE")
                yield connection
        finally:
            connection.close()

    def acquire_snapshot(self, name: str, identity: str):
        with self.reader_registry() as connection:
            state = self.snapshot(name, identity)
            token = uuid.uuid4().hex
            connection.execute("INSERT INTO readers VALUES (?,?,?,?)",
                               (token, name, identity, json.dumps(state[0])))
        return state, ReaderLease(self, token)

    @contextmanager
    def connect(self, create: bool = False):
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=10)
        else:
            # Do not silently create empty publication state when opening an index.
            connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=10)
        try:
            connection.row_factory = sqlite3.Row
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect(create=True) as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY,
                    identity TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS active_documents (
                    collection TEXT NOT NULL,
                    source TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL CHECK(chunk_count > 0),
                    PRIMARY KEY(collection, source)
                );
                CREATE TABLE IF NOT EXISTS publication_receipts (
                    collection TEXT NOT NULL, identity TEXT NOT NULL,
                    generation TEXT NOT NULL, input_hash TEXT NOT NULL,
                    receipt TEXT NOT NULL,
                    PRIMARY KEY(collection, identity, generation)
                );
            """)

    @contextmanager
    def writer(self):
        self.initialize()
        with self.connect(create=True) as connection:
            # All application writers for this database serialize here. Readers
            # continue seeing the old committed pointers while vectors are staged.
            connection.execute("BEGIN IMMEDIATE")
            yield connection

    def bind(self, connection, name: str, collection: Any, reset: bool) -> None:
        identity = str(collection.id)
        row = connection.execute("SELECT identity FROM collections WHERE name=?", (name,)).fetchone()
        if row is not None and row["identity"] == identity:
            return
        if (row is not None or collection.count() > 0) and not reset:
            raise ValueError("Publication state is missing or belongs to another collection; rebuild with --reset while readers are stopped")
        connection.execute("DELETE FROM active_documents WHERE collection=?", (name,))
        connection.execute("INSERT OR REPLACE INTO collections VALUES (?,?)", (name, identity))

    def publish(self, connection, name: str, source: str, revision: str, count: int) -> None:
        connection.execute(
            """INSERT INTO active_documents VALUES (?,?,?,?)
               ON CONFLICT(collection,source) DO UPDATE SET
               revision=excluded.revision, chunk_count=excluded.chunk_count""",
            (name, source, revision, count),
        )

    def receipt(self, connection, name: str, identity: str, generation: str, input_hash: str):
        row = connection.execute(
            "SELECT input_hash,receipt FROM publication_receipts WHERE collection=? AND identity=? AND generation=?",
            (name, identity, generation),
        ).fetchone()
        if row is None:
            return None
        if row["input_hash"] != input_hash:
            raise ValueError("Processing generation was reused for different index input")
        return json.loads(row["receipt"])

    def record_receipt(self, connection, name: str, identity: str, generation: str,
                       input_hash: str, source: str, revision: str, count: int) -> dict:
        receipt = {"schema": 1, "database": str(self.path.parent), "collection": name,
                   "collection_identity": identity, "generation": generation,
                   "input_hash": input_hash, "source": source, "revision": revision,
                   "chunk_count": count, "publication_schema": PUBLICATION_SCHEMA}
        connection.execute("INSERT INTO publication_receipts VALUES (?,?,?,?,?)",
                           (name, identity, generation, input_hash, json.dumps(receipt, sort_keys=True)))
        return receipt

    def verify_receipt(self, receipt: dict) -> None:
        if receipt.get("database") != str(self.path.parent):
            raise ValueError("Receipt database does not match publication store")
        with self.connect() as connection:
            connection.execute("BEGIN")
            identity = connection.execute("SELECT identity FROM collections WHERE name=?",
                                          (receipt["collection"],)).fetchone()
            if identity is None or identity["identity"] != receipt["collection_identity"]:
                raise ValueError("Receipt belongs to an unknown or replaced index")
            stored = self.receipt(connection, receipt["collection"], receipt["collection_identity"],
                                  receipt["generation"], receipt["input_hash"])
            if stored is None or stored != receipt:
                raise ValueError("Receipt is not committed in the publication store")

    def snapshot(self, name: str, identity: str) -> tuple[list[str], int]:
        with self.connect() as connection:
            connection.execute("BEGIN")
            row = connection.execute("SELECT identity FROM collections WHERE name=?", (name,)).fetchone()
            if row is None or row["identity"] != identity:
                raise ValueError("Published index identity is missing or changed; reopen after a controlled rebuild")
            rows = connection.execute(
                "SELECT revision,chunk_count FROM active_documents WHERE collection=? ORDER BY source", (name,)
            ).fetchall()
            return [row["revision"] for row in rows], sum(row["chunk_count"] for row in rows)


def verify_publication_receipt(receipt: dict, *, database: str, collection: str,
                               source: str, generation: str) -> int:
    """Verify shared completion evidence, never trusting a callback's count alone."""
    if (not isinstance(receipt, dict) or type(receipt.get("schema")) is not int or receipt["schema"] != 1
            or receipt.get("publication_schema") != PUBLICATION_SCHEMA
            or receipt.get("generation") != generation or receipt.get("source") != source
            or type(receipt.get("chunk_count")) is not int or receipt["chunk_count"] < 1
            or receipt.get("database") != str(Path(database).resolve())
            or receipt.get("collection") != collection
            or not all(isinstance(receipt.get(key), str) and receipt[key]
                       for key in ("collection_identity", "revision", "input_hash"))):
        raise ValueError("Index receipt does not match the processing generation or target")
    PublicationStore(database).verify_receipt(receipt)
    return receipt["chunk_count"]


class PublishedCollection:
    """Chroma-compatible read view that never searches uncommitted revisions."""

    def __init__(self, collection: Any, database: str, name: str,
                 pinned: tuple[list[str], int] | None = None, lease: ReaderLease | None = None):
        self.collection = collection
        self.store = PublicationStore(database)
        self.name = name
        self.pinned = pinned
        self.lease = lease

    @property
    def metadata(self):
        return self.collection.metadata

    def snapshot(self):
        if self.pinned is not None:
            if self.lease is None or self.lease.closed:
                raise ValueError("Published snapshot is closed or has no reader lease")
            return self
        state, lease = self.store.acquire_snapshot(self.name, str(self.collection.id))
        return PublishedCollection(self.collection, str(self.store.path.parent), self.name, state, lease)

    def close(self):
        if self.lease is not None:
            self.lease.close()

    @contextmanager
    def read_view(self):
        view = self.snapshot()
        try:
            yield view
        finally:
            if view is not self:
                view.close()

    def count(self) -> int:
        with self.read_view() as view:
            return view.pinned[1]

    def query(self, **arguments):
        with self.read_view() as view:
            revisions, count = view.pinned
            if not revisions:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
            revision_filter = {REVISION_FIELD: {"$in": revisions}}
            existing_filter = arguments.get("where")
            arguments["where"] = {"$and": [existing_filter, revision_filter]} if existing_filter else revision_filter
            arguments["n_results"] = min(arguments["n_results"], count)
            return self.collection.query(**arguments)

    def get(self, **arguments):
        with self.read_view() as view:
            revisions, _ = view.pinned
            if not revisions:
                return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
            revision_filter = {REVISION_FIELD: {"$in": revisions}}
            existing_filter = arguments.get("where")
            arguments["where"] = {"$and": [existing_filter, revision_filter]} if existing_filter else revision_filter
            return self.collection.get(**arguments)


def cleanup_revisions(collection: Any, database: str, name: str, *,
                      apply: bool = False, batch_size: int = 500) -> dict:
    """Delete only unreferenced revisions; defaults to a read-only vector preview."""
    from rag_core import validate_collection

    if type(batch_size) is not int or not 1 <= batch_size <= 1000:
        raise ValueError("batch_size must be between 1 and 1000")
    validate_collection(collection)
    store = PublicationStore(database)
    # Missing state must fail before initialize() can create an empty database.
    store.snapshot(name, str(collection.id))
    result = {"apply": apply, "candidate_chunks": 0, "deleted_chunks": 0,
              "protected_revisions": 0, "reader_leases": 0, "legacy_chunks": 0}
    # Lock order: writer then reader registry. Registration reads committed state,
    # so it can finish while BEGIN IMMEDIATE reserves the publication writer.
    # Once both locks are held, no writer or new snapshot can race the deletion.
    with store.writer() as writer, store.reader_registry() as registry:
        identity = writer.execute("SELECT identity FROM collections WHERE name=?", (name,)).fetchone()
        if identity is None or identity["identity"] != str(collection.id):
            raise ValueError("Collection identity changed before cleanup")
        protected = {row["revision"] for row in writer.execute(
            "SELECT revision FROM active_documents WHERE collection=?", (name,)
        )}
        for lease in registry.execute("SELECT revisions FROM readers WHERE collection=? AND identity=?",
                                      (name, str(collection.id))):
            revisions = json.loads(lease["revisions"])
            if not isinstance(revisions, list) or not all(isinstance(value, str) for value in revisions):
                raise ValueError("Malformed reader lease; cleanup refused")
            protected.update(revisions)
            result["reader_leases"] += 1
        result["protected_revisions"] = len(protected)
        candidates: dict[str, int] = {}
        offset = 0
        while True:
            page = collection.get(limit=batch_size, offset=offset, include=["metadatas"])
            ids, metadatas = page["ids"], page["metadatas"]
            if len(ids) != len(metadatas):
                raise ValueError("Invalid collection scan response")
            if not ids:
                break
            for metadata in metadatas:
                revision = (metadata or {}).get(REVISION_FIELD)
                if not isinstance(revision, str) or not revision:
                    result["legacy_chunks"] += 1
                elif revision not in protected:
                    candidates[revision] = candidates.get(revision, 0) + 1
            offset += len(ids)
        result["candidate_chunks"] = sum(candidates.values())
        if apply:
            for revision in sorted(candidates):
                # Restart at offset 0 after deleting: advancing an offset would
                # skip rows as the result set shrinks. Use exact IDs per batch.
                while True:
                    ids = collection.get(where={REVISION_FIELD: revision}, limit=batch_size, include=[])["ids"]
                    if not ids:
                        break
                    collection.delete(ids=ids)
                    if collection.get(ids=ids, include=[])["ids"]:
                        raise RuntimeError("Vector deletion was not confirmed; cleanup stopped")
                    result["deleted_chunks"] += len(ids)
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Preview or remove unreferenced local RAG revisions")
    parser.add_argument("--database", default=str(Path(__file__).resolve().parent / "chroma_data"))
    parser.add_argument("--collection", default="rag_documents")
    parser.add_argument("--apply", action="store_true", help="Delete eligible vectors; omitted means preview only")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 1000:
        parser.error("--batch-size must be between 1 and 1000")
    if not PublicationStore(args.database).path.exists():
        raise ValueError("Publication state is missing; cleanup refused")
    import chromadb

    collection = chromadb.PersistentClient(path=args.database).get_collection(args.collection)
    print(json.dumps(cleanup_revisions(collection, args.database, args.collection,
                                       apply=args.apply, batch_size=args.batch_size)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        raise SystemExit(2)
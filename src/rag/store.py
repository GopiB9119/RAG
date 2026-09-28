"""Single-file SQLite index: documents, chunks, BM25 full text and vectors.

A document version is published in one ACID transaction (new chunks in, old
chunks out), so a reader always sees a complete version of every document.
Dense search runs over an in-memory copy of the vectors that is refreshed
incrementally inside the same read snapshot used for full-text search.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk

logger = logging.getLogger(__name__)

SCHEMA = 1
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    generation TEXT NOT NULL UNIQUE,
    input_hash TEXT NOT NULL,
    chunk_count INTEGER NOT NULL CHECK (chunk_count > 0),
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    source TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    revision INTEGER NOT NULL UNIQUE REFERENCES revisions(id),
    pages INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    content_hash TEXT,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    revision INTEGER NOT NULL REFERENCES revisions(id),
    page INTEGER NOT NULL,
    chunk INTEGER NOT NULL,
    part INTEGER NOT NULL,
    method TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_revision ON chunks(revision);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61 remove_diacritics 2'
);
"""
_STOPWORDS = frozenset(
    "a about above after again against all am an and any are as at be because been before being below "
    "between both but by can could did do does doing down during each few for from further had has have "
    "having he her here hers him his how i if in into is it its itself just me more most my no nor not "
    "now of off on once only or other our out over own please same she should show so some such tell "
    "than that the their them then there these they this those through to too under until up very was "
    "we were what when where which while who whom whose why will with would you your".split()
)
_EDGE_PUNCTUATION = "\"'`.,;:!?()[]{}<>“”‘’«»…-–—/\\|*^~+=&%$#@"


def fts_query(text: str, limit: int = 32) -> str | None:
    """Quote every term: user text can never inject FTS5 syntax, and FTS5 tokenizes any script."""
    terms: list[str] = []
    seen: set[str] = set()
    for raw in text.split():
        term = raw.strip(_EDGE_PUNCTUATION).replace('"', " ").strip()
        key = term.lower()
        if (not term or key in seen or key in _STOPWORDS or not any(ch.isalnum() for ch in term)
                or (len(term) == 1 and not term.isdigit())):
            continue
        seen.add(key)
        terms.append(term)
    return " OR ".join(f'"{term}"' for term in terms[:limit]) or None


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    text: str
    source: str
    title: str
    page: int
    chunk: int
    part: int
    method: str


class _Vectors:
    """Immutable vectors for one index version; ids are ascending within every block."""

    def __init__(self, version: int, max_id: int, blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
        self.version, self.max_id, self.blocks = version, max_id, blocks
        self.count = sum(len(block[0]) for block in blocks)

    def top(self, query: np.ndarray, n: int) -> list[tuple[int, float]]:
        if not self.count or n < 1:
            return []
        ids = np.concatenate([block[0] for block in self.blocks])
        scores = np.concatenate([block[2] @ query for block in self.blocks])
        n = min(n, len(scores))
        best = np.argpartition(-scores, n - 1)[:n]
        best = best[np.argsort(-scores[best], kind="stable")]
        return [(int(ids[index]), float(scores[index])) for index in best]

    def similarity(self, chunk_ids: list[int], query: np.ndarray) -> dict[int, float]:
        wanted = np.asarray(chunk_ids, dtype=np.int64)
        found: dict[int, float] = {}
        for ids, _, vectors in self.blocks:
            if not len(ids) or not len(wanted):
                continue
            positions = np.minimum(np.searchsorted(ids, wanted), len(ids) - 1)
            matches = ids[positions] == wanted
            for chunk_id, position in zip(wanted[matches].tolist(), positions[matches].tolist()):
                found[chunk_id] = float(vectors[position] @ query)
        return found


class Snapshot:
    """Consistent read view: vectors, full text and rows all come from one SQLite snapshot."""

    def __init__(self, connection: sqlite3.Connection, vectors: _Vectors):
        self._connection = connection
        self.vectors = vectors

    @property
    def chunk_count(self) -> int:
        return self.vectors.count

    def dense(self, query: np.ndarray, n: int) -> list[tuple[int, float]]:
        return self.vectors.top(query, n)

    def similarity(self, chunk_ids: list[int], query: np.ndarray) -> dict[int, float]:
        return self.vectors.similarity(chunk_ids, query)

    def lexical(self, text: str, n: int) -> list[tuple[int, float]]:
        match = fts_query(text)
        if match is None or n < 1:
            return []
        try:
            rows = self._connection.execute(
                "SELECT rowid, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts) LIMIT ?", (match, n)).fetchall()
        except sqlite3.OperationalError as error:
            logger.warning("Full-text search skipped: %s", error)
            return []
        return [(row[0], -row[1]) for row in rows]

    def lexical_scores(self, text: str, chunk_ids: list[int]) -> dict[int, float]:
        """BM25 for specific chunks (0 when a chunk shares no query term); same statistics as lexical()."""
        match = fts_query(text)
        if match is None or not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        try:
            rows = self._connection.execute(
                f"SELECT rowid, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ? "
                f"AND rowid IN ({placeholders})", (match, *chunk_ids)).fetchall()
        except sqlite3.OperationalError as error:
            logger.warning("Full-text scoring skipped: %s", error)
            return {}
        return {row[0]: -row[1] for row in rows}

    def fetch(self, chunk_ids: list[int]) -> dict[int, Hit]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._connection.execute(
            f"SELECT c.id, c.text, d.source, d.title, c.page, c.chunk, c.part, c.method FROM chunks c "
            f"JOIN documents d ON d.revision = c.revision WHERE c.id IN ({placeholders})", chunk_ids).fetchall()
        return {row[0]: Hit(*row) for row in rows}


class Index:
    def __init__(self, path: Path, embedding: str, dimensions: int):
        self.path = Path(path).resolve()
        self.embedding, self.dimensions = embedding, dimensions
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._vector_lock = threading.Lock()
        self._vectors = _Vectors(-1, 0, [])
        self._local = threading.local()
        self._writer = self._connect()
        self._writer.executescript(_SCHEMA_SQL)
        with self._transaction(self._writer):
            meta = dict(self._writer.execute("SELECT key, value FROM meta").fetchall())
            if not meta:
                self._writer.executemany("INSERT INTO meta VALUES (?, ?)", [
                    ("schema", str(SCHEMA)), ("index_id", uuid.uuid4().hex), ("embedding", embedding),
                    ("dimensions", str(dimensions)), ("version", "0")])
            elif meta.get("schema") != str(SCHEMA):
                raise ValueError("Index schema is not supported by this version; re-index into a new data directory")
            elif meta.get("embedding") != embedding or meta.get("dimensions") != str(dimensions):
                raise ValueError("Index was built with a different embedding model; "
                                 "re-index into a new data directory or restore the original model setting")
        self.index_id = self._writer.execute("SELECT value FROM meta WHERE key='index_id'").fetchone()[0]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        # FULL: a committed document survives power loss, matching the durable job queue.
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    @contextmanager
    def _transaction(connection: sqlite3.Connection, mode: str = "IMMEDIATE"):
        connection.execute(f"BEGIN {mode}")
        try:
            yield connection
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")

    @staticmethod
    def input_hash(title: str, chunks: list[Chunk]) -> str:
        payload = json.dumps([title, [[c.text, c.page, c.index, c.part, c.method] for c in chunks]],
                             ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _receipt(self, row) -> dict:
        return {"schema": 1, "index_id": self.index_id, "revision": row[0], "source": row[1],
                "generation": row[2], "input_hash": row[3], "chunk_count": row[4]}

    def publish(self, source: str, title: str, pages: int, chunks: list[Chunk], vectors: np.ndarray,
                generation: str, content_hash: str | None = None) -> dict:
        """Atomically replace a document's searchable content; replaying a generation is a no-op."""
        if not source or not chunks or not re.fullmatch(r"[0-9A-Za-z_-]{1,64}", generation):
            raise ValueError("Publishing needs a source, chunks and a valid generation ID")
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(chunks), self.dimensions) or not np.isfinite(vectors).all():
            raise ValueError("Vectors do not match the chunks or the index dimensions")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
            raise ValueError("Vectors must be L2-normalized")
        digest = self.input_hash(title, chunks)
        blobs = vectors.astype("<f4")
        with self._write_lock, self._transaction(self._writer) as db:
            existing = db.execute("SELECT id, source, generation, input_hash, chunk_count FROM revisions "
                                  "WHERE generation=?", (generation,)).fetchone()
            if existing is not None:
                if existing[1] != source or existing[3] != digest:
                    raise ValueError("Generation was reused for different content")
                # Never republish an old generation over a newer version of the document.
                return self._receipt(existing)
            revision = db.execute("INSERT INTO revisions (source, generation, input_hash, chunk_count, created_at) "
                                  "VALUES (?, ?, ?, ?, ?)", (source, generation, digest, len(chunks), time.time())).lastrowid
            db.executemany("INSERT INTO chunks (revision, page, chunk, part, method, text, embedding) "
                           "VALUES (?, ?, ?, ?, ?, ?, ?)",
                           [(revision, c.page, c.index, c.part, c.method, c.text, blobs[i].tobytes())
                            for i, c in enumerate(chunks)])
            db.execute("INSERT INTO chunks_fts (rowid, text) SELECT id, text FROM chunks WHERE revision=?", (revision,))
            old = db.execute("SELECT revision FROM documents WHERE source=?", (source,)).fetchone()
            if old is not None:
                self._delete_revision(db, old[0])
            db.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(source) DO UPDATE SET "
                       "title=excluded.title, revision=excluded.revision, pages=excluded.pages, "
                       "chunk_count=excluded.chunk_count, content_hash=excluded.content_hash, "
                       "updated_at=excluded.updated_at",
                       (source, title, revision, pages, len(chunks), content_hash, time.time()))
            self._bump(db)
            return self._receipt((revision, source, generation, digest, len(chunks)))

    @staticmethod
    def _bump(db: sqlite3.Connection) -> None:
        db.execute("UPDATE meta SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) WHERE key='version'")

    @staticmethod
    def _delete_revision(db: sqlite3.Connection, revision: int) -> None:
        db.execute("INSERT INTO chunks_fts (chunks_fts, rowid, text) "
                   "SELECT 'delete', id, text FROM chunks WHERE revision=?", (revision,))
        db.execute("DELETE FROM chunks WHERE revision=?", (revision,))

    def remove(self, source: str) -> bool:
        with self._write_lock, self._transaction(self._writer) as db:
            row = db.execute("SELECT revision FROM documents WHERE source=?", (source,)).fetchone()
            if row is None:
                return False
            db.execute("DELETE FROM documents WHERE source=?", (source,))
            self._delete_revision(db, row[0])
            self._bump(db)
            return True

    def _reader(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = self._local.connection = self._connect()
        return connection

    def verify_receipt(self, receipt: dict) -> int:
        """Completion evidence must be committed in THIS index, not just returned by a callback."""
        if not isinstance(receipt, dict) or receipt.get("index_id") != self.index_id:
            raise ValueError("Receipt belongs to a different or replaced index")
        row = self._reader().execute("SELECT id, source, generation, input_hash, chunk_count FROM revisions "
                                     "WHERE generation=?", (receipt.get("generation"),)).fetchone()
        if row is None or self._receipt(row) != receipt:
            raise ValueError("Receipt is not committed in the index")
        return receipt["chunk_count"]

    def _vectors_for(self, db: sqlite3.Connection, version: int) -> _Vectors | None:
        """Vectors exactly matching `version`, or None when the cache is already newer."""
        current = self._vectors
        if current.version == version:
            return current
        if current.version > version:
            return None
        with self._vector_lock:
            current = self._vectors
            if current.version >= version:
                return current if current.version == version else None
            active = np.fromiter((row[0] for row in db.execute("SELECT revision FROM documents")), dtype=np.int64)
            blocks = []
            for ids, revisions, vectors in current.blocks:
                keep = np.isin(revisions, active)
                if keep.all():
                    blocks.append((ids, revisions, vectors))
                elif keep.any():
                    blocks.append((ids[keep], revisions[keep], vectors[keep]))
            # IDs only grow, so every row added after the cached version has a larger ID.
            rows = db.execute("SELECT id, revision, embedding FROM chunks WHERE id > ? ORDER BY id",
                              (current.max_id,)).fetchall()
            max_id = current.max_id
            if rows:
                ids = np.fromiter((row[0] for row in rows), dtype=np.int64, count=len(rows))
                revisions = np.fromiter((row[1] for row in rows), dtype=np.int64, count=len(rows))
                vectors = np.frombuffer(b"".join(row[2] for row in rows), dtype="<f4").reshape(len(rows), self.dimensions)
                blocks.append((ids, revisions, vectors))
                max_id = int(ids[-1])
            if len(blocks) > 8:
                blocks = [tuple(np.concatenate([block[part] for block in blocks]) for part in range(3))]
            self._vectors = _Vectors(version, max_id, blocks)
            return self._vectors

    @contextmanager
    def snapshot(self):
        db = self._reader()
        while True:
            db.execute("BEGIN DEFERRED")
            try:
                version = int(db.execute("SELECT value FROM meta WHERE key='version'").fetchone()[0])
                vectors = self._vectors_for(db, version)
            except BaseException:
                db.execute("ROLLBACK")
                raise
            if vectors is not None:
                break
            # Another thread already cached a newer version: take a newer snapshot.
            db.execute("COMMIT")
        try:
            yield Snapshot(db, vectors)
        finally:
            db.execute("COMMIT")

    def stats(self) -> dict:
        documents, chunks, pages = self._reader().execute(
            "SELECT COUNT(*), COALESCE(SUM(chunk_count), 0), COALESCE(SUM(pages), 0) FROM documents").fetchone()
        return {"documents": documents, "chunks": chunks, "pages": pages, "embedding": self.embedding}

    def documents(self, limit: int = 100) -> list[dict]:
        rows = self._reader().execute("SELECT source, title, pages, chunk_count, updated_at FROM documents "
                                      "ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(zip(("source", "title", "pages", "chunks", "updated_at"), row)) for row in rows]

    def close(self) -> None:
        with self._write_lock:
            self._writer.execute("PRAGMA optimize")
            self._writer.close()
        reader = getattr(self._local, "connection", None)
        if reader is not None:
            reader.close()
            self._local.connection = None

"""Durable single-host ingestion queue: SQLite rows plus immutable PDF snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    snapshot TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('queued', 'running', 'ready', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL,
    available_at REAL NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    error_type TEXT,
    chunk_count INTEGER,
    receipt TEXT
);
CREATE INDEX IF NOT EXISTS jobs_due ON jobs(state, available_at, sequence);
CREATE INDEX IF NOT EXISTS jobs_source ON jobs(source, sequence);
CREATE INDEX IF NOT EXISTS jobs_content ON jobs(content_hash, state);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class JobStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.snapshots = self.root / "snapshots"
        self.checkpoints = self.root / "checkpoints"
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "jobs.sqlite3"
        with self.connect() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def consumer_lock(self):
        # The OS owns this lock while the handle is open and releases it if the process dies.
        with (self.root / "consumer.lock").open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise RuntimeError("Another ingestion worker is already running for this data directory") from None
            try:
                yield
            finally:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def enqueue(self, source: Path, max_bytes: int = 100 * 1024 * 1024, max_attempts: int = 3) -> dict:
        if max_bytes < 1 or max_attempts < 1:
            raise ValueError("Size limit and max_attempts must be positive")
        source = Path(source).resolve(strict=True)
        if source.suffix.lower() != ".pdf" or not source.is_file():
            raise ValueError("Input must be a PDF file")
        digest = hashlib.sha256()
        descriptor, temporary_name = tempfile.mkstemp(dir=self.snapshots, suffix=".partial")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
                before = os.fstat(input_file.fileno())
                if before.st_size > max_bytes:
                    raise ValueError("PDF exceeds the upload size limit")
                total = 0
                while block := input_file.read(1024 * 1024):
                    if total == 0 and not block.startswith(b"%PDF-"):
                        raise ValueError("Input does not have a PDF header")
                    total += len(block)
                    if total > max_bytes:
                        raise ValueError("PDF exceeds the upload size limit")
                    digest.update(block)
                    output.write(block)
                after = os.fstat(input_file.fileno())
                if total == 0:
                    raise ValueError("PDF is empty")
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("Input changed while it was being copied; retry when the writer has finished")
                output.flush()
                os.fsync(output.fileno())
            content_hash = digest.hexdigest()
            # Identical bytes share one snapshot; each job keeps its own source identity.
            snapshot = self.snapshots / f"{content_hash}.pdf"
            now = time.time()
            with self.connect() as connection:
                # Reserve the writer before the duplicate check so concurrent submits cannot both insert.
                connection.execute("BEGIN IMMEDIATE")
                previous = connection.execute(
                    "SELECT * FROM jobs WHERE source=? ORDER BY sequence DESC LIMIT 1", (str(source),)).fetchone()
                if previous and previous["content_hash"] == content_hash:
                    return dict(previous)
                if not snapshot.exists():
                    temporary.replace(snapshot)
                job_id = uuid.uuid4().hex
                connection.execute(
                    "INSERT INTO jobs (id, source, content_hash, snapshot, state, max_attempts, available_at, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?)",
                    (job_id, str(source), content_hash, str(snapshot), max_attempts, now, now, now))
                return dict(connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
        finally:
            temporary.unlink(missing_ok=True)

    def recover_interrupted(self) -> int:
        """Caller holds consumer_lock: leftover running jobs lost their worker; attempts are kept."""
        now = time.time()
        with self.connect() as connection:
            return connection.execute(
                "UPDATE jobs SET state=CASE WHEN attempts>=max_attempts THEN 'failed' ELSE 'queued' END, "
                "error_type='WorkerInterrupted', available_at=?, updated_at=? WHERE state='running'",
                (now, now)).rowcount

    def claim(self) -> dict | None:
        now = time.time()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # A newer version of a source waits for its older pending version; other sources proceed.
            row = connection.execute(
                "SELECT * FROM jobs AS candidate WHERE state='queued' AND available_at<=? AND NOT EXISTS ("
                "SELECT 1 FROM jobs AS older WHERE older.source=candidate.source AND older.sequence<candidate.sequence "
                "AND older.state IN ('queued', 'running')) ORDER BY sequence LIMIT 1", (now,)).fetchone()
            if row is None:
                return None
            connection.execute("UPDATE jobs SET state='running', attempts=attempts+1, updated_at=? WHERE id=?",
                               (now, row["id"]))
            return dict(connection.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone())

    def finish(self, job_id: str, chunk_count: int, receipt: dict) -> None:
        if type(chunk_count) is not int or chunk_count < 1 or receipt.get("chunk_count") != chunk_count:
            raise ValueError("A committed receipt with a positive chunk count is required")
        with self.connect() as connection:
            changed = connection.execute(
                "UPDATE jobs SET state='ready', chunk_count=?, error_type=NULL, updated_at=?, receipt=? "
                "WHERE id=? AND state='running'",
                (chunk_count, time.time(), json.dumps(receipt, sort_keys=True), job_id)).rowcount
            if changed != 1:
                raise ValueError("Job is not running")

    def fail(self, job_id: str, error_type: str, *, permanent: bool = False) -> str:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=? AND state='running'", (job_id,)).fetchone()
            if row is None:
                raise ValueError("Job is not running")
            terminal = permanent or row["attempts"] >= row["max_attempts"]
            # Store the next eligible time instead of sleeping in the worker.
            delay = min(300, 5 * 2 ** min(row["attempts"] - 1, 6))
            state = "failed" if terminal else "queued"
            connection.execute("UPDATE jobs SET state=?, error_type=?, available_at=?, updated_at=? WHERE id=?",
                               (state, error_type, time.time() + delay, time.time(), job_id))
            return state

    def release(self, job_id: str) -> None:
        """Return a claimed job that never started (operator interrupt) without spending an attempt."""
        with self.connect() as connection:
            connection.execute("UPDATE jobs SET state='queued', attempts=MAX(attempts-1, 0), updated_at=? "
                               "WHERE id=? AND state='running'", (time.time(), job_id))

    def retry(self, job_id: str) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row or row["state"] != "failed":
                raise ValueError("Only failed jobs can be retried")
            if connection.execute("SELECT 1 FROM jobs WHERE source=? AND sequence>? LIMIT 1",
                                  (row["source"], row["sequence"])).fetchone():
                # Replaying an old version after a newer one would bring back stale facts.
                raise ValueError("A newer version of this document exists; the older version is not replayed")
            connection.execute("UPDATE jobs SET state='queued', attempts=0, available_at=?, updated_at=? WHERE id=?",
                               (time.time(), time.time(), job_id))

    def list_jobs(self, limit: int = 50, state: str | None = None) -> list[dict]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self.connect() as connection:
            if state:
                rows = connection.execute("SELECT * FROM jobs WHERE state=? ORDER BY sequence DESC LIMIT ?",
                                          (state, limit))
            else:
                rows = connection.execute("SELECT * FROM jobs ORDER BY sequence DESC LIMIT ?", (limit,))
            return [dict(row) for row in rows]

    def counts(self) -> dict:
        with self.connect() as connection:
            return {row["state"]: row["count"] for row in connection.execute(
                "SELECT state, COUNT(*) AS count FROM jobs GROUP BY state")}

    def _bind(self, key: str, value: str, message: str) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            if row and row["value"] != value:
                raise ValueError(message)
            connection.execute("INSERT OR IGNORE INTO settings VALUES (?, ?)", (key, value))

    def bind_index(self, index_id: str) -> None:
        # A job that is ready in index A must never count as indexed in a rebuilt index B.
        self._bind("index_id", index_id, "This queue belongs to a different index; "
                   "remove the queue folder together with the index when rebuilding")

    def bind_extraction(self, extraction) -> None:
        from .extraction.models import EXTRACTION_VERSION

        policy = json.dumps({"version": EXTRACTION_VERSION, "options": asdict(extraction)}, sort_keys=True)
        self._bind("extraction_policy", policy, "Extraction settings changed for an existing queue; "
                   "use a new data directory to re-ingest under the new policy")

    def cleanup_ready_artifacts(self, job_id: str) -> bool:
        """Delete the retry snapshot/checkpoints of a ready job; never the original PDF."""
        from .extraction.checkpoints import RangeCheckpoints

        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not job or job["state"] != "ready":
                return False
            digest = job["content_hash"]
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Invalid stored content hash")
            if connection.execute("SELECT 1 FROM jobs WHERE content_hash=? AND state!='ready' LIMIT 1",
                                  (digest,)).fetchone():
                return False
            snapshot = self.snapshots / f"{digest}.pdf"
            checkpoints = RangeCheckpoints.content_directory(self.checkpoints, digest)
            for path in (snapshot, checkpoints):
                if path.is_symlink() or not path.resolve().is_relative_to(self.root):
                    raise OSError("Refusing cleanup outside the queue folder")
            if connection.execute("SELECT 1 FROM jobs WHERE source=? LIMIT 1", (str(snapshot.resolve()),)).fetchone():
                return False
            if checkpoints.exists():
                shutil.rmtree(checkpoints)
            snapshot.unlink(missing_ok=True)
            return True

"""Durable single-host ingestion jobs backed by SQLite and immutable snapshots."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class JobStore:
    def __init__(self, root: Path):
        # SQLite stores DOCUMENT progress. Page/range assignments are recreated
        # on retry; completed ranges are restored from separate checkpoint files.
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots = self.root / "snapshots"
        self.snapshots.mkdir(exist_ok=True)
        self.database = self.root / "jobs.sqlite3"
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    snapshot TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('queued','running','ready','failed')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL,
                    available_at REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    error_type TEXT,
                    chunk_count INTEGER
                );
                CREATE INDEX IF NOT EXISTS jobs_due ON jobs(state, available_at, sequence);
                CREATE INDEX IF NOT EXISTS jobs_source ON jobs(source, sequence);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            # This context commits on success and rolls back on exceptions.
            # Closing the connection is separate, so it belongs in finally.
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def consumer_lock(self):
        # The file's existence is not the lock. The operating system owns the lock
        # while this handle is open and releases it if the process is killed.
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
                raise RuntimeError("An ingestion consumer is already running for this job store") from None
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
        source = source.resolve(strict=True)
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
                    raise ValueError("Input changed during upload; retry once the writer has finished")
                output.flush()
                os.fsync(output.fileno())
            content_hash = digest.hexdigest()
            # Identical content can share snapshot storage. Jobs still retain their
            # original source identity so different documents keep correct citations.
            snapshot = self.snapshots / f"{content_hash}.pdf"
            now = time.time()
            with self.connect() as connection:
                # Reserve the SQLite writer before checking duplicates and inserting.
                # Otherwise two simultaneous submissions could both create a job.
                connection.execute("BEGIN IMMEDIATE")
                previous = connection.execute(
                    "SELECT * FROM jobs WHERE source=? ORDER BY sequence DESC LIMIT 1", (str(source),)
                ).fetchone()
                if previous and previous["content_hash"] == content_hash:
                    # Compare with the LATEST version: A -> B -> A needs a new job.
                    return dict(previous)
                if not snapshot.exists():
                    temporary.replace(snapshot)
                job_id = uuid.uuid4().hex
                connection.execute(
                    """INSERT INTO jobs
                       (id,source,content_hash,snapshot,state,max_attempts,available_at,created_at,updated_at)
                       VALUES (?,?,?,?,'queued',?,?,?,?)""",
                    (job_id, str(source), content_hash, str(snapshot), max_attempts, now, now, now),
                )
                return dict(connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
        finally:
            temporary.unlink(missing_ok=True)

    def recover_interrupted(self) -> int:
        # Caller must own consumer_lock(). A leftover running job means its owner
        # stopped before recording success/failure; preserve its used attempts.
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET state=CASE WHEN attempts>=max_attempts THEN 'failed' ELSE 'queued' END,
                   error_type='WorkerInterrupted', available_at=?, updated_at=? WHERE state='running'""",
                (time.time(), time.time()),
            )
            return cursor.rowcount

    def claim(self) -> dict | None:
        now = time.time()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # A source's newer version waits for its older pending version, even
            # during backoff. Unrelated sources can continue making progress.
            row = connection.execute(
                """SELECT * FROM jobs AS candidate WHERE state='queued' AND available_at<=?
                   AND NOT EXISTS (
                       SELECT 1 FROM jobs AS older WHERE older.source=candidate.source
                       AND older.sequence<candidate.sequence AND older.state IN ('queued','running')
                   ) ORDER BY sequence LIMIT 1""", (now,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE jobs SET state='running',attempts=attempts+1,updated_at=? WHERE id=?", (now, row["id"])
            )
            return dict(connection.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone())

    def finish(self, job_id: str, chunk_count: int) -> None:
        with self.connect() as connection:
            changed = connection.execute(
                """UPDATE jobs SET state='ready',chunk_count=?,error_type=NULL,updated_at=?
                   WHERE id=? AND state='running'""", (chunk_count, time.time(), job_id),
            ).rowcount
            if changed != 1:
                raise ValueError("Job is not running")

    def fail(self, job_id: str, error_type: str) -> None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=? AND state='running'", (job_id,)).fetchone()
            if row is None:
                raise ValueError("Job is not running")
            terminal = row["attempts"] >= row["max_attempts"]
            # Store the next eligible time instead of sleeping and blocking the
            # consumer. watch checks it automatically; work must be run again.
            delay = min(300, 5 * 2 ** min(row["attempts"] - 1, 6))
            connection.execute(
                "UPDATE jobs SET state=?,error_type=?,available_at=?,updated_at=? WHERE id=?",
                ("failed" if terminal else "queued", error_type, time.time() + delay, time.time(), job_id),
            )

    def list_jobs(self, limit: int = 50) -> list[dict]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM jobs ORDER BY sequence DESC LIMIT ?", (limit,)
            )]

    def counts(self) -> dict:
        with self.connect() as connection:
            return {row["state"]: row["count"] for row in connection.execute(
                "SELECT state,COUNT(*) AS count FROM jobs GROUP BY state"
            )}

    def bind_target(self, database: Path, collection: str) -> None:
        import json

        # A ready job in index A must not be treated as already indexed in index B.
        target = json.dumps([str(database.resolve()), collection])
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT value FROM settings WHERE key='target'").fetchone()
            if row and row["value"] != target:
                raise ValueError("This job store is bound to another index; use a separate state directory")
            connection.execute("INSERT OR IGNORE INTO settings VALUES ('target',?)", (target,))

    def retry(self, job_id: str) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row or row["state"] != "failed":
                raise ValueError("Only failed jobs can be retried")
            newer = connection.execute(
                "SELECT id FROM jobs WHERE source=? AND sequence>? LIMIT 1", (row["source"], row["sequence"])
            ).fetchone()
            if newer:
                # Replaying old content after a newer version could replace fresh
                # search results with obsolete facts, so refuse that operation.
                raise ValueError("A newer source version exists; do not replay the older version")
            connection.execute(
                "UPDATE jobs SET state='queued',attempts=0,available_at=?,updated_at=? WHERE id=?",
                (time.time(), time.time(), job_id),
            )

    def cleanup_ready_artifacts(self, job_id: str) -> bool:
        """Retire generated retry data, never original PDFs or indexed chunks."""
        from .checkpoints import RangeCheckpoints

        # Serialize cleanup with enqueue: a new job must not reference a snapshot
        # between our reference check and deletion. Caller also owns consumer_lock.
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not job or job["state"] != "ready":
                return False
            digest = job["content_hash"]
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Invalid stored content hash")
            unfinished = connection.execute(
                "SELECT 1 FROM jobs WHERE content_hash=? AND state!='ready' LIMIT 1", (digest,)
            ).fetchone()
            if unfinished:
                return False
            snapshot = self.snapshots / f"{digest}.pdf"
            checkpoints = RangeCheckpoints.content_directory(self.root / "checkpoints", digest)
            for path in (snapshot, checkpoints):
                if path.is_symlink() or not path.resolve().is_relative_to(self.root):
                    raise OSError("Refusing cleanup outside the job store")
            original_is_snapshot = connection.execute(
                "SELECT 1 FROM jobs WHERE source=? LIMIT 1", (str(snapshot.resolve()),)
            ).fetchone()
            if original_is_snapshot:
                return False
            if checkpoints.exists():
                shutil.rmtree(checkpoints)
            snapshot.unlink(missing_ok=True)
            return True
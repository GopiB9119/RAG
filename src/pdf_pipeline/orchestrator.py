"""Single-coordinator orchestration of Blob uploads and distributed range results."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .distributed import MAX_PDF_BYTES, Manifest, IncompleteDocument, collect_records, dispatch, reconcile
from .job_store import JobStore
from .models import EXTRACTION_VERSION, ExtractionOptions


def emit_event(event: dict) -> None:
    payload = json.dumps(event)
    try:
        print(payload, flush=True)
    except (OSError, ValueError):
        # A disconnected/closed output stream is not an indexing failure. Durable
        # state remains authoritative; do not replay work because logging failed.
        pass


class CoordinatorState(JobStore):
    def __init__(self, root: Path):
        super().__init__(root)
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS cloud_documents (
                    name TEXT PRIMARY KEY, etag TEXT NOT NULL, document_id TEXT NOT NULL,
                    state TEXT NOT NULL, version TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                    next_check REAL NOT NULL, deadline REAL NOT NULL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    error_type TEXT, chunk_count INTEGER
                );
                CREATE INDEX IF NOT EXISTS cloud_due ON cloud_documents(state,next_check);
                CREATE TABLE IF NOT EXISTS cloud_history (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    generation TEXT NOT NULL, event TEXT NOT NULL,
                    snapshot TEXT NOT NULL, recorded_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS cloud_history_generation ON cloud_history(generation,sequence);
            """)
            # Migrate without discarding older observations. Existing ready rows
            # have no publication receipt and must not be treated as cleanup-safe.
            connection.execute("BEGIN IMMEDIATE")
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(cloud_documents)")}
            if "generation" not in columns:
                connection.execute("ALTER TABLE cloud_documents ADD COLUMN generation TEXT")
            if "publication_receipt" not in columns:
                connection.execute("ALTER TABLE cloud_documents ADD COLUMN publication_receipt TEXT")
            for row in connection.execute("SELECT name FROM cloud_documents WHERE generation IS NULL").fetchall():
                connection.execute("UPDATE cloud_documents SET generation=? WHERE name=?", (uuid.uuid4().hex, row["name"]))
                self.record_history(connection, row["name"], "legacy_import")

    def record_history(self, connection, name: str, event: str) -> None:
        row = connection.execute("SELECT * FROM cloud_documents WHERE name=?", (name,)).fetchone()
        connection.execute("INSERT INTO cloud_history(generation,event,snapshot,recorded_at) VALUES (?,?,?,?)",
                           (row["generation"], event, json.dumps(dict(row), sort_keys=True), time.time()))

    def bind_cloud(self, account: str, container: str, queue: str, prefix: str,
                   database: Path, collection: str, pages_per_task: int,
                   extraction: ExtractionOptions | None = None) -> None:
        self.bind_target(database, collection)
        value = json.dumps([account, container, queue, prefix, pages_per_task, EXTRACTION_VERSION,
                    asdict(extraction or ExtractionOptions())], sort_keys=True)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            old = connection.execute("SELECT value FROM settings WHERE key='cloud_target'").fetchone()
            if old and old["value"] != value:
                raise ValueError("Coordinator state belongs to a different cloud configuration")
            connection.execute("INSERT OR IGNORE INTO settings VALUES ('cloud_target',?)", (value,))

    def cursor(self):
        with self.connect() as connection:
            row = connection.execute("SELECT value FROM settings WHERE key='cloud_cursor'").fetchone()
            return json.loads(row["value"]) if row else None

    def observe(self, uploads: list[dict], cursor, timeout: float) -> None:
        now = time.time()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for upload in uploads:
                name, etag = upload["name"], upload["etag"]
                old = connection.execute("SELECT etag FROM cloud_documents WHERE name=?", (name,)).fetchone()
                if old and old["etag"] == etag:
                    continue
                if old:
                    self.record_history(connection, name, "superseded_observation")
                document_id = "blob-" + hashlib.sha256(name.encode()).hexdigest()
                connection.execute("""INSERT INTO cloud_documents
                    (name,etag,document_id,state,next_check,deadline,created_at,updated_at,generation)
                    VALUES (?,?,?,'pending',?,?,?,?,?)
                    ON CONFLICT(name) DO UPDATE SET etag=excluded.etag,state='pending',version=NULL,
                    attempts=0,next_check=excluded.next_check,deadline=excluded.deadline,
                    created_at=excluded.created_at,updated_at=excluded.updated_at,error_type=NULL,chunk_count=NULL,
                    generation=excluded.generation,publication_receipt=NULL""",
                    (name, etag, document_id, now, now + timeout, now, now, uuid.uuid4().hex))
                self.record_history(connection, name, "observed")
            connection.execute("INSERT OR REPLACE INTO settings VALUES ('cloud_cursor',?)", (json.dumps(cursor),))

    def due(self, limit: int) -> list[dict]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM cloud_documents WHERE state IN ('pending','extracting') AND next_check<=? ORDER BY next_check,name LIMIT ?",
                (time.time(), limit),
            )]

    def update(self, job: dict, **values) -> None:
        allowed = {"state", "version", "attempts", "next_check", "error_type", "chunk_count", "publication_receipt"}
        if not values or not set(values) <= allowed:
            raise ValueError("Invalid coordinator state update")
        values["updated_at"] = time.time()
        with self.connect() as connection:
            changed = connection.execute(
                "UPDATE cloud_documents SET " + ",".join(f"{key}=?" for key in values) + " WHERE name=? AND etag=? AND generation=?",
                (*values.values(), job["name"], job["etag"], job["generation"]),
            ).rowcount
            if changed != 1:
                raise RuntimeError("Observed document changed while being processed")
            if any(field in values for field in ("state", "version", "error_type", "publication_receipt")):
                self.record_history(connection, job["name"], "updated")

    def summary(self) -> dict:
        with self.connect() as connection:
            return {row["state"]: row["count"] for row in connection.execute(
                "SELECT state,COUNT(*) AS count FROM cloud_documents GROUP BY state"
            )}

    def status(self, limit: int = 50) -> dict:
        with self.connect() as connection:
            rows = [dict(row) for row in connection.execute(
                "SELECT document_id,generation,state,version,attempts,next_check,deadline,updated_at,error_type,chunk_count,publication_receipt IS NOT NULL AS has_publication_receipt FROM cloud_documents ORDER BY updated_at DESC LIMIT ?", (limit,)
            )]
        return {"counts": self.summary(), "documents": rows}

    def history(self, limit: int = 50) -> list[dict]:
        if not 1 <= limit <= 1000:
            raise ValueError("History limit must be between 1 and 1000")
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM cloud_history ORDER BY sequence DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for row in rows:
            snapshot = json.loads(row["snapshot"])
            result.append({"sequence": row["sequence"], "event": row["event"], "generation": row["generation"],
                           "recorded_at": row["recorded_at"], "document_id": snapshot["document_id"],
                           "state": snapshot["state"], "version": snapshot["version"],
                           "has_publication_receipt": snapshot["publication_receipt"] is not None})
        return result

    def retirement_preview(self, limit: int = 50) -> dict:
        """Inspect evidence only. This milestone cannot authorize cloud deletion."""
        if not 1 <= limit <= 1000:
            raise ValueError("Preview limit must be between 1 and 1000")
        with self.connect() as connection:
            rows = connection.execute("""SELECT snapshot FROM cloud_history WHERE sequence IN (
                SELECT MAX(sequence) FROM cloud_history GROUP BY generation
            ) ORDER BY sequence DESC LIMIT ?""", (limit + 1,)).fetchall()
        generations = []
        for row in rows[:limit]:
            snapshot = json.loads(row["snapshot"])
            blockers = ["cloud_generation_fencing_not_implemented", "shared_cloud_retirement_record_missing"]
            verified = False
            receipt_text = snapshot["publication_receipt"]
            if receipt_text is None:
                blockers.append("committed_publication_receipt_missing")
            else:
                try:
                    self.validate_receipt(snapshot, json.loads(receipt_text))
                    verified = True
                except Exception:
                    blockers.append("publication_receipt_unverifiable")
            if snapshot["state"] != "ready":
                blockers.append("generation_not_ready")
            generations.append({"generation": snapshot["generation"], "document_id": snapshot["document_id"],
                                "version": snapshot["version"], "state": snapshot["state"],
                                "publication_receipt_verified": verified, "eligible_for_deletion": False,
                                "blockers": blockers})
        return {"cloud_deletion_enabled": False, "generations": generations,
                "has_more": len(rows) > limit, "artifact_bytes": None}

    def retry_cloud(self, document_id: str, timeout: float) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Retry timeout must be finite and positive")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            old = connection.execute("SELECT name FROM cloud_documents WHERE document_id=? AND state='failed'", (document_id,)).fetchone()
            if old is None:
                raise ValueError("No failed document with that ID")
            self.record_history(connection, old["name"], "retry_requested")
            changed = connection.execute("""UPDATE cloud_documents SET state='pending',attempts=0,
                next_check=?,deadline=?,updated_at=?,error_type=NULL,version=NULL,chunk_count=NULL,
                generation=?,publication_receipt=NULL WHERE document_id=? AND state='failed'""",
                (time.time(), time.time() + timeout, time.time(), uuid.uuid4().hex, document_id)).rowcount
            if changed != 1:
                raise ValueError("No failed document with that ID")
            self.record_history(connection, old["name"], "retry_generation_created")

    def validate_receipt(self, job: dict, receipt: dict) -> None:
        from index_publication import verify_publication_receipt

        with self.connect() as connection:
            target = connection.execute("SELECT value FROM settings WHERE key='target'").fetchone()
        if not isinstance(receipt, dict):
            raise ValueError("Invalid publication receipt")
        database, collection = json.loads(target["value"]) if target else (receipt.get("database"), receipt.get("collection"))
        if not isinstance(database, str) or not isinstance(collection, str):
            raise ValueError("Missing publication target")
        verify_publication_receipt(receipt, database=database, collection=collection,
                                   source=f"azure-upload:{job['name']}", generation=job["generation"])


def coordinator_cycle(state: CoordinatorState, blobs, sender, inspect_pdf, index_records,
                      *, prefix: str = "incoming/", pages_per_task: int = 10,
                      scan_limit: int = 100, work_limit: int = 10, check_seconds: float = 60,
                      document_timeout: float = 3600, max_attempts: int = 5,
                      require_receipt: bool = False, extraction: ExtractionOptions | None = None) -> dict:
    """Caller owns consumer_lock. One cycle never waits for workers to finish."""
    if not prefix or prefix.startswith("/") or not prefix.endswith("/") or prefix.split("/")[0] in ("sources", "manifests", "results"):
        raise ValueError("Use a dedicated incoming prefix ending in /")
    if (not 1 <= pages_per_task <= 100 or min(scan_limit, work_limit, check_seconds, document_timeout, max_attempts) <= 0
            or not all(math.isfinite(value) for value in (check_seconds, document_timeout))):
        raise ValueError("Coordinator limits must be positive")
    uploads, cursor = blobs.list_uploads(prefix, state.cursor(), scan_limit)
    if any(not upload["name"].startswith(prefix) or not upload["name"].lower().endswith(".pdf")
           or not isinstance(upload["etag"], str) or not upload["etag"] for upload in uploads):
        raise ValueError("Invalid upload listing")
    state.observe(uploads, cursor, document_timeout)
    for job in state.due(work_limit):
        try:
            if time.time() >= job["deadline"]:
                state.update(job, state="failed", error_type="DocumentDeadlineExceeded")
                emit_event({"event": "cloud_document_failed", "document_id": job["document_id"],
                            "error_type": "DocumentDeadlineExceeded"})
                continue
            # ETag equality guards overwrite races. A changed upload is discovered
            # on a later listing cycle; never finalize its superseded observed job.
            if not blobs.upload_matches(job["name"], job["etag"]):
                state.update(job, state="failed", error_type="UploadChangedOrRemoved")
                continue
            if job["state"] == "pending":
                pdf = blobs.read_upload(job["name"], job["etag"], MAX_PDF_BYTES)
                manifest = Manifest(job["document_id"], hashlib.sha256(pdf).hexdigest(), inspect_pdf(pdf), pages_per_task,
                                    extraction=extraction or ExtractionOptions())
                state.update(job, version=manifest.version)
                result = dispatch(blobs, sender, pdf, job["document_id"], manifest.page_count, pages_per_task,
                                  extraction=manifest.extraction)
                state.update(job, state="extracting", version=result["version"], next_check=time.time() + check_seconds)
                emit_event({"event": "cloud_document_dispatched", "document_id": job["document_id"],
                            "version": result["version"], "ranges": result["ranges"]})
                continue
            try:
                records = collect_records(blobs, job["version"])
            except IncompleteDocument:
                reconcile(blobs, sender, job["version"])
                state.update(job, next_check=time.time() + check_seconds)
                continue
            if not blobs.upload_matches(job["name"], job["etag"]):
                state.update(job, state="failed", error_type="UploadChangedOrRemoved")
                continue
            for record in records:
                record["metadata"]["source"] = f"azure-upload:{job['name']}"
                record["metadata"]["title"] = job["name"].rsplit("/", 1)[-1]
                record["metadata"]["processing_generation"] = job["generation"]
            outcome = index_records(records)
            receipt = outcome if isinstance(outcome, dict) else None
            if receipt is not None:
                state.validate_receipt(job, receipt)
            elif require_receipt:
                raise ValueError("Index publisher must return a committed receipt")
            count = receipt["chunk_count"] if receipt is not None else outcome
            if type(count) is not int or count < 1:
                raise ValueError("Indexing returned no chunks")
            # Production publisher receipts are committed with index pointers;
            # replay after this local receipt-write gap returns that same receipt.
            state.update(job, state="ready", chunk_count=count, error_type=None,
                         publication_receipt=json.dumps(receipt, sort_keys=True) if receipt is not None else None)
            emit_event({"event": "cloud_document_ready", "document_id": job["document_id"],
                        "version": job["version"], "chunks": count})
        except Exception as error:
            attempts = job["attempts"] + 1
            state.update(job, attempts=attempts, state="failed" if attempts >= max_attempts else job["state"],
                         error_type=type(error).__name__, next_check=time.time() + min(300, 5 * 2 ** min(attempts - 1, 6)))
            emit_event({"event": "cloud_document_attempt_failed", "document_id": job["document_id"],
                        "attempt": attempts, "error_type": type(error).__name__})
    return state.summary()
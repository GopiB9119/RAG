"""Local single-consumer ingestion MVP. No cloud resources are created."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from threading import Event
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent
# Make the local src package importable from this CLI without installing it globally.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pdf_pipeline.job_store import JobStore


def make_processor(database: Path, collection: str, workers: int, pages_per_task: int = 10) -> Callable:
    # A closure keeps one embedding model alive across documents in this consumer.
    # The PDF worker processes extract text; they do not each load the model.
    model = None

    def process(job: dict) -> int:
        nonlocal model
        from ingest_sources import load_pdf, chunk_records, build_index
        from rag_core import MODEL_NAME
        from sentence_transformers import SentenceTransformer

        snapshot = Path(job["snapshot"])
        # Check the exact saved bytes, not today's contents of the original file.
        # Reading in 1 MiB blocks avoids loading a second full PDF into memory.
        digest = hashlib.sha256()
        with snapshot.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != job["content_hash"]:
            raise ValueError("Snapshot integrity check failed")
        # Extract the snapshot but cite the original source, not its hash filename.
        # Range checkpoints outlive the adapter's temporary output. Retries skip
        # completed ranges for these exact PDF bytes; citations still use source.
        records = load_pdf(snapshot, source=job["source"], workers=workers,
                   pages_per_task=pages_per_task,
                   checkpoint_root=str(snapshot.parent.parent / "checkpoints"))
        chunks = chunk_records(records)
        if model is None:
            model = SentenceTransformer(MODEL_NAME)
        return build_index(chunks, str(database.resolve()), collection, False, model=model)

    return process


def process_due(store: JobStore, processor: Callable, limit: int, stop: Event | None = None) -> dict:
    # Caller owns the consumer lock; the watcher keeps it between scan cycles too.
    summary = {"processed": 0, "ready": 0, "errors": 0}
    for _ in range(limit):
        if stop is not None and stop.is_set():
            break
        job = store.claim()
        if job is None:
            break
        started = time.perf_counter()
        try:
            chunk_count = processor(job)
            if not isinstance(chunk_count, int) or chunk_count < 1:
                raise ValueError("Indexing must return a positive chunk count")
            # A crash after indexing but before ready causes safe at-least-once replay.
            store.finish(job["id"], chunk_count)
            # Index success is durable before cleanup. A cleanup failure must not
            # turn a ready job back into queued work or repeat paid processing.
            try:
                store.cleanup_ready_artifacts(job["id"])
            except Exception as cleanup_error:
                print(json.dumps({"event": "cleanup_deferred", "job_id": job["id"],
                                  "error_type": type(cleanup_error).__name__}), flush=True)
            summary["ready"] += 1
            event = {"event": "job_ready", "job_id": job["id"], "chunks": chunk_count}
        except Exception as error:
            store.fail(job["id"], type(error).__name__)
            summary["errors"] += 1
            event = {"event": "job_attempt_failed", "job_id": job["id"], "error_type": type(error).__name__}
        summary["processed"] += 1
        event.update(attempt=job["attempts"], seconds=round(time.perf_counter() - started, 3))
        print(json.dumps(event), flush=True)
    summary["states"] = store.counts()
    return summary


def consume(store: JobStore, processor: Callable, database: Path, collection: str, limit: int = 100) -> dict:
    if limit < 1:
        raise ValueError("limit must be positive")
    # This lock protects one local queue consumer, not all writers on all machines.
    # Recovery is safe only after the previous consumer no longer holds the lock.
    with store.consumer_lock():
        store.bind_target(database, collection)
        recovered = store.recover_interrupted()
        summary = process_due(store, processor, limit)
        summary["recovered"] = recovered
    return summary


def watch(store: JobStore, processor: Callable, folder: Path, database: Path,
          collection: str, poll_seconds: float = 2, stable_seconds: float = 10,
          limit: int = 1, max_bytes: int = 100 * 1024 * 1024,
          max_attempts: int = 3, stop: Event | None = None) -> None:
    from pdf_pipeline.watcher import FolderScanner

    if not math.isfinite(poll_seconds) or poll_seconds <= 0 or limit < 1:
        raise ValueError("poll_seconds and limit must be positive")
    if database.resolve().is_relative_to(folder.resolve()):
        raise ValueError("Vector database must be outside the watched folder")
    scanner = FolderScanner(folder, store, stable_seconds, max_bytes, max_attempts)
    stop = stop if stop is not None else Event()
    with store.consumer_lock():
        store.bind_target(database, collection)
        recovered = store.recover_interrupted()
        print(json.dumps({"event": "watch_started", "recovered": recovered}), flush=True)
        try:
            while not stop.is_set():
                scanner.scan()
                result = process_due(store, processor, limit, stop)
                if result["processed"]:
                    print(json.dumps({"event": "watch_progress", **result}), flush=True)
                # Event.wait is interruptible; tests inject a controlled event so
                # stability, retries, and shutdown are checked without real delays.
                stop.wait(poll_seconds)
        except KeyboardInterrupt:
            stop.set()
        finally:
            print(json.dumps({"event": "watch_stopped", "states": store.counts()}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Durable local PDF-to-RAG ingestion jobs")
    parser.add_argument("--state-dir", type=Path, default=PROJECT_ROOT / "data" / "jobs")
    commands = parser.add_subparsers(dest="command", required=True)
    enqueue = commands.add_parser("enqueue", help="Snapshot one PDF or all PDFs in a directory")
    enqueue.add_argument("path", type=Path)
    enqueue.add_argument("--max-mb", type=int, default=100)
    enqueue.add_argument("--max-attempts", type=int, default=3)
    work = commands.add_parser("work", help="Process due jobs, then exit; rerun for delayed retries")
    work.add_argument("--workers", type=int, default=4)
    work.add_argument("--pages-per-task", type=int, default=10)
    work.add_argument("--limit", type=int, default=100)
    work.add_argument("--database", type=Path, default=PROJECT_ROOT / "chroma_data")
    work.add_argument("--collection", default="rag_documents")
    watcher = commands.add_parser("watch", help="Automatically enqueue stable PDFs and process due jobs until Ctrl+C")
    watcher.add_argument("path", type=Path, nargs="?", default=PROJECT_ROOT / "data" / "input")
    watcher.add_argument("--workers", type=int, default=2)
    watcher.add_argument("--pages-per-task", type=int, default=10)
    watcher.add_argument("--limit", type=int, default=1, help="Documents processed between folder scans")
    watcher.add_argument("--poll-seconds", type=float, default=2)
    watcher.add_argument("--stable-seconds", type=float, default=10)
    watcher.add_argument("--max-mb", type=int, default=100)
    watcher.add_argument("--max-attempts", type=int, default=3)
    watcher.add_argument("--database", type=Path, default=PROJECT_ROOT / "chroma_data")
    watcher.add_argument("--collection", default="rag_documents")
    status = commands.add_parser("status")
    status.add_argument("--limit", type=int, default=50)
    retry = commands.add_parser("retry", help="Explicitly requeue a failed latest-version job")
    retry.add_argument("job_id")
    args = parser.parse_args()
    for name in ("max_mb", "max_attempts", "workers", "limit", "pages_per_task"):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.command == "status" and args.limit > 1000:
        parser.error("--limit must be at most 1000")
    if args.command == "watch":
        for name in ("poll_seconds", "stable_seconds"):
            if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.command in ("work", "watch"):
        # Fail before claiming work: missing packages must not consume retry attempts.
        missing = [name for name in ("pymupdf", "chromadb", "sentence_transformers")
                   if importlib.util.find_spec(name) is None]
        if missing:
            print(json.dumps({"status": "blocked", "missing_packages": missing}))
            return 2
    store = JobStore(args.state_dir)
    if args.command == "watch":
        watch(store, make_processor(args.database, args.collection, args.workers, args.pages_per_task),
              args.path, args.database, args.collection, args.poll_seconds, args.stable_seconds,
              args.limit, args.max_mb * 1024 * 1024, args.max_attempts)
        return 0
    if args.command == "enqueue":
        source = args.path.resolve(strict=True)
        paths = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf") if source.is_dir() else [source]
        if not paths:
            print(json.dumps({"status": "empty", "submitted": 0}))
            return 2
        rejected = 0
        for path in paths:
            try:
                job = store.enqueue(path, max_bytes=args.max_mb * 1024 * 1024, max_attempts=args.max_attempts)
                print(json.dumps({"job_id": job["id"], "state": job["state"]}))
            except (ValueError, OSError) as error:
                rejected += 1
                print(json.dumps({"event": "upload_rejected", "error_type": type(error).__name__}))
        return 2 if rejected else 0
    if args.command == "status":
        fields = ("id", "state", "attempts", "max_attempts", "created_at", "updated_at", "available_at", "error_type", "chunk_count")
        print(json.dumps({"counts": store.counts(), "jobs": [
            {field: job[field] for field in fields} for job in store.list_jobs(args.limit)
        ]}, indent=2))
        return 0
    if args.command == "retry":
        store.retry(args.job_id)
        print(json.dumps({"job_id": args.job_id, "state": "queued"}))
        return 0
    result = consume(store, make_processor(args.database, args.collection, args.workers, args.pages_per_task),
                     args.database, args.collection, args.limit)
    print(json.dumps(result))
    return 2 if result["errors"] or result["states"].get("failed", 0) else 0


if __name__ == "__main__":
    # Spawned Python processes import modules; this guard prevents rerunning the CLI.
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), file=sys.stderr)
        raise SystemExit(2)
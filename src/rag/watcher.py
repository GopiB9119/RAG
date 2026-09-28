"""Find new or changed PDFs after a quiet period; the durable queue owns deduplication."""

from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path
from typing import Callable

from .jobs import JobStore


class FolderScanner:
    def __init__(self, folder: Path, store: JobStore, emit: Callable[[dict], None], *, stable_seconds: float = 10,
                 max_bytes: int = 100 * 1024 * 1024, max_attempts: int = 3):
        if not math.isfinite(stable_seconds) or stable_seconds <= 0:
            raise ValueError("stable_seconds must be finite and positive")
        self.folder = Path(folder).resolve()
        if store.root.is_relative_to(self.folder):
            raise ValueError("The queue folder must be outside the watched folder")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.store, self.emit = store, emit
        self.stable_seconds, self.max_bytes, self.max_attempts = stable_seconds, max_bytes, max_attempts
        self.observed: dict[Path, tuple[tuple, float]] = {}
        self.submitted: dict[Path, tuple] = {}

    def scan(self) -> int:
        now = time.monotonic()
        present: set[Path] = set()
        submitted = 0
        # Existing files are scanned too, so arrivals while the service was down are not lost.
        for path in self.folder.rglob("*"):
            if path.suffix.lower() != ".pdf" or path.is_symlink():
                continue
            present.add(path)
            try:
                if not path.is_file():
                    continue
                stat = path.stat()
                signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
                previous = self.observed.get(path)
                if previous is None or previous[0] != signature:
                    self.observed[path] = (signature, now)
                    continue
                if self.submitted.get(path) == signature or now - previous[1] < self.stable_seconds:
                    continue
                try:
                    job = self.store.enqueue(path, self.max_bytes, self.max_attempts)
                except sqlite3.OperationalError as error:
                    if getattr(error, "sqlite_errorcode", 0) & 0xFF not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                        raise
                    self.emit({"event": "watch_queue_busy"})
                    continue
                except ValueError as error:
                    # Rejected once per version; a later change to the file makes it eligible again.
                    self.submitted[path] = signature
                    self.emit({"event": "watch_rejected", "file": path.name, "error": str(error)})
                    continue
                self.submitted[path] = signature
                if job["state"] == "queued" and job["attempts"] == 0:
                    submitted += 1
                self.emit({"event": "watch_submitted", "job_id": job["id"], "state": job["state"]})
            except OSError as error:
                # Locked or vanishing files are retried on the next scan.
                self.emit({"event": "watch_file_unavailable", "error_type": type(error).__name__})
        self.observed = {path: state for path, state in self.observed.items() if path in present}
        self.submitted = {path: state for path, state in self.submitted.items() if path in present}
        return submitted

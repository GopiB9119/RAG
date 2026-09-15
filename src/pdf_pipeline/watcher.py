"""Find new/changed PDFs after a quiet period; SQLite owns durable deduplication."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .job_store import JobStore


class FolderScanner:
    def __init__(self, folder: Path, store: JobStore, stable_seconds: float = 10,
                 max_bytes: int = 100 * 1024 * 1024, max_attempts: int = 3):
        if not math.isfinite(stable_seconds) or stable_seconds <= 0:
            raise ValueError("stable_seconds must be finite and positive")
        self.folder = folder.resolve()
        if store.root.is_relative_to(self.folder):
            raise ValueError("Job state must be outside the watched folder")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.stable_seconds = stable_seconds
        self.max_bytes = max_bytes
        self.max_attempts = max_attempts
        self.observed: dict[Path, tuple[tuple, float]] = {}
        self.submitted: dict[Path, tuple] = {}

    def scan(self) -> int:
        now = time.monotonic()
        present = set()
        submitted_count = 0
        # Scan existing files too: arrivals during downtime are not lost. Changes
        # in size/timestamps restart the quiet period before snapshot submission.
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
                except ValueError as error:
                    # Reject this version once. A later file change makes it eligible
                    # again; exhausted extraction retries still require manual retry.
                    self.submitted[path] = signature
                    print(json.dumps({"event": "watch_upload_rejected", "error_type": type(error).__name__}), flush=True)
                    continue
                self.submitted[path] = signature
                submitted_count += 1
                print(json.dumps({"event": "watch_submitted", "job_id": job["id"], "state": job["state"]}), flush=True)
            except OSError as error:
                # Locked, disappearing, or inaccessible files can be retried on the
                # next scan without preventing other PDFs from being discovered.
                print(json.dumps({"event": "watch_file_unavailable", "error_type": type(error).__name__}), flush=True)
        self.observed = {path: state for path, state in self.observed.items() if path in present}
        self.submitted = {path: state for path, state in self.submitted.items() if path in present}
        return submitted_count
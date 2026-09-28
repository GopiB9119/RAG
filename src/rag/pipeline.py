"""Batch ingestion: claim many documents, extract in parallel, embed together, publish one by one."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Callable

import numpy as np

from .chunking import Chunk, chunk_pages, fit_to_token_limit
from .config import Settings
from .extraction.checkpoints import RangeCheckpoints, validate_range_result
from .extraction.models import PageRangeJob, PageRangeResult, split_ranges
from .extraction.pdf import count_pages
from .extraction.pool import ExtractionPool
from .jobs import JobStore
from .store import Index

# Deterministic for the same bytes and policy: retrying cannot succeed.
PERMANENT_ERRORS = {"InvalidPDF", "SnapshotIntegrityError", "NoReadableText", "OCRRequired", "OCRResourceLimit"}


class IngestError(Exception):
    def __init__(self, error_type: str, message: str = ""):
        super().__init__(message or error_type)
        self.error_type = error_type


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class _Document:
    job: dict
    started: float
    pages: int = 0
    ranges: list[PageRangeJob] = field(default_factory=list)
    results: dict[str, PageRangeResult] = field(default_factory=dict)
    checkpoints: RangeCheckpoints | None = None
    chunks: list[Chunk] = field(default_factory=list)
    error: IngestError | None = None

    @property
    def title(self) -> str:
        return Path(self.job["source"]).name


class Ingestor:
    def __init__(self, settings: Settings, queue: JobStore, index: Index, embedder_factory: Callable,
                 emit: Callable[[dict], None], pool: ExtractionPool | None = None):
        self.settings, self.queue, self.index, self.emit = settings, queue, index, emit
        self.extraction = settings.extraction()
        self._embedder_factory = embedder_factory
        self._embedder = None
        self.pool = pool or ExtractionPool(settings.workers, task_timeout=settings.task_timeout)

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = self._embedder_factory()
        return self._embedder

    def _claim_batch(self) -> list[_Document]:
        documents: list[_Document] = []
        pages = 0
        while len(documents) < 256 and (not documents or pages < self.settings.batch_pages):
            job = self.queue.claim()
            if job is None:
                break
            document = _Document(job, time.perf_counter())
            documents.append(document)
            try:
                self._plan(document)
            except IngestError as error:
                document.error = error
            pages += document.pages
        return documents

    def _plan(self, document: _Document) -> None:
        snapshot = Path(document.job["snapshot"])
        try:
            if _sha256(snapshot) != document.job["content_hash"]:
                raise IngestError("SnapshotIntegrityError", "Stored copy of the PDF does not match its hash")
        except OSError as error:
            raise IngestError("SnapshotUnavailable", str(error)) from None
        try:
            document.pages = count_pages(str(snapshot))
        except Exception as error:
            raise IngestError("InvalidPDF", f"PDF cannot be opened: {type(error).__name__}") from None
        if document.pages < 1:
            raise IngestError("InvalidPDF", "PDF has no pages")
        document.ranges = split_ranges(str(snapshot), "doc", document.pages, self.settings.pages_per_task,
                                       self.extraction)
        if len(document.ranges) > 1:
            # Only multi-range documents gain from resuming a failed attempt.
            document.checkpoints = RangeCheckpoints(self.queue.checkpoints, document.job["content_hash"],
                                                    self.settings.pages_per_task, self.extraction)
            for task in document.ranges:
                cached = document.checkpoints.load(task)
                if cached is not None:
                    document.results[task.job_id] = cached

    def _extract(self, documents: list[_Document]) -> None:
        tasks: list[PageRangeJob] = []
        owners: dict[str, tuple[_Document, PageRangeJob]] = {}
        for number, document in enumerate(documents):
            if document.error:
                continue
            for task in document.ranges:
                if task.job_id in document.results:
                    continue
                # Unique IDs across the batch; workers never see the original file name.
                unique = PageRangeJob(f"{number}:{task.job_id}", task.document_id, task.pdf_path,
                                      task.start_page, task.end_page, task.extraction)
                tasks.append(unique)
                owners[unique.job_id] = (document, task)
        for unique_id, outcome in self.pool.map(tasks).items():
            document, task = owners[unique_id]
            if document.error:
                continue
            if isinstance(outcome, BaseException):
                document.error = IngestError(type(outcome).__name__, str(outcome))
                continue
            outcome.job_id = task.job_id
            try:
                validate_range_result(task, outcome)
            except RuntimeError as error:
                document.error = IngestError("InvalidExtractionResult", str(error))
                continue
            failed = [page for page in outcome.pages if not page.success]
            if failed:
                document.error = IngestError(failed[0].error or "PageExtractionFailed",
                                             f"{len(failed)} page(s) failed, first on page {failed[0].page_index + 1}")
                continue
            document.results[task.job_id] = outcome
            if document.checkpoints:
                document.checkpoints.save(task, outcome)

    def _chunk(self, document: _Document) -> None:
        pages = [{"page": page.page_index + 1, "text": page.text, "method": page.extraction_method}
                 for task in document.ranges for page in document.results[task.job_id].pages if page.text.strip()]
        chunks = chunk_pages(pages)
        if not chunks:
            raise IngestError("NoReadableText", "No readable text; enable OCR for scanned PDFs (RAG_OCR=auto)")
        document.chunks = fit_to_token_limit(chunks, self.embedder.count_tokens, self.embedder.max_tokens)

    @staticmethod
    def _guard(documents: list[_Document], action: Callable[[], object]):
        """Turn an unexpected stage failure into a failure of every still-healthy document."""
        try:
            return action()
        except Exception as error:
            failure = error if isinstance(error, IngestError) else IngestError(type(error).__name__, str(error))
            for document in documents:
                if document.error is None:
                    document.error = failure
            return None

    def process_batch(self, stop: Event | None = None) -> dict:
        documents = self._claim_batch()
        summary = {"processed": len(documents), "ready": 0, "failed": 0, "retrying": 0, "pages": 0, "chunks": 0}
        if not documents:
            return summary
        try:
            if stop is not None and stop.is_set():
                raise KeyboardInterrupt
            self._guard(documents, lambda: self._extract(documents))
            for document in documents:
                if document.error is None:
                    self._guard([document], lambda: self._chunk(document))
            ready = [document for document in documents if document.error is None]
            vectors = self._guard(ready, lambda: self.embedder.embed(
                [chunk.text for document in ready for chunk in document.chunks])) if ready else None
            offset = 0
            for document in ready:
                count = len(document.chunks)
                if document.error is None:
                    self._publish(document, vectors[offset:offset + count])
                offset += count
        except KeyboardInterrupt:
            for document in documents:
                self.queue.release(document.job["id"])
            raise
        for document in documents:
            self._settle(document, summary)
        return summary

    def _publish(self, document: _Document, vectors: np.ndarray) -> None:
        job = document.job
        try:
            receipt = self.index.publish(job["source"], document.title, document.pages, document.chunks, vectors,
                                         job["id"], job["content_hash"])
            self.index.verify_receipt(receipt)
            self.queue.finish(job["id"], receipt["chunk_count"], receipt)
        except Exception as error:
            document.error = IngestError(type(error).__name__, str(error))

    def _settle(self, document: _Document, summary: dict) -> None:
        job = document.job
        seconds = round(time.perf_counter() - document.started, 3)
        if document.error is None:
            summary["ready"] += 1
            summary["pages"] += document.pages
            summary["chunks"] += len(document.chunks)
            self.emit({"event": "document_ready", "job_id": job["id"], "title": document.title,
                       "pages": document.pages, "chunks": len(document.chunks), "attempt": job["attempts"],
                       "seconds": seconds})
            try:
                self.queue.cleanup_ready_artifacts(job["id"])
            except Exception as error:
                self.emit({"event": "cleanup_deferred", "job_id": job["id"], "error_type": type(error).__name__})
            return
        error_type = document.error.error_type
        state = self.queue.fail(job["id"], error_type, permanent=error_type in PERMANENT_ERRORS)
        summary["failed" if state == "failed" else "retrying"] += 1
        self.emit({"event": "document_failed" if state == "failed" else "document_retry_scheduled",
                   "job_id": job["id"], "title": document.title, "error_type": error_type,
                   "detail": str(document.error), "attempt": job["attempts"], "seconds": seconds})

    def close(self) -> None:
        self.pool.close()


def run_worker(ingestor: Ingestor, *, stop: Event, scanner=None, poll_seconds: float = 2.0,
               until_idle: bool = False) -> dict:
    """Drain the queue in batches; with a scanner, keep watching until stopped."""
    queue = ingestor.queue
    totals = {"processed": 0, "ready": 0, "failed": 0, "retrying": 0, "pages": 0, "chunks": 0}
    started = time.perf_counter()
    with queue.consumer_lock():
        queue.bind_index(ingestor.index.index_id)
        queue.bind_extraction(ingestor.extraction)
        recovered = queue.recover_interrupted()
        if recovered:
            ingestor.emit({"event": "recovered_interrupted_jobs", "count": recovered})
        next_scan = 0.0
        while not stop.is_set():
            if scanner is not None and time.monotonic() >= next_scan:
                scanner.scan()
                next_scan = time.monotonic() + poll_seconds
            result = ingestor.process_batch(stop)
            for key in totals:
                totals[key] += result[key]
            if result["processed"]:
                # Backlog: continue immediately instead of sleeping between documents.
                continue
            counts = queue.counts()
            if until_idle and not counts.get("queued") and not counts.get("running"):
                break
            stop.wait(poll_seconds)
    elapsed = time.perf_counter() - started
    totals["seconds"] = round(elapsed, 3)
    totals["pages_per_second"] = round(totals["pages"] / elapsed, 2) if elapsed > 0 else 0.0
    return totals

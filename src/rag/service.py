"""One facade over queue, index, embeddings and answers, shared by the CLI and the API."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Event, Lock
from typing import Callable

from .answer import Answer, Answerer
from .config import Settings
from .embeddings import get_spec, load_embedder
from .jobs import JobStore
from .pipeline import Ingestor, run_worker
from .search import Evidence, search
from .store import Index


def pdf_files(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        path = Path(path).resolve(strict=True)
        if path.is_dir():
            found.extend(sorted(item for item in path.rglob("*") if item.suffix.lower() == ".pdf"
                                and item.is_file() and not item.is_symlink()))
        else:
            found.append(path)
    return found


class RAG:
    def __init__(self, settings: Settings, *, embedder_factory: Callable | None = None,
                 answerer: Answerer | None = None):
        self.settings = settings
        spec = get_spec(settings.embedding_model)
        self.index = Index(settings.index_path, spec.identity, spec.dimensions)
        self.queue = JobStore(settings.queue_dir)
        self._embedder_factory = embedder_factory or (
            lambda: load_embedder(settings.embedding_model, settings.model_dir))
        self._embedder = None
        self._embedder_lock = Lock()
        self.answerer = answerer or Answerer(settings)

    @property
    def embedder(self):
        with self._embedder_lock:
            if self._embedder is None:
                self._embedder = self._embedder_factory()
            return self._embedder

    def submit(self, paths: list[Path], emit: Callable[[dict], None]) -> dict:
        totals = {"queued": 0, "already_indexed": 0, "previously_failed": 0, "rejected": 0}
        for path in pdf_files(paths):
            try:
                job = self.queue.enqueue(path, self.settings.max_upload_mb * 1024 * 1024, self.settings.max_attempts)
            except (ValueError, OSError) as error:
                totals["rejected"] += 1
                emit({"event": "rejected", "file": path.name, "error": str(error)})
                continue
            if job["state"] == "ready":
                totals["already_indexed"] += 1
            elif job["state"] == "failed":
                totals["previously_failed"] += 1
                emit({"event": "previously_failed", "file": path.name, "job_id": job["id"],
                      "error_type": job["error_type"], "hint": f"fix the file, or run: rag retry {job['id']}"})
            else:
                totals["queued"] += 1
        return totals

    def ingestor(self, emit: Callable[[dict], None]) -> Ingestor:
        return Ingestor(self.settings, self.queue, self.index, lambda: self.embedder, emit)

    def process(self, emit: Callable[[dict], None], *, stop: Event | None = None, scanner=None,
                until_idle: bool = False) -> dict:
        ingestor = self.ingestor(emit)
        try:
            return run_worker(ingestor, stop=stop or Event(), scanner=scanner, until_idle=until_idle)
        finally:
            ingestor.close()

    def search(self, question: str, *, top_k: int | None = None, mode: str = "hybrid",
               fusion: str | None = None, alpha: float | None = None) -> list[Evidence]:
        return search(self.index, self.embedder, question, top_k=top_k or self.settings.top_k,
                      candidates=self.settings.candidates, min_similarity=self.settings.min_similarity, mode=mode,
                      fusion=fusion or self.settings.fusion,
                      alpha=self.settings.hybrid_alpha if alpha is None else alpha)

    def ask(self, question: str, *, top_k: int | None = None, on_token: Callable[[str], None] | None = None) -> dict:
        started = time.perf_counter()
        evidence = self.search(question, top_k=top_k)
        searched = time.perf_counter()
        answer: Answer = self.answerer.answer(question, evidence, on_token)
        return {"answer": answer.text, "sources": answer.sources, "evidence": evidence,
                "timings": {"search_seconds": round(searched - started, 3),
                            "answer_seconds": round(time.perf_counter() - searched, 3)}}

    def status(self, failures: int = 10) -> dict:
        return {"index": self.index.stats(), "queue": self.queue.counts(),
                "recent_failures": [{"job_id": job["id"], "file": Path(job["source"]).name,
                                     "error_type": job["error_type"], "attempts": job["attempts"]}
                                    for job in self.queue.list_jobs(failures, state="failed")]}

    def close(self) -> None:
        self.index.close()

"""Long-lived extraction processes shared by every document in a run.

Each worker owns a private pipe and runs one range at a time, so a crash or
timeout is attributed to exactly one task: only that worker is replaced.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections import deque
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait

from .models import PageRangeJob, PageRangeResult


class WorkerCrashed(RuntimeError):
    pass


class TaskTimeout(TimeoutError):
    pass


def worker_main(connection: Connection) -> None:
    from .pdf import extract_range

    while True:
        try:
            job = connection.recv()
        except EOFError:
            return
        if job is None:
            return
        connection.send(extract_range(job))


@dataclass
class _Worker:
    process: mp.Process
    connection: Connection
    completed: int = 0
    job: PageRangeJob | None = None
    deadline: float = field(default=0.0)


class ExtractionPool:
    def __init__(self, workers: int, *, task_timeout: float = 300.0, max_tasks_per_worker: int = 250,
                 target=worker_main):
        if workers < 1 or task_timeout <= 0 or max_tasks_per_worker < 1:
            raise ValueError("Pool size, task timeout and recycling limit must be positive")
        self.size = workers
        self.task_timeout = task_timeout
        # Recycling bounds native-library memory growth in long-running services.
        self.max_tasks = max_tasks_per_worker
        self.target = target
        self._context = mp.get_context("spawn")
        self._idle: list[_Worker] = []
        self.started = 0

    def _spawn(self) -> _Worker:
        parent, child = self._context.Pipe()
        process = self._context.Process(target=self.target, args=(child,), daemon=True,
                                        name=f"rag-extract-{self.started + 1}")
        process.start()
        child.close()
        self.started += 1
        return _Worker(process, parent)

    @staticmethod
    def _stop(worker: _Worker, graceful: bool) -> None:
        if graceful and worker.process.is_alive():
            try:
                worker.connection.send(None)
            except OSError:
                pass
            worker.process.join(timeout=5)
        if worker.process.is_alive():
            worker.process.terminate()
            worker.process.join(timeout=5)
        if worker.process.is_alive():
            worker.process.kill()
            worker.process.join(timeout=5)
        worker.connection.close()

    def map(self, jobs: list[PageRangeJob]) -> dict[str, PageRangeResult | BaseException]:
        """Run every job and return its result or its own failure, keyed by job_id."""
        if len({job.job_id for job in jobs}) != len(jobs):
            raise ValueError("Range job IDs must be unique")
        pending = deque(jobs)
        outcomes: dict[str, PageRangeResult | BaseException] = {}
        busy: dict[Connection, _Worker] = {}
        try:
            while pending or busy:
                while pending and len(busy) < self.size:
                    reused = bool(self._idle)
                    if reused:
                        worker = self._idle.pop()
                        if not worker.process.is_alive():
                            self._stop(worker, graceful=False)
                            continue
                    else:
                        worker = self._spawn()
                    job = pending[0]
                    try:
                        worker.connection.send(job)
                    except OSError:
                        self._stop(worker, graceful=False)
                        if not reused:
                            pending.popleft()
                            outcomes[job.job_id] = WorkerCrashed("Could not start an extraction worker")
                        continue
                    pending.popleft()
                    worker.job, worker.deadline = job, time.monotonic() + self.task_timeout
                    busy[worker.connection] = worker
                if not busy:
                    continue
                timeout = max(0.0, min(worker.deadline for worker in busy.values()) - time.monotonic())
                sentinels = {worker.process.sentinel: worker for worker in busy.values()}
                ready = wait([*busy, *sentinels], timeout=timeout)
                finished: set[int] = set()
                for item in ready:
                    worker = busy.get(item) or sentinels.get(item)
                    if worker is None or id(worker) in finished or worker.connection not in busy:
                        continue
                    finished.add(id(worker))
                    job = worker.job
                    del busy[worker.connection]
                    try:
                        result = worker.connection.recv() if worker.connection.poll() else None
                    except (EOFError, OSError):
                        result = None
                    if not isinstance(result, PageRangeResult) or result.job_id != job.job_id:
                        outcomes[job.job_id] = WorkerCrashed("Extraction worker exited or returned an invalid result")
                        self._stop(worker, graceful=False)
                        continue
                    outcomes[job.job_id] = result
                    worker.completed += 1
                    worker.job = None
                    if worker.completed >= self.max_tasks:
                        self._stop(worker, graceful=True)
                    else:
                        self._idle.append(worker)
                now = time.monotonic()
                for connection, worker in list(busy.items()):
                    if worker.deadline <= now:
                        del busy[connection]
                        outcomes[worker.job.job_id] = TaskTimeout(
                            f"Range extraction exceeded {self.task_timeout:g} seconds")
                        self._stop(worker, graceful=False)
        except BaseException:
            # Interrupted mid-run: in-flight workers hold unknown state, never reuse them.
            for worker in busy.values():
                self._stop(worker, graceful=False)
            raise
        return outcomes

    def close(self) -> None:
        while self._idle:
            self._stop(self._idle.pop(), graceful=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

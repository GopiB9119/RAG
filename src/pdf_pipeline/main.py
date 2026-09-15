import multiprocessing as mp
import math
import re
import time
from pathlib import Path
from queue import Empty

from .collector import collect_results
from .checkpoints import RangeCheckpoints, pdf_fingerprint, validate_range_result
from .models import PageResult, PageRangeResult
from .scheduler import create_page_jobs, create_page_range_jobs
from .worker import worker_loop


def run_pipeline(
    pdf_path: str,
    document_id: str,
    workers: int,
    output_root: str,
    *,
    timeout_seconds: float = 300,
    pages_per_task: int = 1,
    checkpoint_root: str | None = None,
    write_outputs: bool = True,
) -> dict:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and greater than zero")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", document_id):
        raise ValueError("document_id must contain only letters, numbers, underscores or hyphens")
    if pages_per_task < 1:
        raise ValueError("pages_per_task must be at least 1")

    pdf_file = Path(pdf_path).resolve()
    if not pdf_file.exists():
        raise FileNotFoundError(f"Input PDF does not exist: {pdf_file}")

    range_mode = pages_per_task > 1 or checkpoint_root is not None
    initial_stat = pdf_file.stat()
    checkpoints = RangeCheckpoints(Path(checkpoint_root), document_id, pdf_fingerprint(pdf_file), pages_per_task) if checkpoint_root else None
    jobs = (create_page_range_jobs(str(pdf_file), document_id, pages_per_task) if range_mode
            else create_page_jobs(pdf_path=str(pdf_file), document_id=document_id))
    if not jobs:
        raise ValueError("The PDF contains no pages.")
    pending = []
    results: list[PageResult] = []
    reused_ranges = 0
    for job in jobs:
        cached = checkpoints.load(job) if checkpoints else None
        if cached is None:
            pending.append(job)
        else:
            results.extend(cached.pages)
            reused_ranges += 1

    # Each worker starts a fresh interpreter. Send serializable jobs rather than
    # sharing an open PDF object; this also supports Windows multiprocessing.
    context = mp.get_context("spawn")
    job_queue = context.Queue()
    result_queue = context.Queue()
    processes: list[mp.Process] = []
    start_time = time.perf_counter()
    # One deadline bounds the whole pool run; it is not renewed for each page.
    deadline = start_time + timeout_seconds
    expected = {job.job_id: job for job in pending}
    if len({job.job_id for job in jobs}) != len(jobs):
        job_queue.close()
        result_queue.close()
        raise ValueError("Page job IDs must be unique")
    received: set[str] = set()
    worker_count = min(workers, len(pending))
    try:
        for worker_number in range(worker_count):
            process = context.Process(
                target=worker_loop,
                args=(job_queue, result_queue),
                name=f"pdf-worker-{worker_number + 1}",
            )
            process.start()
            processes.append(process)

        for job in pending:
            job_queue.put(job)
        # One None sentinel per worker tells every consumer to stop after its jobs.
        for _ in processes:
            job_queue.put(None)

        while len(received) < len(pending):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"PDF extraction exceeded {timeout_seconds} seconds")
            try:
                # A short receive timeout lets the parent check for crashed workers
                # instead of waiting forever for a result that will never arrive.
                result = result_queue.get(timeout=min(0.2, remaining))
            except Empty:
                if any(process.exitcode not in (None, 0) for process in processes):
                    raise RuntimeError("A PDF worker exited unexpectedly")
                if all(process.exitcode is not None for process in processes):
                    raise RuntimeError("PDF workers exited before returning all page results")
                continue
            if not isinstance(result, PageRangeResult if range_mode else PageResult):
                raise RuntimeError("PDF worker returned an invalid result")
            job = expected.get(result.job_id)
            # A count alone is insufficient: two results for one page must not hide
            # a missing page. Match the ID, document, and page before accepting it.
            if (job is None or result.job_id in received or result.document_id != job.document_id):
                raise RuntimeError("PDF worker returned a duplicate or mismatched page result")
            if range_mode:
                validate_range_result(job, result)
                if checkpoints:
                    current_stat = pdf_file.stat()
                    if (initial_stat.st_size, initial_stat.st_mtime_ns) != (current_stat.st_size, current_stat.st_mtime_ns):
                        raise RuntimeError("PDF changed before checkpointing; use an immutable snapshot")
                    checkpoints.save(job, result)
                results.extend(result.pages)
            else:
                if result.page_index != job.page_index:
                    raise RuntimeError("PDF worker returned a mismatched page result")
                results.append(result)
            received.add(result.job_id)

        for process in processes:
            process.join(timeout=max(0, deadline - time.perf_counter()))
            if process.is_alive():
                raise TimeoutError("PDF worker did not shut down before the deadline")
            if process.exitcode != 0:
                raise RuntimeError("A PDF worker exited unexpectedly")
    finally:
        # Cleanup runs on success, timeout, exception, and keyboard interruption.
        # Do not leave native PDF worker processes running after the parent fails.
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        for queue in (job_queue, result_queue):
            queue.cancel_join_thread()
            queue.close()

    elapsed_seconds = time.perf_counter() - start_time
    final_stat = pdf_file.stat()
    if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise RuntimeError("PDF changed during extraction; use an immutable snapshot")
    # Workers finish out of order; the collector sorts pages for reproducible output.
    summary = collect_results(
        document_id=document_id,
        results=results,
        output_root=output_root,
        elapsed_seconds=round(elapsed_seconds, 3),
        worker_count=worker_count,
        write_outputs=write_outputs,
        **({"range_count": len(jobs), "reused_ranges": reused_ranges} if range_mode else {}),
    )
    return summary
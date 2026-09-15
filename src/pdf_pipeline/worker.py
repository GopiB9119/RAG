from multiprocessing.queues import Queue

from .extractor import extract_page, extract_range
from .models import PageJob, PageResult, PageRangeJob


def worker_loop(job_queue: Queue, result_queue: Queue) -> None:
    """Process PageJob objects until the None shutdown sentinel arrives."""
    while True:
        # get() takes one assignment; available workers naturally share the queue.
        # Each process handles one page/range task at a time, not one process per page.
        job: PageJob | PageRangeJob | None = job_queue.get()

        if job is None:
            break

        result = extract_range(job) if isinstance(job, PageRangeJob) else extract_page(job)
        # Completion order can differ from page order. The parent validates results
        # and the collector later restores the original reading sequence.
        result_queue.put(result)
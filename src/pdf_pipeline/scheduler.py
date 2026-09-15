from pathlib import Path

from .models import PageJob, PageRangeJob


def create_page_jobs(pdf_path: str, document_id: str) -> list[PageJob]:
    """Create one PageJob for every page. Page indexes are zero-based."""
    import pymupdf

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"PDF path is not a file: {path}")

    jobs: list[PageJob] = []

    # Inspect once to discover how many tasks to create, then close the file.
    # Actual text extraction belongs to workers, not this scheduling function.
    with pymupdf.open(path) as document:
        page_count = len(document)

    for page_index in range(page_count):
        # 06d pads the human page number: page 1 becomes "000001" in the job ID.
        job_id = f"{document_id}:page:{page_index + 1:06d}"
        jobs.append(
            PageJob(
                job_id=job_id,
                document_id=document_id,
                pdf_path=str(path),
                page_index=page_index,
            )
        )

    return jobs


def split_page_ranges(pdf_path: str, document_id: str, page_count: int,
                      pages_per_task: int = 10) -> list[PageRangeJob]:
    if pages_per_task < 1 or page_count < 1:
        raise ValueError("Page count and pages_per_task must be positive")
    return [PageRangeJob(
        job_id=f"{document_id}:pages:{start:06d}-{min(start + pages_per_task, page_count):06d}",
        document_id=document_id,
        pdf_path=pdf_path,
        start_page=start,
        end_page=min(start + pages_per_task, page_count),
    ) for start in range(0, page_count, pages_per_task)]


def create_page_range_jobs(pdf_path: str, document_id: str,
                           pages_per_task: int = 10) -> list[PageRangeJob]:
    import pymupdf

    path = Path(pdf_path).resolve(strict=True)
    with pymupdf.open(path) as document:
        return split_page_ranges(str(path), document_id, len(document), pages_per_task)
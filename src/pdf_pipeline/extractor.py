from pathlib import Path

from .models import PageJob, PageResult, PageRangeJob, PageRangeResult


def extract_page(job: PageJob) -> PageResult:
    """Extract text from one PDF page. The page_index is zero-based."""
    import pymupdf

    pdf_path = Path(job.pdf_path)

    if not pdf_path.exists():
        return PageResult(
            job_id=job.job_id,
            document_id=job.document_id,
            page_index=job.page_index,
            text="",
            success=False,
            error=f"PDF file does not exist: {pdf_path}",
        )

    try:
        # Own the PDF handle inside this worker and close it even when extraction
        # raises. Reopening for each page is simple but costs time on large PDFs.
        with pymupdf.open(pdf_path) as document:
            page_count = len(document)

            if job.page_index < 0 or job.page_index >= page_count:
                return PageResult(
                    job_id=job.job_id,
                    document_id=job.document_id,
                    page_index=job.page_index,
                    text="",
                    success=False,
                    error=(
                        f"Page index {job.page_index} is outside "
                        f"the document range 0-{page_count - 1}"
                    ),
                )

            page = document[job.page_index]
            # This reads the PDF text layer. It does not recognize words in scans.
            text = page.get_text("text")

            return PageResult(
                job_id=job.job_id,
                document_id=job.document_id,
                page_index=job.page_index,
                text=text,
                success=True,
            )

    except Exception as exc:
        # Normal extraction exceptions become page failures. Native process crashes
        # cannot return this object; the parent detects those through exit codes.
        return PageResult(
            job_id=job.job_id,
            document_id=job.document_id,
            page_index=job.page_index,
            text="",
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def extract_range(job: PageRangeJob) -> PageRangeResult:
    """Open the PDF once for a page range, preserving each page's original index."""
    import pymupdf

    pages = []
    try:
        with pymupdf.open(job.pdf_path) as document:
            if not 0 <= job.start_page < job.end_page <= len(document):
                raise ValueError("Page range is outside the PDF")
            for page_index in range(job.start_page, job.end_page):
                page_id = f"{job.document_id}:page:{page_index + 1:06d}"
                try:
                    text = document[page_index].get_text("text")
                    pages.append(PageResult(page_id, job.document_id, page_index, text, True))
                except Exception as error:
                    pages.append(PageResult(page_id, job.document_id, page_index, "", False, type(error).__name__))
    except Exception as error:
        pages = [PageResult(f"{job.document_id}:page:{page_index + 1:06d}", job.document_id,
                            page_index, "", False, type(error).__name__)
                 for page_index in range(job.start_page, job.end_page)]
    return PageRangeResult(job.job_id, job.document_id, pages)
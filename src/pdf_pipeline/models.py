from dataclasses import dataclass


@dataclass(frozen=True)
class PageJob:
    """A single page extraction task."""

    # frozen=True prevents changing an assignment after it has been created.
    # This is a page task, not the durable document-job row in jobs.sqlite3.
    job_id: str
    document_id: str
    # Send a filename across processes, never an open native PDF handle.
    pdf_path: str
    # Python indexes start at 0; citations shown to readers start at page 1.
    page_index: int


@dataclass
class PageResult:
    """Result returned by a worker after processing one page."""

    # Echo the assignment identity so the parent can reject duplicate/wrong pages.
    job_id: str
    document_id: str
    page_index: int
    text: str
    # success means extraction completed, not that readable text was found.
    # Blank or scanned pages may have success=True and text=""; OCR is separate.
    success: bool
    error: str | None = None


@dataclass(frozen=True)
class PageRangeJob:
    """One serializable range assignment; the end page is exclusive."""

    job_id: str
    document_id: str
    pdf_path: str
    start_page: int
    end_page: int


@dataclass
class PageRangeResult:
    job_id: str
    document_id: str
    pages: list[PageResult]
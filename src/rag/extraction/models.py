from __future__ import annotations

import re
from dataclasses import dataclass, field

# Bump when extraction semantics change: old checkpoints must not be reused.
EXTRACTION_VERSION = 2


@dataclass(frozen=True)
class ExtractionOptions:
    """One extraction policy for every worker; part of checkpoint identity."""

    ocr: str = "off"
    language: str = "eng"
    dpi: int = 300
    max_ocr_pixels: int = 25_000_000

    def __post_init__(self):
        if self.ocr not in ("off", "auto", "always"):
            raise ValueError("ocr must be off, auto, or always")
        if (not isinstance(self.language, str) or len(self.language) > 120
                or not re.fullmatch(r"[a-z][a-z0-9_]*(?:\+[a-z][a-z0-9_]*)*", self.language)):
            raise ValueError("Use OCR language codes such as eng or eng+hin")
        if type(self.dpi) is not int or not 72 <= self.dpi <= 600:
            raise ValueError("OCR dpi must be between 72 and 600")
        if type(self.max_ocr_pixels) is not int or not 1 <= self.max_ocr_pixels <= 100_000_000:
            raise ValueError("max_ocr_pixels must be between 1 and 100000000")


@dataclass(frozen=True)
class PageRangeJob:
    """A serializable range of one PDF; end_page is exclusive and indexes start at 0."""

    job_id: str
    document_id: str
    pdf_path: str
    start_page: int
    end_page: int
    extraction: ExtractionOptions = field(default_factory=ExtractionOptions)


@dataclass
class PageResult:
    job_id: str
    document_id: str
    page_index: int
    text: str
    success: bool
    error: str | None = None
    extraction_method: str = "native"


@dataclass
class PageRangeResult:
    job_id: str
    document_id: str
    pages: list[PageResult]


def split_ranges(pdf_path: str, document_id: str, page_count: int, pages_per_task: int,
                 extraction: ExtractionOptions) -> list[PageRangeJob]:
    if page_count < 1 or pages_per_task < 1:
        raise ValueError("Page count and pages_per_task must be positive")
    return [PageRangeJob(f"{document_id}:pages:{start:06d}-{min(start + pages_per_task, page_count):06d}",
                         document_id, pdf_path, start, min(start + pages_per_task, page_count), extraction)
            for start in range(0, page_count, pages_per_task)]

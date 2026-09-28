"""PyMuPDF text extraction with an explicit OCR policy; runs inside pool workers."""

from __future__ import annotations

import math

from .models import ExtractionOptions, PageRangeJob, PageRangeResult, PageResult


class OCRRequired(ValueError):
    """Native text is absent/unusable or a page-sized image needs recognition."""


class OCRFailed(RuntimeError):
    """OCR could not run or returned no usable text; never indexed as success."""


class OCRResourceLimit(ValueError):
    """The requested raster would exceed the per-page pixel budget."""


def _unusable_text(text: str) -> bool:
    return any(character == "\ufffd" or (ord(character) < 32 and character not in "\t\n\r\f")
               for character in text)


def page_text(page, options: ExtractionOptions) -> tuple[str, str]:
    native_failed = False
    try:
        text = page.get_text("text", sort=True)
    except Exception:
        if options.ocr == "off":
            raise
        native_failed = True
        text = ""
    images = page.get_image_info()
    # A scan often has a page-sized image plus only a page-number text layer.
    page_area = page.rect.width * page.rect.height
    large_image = page_area > 0 and any(
        max(0, min(image["bbox"][2], page.rect.x1) - max(image["bbox"][0], page.rect.x0))
        * max(0, min(image["bbox"][3], page.rect.y1) - max(image["bbox"][1], page.rect.y0))
        >= 0.8 * page_area
        for image in images
    )
    empty = not text.strip()
    if empty and not images and not native_failed and not page.get_drawings():
        return "", "blank"
    needs_ocr = empty or large_image or _unusable_text(text)
    if options.ocr != "always" and not needs_ocr:
        return text, "native"
    if options.ocr == "off":
        raise OCRRequired("Page needs OCR or manual review")
    width, height = (page.rect.width * options.dpi / 72, page.rect.height * options.dpi / 72)
    if (not all(math.isfinite(size) and size > 0 for size in (width, height))
            or math.ceil(width) * math.ceil(height) > options.max_ocr_pixels):
        raise OCRResourceLimit("OCR page exceeds the pixel budget")
    try:
        textpage = page.get_textpage_ocr(language=options.language, dpi=options.dpi, full=True)
        recognized = page.get_text("text", textpage=textpage, sort=True)
    except Exception as error:
        raise OCRFailed("OCR engine or language data is unavailable, or recognition failed") from error
    if not recognized.strip() or _unusable_text(recognized):
        raise OCRFailed("OCR returned no usable text; review this page")
    return recognized, "ocr"


def extract_range(job: PageRangeJob) -> PageRangeResult:
    """Open the PDF once per range; per-page failures are reported, never hidden."""
    import pymupdf

    pages = []
    try:
        with pymupdf.open(job.pdf_path) as document:
            if not 0 <= job.start_page < job.end_page <= len(document):
                raise ValueError("Page range is outside the PDF")
            for index in range(job.start_page, job.end_page):
                page_id = f"{job.document_id}:page:{index + 1:06d}"
                try:
                    text, method = page_text(document[index], job.extraction)
                    pages.append(PageResult(page_id, job.document_id, index, text, True, extraction_method=method))
                except Exception as error:
                    pages.append(PageResult(page_id, job.document_id, index, "", False, type(error).__name__))
    except Exception as error:
        pages = [PageResult(f"{job.document_id}:page:{index + 1:06d}", job.document_id, index, "", False,
                            type(error).__name__) for index in range(job.start_page, job.end_page)]
    return PageRangeResult(job.job_id, job.document_id, pages)


def count_pages(path: str) -> int:
    import pymupdf

    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise ValueError("Encrypted PDF requires a password")
        return len(document)

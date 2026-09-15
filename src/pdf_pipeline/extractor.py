import math

from .models import ExtractionOptions, PageJob, PageResult, PageRangeJob, PageRangeResult


class OCRRequired(ValueError):
    """Native text is absent/unusable or a page-sized image needs recognition."""


class OCRFailed(RuntimeError):
    """OCR could not run or did not return usable text; never index as success."""


class OCRResourceLimit(ValueError):
    """The requested raster would exceed the configured per-page pixel budget."""


def _unusable_text(text: str) -> bool:
    return any(character == "\ufffd" or (ord(character) < 32 and character not in "\t\n\r\f")
               for character in text)


def _page_text(page, options: ExtractionOptions) -> tuple[str, str]:
    native_failed = False
    try:
        text = page.get_text("text", sort=True)
    except Exception:
        if options.ocr == "off":
            raise
        native_failed = True
        text = ""
    images = page.get_image_info()
    # A large image can contain the real page while the native text layer holds
    # only a page number. This conservative trigger is not layout reconstruction.
    page_area = page.rect.width * page.rect.height
    large_image = any(
        max(0, min(image["bbox"][2], page.rect.x1) - max(image["bbox"][0], page.rect.x0))
        * max(0, min(image["bbox"][3], page.rect.y1) - max(image["bbox"][1], page.rect.y0))
        >= 0.8 * page_area
        for image in images
    ) if page_area > 0 else False
    empty = not text.strip()
    if empty and not images and not native_failed and not page.get_drawings():
        return "", "blank"
    needs_ocr = empty or large_image or _unusable_text(text)
    if options.ocr != "always" and not needs_ocr:
        return text, "native"
    if options.ocr == "off":
        raise OCRRequired("Page needs OCR or manual review")

    dimensions = (page.rect.width * options.dpi / 72, page.rect.height * options.dpi / 72)
    if (any(not math.isfinite(size) or size <= 0 for size in dimensions)
            or math.ceil(dimensions[0]) * math.ceil(dimensions[1]) > options.max_ocr_pixels):
        raise OCRResourceLimit("OCR page exceeds the pixel budget")
    try:
        # PyMuPDF renders only the flagged page; Tesseract runs on this machine.
        # Keep the TextPage in memory and do not save images or OCR PDFs.
        textpage = page.get_textpage_ocr(language=options.language, dpi=options.dpi, full=True)
        recognized = page.get_text("text", textpage=textpage, sort=True)
    except Exception as error:
        raise OCRFailed("OCR engine or language data is unavailable, or recognition failed") from error
    if not recognized.strip() or _unusable_text(recognized):
        raise OCRFailed("OCR returned no usable text; review this page")
    return recognized, "ocr"


def extract_page(job: PageJob) -> PageResult:
    """Compatibility API: use the same extraction policy as range workers."""
    result = extract_range(PageRangeJob(job.job_id, job.document_id, job.pdf_path,
                                        job.page_index, job.page_index + 1, job.extraction)).pages[0]
    result.job_id = job.job_id
    return result


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
                    text, method = _page_text(document[page_index], job.extraction)
                    pages.append(PageResult(page_id, job.document_id, page_index, text, True,
                                            extraction_method=method))
                except Exception as error:
                    pages.append(PageResult(page_id, job.document_id, page_index, "", False, type(error).__name__))
    except Exception as error:
        pages = [PageResult(f"{job.document_id}:page:{page_index + 1:06d}", job.document_id,
                            page_index, "", False, type(error).__name__)
                 for page_index in range(job.start_page, job.end_page)]
    return PageRangeResult(job.job_id, job.document_id, pages)
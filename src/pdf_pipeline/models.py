from dataclasses import dataclass, field
import re


EXTRACTION_VERSION = 2


@dataclass(frozen=True)
class ExtractionOptions:
    """One extraction policy shared by local processes and Azure workers."""

    ocr: str = "off"
    language: str = "eng"
    dpi: int = 300
    max_ocr_pixels: int = 25000000

    def __post_init__(self):
        if self.ocr not in ("off", "auto", "always"):
            raise ValueError("ocr must be off, auto, or always")
        if not isinstance(self.language, str) or not re.fullmatch(r"[a-z][a-z0-9_]*(?:\+[a-z][a-z0-9_]*)*", self.language) or len(self.language) > 120:
            raise ValueError("Use OCR language codes such as eng or eng+hin")
        if type(self.dpi) is not int or not 72 <= self.dpi <= 600:
            raise ValueError("OCR dpi must be between 72 and 600")
        if type(self.max_ocr_pixels) is not int or not 1 <= self.max_ocr_pixels <= 100000000:
            raise ValueError("max_ocr_pixels must be between 1 and 100000000")

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--ocr", choices=("off", "auto", "always"), default="off",
                            help="off: reject OCR-needed pages; auto: OCR flagged pages; always: OCR nonblank pages")
        parser.add_argument("--ocr-language", default="eng", help="Installed Tesseract language codes, e.g. eng+hin")
        parser.add_argument("--ocr-dpi", type=int, default=300, help="OCR rendering resolution (72-600)")

    @classmethod
    def from_namespace(cls, args):
        return cls(ocr=args.ocr, language=args.ocr_language, dpi=args.ocr_dpi)


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
    extraction: ExtractionOptions = field(default_factory=ExtractionOptions)


@dataclass
class PageResult:
    """Result returned by a worker after processing one page."""

    # Echo the assignment identity so the parent can reject duplicate/wrong pages.
    job_id: str
    document_id: str
    page_index: int
    text: str
    # Empty native pages pass only when no images/drawings were detected. This is
    # a quality gate, not a guarantee that all visible words were reconstructed.
    success: bool
    error: str | None = None
    extraction_method: str = "native"


@dataclass(frozen=True)
class PageRangeJob:
    """One serializable range assignment; the end page is exclusive."""

    job_id: str
    document_id: str
    pdf_path: str
    start_page: int
    end_page: int
    extraction: ExtractionOptions = field(default_factory=ExtractionOptions)


@dataclass
class PageRangeResult:
    job_id: str
    document_id: str
    pages: list[PageResult]
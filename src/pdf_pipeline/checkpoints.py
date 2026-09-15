"""Content-versioned local range checkpoints; only the parent process writes them."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from .models import EXTRACTION_VERSION, ExtractionOptions, PageRangeJob, PageRangeResult, PageResult


def validate_range_result(job: PageRangeJob, result: PageRangeResult) -> None:
    if (not isinstance(result, PageRangeResult) or result.job_id != job.job_id
            or result.document_id != job.document_id
            or len(result.pages) != job.end_page - job.start_page):
        raise RuntimeError("Invalid page-range result")
    for page_index, page in zip(range(job.start_page, job.end_page), result.pages):
        if (not isinstance(page, PageResult) or page.page_index != page_index
                or page.job_id != f"{job.document_id}:page:{page_index + 1:06d}"
                or page.document_id != job.document_id or not isinstance(page.text, str)
                or type(page.success) is not bool
                or (page.error is not None and not isinstance(page.error, str))
                or (page.success and page.error is not None)
                or page.extraction_method not in ("native", "ocr", "blank")):
            raise RuntimeError("Mismatched page in range result")
        if page.success and (
            (page.extraction_method == "blank") != (not page.text.strip())
            or (page.extraction_method == "ocr" and job.extraction.ocr == "off")
            or (page.extraction_method == "native" and job.extraction.ocr == "always")
        ):
            raise RuntimeError("Page extraction method does not match the policy or text")


def pdf_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class RangeCheckpoints:
    @staticmethod
    def content_directory(root: Path, fingerprint: str) -> Path:
        # Group retry files by PDF content so the durable store can retire them
        # after indexing. Hashing also keeps arbitrary path components out.
        return root / "by-pdf" / hashlib.sha256(fingerprint.encode()).hexdigest()

    def __init__(self, root: Path, document_id: str, fingerprint: str, pages_per_task: int,
                 extraction: ExtractionOptions | None = None):
        # Bump extraction_version when extraction semantics change. A different PDF,
        # document identity, or range size must never reuse this run's checkpoints.
        self.extraction = extraction or ExtractionOptions()
        identity = json.dumps({"extraction_version": EXTRACTION_VERSION, "document_id": document_id,
                       "pdf_sha256": fingerprint, "pages_per_task": pages_per_task,
                       "extraction": asdict(self.extraction)}, sort_keys=True)
        grouped = self.content_directory(root, fingerprint)
        self.identity = hashlib.sha256(identity.encode()).hexdigest()
        # Keep nested paths short on Windows; the full identity is checked inside
        # each new-format file, not inferred from this shortened directory name.
        self.directory = grouped / self.identity[:24]
        # Keep old checkpoints readable after this storage-layout change.
        self.legacy_directory = root / hashlib.sha256(identity.encode()).hexdigest()
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, job: PageRangeJob) -> Path:
        if job.extraction != self.extraction:
            raise ValueError("Checkpoint policy does not match the range job")
        return self.directory / f"{job.start_page:06d}-{job.end_page:06d}.json"

    def load(self, job: PageRangeJob) -> PageRangeResult | None:
        path = self.path(job)
        if not path.exists():
            path = self.legacy_directory / path.name
            if not path.exists():
                return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if path.parent == self.directory and envelope.get("identity") != self.identity:
                return None
            payload = envelope["result"]
            encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            if hashlib.sha256(encoded).hexdigest() != envelope["sha256"]:
                return None
            result = PageRangeResult(payload["job_id"], payload["document_id"],
                                     [PageResult(**page) for page in payload["pages"]])
            validate_range_result(job, result)
            return result if all(page.success for page in result.pages) else None
        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError):
            # A malformed/truncated checkpoint is not evidence of completion.
            # Re-extract it; a later successful save replaces it atomically.
            return None

    def save(self, job: PageRangeJob, result: PageRangeResult) -> None:
        validate_range_result(job, result)
        if not all(page.success for page in result.pages):
            return
        payload = asdict(result)
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        envelope = {"identity": self.identity, "sha256": hashlib.sha256(encoded).hexdigest(), "result": payload}
        descriptor, name = tempfile.mkstemp(dir=self.directory, suffix=".partial")
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(envelope, output, ensure_ascii=False)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(self.path(job))
        finally:
            temporary.unlink(missing_ok=True)
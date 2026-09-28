"""Validated range results and content-addressed retry checkpoints (parent process only)."""

from __future__ import annotations

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
    for index, page in zip(range(job.start_page, job.end_page), result.pages):
        if (not isinstance(page, PageResult) or page.page_index != index
                or page.job_id != f"{job.document_id}:page:{index + 1:06d}"
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


class RangeCheckpoints:
    """Successful ranges survive a failed attempt, so a retry only re-extracts what failed."""

    @staticmethod
    def content_directory(root: Path, content_hash: str) -> Path:
        return root / hashlib.sha256(content_hash.encode()).hexdigest()[:32]

    def __init__(self, root: Path, content_hash: str, pages_per_task: int, extraction: ExtractionOptions):
        self.extraction = extraction
        identity = json.dumps({"extraction_version": EXTRACTION_VERSION, "pdf_sha256": content_hash,
                               "pages_per_task": pages_per_task, "extraction": asdict(extraction)}, sort_keys=True)
        self.identity = hashlib.sha256(identity.encode()).hexdigest()
        self.directory = self.content_directory(root, content_hash) / self.identity[:24]

    def _path(self, job: PageRangeJob) -> Path:
        if job.extraction != self.extraction:
            raise ValueError("Checkpoint policy does not match the range job")
        return self.directory / f"{job.start_page:06d}-{job.end_page:06d}.json"

    def load(self, job: PageRangeJob) -> PageRangeResult | None:
        path = self._path(job)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if envelope.get("identity") != self.identity:
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
            # A damaged checkpoint is not evidence of completion; extract again.
            return None

    def save(self, job: PageRangeJob, result: PageRangeResult) -> None:
        validate_range_result(job, result)
        if not all(page.success for page in result.pages):
            return
        payload = asdict(result)
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        envelope = {"identity": self.identity, "sha256": hashlib.sha256(encoded).hexdigest(), "result": payload}
        self.directory.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=self.directory, suffix=".partial")
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(envelope, output, ensure_ascii=False)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(self._path(job))
        finally:
            temporary.unlink(missing_ok=True)

"""Cloud-independent manifest and completion protocol for distributed extraction."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .checkpoints import validate_range_result
from .models import PageRangeJob, PageRangeResult, PageResult


MAX_PDF_BYTES = 100 * 1024 * 1024
MAX_RESULT_BYTES = 16 * 1024 * 1024
MAX_PAGES = 100000
SCHEMA = 1


class InvalidTask(ValueError):
    pass


class IncompleteDocument(RuntimeError):
    pass


class BlobStore(Protocol):
    def read(self, name: str, limit: int) -> bytes | None: ...
    def create(self, name: str, data: bytes) -> bool: ...


class TaskSender(Protocol):
    def send(self, task: dict, message_id: str) -> None: ...


def encode(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Manifest:
    document_id: str
    pdf_sha256: str
    page_count: int
    pages_per_task: int
    schema: int = SCHEMA

    def __post_init__(self):
        if not isinstance(self.document_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", self.document_id):
            raise InvalidTask("Invalid document ID")
        if not isinstance(self.pdf_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.pdf_sha256):
            raise InvalidTask("Invalid PDF hash")
        if type(self.page_count) is not int or not 1 <= self.page_count <= MAX_PAGES:
            raise InvalidTask("Invalid page count")
        if type(self.pages_per_task) is not int or not 1 <= self.pages_per_task <= 100:
            raise InvalidTask("Range size must be between 1 and 100")
        if type(self.schema) is not int or self.schema != SCHEMA:
            raise InvalidTask("Unsupported manifest schema")

    @property
    def version(self) -> str:
        return sha256(encode(asdict(self)))

    @property
    def source_blob(self) -> str:
        return f"sources/{self.pdf_sha256}.pdf"

    def tasks(self) -> list[dict]:
        return [{"schema": SCHEMA, "version": self.version, "start_page": start,
                 "end_page": min(start + self.pages_per_task, self.page_count)}
                for start in range(0, self.page_count, self.pages_per_task)]

    def job(self, task: dict, pdf_path: str) -> PageRangeJob:
        start, end = task["start_page"], task["end_page"]
        if (start < 0 or start >= self.page_count or start % self.pages_per_task != 0
                or end != min(start + self.pages_per_task, self.page_count)):
            raise InvalidTask("Range does not match manifest")
        return PageRangeJob(f"{self.document_id}:pages:{start:06d}-{end:06d}",
                            self.document_id, pdf_path, start, end)


def validate_task(task: dict) -> None:
    if not isinstance(task, dict) or set(task) != {"schema", "version", "start_page", "end_page"}:
        raise InvalidTask("Invalid task shape")
    if type(task["schema"]) is not int or task["schema"] != SCHEMA:
        raise InvalidTask("Unsupported task schema")
    if not isinstance(task["version"], str) or not re.fullmatch(r"[0-9a-f]{64}", task["version"]):
        raise InvalidTask("Invalid manifest reference")
    if any(type(task[field]) is not int for field in ("start_page", "end_page")):
        raise InvalidTask("Page bounds must be integers")


def load_manifest(blobs: BlobStore, version: str) -> Manifest:
    if not re.fullmatch(r"[0-9a-f]{64}", version):
        raise InvalidTask("Invalid manifest reference")
    data = blobs.read(f"manifests/{version}.json", 1024 * 1024)
    if data is None:
        raise IncompleteDocument("Manifest is unavailable")
    try:
        manifest = Manifest(**json.loads(data))
    except (TypeError, ValueError) as error:
        raise InvalidTask("Invalid manifest") from error
    if manifest.version != version:
        raise InvalidTask("Manifest identity mismatch")
    return manifest


def result_name(task: dict) -> str:
    return f"results/{task['version']}/{task['start_page']:06d}-{task['end_page']:06d}.json"


def read_result(blobs: BlobStore, manifest: Manifest, task: dict) -> PageRangeResult | None:
    data = blobs.read(result_name(task), MAX_RESULT_BYTES)
    if data is None:
        return None
    try:
        envelope = json.loads(data)
        payload = envelope["result"]
        if envelope["version"] != manifest.version or sha256(encode(payload)) != envelope["sha256"]:
            raise ValueError("Result checksum mismatch")
        result = PageRangeResult(payload["job_id"], payload["document_id"],
                                 [PageResult(**page) for page in payload["pages"]])
        validate_range_result(manifest.job(task, ""), result)
        if not all(page.success for page in result.pages):
            raise ValueError("Incomplete extraction")
        return result
    except (ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
        raise InvalidTask("Stored range result is invalid; operator repair required") from error


def create_verified(blobs: BlobStore, name: str, data: bytes, limit: int) -> None:
    if not blobs.create(name, data) and blobs.read(name, limit) != data:
        raise InvalidTask("Existing immutable blob differs from expected content")


def dispatch(blobs: BlobStore, sender: TaskSender, pdf: bytes, document_id: str,
             page_count: int, pages_per_task: int = 10) -> dict:
    if not pdf.startswith(b"%PDF-") or len(pdf) > MAX_PDF_BYTES:
        raise InvalidTask("PDF header or size limit check failed")
    manifest = Manifest(document_id, sha256(pdf), page_count, pages_per_task)
    # Save source and manifest BEFORE sending. If sending fails halfway, repeating
    # dispatch or reconcile sends missing work again; completed ranges are skipped.
    create_verified(blobs, manifest.source_blob, pdf, MAX_PDF_BYTES)
    create_verified(blobs, f"manifests/{manifest.version}.json", encode(asdict(manifest)), 1024 * 1024)
    return reconcile(blobs, sender, manifest.version)


def reconcile(blobs: BlobStore, sender: TaskSender, version: str) -> dict:
    manifest = load_manifest(blobs, version)
    sent = 0
    for task in manifest.tasks():
        if read_result(blobs, manifest, task) is None:
            sender.send(task, sha256(encode(task)))
            sent += 1
    return {"version": version, "ranges": len(manifest.tasks()), "sent": sent}


def process_task(blobs: BlobStore, task: dict, extractor) -> str:
    validate_task(task)
    manifest = load_manifest(blobs, task["version"])
    manifest.job(task, "")
    if read_result(blobs, manifest, task) is not None:
        return "already_complete"
    pdf = blobs.read(manifest.source_blob, MAX_PDF_BYTES)
    if pdf is None:
        raise IncompleteDocument("Source PDF is unavailable")
    if sha256(pdf) != manifest.pdf_sha256:
        raise InvalidTask("Source PDF hash mismatch")
    with tempfile.TemporaryDirectory(prefix="azure-pdf-range-") as directory:
        path = Path(directory) / "source.pdf"
        path.write_bytes(pdf)
        job = manifest.job(task, str(path))
        result = extractor(job)
    validate_range_result(job, result)
    if not all(page.success for page in result.pages):
        raise RuntimeError("Range extraction did not succeed for every page")
    payload = asdict(result)
    data = encode({"version": manifest.version, "sha256": sha256(encode(payload)), "result": payload})
    if len(data) > MAX_RESULT_BYTES:
        raise InvalidTask("Range output exceeds size limit; use smaller tasks")
    # Duplicate workers may race. The first valid result wins; validate the winner
    # rather than counting completions or overwriting an existing object.
    blobs.create(result_name(task), data)
    if read_result(blobs, manifest, task) is None:
        raise IncompleteDocument("Range result publication was not confirmed")
    return "completed"


def collect_records(blobs: BlobStore, version: str) -> list[dict]:
    manifest = load_manifest(blobs, version)
    records = []
    for task in manifest.tasks():
        result = read_result(blobs, manifest, task)
        if result is None:
            raise IncompleteDocument("Not every page range has completed")
        for page in result.pages:
            text = " ".join(page.text.split())
            if text:
                records.append({"text": text, "metadata": {
                    "source": f"{manifest.document_id}.pdf", "title": f"{manifest.document_id}.pdf",
                    "type": "pdf", "page": page.page_index + 1, "version": version,
                }})
    if not records:
        raise ValueError("PDF contains no readable text; OCR is required")
    return records
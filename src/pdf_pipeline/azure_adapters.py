"""Azure I/O adapters. Imports are lazy so protocol tests need no cloud SDKs."""

from __future__ import annotations

import json
import multiprocessing as mp
import tempfile
from dataclasses import asdict
from pathlib import Path

from .distributed import InvalidTask, MAX_RESULT_BYTES, encode, process_task
from .models import PageRangeResult, PageResult


class AzureBlobs:
    def __init__(self, container):
        self.container = container

    def read(self, name: str, limit: int) -> bytes | None:
        from azure.core.exceptions import ResourceNotFoundError

        blob = self.container.get_blob_client(name)
        try:
            if blob.get_blob_properties().size > limit:
                raise InvalidTask("Blob exceeds permitted size")
            result = bytearray()
            for chunk in blob.download_blob(max_concurrency=1).chunks():
                result.extend(chunk)
                if len(result) > limit:
                    raise InvalidTask("Blob exceeds permitted size")
            return bytes(result)
        except ResourceNotFoundError:
            return None

    def create(self, name: str, data: bytes) -> bool:
        from azure.core.exceptions import ResourceExistsError

        try:
            self.container.get_blob_client(name).upload_blob(data, overwrite=False)
            return True
        except ResourceExistsError:
            return False


class AzureSender:
    def __init__(self, sender):
        self.sender = sender

    def send(self, task: dict, message_id: str) -> None:
        from azure.servicebus import ServiceBusMessage

        self.sender.send_messages(ServiceBusMessage(
            encode(task), message_id=message_id, content_type="application/json",
        ))


def _extract_child(job, result_path):
    from .extractor import extract_range

    try:
        output = encode({"success": True, "result": asdict(extract_range(job))})
        if len(output) > MAX_RESULT_BYTES:
            raise ValueError("Range result is too large")
    except Exception as error:
        output = encode({"success": False, "error_type": type(error).__name__})
    Path(result_path).write_bytes(output)


def isolated_extract(job, timeout_seconds: float = 300):
    import math

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Extraction timeout must be finite and positive")
    context = mp.get_context("spawn")
    # A private result file avoids waiting on a partly written pipe message. The
    # parent waits for process exit before reading and can terminate a stuck child.
    with tempfile.TemporaryDirectory(prefix="azure-extractor-") as directory:
        result_path = Path(directory) / "result.json"
        process = context.Process(target=_extract_child, args=(job, str(result_path)))
        started = False
        try:
            process.start()
            started = True
            process.join(timeout=timeout_seconds)
            if process.is_alive():
                raise TimeoutError("Native range extraction timed out")
            if process.exitcode != 0 or not result_path.is_file():
                raise RuntimeError("Extraction process exited without a result")
            if result_path.stat().st_size > MAX_RESULT_BYTES:
                raise InvalidTask("Range result is too large")
            output = json.loads(result_path.read_bytes())
            if not output["success"]:
                raise RuntimeError("Native range extraction failed")
            result = output["result"]
            return PageRangeResult(result["job_id"], result["document_id"],
                                   [PageResult(**page) for page in result["pages"]])
        finally:
            if started:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)


def handle_message(receiver, message, blobs, extractor=isolated_extract) -> str:
    try:
        body = bytearray()
        for part in message.body:
            body.extend(part)
            if len(body) > 64 * 1024:
                raise InvalidTask("Task message is too large")
        try:
            task = json.loads(bytes(body))
        except (ValueError, UnicodeError):
            raise InvalidTask("Task message is not valid JSON") from None
        state = process_task(blobs, task, extractor)
    except InvalidTask:
        receiver.dead_letter_message(message, reason="InvalidRangeTask",
                                     error_description="Task or stored content failed validation")
        return "dead_lettered"
    except Exception:
        # Delivery limits on the queue eventually dead-letter repeat failures.
        # Do not print exceptions that might include URLs or credentials.
        receiver.abandon_message(message)
        return "retry_requested"
    # If settlement fails after publication, redelivery validates the saved result
    # and completes without repeating extraction. Do not undo the stored result.
    receiver.complete_message(message)
    return state
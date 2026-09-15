from dataclasses import replace

import pytest

from pdf_pipeline.distributed import (
    InvalidTask, IncompleteDocument, collect_records, dispatch, process_task, reconcile,
)
from pdf_pipeline.models import PageRangeResult, PageResult


class MemoryBlobs:
    def __init__(self):
        self.objects = {}

    def read(self, name, limit):
        value = self.objects.get(name)
        if value is not None and len(value) > limit:
            raise ValueError("Too large")
        return value

    def create(self, name, data):
        if name in self.objects:
            return False
        self.objects[name] = data
        return True


class MemorySender:
    def __init__(self):
        self.messages = []

    def send(self, task, message_id):
        self.messages.append((task, message_id))


def extracted(job):
    return PageRangeResult(job.job_id, job.document_id, [
        PageResult(f"{job.document_id}:page:{number + 1:06d}", job.document_id, number,
                   f"Page {number + 1} text", True)
        for number in range(job.start_page, job.end_page)
    ])


def test_distributed_ranges_are_idempotent_and_ordered():
    blobs, sender = MemoryBlobs(), MemorySender()
    report = dispatch(blobs, sender, b"%PDF-1.7\nfixture", "report", 23, 10)
    assert report["ranges"] == 3
    assert [(task["start_page"], task["end_page"]) for task, _ in sender.messages] == [(0, 10), (10, 20), (20, 23)]
    with pytest.raises(IncompleteDocument):
        collect_records(blobs, report["version"])
    for task, _ in reversed(sender.messages):
        assert process_task(blobs, task, extracted) == "completed"
        assert process_task(blobs, task, lambda job: pytest.fail("Duplicate extraction")) == "already_complete"
    records = collect_records(blobs, report["version"])
    assert [record["metadata"]["page"] for record in records] == list(range(1, 24))
    assert all(record["metadata"]["source"] == "report.pdf" for record in records)
    assert reconcile(blobs, sender, report["version"])["sent"] == 0


def test_reconcile_resends_only_missing_after_interruption():
    blobs, sender = MemoryBlobs(), MemorySender()
    report = dispatch(blobs, sender, b"%PDF-1.7\nfixture", "report", 25, 10)
    process_task(blobs, sender.messages[0][0], extracted)
    replacement = MemorySender()
    assert reconcile(blobs, replacement, report["version"])["sent"] == 2
    assert replacement.messages == sender.messages[1:]
    with pytest.raises(IncompleteDocument):
        collect_records(blobs, report["version"])


def test_failed_range_never_publishes_result():
    blobs, sender = MemoryBlobs(), MemorySender()
    report = dispatch(blobs, sender, b"%PDF-1.7\nfixture", "report", 5, 10)

    def failed(job):
        result = extracted(job)
        result.pages[0] = replace(result.pages[0], success=False, error="ValueError")
        return result

    with pytest.raises(RuntimeError):
        process_task(blobs, sender.messages[0][0], failed)
    with pytest.raises(IncompleteDocument):
        collect_records(blobs, report["version"])


def test_changed_pdf_is_a_separate_version():
    blobs, sender = MemoryBlobs(), MemorySender()
    first = dispatch(blobs, sender, b"%PDF-1.7\nfirst", "report", 1)
    process_task(blobs, sender.messages[0][0], extracted)
    second = dispatch(blobs, sender, b"%PDF-1.7\nsecond", "report", 1)
    assert first["version"] != second["version"]
    with pytest.raises(IncompleteDocument):
        collect_records(blobs, second["version"])


def test_forged_bounds_and_corrupt_source_are_rejected():
    blobs, sender = MemoryBlobs(), MemorySender()
    dispatch(blobs, sender, b"%PDF-1.7\nfixture", "report", 20)
    task = sender.messages[0][0]
    with pytest.raises(InvalidTask):
        process_task(blobs, {**task, "end_page": 19}, extracted)
    source = next(name for name in blobs.objects if name.startswith("sources/"))
    blobs.objects[source] = b"modified"
    with pytest.raises(InvalidTask, match="hash mismatch"):
        process_task(blobs, task, extracted)


class Receiver:
    def __init__(self):
        self.events = []

    def complete_message(self, message):
        self.events.append("complete")

    def abandon_message(self, message):
        self.events.append("abandon")

    def dead_letter_message(self, message, **kwargs):
        self.events.append("dead_letter")


def test_servicebus_completes_only_after_result_publication():
    from types import SimpleNamespace
    from pdf_pipeline.azure_adapters import handle_message
    from pdf_pipeline.distributed import encode, result_name

    blobs, sender = MemoryBlobs(), MemorySender()
    dispatch(blobs, sender, b"%PDF-1.7\nfixture", "report", 2)
    task = sender.messages[0][0]
    message = SimpleNamespace(body=[encode(task)])

    class CheckingReceiver(Receiver):
        def complete_message(self, message):
            assert result_name(task) in blobs.objects
            super().complete_message(message)

    receiver = CheckingReceiver()
    assert handle_message(receiver, message, blobs, extracted) == "completed"
    assert receiver.events == ["complete"]


def test_transient_failure_abandons_and_invalid_task_deadletters():
    from types import SimpleNamespace
    from pdf_pipeline.azure_adapters import handle_message
    from pdf_pipeline.distributed import encode

    blobs, sender = MemoryBlobs(), MemorySender()
    dispatch(blobs, sender, b"%PDF-1.7\nfixture", "report", 2)
    receiver = Receiver()

    def broken(job):
        raise TimeoutError

    handle_message(receiver, SimpleNamespace(body=[encode(sender.messages[0][0])]), blobs, broken)
    handle_message(receiver, SimpleNamespace(body=[b"invalid JSON"]), blobs, broken)
    assert receiver.events == ["abandon", "dead_letter"]


def test_lost_acknowledgment_is_safe_to_redeliver():
    from types import SimpleNamespace
    from pdf_pipeline.azure_adapters import handle_message
    from pdf_pipeline.distributed import encode

    blobs, sender = MemoryBlobs(), MemorySender()
    dispatch(blobs, sender, b"%PDF-1.7\nfixture", "report", 2)
    message = SimpleNamespace(body=[encode(sender.messages[0][0])])

    class LostLock(Receiver):
        def complete_message(self, message):
            raise RuntimeError("Lock expired")

    with pytest.raises(RuntimeError):
        handle_message(LostLock(), message, blobs, extracted)
    receiver = Receiver()
    assert handle_message(receiver, message, blobs, lambda job: pytest.fail("Re-extracted after lost ACK")) == "already_complete"
    assert receiver.events == ["complete"]


def test_azure_cli_preflight_without_cloud_calls(monkeypatch, capsys):
    import sys
    import azure_pipeline

    monkeypatch.setattr(azure_pipeline, "dependency_missing", lambda name: True)
    monkeypatch.setattr(sys, "argv", ["azure_pipeline.py", "worker", "--once"])
    assert azure_pipeline.main() == 2
    assert "missing_packages" in capsys.readouterr().out


def successful_child(job, path):
    from pathlib import Path
    from dataclasses import asdict
    from pdf_pipeline.distributed import encode

    Path(path).write_bytes(encode({"success": True, "result": asdict(extracted(job))}))


def crashed_child(job, path):
    import os

    os._exit(7)


def stuck_child(job, path):
    from threading import Event

    Event().wait()


@pytest.mark.parametrize("target,error", [(successful_child, None), (crashed_child, RuntimeError), (stuck_child, TimeoutError)])
def test_azure_worker_subprocess_is_bounded(monkeypatch, target, error):
    import multiprocessing as mp
    from pdf_pipeline.models import PageRangeJob
    import pdf_pipeline.azure_adapters as adapters

    before = {process.pid for process in mp.active_children()}
    monkeypatch.setattr(adapters, "_extract_child", target)
    job = PageRangeJob("range", "demo", "synthetic.pdf", 0, 2)
    if error:
        with pytest.raises(error):
            adapters.isolated_extract(job, timeout_seconds=1 if target is stuck_child else 10)
    else:
        assert adapters.isolated_extract(job, timeout_seconds=10) == extracted(job)
    assert {process.pid for process in mp.active_children()} <= before


def test_mid_dispatch_failure_is_reconcilable():
    blobs = MemoryBlobs()

    class InterruptedSender(MemorySender):
        def send(self, task, message_id):
            if self.messages:
                raise ConnectionError
            super().send(task, message_id)

    sender = InterruptedSender()
    with pytest.raises(ConnectionError):
        dispatch(blobs, sender, b"%PDF-1.7\nfixture", "report", 21)
    version = sender.messages[0][0]["version"]
    replacement = MemorySender()
    assert reconcile(blobs, replacement, version)["sent"] == 3


def test_blob_adapter_bounds_reads_and_uses_create_only(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from pdf_pipeline.azure_adapters import AzureBlobs

    errors = ModuleType("azure.core.exceptions")
    errors.ResourceNotFoundError = type("ResourceNotFoundError", (Exception,), {})
    errors.ResourceExistsError = type("ResourceExistsError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "azure.core.exceptions", errors)

    class Blob:
        data = None

        def get_blob_properties(self):
            if self.data is None:
                raise errors.ResourceNotFoundError()
            return SimpleNamespace(size=len(self.data))

        def download_blob(self, **kwargs):
            assert kwargs["max_concurrency"] == 1
            return SimpleNamespace(chunks=lambda: iter([self.data]))

        def upload_blob(self, data, overwrite):
            assert overwrite is False
            if self.data is not None:
                raise errors.ResourceExistsError()
            self.data = data

    blob = Blob()
    adapter = AzureBlobs(SimpleNamespace(get_blob_client=lambda name: blob))
    assert adapter.read("result", 10) is None
    assert adapter.create("result", b"data") is True
    assert adapter.create("result", b"new") is False
    assert adapter.read("result", 10) == b"data"
    with pytest.raises(InvalidTask, match="size"):
        adapter.read("result", 2)


def test_corrupt_result_blocks_collection():
    from pdf_pipeline.distributed import result_name

    blobs, sender = MemoryBlobs(), MemorySender()
    report = dispatch(blobs, sender, b"%PDF-1.7\nfixture", "report", 1)
    task = sender.messages[0][0]
    process_task(blobs, task, extracted)
    blobs.objects[result_name(task)] = b"{broken"
    with pytest.raises(InvalidTask, match="Stored range"):
        collect_records(blobs, report["version"])
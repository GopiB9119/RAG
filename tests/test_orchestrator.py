from pdf_pipeline.orchestrator import CoordinatorState, coordinator_cycle
from pdf_pipeline.distributed import process_task
from test_distributed import MemoryBlobs, MemorySender, extracted
import pytest


class UploadBlobs(MemoryBlobs):
    def __init__(self):
        super().__init__()
        self.uploads = {"incoming/report.pdf": ('"etag1"', b"%PDF-1.7\ninput")}

    def list_uploads(self, prefix, cursor, limit):
        return [{"name": name, "etag": value[0]} for name, value in self.uploads.items()], None

    def upload_matches(self, name, etag):
        return name in self.uploads and self.uploads[name][0] == etag

    def read_upload(self, name, etag, limit):
        assert self.upload_matches(name, etag)
        return self.uploads[name][1]


def test_upload_to_distributed_extraction_to_index_is_automatic(tmp_path, monkeypatch):
    import pdf_pipeline.orchestrator as module

    now = [100.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    state = CoordinatorState(tmp_path / "state")
    blobs, sender = UploadBlobs(), MemorySender()
    indexed = []

    def index(records):
        indexed.append(records)
        return len(records)

    def cycle():
        with state.consumer_lock():
            return coordinator_cycle(state, blobs, sender, lambda pdf: 23, index, check_seconds=1)

    assert cycle() == {"extracting": 1}
    assert len(sender.messages) == 3
    process_task(blobs, sender.messages[0][0], extracted)
    now[0] += 1
    assert cycle() == {"extracting": 1}
    assert indexed == []
    for task, _ in sender.messages:
        process_task(blobs, task, extracted)
    now[0] += 1
    assert cycle() == {"ready": 1}
    assert len(indexed) == 1 and len(indexed[0]) == 23
    assert indexed[0][0]["metadata"]["source"] == "azure-upload:incoming/report.pdf"
    assert indexed[0][0]["metadata"]["title"] == "report.pdf"
    assert cycle() == {"ready": 1}
    assert len(indexed) == 1


def test_restart_reconciles_partial_dispatch(tmp_path, monkeypatch):
    import pdf_pipeline.orchestrator as module

    now = [100.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    state = CoordinatorState(tmp_path / "state")
    blobs = UploadBlobs()

    class FailingSender(MemorySender):
        def send(self, task, message_id):
            if self.messages:
                raise ConnectionError("sensitive detail")
            super().send(task, message_id)

    sender = FailingSender()
    with state.consumer_lock():
        coordinator_cycle(state, blobs, sender, lambda pdf: 20, lambda records: len(records))
    assert state.summary() == {"pending": 1}
    restarted = CoordinatorState(tmp_path / "state")
    now[0] += 6
    replacement = MemorySender()
    with restarted.consumer_lock():
        coordinator_cycle(restarted, blobs, replacement, lambda pdf: 20, lambda records: len(records))
    assert restarted.summary() == {"extracting": 1}
    assert replacement.messages[0] == sender.messages[0]


def test_new_etag_supersedes_unfinished_version(tmp_path, monkeypatch):
    import pdf_pipeline.orchestrator as module

    now = [100.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    state = CoordinatorState(tmp_path / "state")
    blobs, sender = UploadBlobs(), MemorySender()
    indexed = []
    with state.consumer_lock():
        coordinator_cycle(state, blobs, sender, lambda pdf: 1, lambda rows: indexed.append(rows) or len(rows), check_seconds=1)
        old_task = sender.messages[0][0]
        blobs.uploads["incoming/report.pdf"] = ('"etag2"', b"%PDF-1.7\nupdated")
        now[0] += 2
        coordinator_cycle(state, blobs, sender, lambda pdf: 1, lambda rows: indexed.append(rows) or len(rows), check_seconds=1)
        new_task = sender.messages[-1][0]
        process_task(blobs, old_task, extracted)
        now[0] += 2
        coordinator_cycle(state, blobs, sender, lambda pdf: 1, lambda rows: indexed.append(rows) or len(rows), check_seconds=1)
        assert indexed == []
        process_task(blobs, new_task, extracted)
        now[0] += 2
        coordinator_cycle(state, blobs, sender, lambda pdf: 1, lambda rows: indexed.append(rows) or len(rows), check_seconds=1)
    assert indexed[0][0]["metadata"]["version"] == new_task["version"]


def test_document_deadline_stops_endless_reconciliation(tmp_path, monkeypatch):
    import pdf_pipeline.orchestrator as module

    now = [100.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    state = CoordinatorState(tmp_path / "state")
    blobs, sender = UploadBlobs(), MemorySender()
    with state.consumer_lock():
        coordinator_cycle(state, blobs, sender, lambda pdf: 1, lambda records: 1, check_seconds=1, document_timeout=5)
        now[0] += 6
        assert coordinator_cycle(state, blobs, sender, lambda pdf: 1, lambda records: 1) == {"failed": 1}
    assert len(sender.messages) == 1


def test_coordinate_preflight_does_not_contact_cloud(monkeypatch, capsys):
    import sys
    import azure_pipeline

    monkeypatch.setattr(azure_pipeline, "dependency_missing", lambda name: True)
    monkeypatch.setattr(sys, "argv", ["azure_pipeline.py", "coordinate", "--once"])
    assert azure_pipeline.main() == 2
    assert "missing_packages" in capsys.readouterr().out


def test_index_failures_are_bounded_and_do_not_mark_ready(tmp_path, monkeypatch, capsys):
    import pdf_pipeline.orchestrator as module

    now = [100.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    state = CoordinatorState(tmp_path / "state")
    blobs, sender = UploadBlobs(), MemorySender()

    def broken_index(records):
        raise RuntimeError("private endpoint contents")

    with state.consumer_lock():
        coordinator_cycle(state, blobs, sender, lambda pdf: 1, broken_index, check_seconds=1, max_attempts=2)
        process_task(blobs, sender.messages[0][0], extracted)
        now[0] += 2
        coordinator_cycle(state, blobs, sender, lambda pdf: 1, broken_index, check_seconds=1, max_attempts=2)
        assert state.summary() == {"extracting": 1}
        now[0] += 6
        coordinator_cycle(state, blobs, sender, lambda pdf: 1, broken_index, check_seconds=1, max_attempts=2)
        assert state.summary() == {"failed": 1}
        document_id = state.status()["documents"][0]["document_id"]
        state.retry_cloud(document_id, timeout=60)
        assert state.summary() == {"pending": 1}
    assert "private endpoint contents" not in capsys.readouterr().out


def test_ready_receipt_survives_coordinator_restart(tmp_path, monkeypatch):
    import pdf_pipeline.orchestrator as module

    now = [100.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    state = CoordinatorState(tmp_path / "state")
    blobs, sender = UploadBlobs(), MemorySender()
    with state.consumer_lock():
        coordinator_cycle(state, blobs, sender, lambda pdf: 1, lambda rows: len(rows), check_seconds=1)
        process_task(blobs, sender.messages[0][0], extracted)
        now[0] += 2
        coordinator_cycle(state, blobs, sender, lambda pdf: 1, lambda rows: len(rows), check_seconds=1)
    restarted = CoordinatorState(tmp_path / "state")
    with restarted.consumer_lock():
        coordinator_cycle(restarted, blobs, sender, lambda pdf: pytest.fail("Repeated dispatch"),
                          lambda rows: pytest.fail("Repeated indexing"))
    assert restarted.summary() == {"ready": 1}


@pytest.mark.parametrize("failure", [BrokenPipeError, ValueError])
def test_log_failure_never_requeues_completed_indexing(tmp_path, monkeypatch, failure):
    import pdf_pipeline.orchestrator as module

    now = [100.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    state = CoordinatorState(tmp_path / "state")
    blobs, sender = UploadBlobs(), MemorySender()
    indexed = []

    def output_closed(*args, **kwargs):
        raise failure("Output stream unavailable")

    monkeypatch.setattr(module, "print", output_closed, raising=False)
    with state.consumer_lock():
        coordinator_cycle(state, blobs, sender, lambda pdf: 1,
                          lambda rows: indexed.append(rows) or len(rows), check_seconds=1)
        assert state.summary() == {"extracting": 1}
        assert state.status()["documents"][0]["attempts"] == 0
        process_task(blobs, sender.messages[0][0], extracted)
        now[0] += 2
        coordinator_cycle(state, blobs, sender, lambda pdf: 1,
                          lambda rows: indexed.append(rows) or len(rows), check_seconds=1)
        assert state.summary() == {"ready": 1}
        assert state.status()["documents"][0]["attempts"] == 0
        coordinator_cycle(state, blobs, sender, lambda pdf: pytest.fail("Repeated dispatch"),
                          lambda rows: pytest.fail("Repeated indexing"))
    assert len(indexed) == 1


def test_log_failure_preserves_processing_error_and_retry_budget(tmp_path, monkeypatch):
    import pdf_pipeline.orchestrator as module

    state = CoordinatorState(tmp_path / "state")

    def output_closed(*args, **kwargs):
        raise BrokenPipeError

    def invalid_pdf(pdf):
        raise ValueError("Parsing failed")

    monkeypatch.setattr(module, "print", output_closed, raising=False)
    with state.consumer_lock():
        coordinator_cycle(state, UploadBlobs(), MemorySender(), invalid_pdf, lambda rows: 1, max_attempts=1)
    document = state.status()["documents"][0]
    assert document["state"] == "failed"
    assert document["error_type"] == "ValueError"
    assert document["attempts"] == 1


def test_cloud_state_target_cannot_change(tmp_path):
    state = CoordinatorState(tmp_path / "state")
    with state.consumer_lock():
        state.bind_cloud("account", "container", "queue", "incoming/", tmp_path / "index", "docs", 10)
        with pytest.raises(ValueError, match="different cloud"):
            state.bind_cloud("account", "container", "queue", "other/", tmp_path / "index", "docs", 10)


def test_scan_cursor_is_persisted_for_bounded_listing(tmp_path):
    state = CoordinatorState(tmp_path / "state")
    state.observe([], "next-page", timeout=60)
    assert CoordinatorState(tmp_path / "state").cursor() == "next-page"
    state.observe([], None, timeout=60)
    assert state.cursor() is None


def test_blob_upload_adapter_uses_etag_and_pagination(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from pdf_pipeline.azure_adapters import AzureBlobs

    core = ModuleType("azure.core")
    core.MatchConditions = SimpleNamespace(IfNotModified="must-match")
    monkeypatch.setitem(sys.modules, "azure.core", core)
    calls = []

    class Pages:
        continuation_token = "next-page"

        def __next__(self):
            return iter([SimpleNamespace(name="incoming/report.PDF", etag='"etag"'),
                         SimpleNamespace(name="incoming/upload.partial", etag='"partial"')])

    class Listing:
        def by_page(self, continuation_token):
            assert continuation_token == "previous-page"
            return Pages()

    class Blob:
        def get_blob_properties(self):
            return SimpleNamespace(size=4)

        def download_blob(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(chunks=lambda: iter([b"data"]))

    class Container:
        def list_blobs(self, **kwargs):
            assert kwargs == {"name_starts_with": "incoming/", "results_per_page": 10}
            return Listing()

        def get_blob_client(self, name):
            return Blob()

    adapter = AzureBlobs(Container())
    uploads, cursor = adapter.list_uploads("incoming/", "previous-page", 10)
    assert uploads == [{"name": "incoming/report.PDF", "etag": '"etag"'}]
    assert cursor == "next-page"
    assert adapter.read_upload("incoming/report.PDF", '"etag"', 4) == b"data"
    assert calls == [{"etag": '"etag"', "match_condition": "must-match", "max_concurrency": 1}]


def test_history_preserves_overwritten_generation_and_rejects_stale_update(tmp_path):
    state = CoordinatorState(tmp_path / "state")
    upload = {"name": "incoming/report.pdf", "etag": "etag1"}
    state.observe([upload], None, 60)
    first = state.due(1)[0]
    state.update(first, version="a" * 64, state="ready", chunk_count=2)
    state.observe([{**upload, "etag": "etag2"}], None, 60)
    second = state.due(1)[0]
    assert first["generation"] != second["generation"]
    historical = [row for row in state.history() if row["generation"] == first["generation"]]
    assert any(row["event"] == "superseded_observation" and row["state"] == "ready" and row["version"] == "a" * 64 for row in historical)
    with pytest.raises(RuntimeError, match="changed"):
        state.update(first, state="failed")
    assert state.due(1)[0]["state"] == "pending"
    assert CoordinatorState(tmp_path / "state").history() == state.history()


def test_explicit_retry_has_new_generation_and_preserves_failed_history(tmp_path):
    state = CoordinatorState(tmp_path / "state")
    state.observe([{"name": "incoming/report.pdf", "etag": "etag1"}], None, 60)
    first = state.due(1)[0]
    state.update(first, state="failed", error_type="TimeoutError")
    state.retry_cloud(first["document_id"], 60)
    second = state.due(1)[0]
    assert second["generation"] != first["generation"]
    assert second["publication_receipt"] is None and second["version"] is None
    assert any(row["event"] == "retry_requested" and row["state"] == "failed"
               and row["generation"] == first["generation"] for row in state.history())


def test_old_coordinator_state_migrates_without_inventing_receipt(tmp_path):
    import sqlite3

    root = tmp_path / "state"
    root.mkdir()
    with sqlite3.connect(root / "jobs.sqlite3") as connection:
        connection.executescript("""
            CREATE TABLE cloud_documents (
                name TEXT PRIMARY KEY, etag TEXT, document_id TEXT, state TEXT,
                version TEXT, attempts INTEGER, next_check REAL, deadline REAL,
                created_at REAL, updated_at REAL, error_type TEXT, chunk_count INTEGER
            );
            INSERT INTO cloud_documents VALUES ('incoming/report.pdf','etag1','report','ready',
                'old-version',1,0,60,0,0,NULL,5);
        """)
    state = CoordinatorState(root)
    row = state.status()["documents"][0]
    assert row["state"] == "ready" and row["chunk_count"] == 5
    assert len(row["generation"]) == 32 and row["has_publication_receipt"] == 0
    assert state.history()[0]["event"] == "legacy_import"
    assert len(CoordinatorState(root).history()) == 1


def test_index_receipt_recovers_crash_gap_without_second_embedding(tmp_path, monkeypatch):
    import ingest_sources
    import pdf_pipeline.orchestrator as module
    from test_rag import FakeCollection, install_index_fakes

    now = [100.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    state = CoordinatorState(tmp_path / "state")
    database = tmp_path / "index"
    state.bind_target(database, "documents")
    blobs, sender = UploadBlobs(), MemorySender()
    collection = FakeCollection()
    model = install_index_fakes(monkeypatch, collection)

    def index(records):
        return ingest_sources.build_index(ingest_sources.chunk_records(records), str(database), "documents", False,
                                          publication_generation=records[0]["metadata"]["processing_generation"])

    with state.consumer_lock():
        coordinator_cycle(state, blobs, sender, lambda pdf: 1, index, check_seconds=1, require_receipt=True)
        process_task(blobs, sender.messages[0][0], extracted)
        original_update = state.update

        def fail_ready(job, **values):
            if values.get("state") == "ready":
                raise OSError("Receipt persistence interrupted")
            return original_update(job, **values)

        monkeypatch.setattr(state, "update", fail_ready)
        now[0] += 2
        coordinator_cycle(state, blobs, sender, lambda pdf: 1, index, check_seconds=1, require_receipt=True)
        assert state.summary() == {"extracting": 1}
        assert len(model.batches) == 1
    restarted = CoordinatorState(state.root)
    now[0] += 6
    with restarted.consumer_lock():
        coordinator_cycle(restarted, blobs, sender, lambda pdf: 1, index, check_seconds=1, require_receipt=True)
    assert restarted.summary() == {"ready": 1}
    assert restarted.status()["documents"][0]["has_publication_receipt"] == 1
    assert len(model.batches) == 1 and collection.count() == 1
    preview = restarted.retirement_preview()
    assert preview["generations"][0]["publication_receipt_verified"] is True
    assert preview["generations"][0]["eligible_for_deletion"] is False
    assert preview["cloud_deletion_enabled"] is False


def test_required_receipt_cannot_be_replaced_by_chunk_count(tmp_path, monkeypatch):
    import pdf_pipeline.orchestrator as module

    now = [100.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    state = CoordinatorState(tmp_path / "state")
    blobs, sender = UploadBlobs(), MemorySender()
    with state.consumer_lock():
        coordinator_cycle(state, blobs, sender, lambda pdf: 1, lambda rows: 1, check_seconds=1, require_receipt=True)
        process_task(blobs, sender.messages[0][0], extracted)
        now[0] += 2
        coordinator_cycle(state, blobs, sender, lambda pdf: 1, lambda rows: 1, check_seconds=1,
                          require_receipt=True, max_attempts=1)
    assert state.summary() == {"failed": 1}
    assert state.status()["documents"][0]["has_publication_receipt"] == 0


def test_retirement_preview_is_bounded_and_legacy_is_not_eligible(tmp_path):
    state = CoordinatorState(tmp_path / "state")
    for number in range(3):
        state.observe([{"name": f"incoming/private-{number}.pdf", "etag": "etag"}], None, 60)
    preview = state.retirement_preview(limit=2)
    assert preview["has_more"] is True
    assert preview["artifact_bytes"] is None
    assert len(preview["generations"]) == 2
    assert all("committed_publication_receipt_missing" in row["blockers"]
               and row["eligible_for_deletion"] is False for row in preview["generations"])
    assert "private-" not in str(preview)


def test_coordinator_dispatches_shared_policy_and_rejects_changes(tmp_path):
    from pdf_pipeline.models import ExtractionOptions
    from pdf_pipeline.distributed import load_manifest

    state = CoordinatorState(tmp_path / "state")
    policy = ExtractionOptions(ocr="auto", language="eng+hin", dpi=200)
    blobs, sender = UploadBlobs(), MemorySender()
    with state.consumer_lock():
        state.bind_cloud("account", "container", "queue", "incoming/", tmp_path / "index", "docs", 10, policy)
        coordinator_cycle(state, blobs, sender, lambda pdf: 1, lambda rows: len(rows), extraction=policy)
        manifest = load_manifest(blobs, sender.messages[0][0]["version"])
        assert manifest.extraction == policy
        with pytest.raises(ValueError, match="different cloud"):
            state.bind_cloud("account", "container", "queue", "incoming/", tmp_path / "index", "docs", 10, ExtractionOptions())
    assert state.summary() == {"extracting": 1}


def test_overwritten_upload_before_finalize_is_not_indexed(tmp_path, monkeypatch):
    import pdf_pipeline.orchestrator as module

    now = [100.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    state = CoordinatorState(tmp_path / "state")
    blobs, sender = UploadBlobs(), MemorySender()
    with state.consumer_lock():
        coordinator_cycle(state, blobs, sender, lambda pdf: 1, lambda rows: len(rows), check_seconds=1)
        process_task(blobs, sender.messages[0][0], extracted)
        now[0] += 2
        monkeypatch.setattr(blobs, "upload_matches", lambda *args: False)
        coordinator_cycle(state, blobs, sender, lambda pdf: 1, lambda rows: pytest.fail("Indexed removed upload"))
    assert state.summary() == {"failed": 1}
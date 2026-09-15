from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from pdf_pipeline.job_store import JobStore


def pdf(tmp_path, name="source.pdf", text=b"first"):
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.7\n" + text)
    return path


def test_enqueue_deduplicates_and_keeps_snapshot(tmp_path):
    source = pdf(tmp_path)
    store = JobStore(tmp_path / "state")
    first = store.enqueue(source)
    assert store.enqueue(source)["id"] == first["id"]
    source.write_bytes(b"%PDF-1.7\nsecond")
    second = store.enqueue(source)
    assert second["id"] != first["id"]
    from pathlib import Path

    assert Path(first["snapshot"]).read_bytes() == b"%PDF-1.7\nfirst"
    assert store.counts() == {"queued": 2}


def test_duplicate_submissions_are_serialized(tmp_path):
    source = pdf(tmp_path)
    store = JobStore(tmp_path / "state")
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = list(pool.map(lambda unused: store.enqueue(source), range(8)))
    assert len({job["id"] for job in jobs}) == 1


def test_recovery_and_bounded_retry(tmp_path, monkeypatch):
    import pdf_pipeline.job_store as module

    clock = [100.0]
    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    store = JobStore(tmp_path / "state")
    queued = store.enqueue(pdf(tmp_path), max_attempts=2)
    with store.consumer_lock():
        assert store.claim()["attempts"] == 1
    reopened = JobStore(tmp_path / "state")
    with reopened.consumer_lock():
        assert reopened.recover_interrupted() == 1
        assert reopened.claim()["attempts"] == 2
        reopened.fail(queued["id"], "TimeoutError")
        clock[0] += 1000
        assert reopened.claim() is None
    assert reopened.counts() == {"failed": 1}


def test_backoff_and_source_version_ordering(tmp_path, monkeypatch):
    import pdf_pipeline.job_store as module

    clock = [100.0]
    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    store = JobStore(tmp_path / "state")
    source = pdf(tmp_path)
    first = store.enqueue(source)
    source.write_bytes(b"%PDF-1.7\nsecond")
    second = store.enqueue(source)
    other = store.enqueue(pdf(tmp_path, "other.pdf"))
    with store.consumer_lock():
        assert store.claim()["id"] == first["id"]
        store.fail(first["id"], "TimeoutError")
        assert store.claim()["id"] == other["id"]
        store.finish(other["id"], 3)
        assert store.claim() is None
        clock[0] += 5
        assert store.claim()["id"] == first["id"]
        store.finish(first["id"], 2)
        assert store.claim()["id"] == second["id"]


def test_only_one_consumer_can_hold_lock(tmp_path):
    store = JobStore(tmp_path / "state")
    with store.consumer_lock():
        with pytest.raises(RuntimeError, match="already running"):
            with JobStore(tmp_path / "state").consumer_lock():
                pytest.fail("Second consumer acquired lock")
    with store.consumer_lock():
        pass


@pytest.mark.parametrize("content,limit", [(b"not a PDF", 100), (b"", 100), (b"%PDF-1.7\nlarge", 4)])
def test_invalid_upload_does_not_queue_work(tmp_path, content, limit):
    source = tmp_path / "invalid.pdf"
    source.write_bytes(content)
    store = JobStore(tmp_path / "state")
    with pytest.raises(ValueError):
        store.enqueue(source, max_bytes=limit)
    assert store.counts() == {}
    assert list(store.snapshots.iterdir()) == []


def test_consumer_runs_jobs_and_does_not_repeat_ready_jobs(tmp_path, capsys):
    from jobs import consume

    store = JobStore(tmp_path / "state")
    job = store.enqueue(pdf(tmp_path))
    seen = []

    def processor(current):
        seen.append(current["id"])
        return 5

    result = consume(store, processor, tmp_path / "index", "documents")
    assert result["ready"] == 1
    assert consume(store, processor, tmp_path / "index", "documents")["processed"] == 0
    assert seen == [job["id"]]
    output = capsys.readouterr().out
    assert "job_ready" in output
    assert str(tmp_path) not in output
    assert store.list_jobs()[0]["chunk_count"] == 5


def test_consumer_errors_do_not_leak_secrets(tmp_path, capsys):
    from jobs import consume

    store = JobStore(tmp_path / "state")
    store.enqueue(pdf(tmp_path), max_attempts=1)

    def broken(job):
        raise RuntimeError("secret document contents")

    result = consume(store, broken, tmp_path / "index", "documents")
    assert result["errors"] == 1
    assert result["states"] == {"failed": 1}
    assert "secret document contents" not in capsys.readouterr().out


def test_store_rejects_target_change(tmp_path):
    store = JobStore(tmp_path / "state")
    store.bind_target(tmp_path / "index", "first")
    with pytest.raises(ValueError, match="another index"):
        store.bind_target(tmp_path / "index", "second")


def test_retry_only_latest_failed_version(tmp_path):
    store = JobStore(tmp_path / "state")
    source = pdf(tmp_path)
    first = store.enqueue(source, max_attempts=1)
    with store.consumer_lock():
        store.claim()
        store.fail(first["id"], "TimeoutError")
    store.retry(first["id"])
    assert store.counts() == {"queued": 1}
    with store.consumer_lock():
        store.claim()
        store.fail(first["id"], "TimeoutError")
    source.write_bytes(b"%PDF-1.7\nnew")
    store.enqueue(source)
    with pytest.raises(ValueError, match="newer source version"):
        store.retry(first["id"])


def test_work_preflight_does_not_claim_queued_jobs(tmp_path, monkeypatch, capsys):
    import sys
    import jobs

    store = JobStore(tmp_path / "state")
    store.enqueue(pdf(tmp_path))
    monkeypatch.setattr(jobs.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(sys, "argv", ["jobs.py", "--state-dir", str(store.root), "work"])
    assert jobs.main() == 2
    assert store.counts() == {"queued": 1}
    assert store.list_jobs()[0]["attempts"] == 0


def test_killed_process_releases_lock_and_job_is_recovered(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    from jobs import consume

    store = JobStore(tmp_path / "state")
    queued = store.enqueue(pdf(tmp_path))
    script = (
        "import sys; from threading import Event; sys.path.insert(0,sys.argv[1]); "
        "from pathlib import Path; from pdf_pipeline.job_store import JobStore\n"
        "store=JobStore(Path(sys.argv[2]))\n"
        "with store.consumer_lock():\n"
        "    store.claim()\n"
        "    print('claimed',flush=True)\n"
        "    Event().wait()\n"
    )
    with pytest.raises(subprocess.TimeoutExpired) as error:
        subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve().parents[1] / "src"), str(store.root)],
                       capture_output=True, timeout=3)
    assert b"claimed" in error.value.stdout
    assert store.counts() == {"running": 1}
    result = consume(store, lambda job: 2, tmp_path / "index", "documents")
    assert result["recovered"] == 1
    assert result["ready"] == 1
    assert store.list_jobs()[0]["id"] == queued["id"]
    assert store.list_jobs()[0]["attempts"] == 2


def test_processor_verifies_snapshot_preserves_source_and_reuses_model(tmp_path, monkeypatch):
    from pathlib import Path
    import ingest_sources
    from jobs import make_processor
    from test_rag import FakeCollection, install_index_fakes

    store = JobStore(tmp_path / "state")
    source = pdf(tmp_path)
    job = store.enqueue(source)
    collection = FakeCollection()
    model = install_index_fakes(monkeypatch, collection)

    def load_pdf(snapshot, *, source, workers, checkpoint_root, pages_per_task, extraction):
        assert str(snapshot) == job["snapshot"]
        assert source == job["source"]
        assert workers == 2
        assert checkpoint_root == str(store.root / "checkpoints")
        assert pages_per_task == 10
        assert extraction.ocr == "off"
        return [{"text": "Extracted text.", "metadata": {"source": source, "page": 1, "type": "pdf"}}]

    monkeypatch.setattr(ingest_sources, "load_pdf", load_pdf)
    process = make_processor(tmp_path / "index", "documents", 2)
    first = process(job)
    assert process(job) == first
    assert first["chunk_count"] == 1 and first["generation"] == job["id"]
    assert len(model.batches) == 1
    Path(job["snapshot"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity check"):
        process(job)


def test_scanner_waits_for_stability_and_detects_changes(tmp_path, monkeypatch):
    import pdf_pipeline.watcher as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    folder = tmp_path / "input"
    folder.mkdir()
    source = pdf(folder)
    store = JobStore(tmp_path / "state")
    scanner = module.FolderScanner(folder, store, stable_seconds=10)
    assert scanner.scan() == 0
    clock[0] = 9
    source.write_bytes(b"%PDF-1.7\nstill copying")
    assert scanner.scan() == 0
    clock[0] = 18
    assert scanner.scan() == 0
    clock[0] = 19
    assert scanner.scan() == 1
    clock[0] = 30
    assert scanner.scan() == 0
    source.write_bytes(b"%PDF-1.7\nfinished new revision")
    assert scanner.scan() == 0
    clock[0] = 40
    assert scanner.scan() == 1
    assert store.counts() == {"queued": 2}


def test_scanner_ignores_partial_files_and_reconciles_restart(tmp_path, monkeypatch):
    import pdf_pipeline.watcher as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    folder = tmp_path / "input"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    partial = nested / "new.partial"
    partial.write_bytes(b"%PDF-1.7\ncomplete")
    store = JobStore(tmp_path / "state")
    scanner = module.FolderScanner(folder, store, stable_seconds=1)
    assert scanner.scan() == 0
    partial.rename(nested / "new.PDF")
    assert scanner.scan() == 0
    clock[0] = 2
    assert scanner.scan() == 1
    restarted = module.FolderScanner(folder, store, stable_seconds=1)
    assert restarted.scan() == 0
    clock[0] = 4
    assert restarted.scan() == 1
    assert len(store.list_jobs()) == 1


def test_watcher_automatically_retries_and_holds_lock(tmp_path, monkeypatch):
    import jobs
    import pdf_pipeline.watcher as scanner_module
    import pdf_pipeline.job_store as store_module

    clock = [100.0]
    monkeypatch.setattr(scanner_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(store_module.time, "time", lambda: clock[0])
    folder = tmp_path / "input"
    folder.mkdir()
    pdf(folder)
    store = JobStore(tmp_path / "state")
    attempts = []

    class ControlledStop:
        def is_set(self):
            return clock[0] >= 110

        def wait(self, seconds):
            with pytest.raises(RuntimeError, match="already running"):
                with store.consumer_lock():
                    pass
            clock[0] += seconds

    def processor(job):
        attempts.append(job["attempts"])
        if len(attempts) == 1:
            raise RuntimeError("temporary failure")
        return 2

    jobs.watch(store, processor, folder, tmp_path / "index", "documents",
               poll_seconds=1, stable_seconds=1, stop=ControlledStop())
    assert attempts == [1, 2]
    assert store.counts() == {"ready": 1}
    with store.consumer_lock():
        pass


def test_scanner_rejects_invalid_file_without_blocking_valid_pdf(tmp_path, monkeypatch, capsys):
    import pdf_pipeline.watcher as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    folder = tmp_path / "input"
    folder.mkdir()
    (folder / "bad.pdf").write_bytes(b"invalid")
    pdf(folder, "good.pdf")
    store = JobStore(tmp_path / "state")
    scanner = module.FolderScanner(folder, store, stable_seconds=1)
    scanner.scan()
    clock[0] = 2
    assert scanner.scan() == 1
    assert scanner.scan() == 0
    assert store.counts() == {"queued": 1}
    assert capsys.readouterr().out.count("watch_upload_rejected") == 1


@pytest.mark.parametrize("code", [sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, 517])
def test_scanner_retries_queue_lock_without_losing_file(tmp_path, monkeypatch, capsys, code):
    import pdf_pipeline.watcher as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    folder = tmp_path / "input"
    folder.mkdir()
    source = pdf(folder)
    store = JobStore(tmp_path / "state")
    scanner = module.FolderScanner(folder, store, stable_seconds=1)
    scanner.scan()
    clock[0] = 2
    original_enqueue = store.enqueue

    def busy(*args):
        error = sqlite3.OperationalError("private path details")
        error.sqlite_errorcode = code
        raise error

    monkeypatch.setattr(store, "enqueue", busy)
    assert scanner.scan() == 0
    assert source not in scanner.submitted
    output = capsys.readouterr().out
    assert "watch_queue_busy" in output and "private path details" not in output
    monkeypatch.setattr(store, "enqueue", original_enqueue)
    assert scanner.scan() == 1
    assert scanner.scan() == 0
    assert store.counts() == {"queued": 1}


def test_scanner_does_not_hide_database_faults(tmp_path, monkeypatch):
    import pdf_pipeline.watcher as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    folder = tmp_path / "input"
    folder.mkdir()
    pdf(folder)
    store = JobStore(tmp_path / "state")
    scanner = module.FolderScanner(folder, store, stable_seconds=1)
    scanner.scan()
    clock[0] = 2

    def broken(*args):
        error = sqlite3.OperationalError("Database I/O failure")
        error.sqlite_errorcode = sqlite3.SQLITE_IOERR
        raise error

    monkeypatch.setattr(store, "enqueue", broken)
    with pytest.raises(sqlite3.OperationalError):
        scanner.scan()


def test_watch_preflight_does_not_start_or_claim_work(tmp_path, monkeypatch, capsys):
    import sys
    import jobs

    store = JobStore(tmp_path / "state")
    store.enqueue(pdf(tmp_path))
    monkeypatch.setattr(jobs.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(sys, "argv", ["jobs.py", "--state-dir", str(store.root), "watch", str(tmp_path / "input")])
    assert jobs.main() == 2
    assert store.list_jobs()[0]["attempts"] == 0
    assert "blocked" in capsys.readouterr().out
    assert not (tmp_path / "input").exists()


def test_watch_interrupt_releases_lock_and_leaves_job_recoverable(tmp_path, monkeypatch):
    from threading import Event
    import jobs

    folder = tmp_path / "input"
    folder.mkdir()
    store = JobStore(tmp_path / "state")
    store.enqueue(pdf(folder))

    def interrupted(job):
        raise KeyboardInterrupt

    jobs.watch(store, interrupted, folder, tmp_path / "index", "documents", stop=Event())
    assert store.counts() == {"running": 1}
    result = jobs.consume(store, lambda job: 2, tmp_path / "index", "documents")
    assert result["recovered"] == 1 and result["ready"] == 1


@pytest.mark.parametrize("option,value", [("--poll-seconds", "0"), ("--stable-seconds", "nan")])
def test_watch_rejects_invalid_intervals(option, value, monkeypatch):
    import sys
    import jobs

    monkeypatch.setattr(sys, "argv", ["jobs.py", "watch", option, value])
    with pytest.raises(SystemExit) as error:
        jobs.main()
    assert error.value.code == 2


def test_watch_rejects_state_inside_input(tmp_path):
    from pdf_pipeline.watcher import FolderScanner

    folder = tmp_path / "input"
    store = JobStore(folder / "state")
    with pytest.raises(ValueError, match="outside"):
        FolderScanner(folder, store)


def test_ready_cleanup_preserves_original_and_unfinished_references(tmp_path):
    from pathlib import Path
    from pdf_pipeline.checkpoints import RangeCheckpoints

    store = JobStore(tmp_path / "state")
    first_source = pdf(tmp_path, "first.pdf")
    second_source = pdf(tmp_path, "second.pdf")
    first = store.enqueue(first_source)
    second = store.enqueue(second_source)
    assert first["content_hash"] == second["content_hash"]
    checkpoint_dir = RangeCheckpoints.content_directory(store.root / "checkpoints", first["content_hash"])
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "result.json").write_text("checkpoint")
    with store.consumer_lock():
        store.claim()
        store.finish(first["id"], 1)
        assert store.cleanup_ready_artifacts(first["id"]) is False
        assert Path(first["snapshot"]).exists() and checkpoint_dir.exists()
        store.claim()
        store.finish(second["id"], 1)
        assert store.cleanup_ready_artifacts(second["id"]) is True
    assert first_source.exists() and second_source.exists()
    assert not Path(first["snapshot"]).exists()
    assert not checkpoint_dir.exists()
    assert store.enqueue(first_source)["id"] == first["id"]


def test_failed_job_preserves_retry_artifacts(tmp_path):
    from pathlib import Path

    store = JobStore(tmp_path / "state")
    first = store.enqueue(pdf(tmp_path), max_attempts=1)
    with store.consumer_lock():
        store.claim()
        store.fail(first["id"], "TimeoutError")
        assert store.cleanup_ready_artifacts(first["id"]) is False
    assert Path(first["snapshot"]).exists()


@pytest.mark.parametrize("failure", [PermissionError, sqlite3.OperationalError])
def test_cleanup_failure_does_not_requeue_success(tmp_path, monkeypatch, capsys, failure):
    from jobs import consume

    store = JobStore(tmp_path / "state")
    store.enqueue(pdf(tmp_path))

    def blocked_cleanup(job_id):
        raise failure("Private details")

    monkeypatch.setattr(store, "cleanup_ready_artifacts", blocked_cleanup)
    result = consume(store, lambda job: 1, tmp_path / "index", "documents")
    assert result["ready"] == 1 and result["errors"] == 0
    assert store.counts() == {"ready": 1}
    output = capsys.readouterr().out
    assert "cleanup_deferred" in output and "Private details" not in output


@pytest.mark.parametrize("failure", [BrokenPipeError, ValueError])
def test_cleanup_log_failure_cannot_requeue_ready_job(tmp_path, monkeypatch, failure):
    import jobs

    store = JobStore(tmp_path / "state")
    store.enqueue(pdf(tmp_path))

    def cleanup_failed(job_id):
        raise PermissionError("private details")

    def output_closed(*args, **kwargs):
        raise failure("closed output")

    monkeypatch.setattr(store, "cleanup_ready_artifacts", cleanup_failed)
    monkeypatch.setattr(jobs, "print", output_closed, raising=False)
    result = jobs.consume(store, lambda job: 1, tmp_path / "index", "documents")
    assert result["ready"] == 1 and result["errors"] == 0
    assert store.counts() == {"ready": 1}
    assert store.list_jobs()[0]["attempts"] == 1


def test_queue_extraction_policy_prevents_silent_config_reuse(tmp_path):
    from pdf_pipeline.models import ExtractionOptions

    store = JobStore(tmp_path / "state")
    source = pdf(tmp_path)
    queued = store.enqueue(source)
    native = ExtractionOptions()
    with store.consumer_lock():
        store.bind_extraction(native)
        store.claim()
        store.finish(queued["id"], 1)
        store.bind_extraction(native)
        with pytest.raises(ValueError, match="policy changed"):
            store.bind_extraction(ExtractionOptions(ocr="auto"))
    assert source.is_file()
    assert store.counts() == {"ready": 1}
    legacy = JobStore(tmp_path / "legacy")
    legacy.enqueue(source)
    with legacy.consumer_lock():
        legacy.claim()
        with pytest.raises(ValueError, match="unversioned"):
            legacy.bind_extraction(native)


def test_watch_cli_passes_one_shared_ocr_policy(tmp_path, monkeypatch):
    import sys
    import jobs
    from pdf_pipeline.models import ExtractionOptions

    seen = []
    monkeypatch.setattr(jobs.importlib.util, "find_spec", lambda name: object())

    def processor(database, collection, workers, pages_per_task, extraction):
        seen.append(extraction)
        return object()

    def watcher(*args, **kwargs):
        seen.append(kwargs["extraction"])
        assert kwargs["require_receipt"] is True

    monkeypatch.setattr(jobs, "make_processor", processor)
    monkeypatch.setattr(jobs, "watch", watcher)
    monkeypatch.setattr(sys, "argv", ["jobs.py", "--state-dir", str(tmp_path / "state"), "watch",
                                     str(tmp_path / "input"), "--ocr", "auto", "--ocr-language", "eng+hin",
                                     "--ocr-dpi", "200"])
    assert jobs.main() == 0
    assert seen == [ExtractionOptions(ocr="auto", language="eng+hin", dpi=200)] * 2


def test_local_publication_receipt_recovers_ready_write_gap(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import ingest_sources
    import jobs
    import pdf_pipeline.job_store as store_module
    from test_rag import FakeCollection, install_index_fakes

    now = [100.0]
    monkeypatch.setattr(store_module.time, "time", lambda: now[0])
    store = JobStore(tmp_path / "state")
    source = pdf(tmp_path)
    queued = store.enqueue(source)
    collection = FakeCollection()
    model = install_index_fakes(monkeypatch, collection)
    monkeypatch.setattr(ingest_sources, "load_pdf", lambda path, **kwargs: [
        {"text": "Extracted page.", "metadata": {"source": kwargs["source"], "page": 1}},
    ])
    processor = jobs.make_processor(tmp_path / "index", "documents", 2)

    def interrupted_finish(*args, **kwargs):
        raise OSError("Ready-state write failed")

    monkeypatch.setattr(store, "finish", interrupted_finish)
    first = jobs.consume(store, processor, tmp_path / "index", "documents", require_receipt=True)
    assert first["errors"] == 1 and first["states"] == {"queued": 1}
    assert len(model.batches) == 1 and collection.count() == 1
    assert Path(queued["snapshot"]).exists()
    reopened = JobStore(store.root)
    now[0] += 6
    retry = jobs.consume(reopened, processor, tmp_path / "index", "documents", require_receipt=True)
    assert retry["ready"] == 1 and retry["errors"] == 0
    assert len(model.batches) == 1 and collection.count() == 1
    stored = reopened.list_jobs()[0]
    receipt = json.loads(stored["publication_receipt"])
    assert receipt["generation"] == queued["id"] and receipt["source"] == str(source.resolve())
    assert receipt["chunk_count"] == stored["chunk_count"] == 1
    assert source.exists() and not Path(queued["snapshot"]).exists()


def test_production_local_consumer_requires_committed_receipt(tmp_path):
    from pathlib import Path
    from jobs import consume

    store = JobStore(tmp_path / "state")
    queued = store.enqueue(pdf(tmp_path), max_attempts=1)
    result = consume(store, lambda job: 1, tmp_path / "index", "documents", require_receipt=True)
    assert result["errors"] == 1 and store.counts() == {"failed": 1}
    assert Path(queued["snapshot"]).exists()
    assert store.list_jobs()[0]["publication_receipt"] is None


def test_local_completion_rejects_another_jobs_receipt(tmp_path, monkeypatch):
    from pathlib import Path
    from jobs import consume
    from ingest_sources import DocumentPublisher
    from test_rag import FakeCollection, install_index_fakes

    collection = FakeCollection()
    install_index_fakes(monkeypatch, collection)
    store = JobStore(tmp_path / "state")
    source = pdf(tmp_path)
    queued = store.enqueue(source, max_attempts=1)
    publisher = DocumentPublisher(str(tmp_path / "index"), "documents")
    wrong_receipt = publisher.publish(
        [{"text": "Page text.", "metadata": {"source": str(source.resolve()), "page": 1}}], "a" * 32,
    )
    result = consume(store, lambda job: wrong_receipt, tmp_path / "index", "documents", require_receipt=True)
    assert result["errors"] == 1 and store.counts() == {"failed": 1}
    assert Path(queued["snapshot"]).exists()


def test_legacy_local_jobs_migrate_without_invented_receipts(tmp_path):
    root = tmp_path / "state"
    root.mkdir()
    with sqlite3.connect(root / "jobs.sqlite3") as connection:
        connection.executescript("""
            CREATE TABLE jobs (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
                source TEXT, content_hash TEXT, snapshot TEXT, state TEXT,
                attempts INTEGER, max_attempts INTEGER, available_at REAL,
                created_at REAL, updated_at REAL, error_type TEXT, chunk_count INTEGER
            );
            INSERT INTO jobs (id,source,content_hash,snapshot,state,attempts,max_attempts,
                available_at,created_at,updated_at,chunk_count)
                VALUES ('old-job','old.pdf','hash','snapshot.pdf','ready',1,3,0,0,0,5);
        """)
    store = JobStore(root)
    old = store.list_jobs()[0]
    assert old["state"] == "ready" and old["chunk_count"] == 5
    assert old["publication_receipt"] is None
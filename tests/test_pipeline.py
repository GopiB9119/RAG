import json
import subprocess
import sys
from pathlib import Path

import pytest

from pdf_pipeline.collector import collect_results
from pdf_pipeline.models import PageResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_collector_orders_pages_and_writes_outputs(tmp_path):
    results = [
        PageResult("demo:2", "demo", 1, "Second page", True),
        PageResult("demo:1", "demo", 0, "First page", True),
    ]
    summary = collect_results("demo", results, str(tmp_path))
    output = tmp_path / "demo"
    text = (output / "document.txt").read_text(encoding="utf-8")
    pages = [json.loads(line) for line in (output / "pages.jsonl").read_text(encoding="utf-8").splitlines()]
    assert text.index("First page") < text.index("Second page")
    assert [page["page_number"] for page in pages] == [1, 2]
    assert summary["successful_pages"] == 2
    assert summary["status"] == "complete"
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == summary


def test_collector_reports_failed_pages(tmp_path):
    summary = collect_results(
        "demo", [PageResult("demo:1", "demo", 0, "", False, "Missing PDF")], str(tmp_path)
    )
    assert summary["status"] == "incomplete"
    assert summary["failed_pages"] == [1]
    assert summary["failed_pages_count"] == 1


def test_collector_in_memory_mode_creates_no_exports(tmp_path):
    summary = collect_results("document", [PageResult("page:1", "document", 0, "Text", True)],
                              str(tmp_path), write_outputs=False)
    assert summary["status"] == "complete"
    assert summary["output_files"] == {}
    assert summary["pages"][0]["page_number"] == 1
    assert list(tmp_path.iterdir()) == []


def test_worker_pool_extracts_three_pages(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")

    pdf_path = tmp_path / "test.pdf"
    with pymupdf.open() as document:
        for page_number in range(3):
            page = document.new_page()
            page.insert_text((72, 72), f"This is test page {page_number + 1}")
        document.save(pdf_path)

    result = subprocess.run(
        [sys.executable, "-c",
         "import json,sys; sys.path.insert(0,sys.argv[1]); "
         "from pdf_pipeline.main import run_pipeline; "
         "print(json.dumps(run_pipeline(sys.argv[2], 'test-document', 2, sys.argv[3])))",
         str(PROJECT_ROOT / "src"), str(pdf_path), str(tmp_path / "output")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["successful_pages"] == 3
    assert summary["failed_pages_count"] == 0
    assert summary["worker_count"] == 2
    output = tmp_path / "output" / "test-document"
    pages = [json.loads(line) for line in (output / "pages.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [page["page_index"] for page in pages] == [0, 1, 2]
    for page_number, page in enumerate(pages, start=1):
        assert f"This is test page {page_number}" in page["text"]


@pytest.mark.parametrize("workers", [0, -1])
def test_pipeline_rejects_invalid_worker_count(workers):
    from pdf_pipeline.main import run_pipeline

    with pytest.raises(ValueError, match="workers must be at least 1"):
        run_pipeline("missing.pdf", "test-document", workers, "unused")


def test_rag_uses_worker_pool_and_preserves_citations(tmp_path, monkeypatch):
    from types import ModuleType
    import ingest_sources

    calls = []
    pipeline = ModuleType("pdf_pipeline.main")

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs)
        return collect_results(
            kwargs["document_id"],
            [PageResult("page:1", kwargs["document_id"], 0, "First  page.", True),
             PageResult("page:2", kwargs["document_id"], 1, "", True),
             PageResult("page:3", kwargs["document_id"], 2, "Third page.", True)],
            kwargs["output_root"], write_outputs=kwargs["write_outputs"],
        )

    pipeline.run_pipeline = fake_run_pipeline
    monkeypatch.setitem(sys.modules, "pdf_pipeline.main", pipeline)
    source = tmp_path / "report.pdf"
    records = ingest_sources.load_pdf(source)
    assert calls[0]["pdf_path"] == str(source.resolve())
    assert calls[0]["workers"] == 4
    assert [record["metadata"]["page"] for record in records] == [1, 3]
    assert all(record["metadata"]["source"] == str(source.resolve()) for record in records)
    chunks = ingest_sources.chunk_records(records)
    assert chunks[0]["text"] == "First page."
    assert chunks[0]["metadata"]["title"] == "report.pdf"
    assert chunks[0]["metadata"]["chunk"] == 0
    assert calls[0]["write_outputs"] is False


def test_downloaded_pdf_uses_worker_pool_with_original_url(monkeypatch):
    from types import ModuleType
    import ingest_sources

    paths = []
    pipeline = ModuleType("pdf_pipeline.main")

    def fake_run_pipeline(**kwargs):
        path = Path(kwargs["pdf_path"])
        paths.append(path)
        assert kwargs["workers"] == 2
        assert path.read_bytes() == b"downloaded PDF content"
        return collect_results(
            kwargs["document_id"],
            [PageResult("page:1", kwargs["document_id"], 0, "Downloaded text.", True)],
            kwargs["output_root"], write_outputs=kwargs["write_outputs"],
        )

    pipeline.run_pipeline = fake_run_pipeline
    monkeypatch.setitem(sys.modules, "pdf_pipeline.main", pipeline)
    url = "https://example.com/report.pdf"
    records = ingest_sources.extract_pdf(b"downloaded PDF content", url, workers=2)
    assert records[0]["metadata"]["source"] == url
    assert records[0]["metadata"]["page"] == 1
    assert not paths[0].exists()


def test_rag_rejects_incomplete_extraction(monkeypatch):
    from types import ModuleType
    import ingest_sources

    pipeline = ModuleType("pdf_pipeline.main")
    pipeline.run_pipeline = lambda **kwargs: {"status": "incomplete", "failed_pages": [2]}
    monkeypatch.setitem(sys.modules, "pdf_pipeline.main", pipeline)
    with pytest.raises(RuntimeError, match="PDF extraction failed on pages"):
        ingest_sources.load_pdf(Path("report.pdf"))


def test_ingestion_cli_connects_pipeline_to_index(tmp_path, monkeypatch):
    from types import ModuleType
    import ingest_sources

    pdf_path = tmp_path / "report.pdf"
    pdf_path.touch()
    calls = []
    indexed = []
    pipeline = ModuleType("pdf_pipeline.main")

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs)
        return collect_results(
            kwargs["document_id"],
            [PageResult("page:1", kwargs["document_id"], 0, "Revenue increased.", True)],
            kwargs["output_root"], write_outputs=kwargs["write_outputs"],
        )

    def fake_build_index(chunks, database, collection_name, reset):
        indexed.extend(chunks)
        assert database == str(tmp_path / "index")
        assert collection_name == "pipeline-test"
        assert reset is False
        return len(chunks)

    pipeline.run_pipeline = fake_run_pipeline
    monkeypatch.setitem(sys.modules, "pdf_pipeline.main", pipeline)
    monkeypatch.setattr(ingest_sources, "build_index", fake_build_index)
    monkeypatch.setattr(sys, "argv", [
        "ingest_sources.py", "--pdf-dir", str(tmp_path), "--workers", "2",
        "--urls-file", str(tmp_path / "no-urls.txt"),
        "--database", str(tmp_path / "index"), "--collection", "pipeline-test",
    ])
    ingest_sources.main()
    assert len(calls) == 1
    assert calls[0]["workers"] == 2
    assert indexed == [{"text": "Revenue increased.", "metadata": {
        "source": str(pdf_path.resolve()), "title": "report.pdf",
        "type": "pdf", "page": 1, "chunk": 0,
    }}]


@pytest.mark.parametrize("workers", ["0", "-1"])
def test_ingestion_cli_rejects_invalid_workers(workers, monkeypatch):
    import ingest_sources

    monkeypatch.setattr(sys, "argv", ["ingest_sources.py", "--workers", workers])
    with pytest.raises(SystemExit) as error:
        ingest_sources.parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize("text", [
    " ".join(f"word{number}" for number in range(100)),
    "x" * 301,
    "First sentence here. Second sentence here. Third sentence here.",
])
def test_chunks_respect_limit_without_losing_text(text):
    from ingest_sources import split_into_chunks

    chunks = split_into_chunks(text, chunk_size=30, overlap=0)
    assert all(0 < len(chunk) <= 30 for chunk in chunks)
    assert "".join("".join(chunks).split()) == "".join(text.split())


def test_chunk_overlap_respects_limit():
    from ingest_sources import split_into_chunks

    text = "First sentence here. Second sentence here. Third sentence here."
    chunks = split_into_chunks(text, chunk_size=30, overlap=2)
    assert all(len(chunk) <= 30 for chunk in chunks)
    assert all(sentence in " ".join(chunks) for sentence in (
        "First sentence here.", "Second sentence here.", "Third sentence here.",
    ))


def simulated_page_worker(job_queue, result_queue):
    while True:
        job = job_queue.get()
        if job is None:
            return
        result_queue.put(PageResult(job.job_id, job.document_id, job.page_index, "Page text.", True))


def crashed_page_worker(job_queue, result_queue):
    import os

    os._exit(7)


def stuck_page_worker(job_queue, result_queue):
    from threading import Event

    Event().wait()


def duplicate_page_worker(job_queue, result_queue):
    job = job_queue.get()
    result = PageResult(job.job_id, job.document_id, job.page_index, "Page text.", True)
    result_queue.put(result)
    result_queue.put(result)


@pytest.mark.parametrize("target,error", [
    (simulated_page_worker, None),
    (crashed_page_worker, RuntimeError),
    (stuck_page_worker, TimeoutError),
    (duplicate_page_worker, RuntimeError),
])
def test_worker_lifecycle_with_real_processes(tmp_path, monkeypatch, target, error):
    import multiprocessing as mp
    import pdf_pipeline.main as pipeline
    from pdf_pipeline.models import PageJob

    path = tmp_path / "placeholder.pdf"
    path.touch()
    jobs = [PageJob(f"demo:{number}", "demo", str(path), number) for number in range(2)]
    monkeypatch.setattr(pipeline, "create_page_jobs", lambda **kwargs: jobs)
    monkeypatch.setattr(pipeline, "worker_loop", target)
    before = {process.pid for process in mp.active_children()}
    arguments = dict(pdf_path=str(path), document_id="demo", workers=1,
                     output_root=str(tmp_path), timeout_seconds=1 if target is stuck_page_worker else 15)
    if error:
        with pytest.raises(error):
            pipeline.run_pipeline(**arguments)
        assert not (tmp_path / "demo" / "summary.json").exists()
    else:
        summary = pipeline.run_pipeline(**arguments)
        assert summary["successful_pages"] == 2
        assert summary["worker_count"] == 1
        assert json.loads((tmp_path / "demo" / "summary.json").read_text()) == summary
    assert {process.pid for process in mp.active_children()} <= before


def test_live_pdf_extraction_index_and_retrieval(tmp_path, monkeypatch):
    import os

    if os.environ.get("RUN_RAG_LIVE") != "1":
        pytest.skip("Set RUN_RAG_LIVE=1 to run native PDF, embedding-model and Chroma integration")
    import pymupdf
    import chromadb
    from sentence_transformers import SentenceTransformer
    from ingest_sources import load_pdf, chunk_records, build_index
    from rag_core import MODEL_NAME, retrieve, validate_collection

    pdf_path = tmp_path / "synthetic-budget.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "The ORBIT42 project budget is 731 credits.")
        document.new_page()
        page = document.new_page()
        page.insert_text((72, 72), "The ORBIT42 project owner is Mira.")
        document.save(pdf_path)

    records = load_pdf(pdf_path, workers=2)
    assert [record["metadata"]["page"] for record in records] == [1, 3]
    chunks = chunk_records(records)
    database = str(tmp_path / "live-index")
    assert build_index(chunks, database, "live_test", False) == len(chunks)
    client = chromadb.PersistentClient(path=database)
    collection = client.get_collection("live_test")
    validate_collection(collection)
    assert collection.count() == len(chunks)
    model = SentenceTransformer(MODEL_NAME)
    monkeypatch.setenv("RAG_TOP_K", "4")
    monkeypatch.setenv("RAG_MAX_DISTANCE", "1.6")
    evidence = retrieve("What is the ORBIT42 project budget?", collection, model, collection.count())
    assert any("731 credits" in text and metadata["page"] == 1
               and metadata["source"] == str(pdf_path.resolve()) for text, metadata, _ in evidence)

    updated = [{**chunks[0], "text": "The ORBIT42 project budget is now 982 credits."}]
    assert build_index(updated, database, "live_test", False) == 1
    assert collection.count() == 1
    stored = collection.get(include=["documents"])["documents"]
    assert stored == [updated[0]["text"]]

    if os.environ.get("RUN_RAG_AZURE") == "1":
        from rag_core import generate_answer

        evidence = retrieve("What is the ORBIT42 project budget?", collection, model, collection.count())
        try:
            answer = generate_answer("What is the ORBIT42 project budget?", evidence)
        except Exception as error:
            pytest.fail(f"Azure live request failed ({type(error).__name__}); credentials and response details hidden", pytrace=False)
        assert "982" in answer
        assert "Source:" in answer


def test_baseline_reports_missing_dependencies_without_processing(tmp_path, monkeypatch):
    import baseline

    monkeypatch.setattr(baseline.importlib.util, "find_spec", lambda name: None)
    report, path = baseline.run_baseline(tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == report
    assert saved["status"] == "blocked"
    assert saved["missing_packages"] == list(baseline.REQUIRED_MODULES)
    assert saved["expected_pages"] == 300
    assert saved["stages"] == []
    assert not (path.parent / "input").exists()
    assert not (path.parent / "index").exists()


def test_baseline_fact_identifiers_are_unique():
    from baseline import DOCUMENT_COUNT, PAGES_PER_DOCUMENT, page_fact

    facts = [page_fact(document, page) for document in range(1, DOCUMENT_COUNT + 1)
             for page in range(1, PAGES_PER_DOCUMENT + 1)]
    assert len(facts) == 300
    assert len({reference for reference, _, _ in facts}) == 300
    assert all(reference in text and amount in text for reference, amount, text in facts)


def test_baseline_refuses_invalid_worker_count(tmp_path):
    from baseline import run_baseline

    with pytest.raises(ValueError, match="workers must be at least 1"):
        run_baseline(tmp_path, workers=0)
    assert list(tmp_path.iterdir()) == []


def test_baseline_runs_have_separate_output_directories(tmp_path, monkeypatch):
    import baseline

    monkeypatch.setattr(baseline.importlib.util, "find_spec", lambda name: None)
    _, first = baseline.run_baseline(tmp_path)
    _, second = baseline.run_baseline(tmp_path)
    assert first != second
    assert first.is_file() and second.is_file()


@pytest.mark.parametrize("azure", [False, True])
def test_baseline_full_control_flow_with_service_doubles(tmp_path, monkeypatch, azure):
    from types import ModuleType, SimpleNamespace
    import baseline
    import ingest_sources
    import rag_core

    monkeypatch.setattr(baseline.importlib.util, "find_spec", lambda name: object())
    fixtures = []
    expected = {}
    indexed = []
    answers = []

    def fake_fixture(directory):
        for document_number in range(1, 6):
            path = directory / f"synthetic-{document_number:02d}.pdf"
            questions = []
            for page_number in (1, 30, 60):
                reference, amount, text = baseline.page_fact(document_number, page_number)
                question = f"Budget for {reference}?"
                questions.append({"question": question, "reference": reference,
                                  "expected_answer": amount, "source": str(path.resolve()), "page": page_number})
                expected[question] = (text, {"source": str(path.resolve()), "page": page_number}, 0.1)
            fixtures.append({"path": path, "number": document_number, "questions": questions})
        return fixtures

    def fake_load_pdf(path, workers, pages_per_task, checkpoint_root):
        assert workers == 2
        assert pages_per_task == 10
        assert Path(checkpoint_root).name == "checkpoints"
        number = next(fixture["number"] for fixture in fixtures if fixture["path"] == path)
        return [{"text": baseline.page_fact(number, page)[2],
                 "metadata": {"source": str(path.resolve()), "page": page, "title": path.name, "type": "pdf"}}
                for page in range(1, 61)]

    def fake_index(chunks, database, collection_name, reset):
        assert Path(database).parent.parent == tmp_path
        assert collection_name == "baseline_documents" and reset is False
        indexed.extend(chunks)
        return len(chunks)

    def fake_answer(question, evidence):
        answers.append(question)
        text, metadata, _ = expected[question]
        return f"{text} Source: {Path(metadata['source']).name}, page {metadata['page']}."

    collection = SimpleNamespace(metadata=dict(rag_core.COLLECTION_METADATA), count=lambda: len(indexed))
    chroma = ModuleType("chromadb")
    chroma.PersistentClient = lambda **kwargs: SimpleNamespace(get_collection=lambda name: collection)
    transformers = ModuleType("sentence_transformers")
    transformers.SentenceTransformer = lambda name: object()
    monkeypatch.setitem(sys.modules, "chromadb", chroma)
    monkeypatch.setitem(sys.modules, "sentence_transformers", transformers)
    monkeypatch.setattr(baseline, "create_fixture", fake_fixture)
    monkeypatch.setattr(ingest_sources, "load_pdf", fake_load_pdf)
    monkeypatch.setattr(ingest_sources, "build_index", fake_index)
    monkeypatch.setattr(rag_core, "retrieve", lambda question, *args: [expected[question]])
    monkeypatch.setattr(rag_core, "generate_answer", fake_answer)
    for setting in rag_core.REQUIRED_AZURE_SETTINGS:
        monkeypatch.setenv(setting, "test-only")
    report, path = baseline.run_baseline(tmp_path, workers=2, azure=azure)
    assert report["status"] == ("passed" if azure else "passed_without_azure")
    assert report["extracted_pages"] == report["indexed_chunks"] == 300
    assert len(report["documents"]) == 5
    assert len(report["questions"]) == 15
    assert len(answers) == (15 if azure else 0)
    assert json.loads(path.read_text()) == report


def test_baseline_sanitizes_unexpected_failures(tmp_path, monkeypatch):
    import baseline

    def broken_dependency(name):
        raise RuntimeError("sensitive details must not appear in reports")

    monkeypatch.setattr(baseline.importlib.util, "find_spec", broken_dependency)
    report, path = baseline.run_baseline(tmp_path)
    assert report["status"] == "failed"
    assert report["failed_stage"] == "preflight"
    assert report["error_type"] == "RuntimeError"
    assert "sensitive details" not in path.read_text()


@pytest.mark.parametrize("page_count,range_size", [(1, 10), (60, 10), (63, 10), (1001, 32)])
def test_range_jobs_cover_every_page_exactly_once(page_count, range_size):
    from pdf_pipeline.scheduler import split_page_ranges

    jobs = split_page_ranges("input.pdf", "demo", page_count, range_size)
    assert [page for job in jobs for page in range(job.start_page, job.end_page)] == list(range(page_count))
    assert len({job.job_id for job in jobs}) == len(jobs)
    assert all(0 < job.end_page - job.start_page <= range_size for job in jobs)


def test_range_extractor_opens_pdf_once_and_preserves_page_errors(monkeypatch):
    from types import ModuleType
    from pdf_pipeline.models import PageRangeJob
    from pdf_pipeline.extractor import extract_range

    opens = []

    class FakePage:
        def __init__(self, number):
            self.number = number

        def get_text(self, mode):
            if self.number == 2:
                raise ValueError("Unusable page")
            return f"Page {self.number + 1}"

    class FakeDocument:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __len__(self):
            return 5

        def __getitem__(self, index):
            return FakePage(index)

    native = ModuleType("pymupdf")
    native.open = lambda path: (opens.append(path), FakeDocument())[1]
    monkeypatch.setitem(sys.modules, "pymupdf", native)
    result = extract_range(PageRangeJob("range", "demo", "input.pdf", 1, 4))
    assert opens == ["input.pdf"]
    assert [page.page_index for page in result.pages] == [1, 2, 3]
    assert [page.success for page in result.pages] == [True, False, True]
    assert result.pages[0].text == "Page 2"


def range_result(job):
    from pdf_pipeline.models import PageRangeResult

    return PageRangeResult(job.job_id, job.document_id, [
        PageResult(f"{job.document_id}:page:{number + 1:06d}", job.document_id, number, f"Page {number + 1}", True)
        for number in range(job.start_page, job.end_page)
    ])


def test_checkpoints_validate_identity_and_detect_corruption(tmp_path):
    from pdf_pipeline.checkpoints import RangeCheckpoints
    from pdf_pipeline.scheduler import split_page_ranges

    job = split_page_ranges("pdf", "demo", 3, 2)[0]
    checkpoints = RangeCheckpoints(tmp_path, "demo", "versionA", 2)
    result = range_result(job)
    checkpoints.save(job, result)
    assert checkpoints.load(job) == result
    assert RangeCheckpoints(tmp_path, "demo", "versionB", 2).load(job) is None
    assert RangeCheckpoints(tmp_path, "demo", "versionA", 3).load(job) is None
    payload = json.loads(checkpoints.path(job).read_text())
    payload["result"]["pages"][0]["text"] = "damaged"
    checkpoints.path(job).write_text(json.dumps(payload))
    assert checkpoints.load(job) is None
    checkpoints.path(job).write_text("{truncated")
    assert checkpoints.load(job) is None


def test_failed_and_mismatched_ranges_are_not_checkpointed(tmp_path):
    from pdf_pipeline.checkpoints import RangeCheckpoints
    from pdf_pipeline.scheduler import split_page_ranges

    job = split_page_ranges("pdf", "demo", 3, 2)[0]
    checkpoints = RangeCheckpoints(tmp_path, "demo", "versionA", 2)
    result = range_result(job)
    result.pages[0].success = False
    result.pages[0].error = "ValueError"
    checkpoints.save(job, result)
    assert checkpoints.load(job) is None
    result.pages[0].page_index = 99
    with pytest.raises(RuntimeError, match="Mismatched page"):
        checkpoints.save(job, result)


def successful_range_worker(job_queue, result_queue):
    while True:
        job = job_queue.get()
        if job is None:
            return
        result_queue.put(range_result(job))


def incomplete_range_worker(job_queue, result_queue):
    job = job_queue.get()
    result_queue.put(range_result(job))


def test_range_pool_resumes_after_worker_exit(tmp_path, monkeypatch):
    import pdf_pipeline.main as pipeline
    from pdf_pipeline.scheduler import split_page_ranges

    path = tmp_path / "input.pdf"
    path.write_bytes(b"synthetic test placeholder")
    monkeypatch.setattr(pipeline, "create_page_range_jobs", lambda path, document_id, size:
                        split_page_ranges(path, document_id, 5, size))
    arguments = dict(pdf_path=str(path), document_id="demo", workers=1,
                     output_root=str(tmp_path / "output"), pages_per_task=2,
                     checkpoint_root=str(tmp_path / "checkpoints"), timeout_seconds=15)
    monkeypatch.setattr(pipeline, "worker_loop", incomplete_range_worker)
    with pytest.raises(RuntimeError, match="before returning all"):
        pipeline.run_pipeline(**arguments)
    assert len(list((tmp_path / "checkpoints").rglob("*.json"))) == 1
    monkeypatch.setattr(pipeline, "worker_loop", successful_range_worker)
    result = pipeline.run_pipeline(**arguments)
    assert result["successful_pages"] == 5
    assert result["range_count"] == 3 and result["reused_ranges"] == 1
    all_cached = pipeline.run_pipeline(**arguments)
    assert all_cached["reused_ranges"] == 3 and all_cached["worker_count"] == 0
    records = [json.loads(line) for line in (tmp_path / "output" / "demo" / "pages.jsonl").read_text().splitlines()]
    assert [record["page_index"] for record in records] == list(range(5))
    path.write_bytes(b"new synthetic content")
    changed = pipeline.run_pipeline(**arguments)
    assert changed["reused_ranges"] == 0


def test_rag_adapter_forwards_checkpoint_settings(tmp_path, monkeypatch):
    from types import ModuleType
    import ingest_sources

    pipeline = ModuleType("pdf_pipeline.main")

    def fake_pipeline(**kwargs):
        assert kwargs["pages_per_task"] == 10
        assert kwargs["checkpoint_root"] == str(tmp_path / "checkpoints")
        return collect_results(kwargs["document_id"], [
            PageResult("page:1", kwargs["document_id"], 0, "Range extracted text.", True),
        ], kwargs["output_root"], write_outputs=kwargs["write_outputs"])

    pipeline.run_pipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "pdf_pipeline.main", pipeline)
    records = ingest_sources.load_pdf(tmp_path / "snapshot.pdf", source="original.pdf", workers=2,
                                     checkpoint_root=str(tmp_path / "checkpoints"), pages_per_task=10)
    assert records[0]["metadata"]["source"] == "original.pdf"
    assert records[0]["metadata"]["page"] == 1


def test_live_native_range_pipeline_and_cached_rerun(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    from pdf_pipeline.main import run_pipeline

    path = tmp_path / "ranges.pdf"
    with pymupdf.open() as document:
        for number in range(23):
            page = document.new_page()
            page.insert_text((72, 72), f"Native page {number + 1}")
        document.save(path)
    arguments = dict(pdf_path=str(path), document_id="native", workers=2,
                     output_root=str(tmp_path / "output"), pages_per_task=10,
                     checkpoint_root=str(tmp_path / "checkpoints"), timeout_seconds=60)
    summary = run_pipeline(**arguments)
    assert summary["range_count"] == 3
    assert summary["successful_pages"] == 23 and summary["reused_ranges"] == 0
    repeat = run_pipeline(**arguments)
    assert repeat["reused_ranges"] == 3 and repeat["worker_count"] == 0
    pages = [json.loads(line) for line in (tmp_path / "output" / "native" / "pages.jsonl").read_text().splitlines()]
    assert len(pages) == 23
    assert all(f"Native page {number + 1}" in page["text"] for number, page in enumerate(pages))


def duplicate_range_worker(job_queue, result_queue):
    job = job_queue.get()
    result_queue.put(range_result(job))
    result_queue.put(range_result(job))


def test_duplicate_range_completion_cannot_hide_missing_ranges(tmp_path, monkeypatch):
    import pdf_pipeline.main as pipeline
    from pdf_pipeline.scheduler import split_page_ranges

    path = tmp_path / "input.pdf"
    path.touch()
    monkeypatch.setattr(pipeline, "create_page_range_jobs", lambda path, document_id, size:
                        split_page_ranges(path, document_id, 5, size))
    monkeypatch.setattr(pipeline, "worker_loop", duplicate_range_worker)
    with pytest.raises(RuntimeError, match="duplicate or mismatched"):
        pipeline.run_pipeline(str(path), "demo", 1, str(tmp_path / "output"),
                              pages_per_task=2, checkpoint_root=str(tmp_path / "checkpoints"), timeout_seconds=15)
    assert not (tmp_path / "output" / "demo" / "summary.json").exists()
    assert len(list((tmp_path / "checkpoints").rglob("*.json"))) == 1
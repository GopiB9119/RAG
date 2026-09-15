"""Repeatable synthetic five-PDF baseline; never modifies the user's index."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENT_COUNT = 5
PAGES_PER_DOCUMENT = 60
REQUIRED_MODULES = ("pymupdf", "chromadb", "sentence_transformers")


def page_fact(document_number: int, page_number: int) -> tuple[str, str, str]:
    reference = f"DOC{document_number:03d}PAGE{page_number:03d}"
    amount = str(10000 + document_number * 1000 + page_number * 7)
    text = f"The approved budget for reference {reference} is {amount} credits."
    return reference, amount, text


def create_fixture(directory: Path) -> list[dict[str, Any]]:
    import pymupdf

    directory.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for document_number in range(1, DOCUMENT_COUNT + 1):
        path = directory / f"synthetic-{document_number:02d}.pdf"
        with pymupdf.open() as document:
            for page_number in range(1, PAGES_PER_DOCUMENT + 1):
                _, _, text = page_fact(document_number, page_number)
                page = document.new_page()
                page.insert_text((72, 72), text)
            document.save(path)
        questions = []
        for page_number in (1, PAGES_PER_DOCUMENT // 2, PAGES_PER_DOCUMENT):
            reference, amount, _ = page_fact(document_number, page_number)
            questions.append({
                "question": f"What is the approved budget for reference {reference}?",
                "reference": reference,
                "expected_answer": amount,
                "source": str(path.resolve()),
                "page": page_number,
            })
        fixtures.append({"path": path, "number": document_number, "questions": questions})
    return fixtures


def run_baseline(output_root: Path, workers: int = 4, azure: bool = False) -> tuple[dict, Path]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    output_root.mkdir(parents=True, exist_ok=True)
    run_directory = Path(tempfile.mkdtemp(prefix="baseline-", dir=output_root)).resolve()
    report_path = run_directory / "report.json"
    report: dict[str, Any] = {
        "status": "running",
        "synthetic": True,
        "python": sys.version.split()[0],
        "document_count": DOCUMENT_COUNT,
        "pages_per_document": PAGES_PER_DOCUMENT,
        "expected_pages": DOCUMENT_COUNT * PAGES_PER_DOCUMENT,
        "workers": workers,
        "pages_per_task": 10,
        "azure_requested": azure,
        "stages": [],
        "documents": [],
        "questions": [],
    }
    stage = "preflight"
    started = time.perf_counter()

    def save_report() -> None:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(report_path)

    def complete_stage(name: str, since: float) -> None:
        report["stages"].append({"name": name, "status": "passed",
                                 "seconds": round(time.perf_counter() - since, 3)})
        save_report()
        print(f"{name}: passed ({report['stages'][-1]['seconds']} seconds)", flush=True)

    try:
        missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
        if missing:
            report.update(status="blocked", blocked_stage=stage, missing_packages=missing)
            return report, report_path

        import chromadb
        from sentence_transformers import SentenceTransformer
        from ingest_sources import build_index, chunk_records, load_pdf
        from rag_core import MODEL_NAME, REQUIRED_AZURE_SETTINGS, generate_answer, retrieve, open_published_collection

        if azure:
            missing_settings = [name for name in REQUIRED_AZURE_SETTINGS if not os.environ.get(name)]
            if missing_settings:
                report.update(status="blocked", blocked_stage=stage, missing_settings=missing_settings)
                return report, report_path
        report["embedding_model"] = MODEL_NAME
        report["retrieval_settings"] = {
            "top_k": os.environ.get("RAG_TOP_K", "8"),
            "max_distance": os.environ.get("RAG_MAX_DISTANCE", "1.6"),
        }
        complete_stage(stage, started)

        stage = "generate_pdfs"
        since = time.perf_counter()
        fixtures = create_fixture(run_directory / "input")
        complete_stage(stage, since)

        stage = "extract_and_chunk"
        since = time.perf_counter()
        chunks = []
        for fixture in fixtures:
            document_started = time.perf_counter()
            records = load_pdf(fixture["path"], workers=workers, pages_per_task=10,
                               checkpoint_root=str(run_directory / "checkpoints"))
            if [record["metadata"]["page"] for record in records] != list(range(1, PAGES_PER_DOCUMENT + 1)):
                raise ValueError("Extracted page count or order mismatch")
            for page_number, record in enumerate(records, start=1):
                _, _, expected_text = page_fact(fixture["number"], page_number)
                if record["text"] != expected_text or record["metadata"]["source"] != str(fixture["path"].resolve()):
                    raise ValueError("Extracted text or source mismatch")
            document_chunks = chunk_records(records)
            if len(document_chunks) != PAGES_PER_DOCUMENT:
                raise ValueError("Unexpected synthetic chunk count")
            chunks.extend(document_chunks)
            report["documents"].append({
                "file": fixture["path"].name, "status": "passed",
                "pages": len(records), "chunks": len(document_chunks),
                "seconds": round(time.perf_counter() - document_started, 3),
            })
            save_report()
        extraction_seconds = time.perf_counter() - since
        report["extracted_pages"] = sum(document["pages"] for document in report["documents"])
        report["pages_per_second"] = round(report["extracted_pages"] / max(extraction_seconds, 0.000001), 3)
        complete_stage(stage, since)

        stage = "embed_and_index"
        since = time.perf_counter()
        database = str(run_directory / "index")
        stored = build_index(chunks, database, "baseline_documents", False)
        collection = chromadb.PersistentClient(path=database).get_collection("baseline_documents")
        collection = open_published_collection(collection, database, "baseline_documents")
        if stored < len(chunks) or collection.count() != stored:
            raise ValueError("Published vector count does not match token-prepared chunks")
        report["indexed_chunks"] = stored
        complete_stage(stage, since)

        stage = "retrieve"
        since = time.perf_counter()
        model = SentenceTransformer(MODEL_NAME)
        retrieved_questions = []
        for fixture in fixtures:
            for question in fixture["questions"]:
                query_started = time.perf_counter()
                evidence = retrieve(question["question"], collection, model, collection.count())
                matched = any(
                    question["reference"] in text and question["expected_answer"] in text
                    and metadata["source"] == question["source"] and metadata["page"] == question["page"]
                    for text, metadata, _ in evidence
                )
                result = {"reference": question["reference"], "page": question["page"],
                          "retrieval_passed": matched,
                          "retrieval_seconds": round(time.perf_counter() - query_started, 3)}
                report["questions"].append(result)
                retrieved_questions.append((question, evidence, result))
                save_report()
        if not all(result["retrieval_passed"] for result in report["questions"]):
            raise ValueError("Some expected facts or page citations were not retrieved")
        complete_stage(stage, since)

        if azure:
            stage = "azure_answers"
            since = time.perf_counter()
            for question, evidence, result in retrieved_questions:
                answer_started = time.perf_counter()
                answer = generate_answer(question["question"], evidence)
                result["answer_passed"] = question["expected_answer"] in answer and Path(question["source"]).name in answer
                result["answer_seconds"] = round(time.perf_counter() - answer_started, 3)
                save_report()
            if not all(result["answer_passed"] for result in report["questions"]):
                raise ValueError("Some answers lack the expected value or source filename")
            complete_stage(stage, since)
        else:
            report["stages"].append({"name": "azure_answers", "status": "not_run"})
        report["status"] = "passed" if azure else "passed_without_azure"
    except Exception as error:
        report.update(status="failed", failed_stage=stage, error_type=type(error).__name__)
    finally:
        save_report()
    return report, report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Test five synthetic 60-page PDFs against the real local RAG pipeline")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "output")
    parser.add_argument("--azure", action="store_true", help="Send 15 synthetic questions to Azure; API charges may apply")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    report, report_path = run_baseline(args.output_root, args.workers, args.azure)
    print(f"Status: {report['status']}")
    print(f"Report: {report_path}")
    if report.get("missing_packages"):
        print("Missing packages: " + ", ".join(report["missing_packages"]))
    if report.get("missing_settings"):
        print("Missing settings: " + ", ".join(report["missing_settings"]))
    if report.get("failed_stage"):
        print(f"Failed stage: {report['failed_stage']} ({report['error_type']})")
    return 0 if report["status"] in ("passed", "passed_without_azure") else 2


if __name__ == "__main__":
    raise SystemExit(main())
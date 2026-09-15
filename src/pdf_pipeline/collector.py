import json
import re
from pathlib import Path

from .models import PageResult


def collect_results(
    document_id: str,
    results: list[PageResult],
    output_root: str = "data/output",
    *,
    elapsed_seconds: float | None = None,
    worker_count: int | None = None,
    range_count: int | None = None,
    reused_ranges: int | None = None,
    write_outputs: bool = True,
) -> dict:
    """Sort page results, write output files, and return a summary."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", document_id):
        raise ValueError("Invalid document_id")
    # Restricting document_id above prevents paths such as ../../other-folder.
    output_directory = Path(output_root) / document_id

    # Workers may finish page 10 before page 2; output still follows page order.
    ordered_results = sorted(results, key=lambda result: result.page_index)
    document_text_path = output_directory / "document.txt"
    pages_jsonl_path = output_directory / "pages.jsonl"
    summary_json_path = output_directory / "summary.json"

    successful_results = [result for result in ordered_results if result.success]
    failed_results = [result for result in ordered_results if not result.success]
    pages = [{"job_id": result.job_id, "document_id": result.document_id,
              "page_index": result.page_index, "page_number": result.page_index + 1,
              "text": result.text, "success": result.success, "error": result.error,
              "extraction_method": result.extraction_method}
             for result in ordered_results]

    failed_pages = [result.page_index + 1 for result in failed_results]
    summary = {
        "document_id": document_id,
        "total_pages_processed": len(ordered_results),
        "successful_pages": len(successful_results),
        "failed_pages_count": len(failed_results),
        "failed_pages": failed_pages,
        "status": "complete" if not failed_results else "incomplete",
        "extraction_methods": {method: sum(result.extraction_method == method for result in successful_results)
                       for method in ("native", "ocr", "blank")},
        "output_files": {
            "document_text": str(document_text_path),
            "pages_jsonl": str(pages_jsonl_path),
            "summary": str(summary_json_path),
        } if write_outputs else {},
    }
    if elapsed_seconds is not None:
        # Add metrics BEFORE saving, so the file and returned dict agree.
        summary["elapsed_seconds"] = elapsed_seconds
    if worker_count is not None:
        summary["worker_count"] = worker_count
    if range_count is not None:
        summary["range_count"] = range_count
        summary["reused_ranges"] = reused_ranges
    if write_outputs:
        output_directory.mkdir(parents=True, exist_ok=True)
        with document_text_path.open("w", encoding="utf-8") as text_file:
            for result in successful_results:
                text_file.write(f"\n\n===== PAGE {result.page_index + 1} =====\n\n")
                text_file.write(result.text)
        with pages_jsonl_path.open("w", encoding="utf-8") as pages_file:
            for page in pages:
                pages_file.write(json.dumps(page, ensure_ascii=False) + "\n")
        summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        # Ingestion consumes these records directly; there is no need to create
        # document.txt and pages.jsonl just to read them back immediately.
        summary["pages"] = pages
    return summary
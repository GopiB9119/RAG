"""Explicit Azure extraction commands; this does not create cloud resources."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from contextlib import ExitStack
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pdf_pipeline.distributed import MAX_PDF_BYTES, collect_records, dispatch, reconcile, load_manifest


def dependency_missing(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except ModuleNotFoundError:
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Dispatch and extract PDF ranges across Azure workers")
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("dispatch")
    submit.add_argument("pdf", type=Path)
    submit.add_argument("--document-id", required=True)
    submit.add_argument("--pages-per-task", type=int, default=10)
    resume = commands.add_parser("reconcile", help="Resend range tasks with no valid stored result")
    resume.add_argument("version")
    worker = commands.add_parser("worker")
    worker.add_argument("--once", action="store_true", help="Receive at most one message, then exit")
    worker.add_argument("--timeout-seconds", type=int, default=300)
    collect = commands.add_parser("collect", help="Collect a completed document and optionally index it centrally")
    collect.add_argument("version")
    collect.add_argument("--output-dir", type=Path, default=ROOT / "data" / "output" / "azure")
    collect.add_argument("--export", action="store_true", help="Explicitly save a local page-record JSONL copy")
    collect.add_argument("--index", action="store_true", help="Write to ONE coordinator's local Chroma index")
    collect.add_argument("--database", default=str(ROOT / "chroma_data"))
    collect.add_argument("--collection", default="rag_documents")
    args = parser.parse_args()
    if args.command == "dispatch" and not 1 <= args.pages_per_task <= 100:
        parser.error("--pages-per-task must be between 1 and 100")
    if args.command == "dispatch" and not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", args.document_id):
        parser.error("--document-id must be 1-100 letters, digits, underscores or hyphens")
    if args.command == "worker" and not 1 <= args.timeout_seconds <= 900:
        parser.error("--timeout-seconds must be between 1 and 900")
    if args.command in ("collect", "reconcile") and not re.fullmatch(r"[0-9a-f]{64}", args.version):
        parser.error("version must be the 64-character hash returned by dispatch")

    modules = ["azure.identity", "azure.storage.blob", "azure.servicebus", "dotenv"]
    if args.command in ("dispatch", "worker"):
        modules.append("pymupdf")
    if args.command == "collect" and args.index:
        modules.extend(["chromadb", "sentence_transformers"])
    missing = [name for name in modules if dependency_missing(name)]
    if missing:
        print(json.dumps({"status": "blocked", "missing_packages": missing}))
        return 2
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    required = ["AZURE_STORAGE_ACCOUNT_URL", "AZURE_STORAGE_CONTAINER"]
    if args.command != "collect":
        required += ["AZURE_SERVICEBUS_NAMESPACE", "AZURE_SERVICEBUS_QUEUE"]
    missing_settings = [name for name in required if not os.environ.get(name)]
    if missing_settings:
        print(json.dumps({"status": "blocked", "missing_settings": missing_settings}))
        return 2

    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient
    from azure.servicebus import ServiceBusClient, AutoLockRenewer, ServiceBusReceiveMode
    from pdf_pipeline.azure_adapters import AzureBlobs, AzureSender, handle_message, isolated_extract

    with ExitStack() as stack:
        credential = stack.enter_context(DefaultAzureCredential(exclude_interactive_browser_credential=True))
        storage = stack.enter_context(BlobServiceClient(
            account_url=os.environ["AZURE_STORAGE_ACCOUNT_URL"], credential=credential,
            connection_timeout=10, read_timeout=60, retry_total=3,
        ))
        blobs = AzureBlobs(storage.get_container_client(os.environ["AZURE_STORAGE_CONTAINER"]))
        if args.command == "collect":
            # Check every expected range before returning ANY records for indexing.
            records = collect_records(blobs, args.version)
            manifest = load_manifest(blobs, args.version)
            output = None
            if args.export:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                output = args.output_dir / f"{args.version}.jsonl"
                with output.open("w", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            indexed = None
            if args.index:
                from ingest_sources import build_index, chunk_records

                indexed = build_index(chunk_records(records), args.database, args.collection, False)
            print(json.dumps({"status": "collected", "version": args.version,
                              "expected_pages": manifest.page_count, "readable_pages": len(records),
                              "indexed_chunks": indexed, "output": str(output) if output else None}))
            return 0

        bus = stack.enter_context(ServiceBusClient(
            fully_qualified_namespace=os.environ["AZURE_SERVICEBUS_NAMESPACE"], credential=credential,
            retry_total=3,
        ))
        queue = os.environ["AZURE_SERVICEBUS_QUEUE"]
        if args.command in ("dispatch", "reconcile"):
            sender = AzureSender(stack.enter_context(bus.get_queue_sender(queue)))
            if args.command == "reconcile":
                result = reconcile(blobs, sender, args.version)
            else:
                import pymupdf

                with args.pdf.open("rb") as source:
                    pdf = source.read(MAX_PDF_BYTES + 1)
                if len(pdf) > MAX_PDF_BYTES or not pdf.startswith(b"%PDF-"):
                    raise ValueError("Invalid or oversized PDF")
                # Count pages from the same bytes uploaded; a changing original file
                # cannot give a manifest for a different uploaded version.
                with pymupdf.open(stream=pdf, filetype="pdf") as document:
                    page_count = len(document)
                result = dispatch(blobs, sender, pdf, args.document_id, page_count, args.pages_per_task)
            print(json.dumps(result))
            return 0

        receiver = stack.enter_context(bus.get_queue_receiver(
            queue_name=queue, receive_mode=ServiceBusReceiveMode.PEEK_LOCK, prefetch_count=0,
        ))
        print(json.dumps({"event": "azure_worker_started"}), flush=True)
        while True:
            messages = receiver.receive_messages(max_message_count=1, max_wait_time=10)
            for message in messages:
                started = time.perf_counter()
                try:
                    with AutoLockRenewer(max_lock_renewal_duration=args.timeout_seconds + 180) as renewer:
                        renewer.register(receiver, message)
                        state = handle_message(receiver, message, blobs,
                                               lambda job: isolated_extract(job, args.timeout_seconds))
                    print(json.dumps({"event": "azure_range_settled", "status": state,
                                      "seconds": round(time.perf_counter() - started, 3)}), flush=True)
                except Exception as error:
                    print(json.dumps({"event": "azure_settlement_failed", "error_type": type(error).__name__}), flush=True)
                    if args.once:
                        return 2
            if args.once:
                return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"event": "azure_worker_stopped"}))
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), file=sys.stderr)
        raise SystemExit(2)
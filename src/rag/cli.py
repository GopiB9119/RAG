"""`rag` command: one entry point for ingesting, asking, serving and checking status."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from threading import Event

from . import __version__
from .config import load_settings

_LABELS = {"document_ready": "ready  ", "document_failed": "FAILED ", "document_retry_scheduled": "retry  ",
           "rejected": "skipped", "previously_failed": "FAILED "}


class Printer:
    def __init__(self, as_json: bool):
        self.as_json = as_json

    def __call__(self, event: dict) -> None:
        try:
            if self.as_json:
                print(json.dumps(event, ensure_ascii=False, default=str), flush=True)
                return
            name = event.get("event")
            label = _LABELS.get(name)
            if name == "document_ready":
                print(f"{label} {event['title']}  ({event['pages']} pages, {event['chunks']} chunks, "
                      f"{event['seconds']} s)", flush=True)
            elif label:
                reason = event.get("error_type") or ""
                detail = event.get("detail") or event.get("error") or event.get("hint") or ""
                print(f"{label} {event.get('title') or event.get('file')}  {reason} {detail}".rstrip(), flush=True)
            elif name and name.startswith("watch_") and name != "watch_submitted":
                print(f"watch   {json.dumps(event, ensure_ascii=False)}", flush=True)
        except (OSError, ValueError):
            pass  # a closed console must not fail ingestion; queue state is authoritative


def _service(args):
    from .service import RAG

    settings = load_settings(data_dir=args.data_dir, workers=getattr(args, "workers", None),
                             ocr=getattr(args, "ocr", None))
    return RAG(settings)


def _print_summary(result: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result), flush=True)
        return
    print(f"\ndone: {result['ready']} ready, {result['failed']} failed, {result['retrying']} will retry | "
          f"{result['pages']} pages, {result['chunks']} chunks in {result['seconds']} s "
          f"({result['pages_per_second']} pages/s)", flush=True)


def cmd_ingest(args) -> int:
    rag = _service(args)
    emit = Printer(args.json)
    try:
        submitted = rag.submit([Path(path) for path in args.paths], emit)
        if not args.json:
            print(f"queued {submitted['queued']} new PDF(s); {submitted['already_indexed']} already indexed; "
                  f"{submitted['rejected']} rejected", flush=True)
        result = rag.process(emit, until_idle=True)
        _print_summary(result, args.json)
        return 0 if not result["failed"] and not submitted["rejected"] else 2
    finally:
        rag.close()


def cmd_watch(args) -> int:
    from .watcher import FolderScanner

    rag = _service(args)
    emit = Printer(args.json)
    folder = Path(args.folder).resolve() if args.folder else rag.settings.input_dir
    if rag.settings.data_dir.resolve().is_relative_to(folder):
        raise SystemExit("The data directory must not be inside the watched folder")
    scanner = FolderScanner(folder, rag.queue, emit, stable_seconds=args.stable_seconds,
                            max_bytes=rag.settings.max_upload_mb * 1024 * 1024, max_attempts=rag.settings.max_attempts)
    print(f"watching {folder}  (drop PDFs here; Ctrl+C to stop)", flush=True)
    stop = Event()
    try:
        _print_summary(rag.process(emit, stop=stop, scanner=scanner), args.json)
    except KeyboardInterrupt:
        stop.set()
        print("\nstopped; unfinished documents resume next time", flush=True)
    finally:
        rag.close()
    return 0


def _show_answer(result: dict, streamed: bool) -> None:
    if not streamed:
        print(result["answer"])
    print()
    for source in result["sources"]:
        print(f"  [{source['n']}] {source['title']}, page {source['page']}  (similarity {source['similarity']:.2f})")


def _ask(rag, question: str) -> None:
    streamed = sys.stdout.isatty()
    result = rag.ask(question, on_token=(lambda token: print(token, end="", flush=True)) if streamed else None)
    _show_answer(result, streamed)


def cmd_ask(args) -> int:
    rag = _service(args)
    try:
        if args.json:
            result = rag.ask(args.question)
            result["evidence"] = [vars(item) for item in result["evidence"]]
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _ask(rag, args.question)
        return 0
    finally:
        rag.close()


def cmd_chat(args) -> int:
    rag = _service(args)
    stats = rag.index.stats()
    print(f"{stats['documents']} documents, {stats['chunks']} passages indexed. Ask a question ('exit' to quit).")
    try:
        while True:
            try:
                question = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if question.lower() in {"exit", "quit"}:
                return 0
            if question:
                print("\nRAG: ", end="", flush=True)
                try:
                    _ask(rag, question)
                except RuntimeError as error:
                    print(f"\n{error}")
    finally:
        rag.close()


def cmd_search(args) -> int:
    rag = _service(args)
    try:
        results = rag.search(args.query, top_k=args.top_k, mode=args.mode)
        if args.json:
            print(json.dumps([vars(item) for item in results], ensure_ascii=False, indent=2))
            return 0
        for number, item in enumerate(results, start=1):
            print(f"[{number}] {item.citation()}  similarity={item.similarity:.3f} "
                  f"dense_rank={item.dense_rank} keyword_rank={item.lexical_rank}")
            print(f"    {item.text[:300]}{'...' if len(item.text) > 300 else ''}")
        if not results:
            print("no matching passages")
        return 0
    finally:
        rag.close()


def cmd_status(args) -> int:
    rag = _service(args)
    try:
        print(json.dumps(rag.status(), indent=2, default=str))
        return 0
    finally:
        rag.close()


def cmd_retry(args) -> int:
    rag = _service(args)
    try:
        jobs = [job["id"] for job in rag.queue.list_jobs(1000, state="failed")] if args.all else args.job_ids
        for job_id in jobs:
            rag.queue.retry(job_id)
        print(f"requeued {len(jobs)} job(s); run `rag ingest` or keep `rag watch`/`rag serve` running")
        return 0
    finally:
        rag.close()


def cmd_remove(args) -> int:
    rag = _service(args)
    try:
        matches = [doc["source"] for doc in rag.index.documents(100000)
                   if doc["source"] == args.document or doc["title"] == args.document]
        if len(matches) != 1:
            raise SystemExit(f"{len(matches)} documents match {args.document!r}; pass the full source path")
        rag.index.remove(matches[0])
        print(f"removed {matches[0]} from the index (the PDF file itself is untouched)")
        return 0
    finally:
        rag.close()


def cmd_models(args) -> int:
    from .embeddings import ensure_model, get_spec

    settings = load_settings(data_dir=args.data_dir)
    print(ensure_model(get_spec(settings.embedding_model), settings.model_dir))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .api import create_app

    rag = _service(args)
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    if not loopback and not rag.settings.api_key:
        raise SystemExit("Refusing to listen on a network interface without RAG_API_KEY; set it or use --host 127.0.0.1")
    app = create_app(rag, watch=args.watch)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_eval(args) -> int:
    from .evaluation import run_beir

    settings = load_settings(data_dir=args.data_dir, workers=args.workers)
    report = run_beir(settings, args.dataset, limit_queries=args.queries, answer_sample=args.answers)
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="Ask questions about your PDF collection.")
    parser.add_argument("--version", action="version", version=f"rag {__version__}")
    # Shared flags are accepted before or after the command; SUPPRESS keeps an earlier value.
    common = argparse.ArgumentParser(add_help=False)
    for target, default in ((parser, None), (common, argparse.SUPPRESS)):
        target.add_argument("--data-dir", type=Path, default=default,
                            help="index, queue and model folder (default: ./data or RAG_DATA_DIR)")
        target.add_argument("--json", action="store_true", default=default or False, help="machine-readable output")
        target.add_argument("-v", "--verbose", action="store_true", default=default or False)
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help_text: str):
        return commands.add_parser(name, help=help_text, parents=[common])

    ingest = command("ingest", "index PDFs or folders now and wait until done")
    ingest.add_argument("paths", nargs="+")
    ingest.add_argument("--workers", type=int, help="parallel extraction processes")
    ingest.add_argument("--ocr", choices=("off", "auto", "always"), help="OCR policy for scanned pages")
    ingest.set_defaults(handler=cmd_ingest)

    watch = command("watch", "keep indexing PDFs dropped into a folder")
    watch.add_argument("folder", nargs="?")
    watch.add_argument("--workers", type=int)
    watch.add_argument("--ocr", choices=("off", "auto", "always"))
    watch.add_argument("--stable-seconds", type=float, default=10.0, help="quiet period before a file is picked up")
    watch.set_defaults(handler=cmd_watch)

    ask = command("ask", "answer one question with citations")
    ask.add_argument("question")
    ask.set_defaults(handler=cmd_ask)
    command("chat", "interactive questions").set_defaults(handler=cmd_chat)

    find = command("search", "show the passages retrieval finds (no LLM call)")
    find.add_argument("query")
    find.add_argument("--top-k", type=int, default=8)
    find.add_argument("--mode", choices=("hybrid", "dense", "lexical"), default="hybrid")
    find.set_defaults(handler=cmd_search)

    serve = command("serve", "HTTP API + web page + background indexing")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--workers", type=int)
    serve.add_argument("--ocr", choices=("off", "auto", "always"))
    serve.add_argument("--watch", action="store_true", help="also index PDFs dropped into RAG_INPUT_DIR")
    serve.set_defaults(handler=cmd_serve)

    command("status", "index size, queue state and recent failures").set_defaults(handler=cmd_status)
    retry = command("retry", "requeue failed documents")
    retry.add_argument("job_ids", nargs="*")
    retry.add_argument("--all", action="store_true", help="every failed document")
    retry.set_defaults(handler=cmd_retry)
    remove = command("remove", "remove a document from the index")
    remove.add_argument("document", help="file name or full source path")
    remove.set_defaults(handler=cmd_remove)
    command("models", "download the pinned embedding model (for offline images)").set_defaults(handler=cmd_models)

    evaluate = command("eval", "measure retrieval accuracy and speed on a public benchmark")
    evaluate.add_argument("--dataset", default="scifact")
    evaluate.add_argument("--queries", type=int, default=0, help="limit the number of queries (0 = all)")
    evaluate.add_argument("--answers", type=int, default=0, help="also send N questions to the answer model")
    evaluate.add_argument("--workers", type=int)
    evaluate.set_defaults(handler=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    if getattr(args, "workers", None) is not None and not 1 <= args.workers <= 64:
        raise SystemExit("--workers must be between 1 and 64")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        return args.handler(args)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

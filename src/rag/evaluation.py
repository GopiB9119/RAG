"""Accuracy + speed benchmark on a public BEIR dataset, run through the real PDF pipeline.

Every benchmark document becomes a real PDF, is ingested by the production
pipeline into an isolated data directory, and each test query is scored
against human relevance labels (nDCG@10, Recall@10/100, MRR@10).
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path

from .config import Settings

DATASETS = {
    "scifact": ("https://huggingface.co/datasets/BeIR/scifact/resolve/main/corpus.jsonl.gz",
                "https://huggingface.co/datasets/BeIR/scifact/resolve/main/queries.jsonl.gz",
                "https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/main/test.tsv",
                "https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/main/train.tsv"),
}
CONFIGS = {"dense": {"mode": "dense"}, "keyword": {"mode": "lexical"}, "hybrid_rrf": {"fusion": "rrf"},
           **{f"hybrid_convex_{alpha}": {"fusion": "convex", "alpha": alpha} for alpha in (0.3, 0.5, 0.7)}}


def _fetch(url: str, target: Path) -> Path:
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
            while block := response.read(1 << 20):
                output.write(block)
        partial.replace(target)
    return target


def _jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_pdf(path: Path, title: str, text: str) -> None:
    import pymupdf

    words = f"{title}. {text}".split()
    pages, current = [], []
    for word in words:
        current.append(word)
        if sum(len(item) + 1 for item in current) > 2500:
            pages.append(" ".join(current))
            current = []
    if current:
        pages.append(" ".join(current))
    with pymupdf.open() as document:
        for content in pages:
            page = document.new_page()
            rect = page.rect + (50, 50, -50, -50)
            for size in (10, 8, 6):
                if page.insert_textbox(rect, content, fontsize=size) >= 0:
                    break
            else:
                raise ValueError("Benchmark page text does not fit")
        document.save(path)


def _metrics(ranked: list[str], relevant: dict[str, int]) -> dict:
    gains = [relevant.get(doc, 0) for doc in ranked[:10]]
    dcg = sum(gain / math.log2(position + 2) for position, gain in enumerate(gains))
    ideal = sorted(relevant.values(), reverse=True)[:10]
    idcg = sum(gain / math.log2(position + 2) for position, gain in enumerate(ideal))
    first = next((position for position, doc in enumerate(ranked[:10]) if relevant.get(doc, 0) > 0), None)
    wanted = {doc for doc, gain in relevant.items() if gain > 0}
    return {"ndcg@10": dcg / idcg if idcg else 0.0, "mrr@10": 0.0 if first is None else 1.0 / (first + 1),
            "recall@10": len(wanted & set(ranked[:10])) / len(wanted),
            "recall@100": len(wanted & set(ranked[:100])) / len(wanted)}


def _qrels(path: Path) -> dict[str, dict[str, int]]:
    judged: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            judged.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return judged


def _evaluate(rag, queries: dict[str, str], qrels: dict[str, dict[str, int]], selected: list[str], config: dict) -> dict:
    scores, latencies = [], []
    for query_id in selected:
        began = time.perf_counter()
        hits = rag.search(queries[query_id], top_k=100, **config)
        latencies.append(time.perf_counter() - began)
        scores.append(_metrics(list(dict.fromkeys(Path(hit.source).stem for hit in hits)), qrels[query_id]))
    latencies.sort()
    return {**{name: round(statistics.fmean(score[name] for score in scores), 4) for name in scores[0]},
            "latency_ms_p50": round(1000 * latencies[len(latencies) // 2], 1),
            "latency_ms_p95": round(1000 * latencies[int(0.95 * (len(latencies) - 1))], 1)}


def run_beir(settings: Settings, dataset: str = "scifact", *, limit_queries: int = 0, answer_sample: int = 0) -> dict:
    from .service import RAG

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset; choose one of: {', '.join(DATASETS)}")
    root = settings.data_dir / "eval" / dataset
    corpus_url, queries_url, test_url, train_url = DATASETS[dataset]
    corpus = _jsonl(_fetch(corpus_url, root / "download" / "corpus.jsonl.gz"))
    queries = {row["_id"]: row["text"] for row in _jsonl(_fetch(queries_url, root / "download" / "queries.jsonl.gz"))}
    test = _qrels(_fetch(test_url, root / "download" / "test.tsv"))
    train = _qrels(_fetch(train_url, root / "download" / "train.tsv"))

    pdf_dir = root / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for row in corpus:
        target = pdf_dir / f"{row['_id']}.pdf"
        if not target.exists():
            _write_pdf(target, row.get("title") or "", row.get("text") or "")
    render_seconds = time.perf_counter() - started

    # Isolated data directory: the benchmark never touches the user's own index.
    bench = settings.with_overrides(data_dir=root / "rag", input_dir=pdf_dir, model_dir=settings.model_dir)
    rag = RAG(bench)
    try:
        events = {"failed": 0}

        def count(event: dict) -> None:
            if event.get("event") == "document_failed":
                events["failed"] += 1

        rag.submit([pdf_dir], count)
        ingest = rag.process(count, until_idle=True)
        test_ids = [query_id for query_id in test if query_id in queries]
        train_ids = [query_id for query_id in train if query_id in queries]
        if limit_queries:
            test_ids, train_ids = test_ids[:limit_queries], train_ids[:limit_queries]
        report = {"dataset": dataset, "documents": len(corpus), "test_queries": len(test_ids),
                  "train_queries": len(train_ids), "embedding_model": bench.embedding_model,
                  "extraction_workers": bench.workers, "pdf_render_seconds": round(render_seconds, 1),
                  "ingest": ingest, "failed_documents": events["failed"], "index": rag.index.stats(),
                  "train": {}, "test": {}}
        rag.search("warm up", top_k=1)
        # Choose settings on the training queries; the test numbers are never used for tuning.
        for name, config in CONFIGS.items():
            report["train"][name] = _evaluate(rag, queries, train, train_ids, config)
            report["test"][name] = _evaluate(rag, queries, test, test_ids, config)
        best = max((name for name in CONFIGS if name.startswith("hybrid")),
                   key=lambda name: report["train"][name]["ndcg@10"])
        report["selected_on_train"] = best
        if answer_sample:
            report["answers"] = []
            for query_id in test_ids[:answer_sample]:
                result = rag.ask(f"What do the sources say about this claim: {queries[query_id]}")
                cited = [Path(source["source"]).stem for source in result["sources"]]
                report["answers"].append({"query_id": query_id, "claim": queries[query_id],
                                          "answer": result["answer"],
                                          "cited_relevant": any(doc in test[query_id] for doc in cited),
                                          "seconds": result["timings"]})
        return report
    finally:
        rag.close()

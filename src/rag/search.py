"""Hybrid retrieval: exact dense search + BM25, fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass

from .store import Index

RRF_K = 60


@dataclass(frozen=True)
class Evidence:
    text: str
    source: str
    title: str
    page: int
    chunk: int
    part: int
    method: str
    similarity: float
    score: float
    dense_rank: int | None
    lexical_rank: int | None

    def citation(self) -> str:
        return f"{self.title}, page {self.page}"


def fuse(dense: dict[int, float], lexical: dict[int, float], fusion: str, alpha: float,
         dense_rank: dict[int, int], lexical_rank: dict[int, int]) -> dict[int, float]:
    candidates = set(dense_rank) | set(lexical_rank)
    if fusion == "rrf":
        return {chunk_id: sum(1.0 / (RRF_K + ranks[chunk_id]) for ranks in (dense_rank, lexical_rank)
                              if chunk_id in ranks) for chunk_id in candidates}
    # Convex combination of max-normalized scores; theoretical minimum 0 for both signals.
    top_dense = max((max(dense.get(chunk_id, 0.0), 0.0) for chunk_id in candidates), default=0.0) or 1.0
    top_lexical = max((lexical.get(chunk_id, 0.0) for chunk_id in candidates), default=0.0) or 1.0
    return {chunk_id: alpha * max(dense.get(chunk_id, 0.0), 0.0) / top_dense
            + (1.0 - alpha) * lexical.get(chunk_id, 0.0) / top_lexical for chunk_id in candidates}


def search(index: Index, embedder, question: str, *, top_k: int = 8, candidates: int = 50,
           min_similarity: float = 0.2, mode: str = "hybrid", fusion: str = "convex",
           alpha: float = 0.5) -> list[Evidence]:
    if mode not in ("hybrid", "dense", "lexical") or fusion not in ("convex", "rrf") or not 0 <= alpha <= 1:
        raise ValueError("mode must be hybrid/dense/lexical, fusion convex/rrf and alpha within 0..1")
    question = " ".join(question.split())
    if not question or top_k < 1:
        return []
    query = embedder.embed_query(question)
    with index.snapshot() as snapshot:
        if not snapshot.chunk_count:
            return []
        dense_hits = snapshot.dense(query, candidates) if mode != "lexical" else []
        lexical_hits = snapshot.lexical(question, candidates) if mode != "dense" else []
        dense_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(dense_hits, start=1)}
        lexical_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(lexical_hits, start=1)}
        pool = list(dict.fromkeys([*dense_rank, *lexical_rank]))
        similarity = dict(dense_hits)
        similarity.update(snapshot.similarity([chunk_id for chunk_id in pool if chunk_id not in similarity], query))
        keyword = dict(lexical_hits)
        if mode == "hybrid":
            keyword.update(snapshot.lexical_scores(question, [chunk_id for chunk_id in pool if chunk_id not in keyword]))
            scores = fuse(similarity, keyword, fusion, alpha, dense_rank, lexical_rank)
        else:
            scores = similarity if mode == "dense" else keyword
        # Keyword matches still need some semantic relation, so unrelated text never reaches the LLM.
        ordered = sorted((chunk_id for chunk_id in pool if similarity.get(chunk_id, -1.0) >= min_similarity),
                         key=lambda chunk_id: (-scores[chunk_id], -similarity[chunk_id]))
        shortlist = ordered[:top_k * 3]
        rows = snapshot.fetch(shortlist)
    results: list[Evidence] = []
    seen: set[str] = set()
    for chunk_id in shortlist:
        hit = rows.get(chunk_id)
        if hit is None:
            continue
        fingerprint = " ".join(hit.text.split()).lower()
        if fingerprint in seen:
            continue  # the same passage duplicated across documents
        seen.add(fingerprint)
        results.append(Evidence(hit.text, hit.source, hit.title, hit.page, hit.chunk, hit.part, hit.method,
                                round(similarity[chunk_id], 4), round(scores[chunk_id], 6),
                                dense_rank.get(chunk_id), lexical_rank.get(chunk_id)))
        if len(results) == top_k:
            break
    return results

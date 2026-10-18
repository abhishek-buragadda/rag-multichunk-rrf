"""Passage-level evaluation on QASPER: how much annotated evidence fits in a fixed context budget.

For each question we retrieve only inside its paper, take chunks in rank order
and add their words to the LLM context until a word budget is spent. Words already
in the context (from overlapping chunks) are free, so larger or overlapping
chunks get no unfair advantage. An evidence paragraph counts as found when at
least half of its words are in the context. The score is evidence recall, taking
the best-matching annotator.

Fusion strategies compared against single sizes:

* RRF(interleave) - plain RRF over chunk ids. Chunks of different sizes never
  share an id, so this just interleaves the per-size rankings.
* RRF(span) - overlap-aware RRF. Every word of the paper gets, from each size,
  the rank of the best chunk containing it; the word's score is the RRF sum over
  sizes. Candidate chunks from all sizes are then ranked by the mean score of
  their words, so text that ranks well at every granularity comes first.
"""

from __future__ import annotations

import re

import numpy as np
from tqdm import tqdm

from .fusion import reciprocal_rank_fusion
from .long_corpora import QasperCorpus
from .pipeline import Retriever

BUDGETS = (128, 256, 512, 1024)
EVIDENCE_COVERAGE = 0.5


class _WordIndex:
    """Maps char spans of one document to word-index ranges."""

    def __init__(self, text: str):
        self.starts = np.array([m.start() for m in re.finditer(r"\S+", text)])
        self.n = len(self.starts)

    def span(self, start: int, end: int) -> tuple[int, int]:
        return int(np.searchsorted(self.starts, start)), int(np.searchsorted(self.starts, end))


def _fill_budget(ranked_spans: list[tuple[int, int]], n_words: int, budget: int) -> np.ndarray:
    covered = np.zeros(n_words, dtype=bool)
    used = 0
    for ws, we in ranked_spans:
        new = np.flatnonzero(~covered[ws:we]) + ws
        if used + len(new) > budget:
            covered[new[: budget - used]] = True
            break
        covered[new] = True
        used += len(new)
    return covered


def _evidence_recall(covered: np.ndarray, annotators: list[list[tuple[int, int]]]) -> float:
    best = 0.0
    for spans in annotators:
        found = [covered[ws:we].mean() >= EVIDENCE_COVERAGE for ws, we in spans if we > ws]
        if found:
            best = max(best, float(np.mean(found)))
    return best


def _span_fusion(per_size: dict[int, list[dict]], widx: _WordIndex, rrf_k: int) -> list[dict]:
    fused = np.zeros(widx.n)
    for rows in per_size.values():
        best_rank = np.full(widx.n, np.inf)
        for rank, r in enumerate(rows, start=1):
            ws, we = r["_w"]
            best_rank[ws:we] = np.minimum(best_rank[ws:we], rank)
        fused += np.where(np.isfinite(best_rank), 1.0 / (rrf_k + best_rank), 0.0)
    candidates = [r for rows in per_size.values() for r in rows if r["_w"][1] > r["_w"][0]]
    return sorted(candidates, key=lambda r: fused[r["_w"][0] : r["_w"][1]].mean(), reverse=True)


def evaluate_passages(
    retriever: Retriever, corpus: QasperCorpus, budgets=BUDGETS, rrf_k: int = 60, per_size_k: int = 400
) -> dict[str, dict[str, np.ndarray]]:
    docs = {d.doc_id: d for d in corpus.docs}
    widx_cache: dict[str, _WordIndex] = {}
    qids = list(corpus.queries)
    vecs = retriever.embedder.embed_queries([corpus.queries[q] for q in qids])

    systems = [f"size={s}" for s in retriever.sizes] + ["RRF(interleave)", "RRF(span)"]
    scores = {name: {f"EvR@{b}w": [] for b in budgets} for name in systems}

    for qid, vec in tqdm(zip(qids, vecs), total=len(qids), desc="Evaluating"):
        doc_id = corpus.query_doc[qid]
        if doc_id not in widx_cache:
            widx_cache[doc_id] = _WordIndex(docs[doc_id].text)
        widx = widx_cache[doc_id]
        annotators = [[widx.span(s, e) for s, e in spans] for spans in corpus.evidence[qid]]

        per_size = retriever.search_all_sizes(vec, per_size_k, where=f"doc_id = '{doc_id}'")
        for rows in per_size.values():
            for r in rows:
                r["_w"] = widx.span(r["start"], r["end"])

        runs = {f"size={s}": rows for s, rows in per_size.items()}
        by_id = {r["chunk_id"]: r for rows in per_size.values() for r in rows}
        interleaved = reciprocal_rank_fusion(
            {str(s): [r["chunk_id"] for r in rows] for s, rows in per_size.items()}, k=rrf_k
        )
        runs["RRF(interleave)"] = [by_id[c] for c, _ in interleaved]
        runs["RRF(span)"] = _span_fusion(per_size, widx, rrf_k)

        for name, ranked in runs.items():
            spans = [r["_w"] for r in ranked]
            for b in budgets:
                covered = _fill_budget(spans, widx.n, b)
                scores[name][f"EvR@{b}w"].append(_evidence_recall(covered, annotators))

    return {name: {m: np.array(v) for m, v in ms.items()} for name, ms in scores.items()}

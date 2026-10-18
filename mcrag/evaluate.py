"""Compare each single chunk size against RRF-fused rankings.

Every function returns per-query scores ({system: {metric: np.ndarray}}) so
that differences between systems can be tested for significance.
"""

from __future__ import annotations

import math

import numpy as np
from tqdm import tqdm

from .corpus import Corpus
from .fusion import dedupe_preserving_order, reciprocal_rank_fusion
from .pipeline import Retriever

RRF = "RRF(all sizes)"


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    return len(set(ranked[:k]) & rels.keys()) / len(rels) if rels else 0.0


def hit_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    return float(any(d in rels for d in ranked[:k]))


def mrr_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    for i, d in enumerate(ranked[:k]):
        if d in rels:
            return 1.0 / (i + 1)
    return 0.0


DOC_METRICS = {
    "nDCG@10": lambda r, q: ndcg_at_k(r, q, 10),
    "MRR@10": lambda r, q: mrr_at_k(r, q, 10),
    "Hit@1": lambda r, q: hit_at_k(r, q, 1),
    "R@10": lambda r, q: recall_at_k(r, q, 10),
    "R@100": lambda r, q: recall_at_k(r, q, 100),
}


def evaluate(
    retriever: Retriever, corpus: Corpus, per_size_k: int = 100, rrf_k: int = 60, metrics=None
) -> dict[str, dict[str, np.ndarray]]:
    """Document-level retrieval over the whole corpus (SciFact, Seinfeld)."""
    metrics = metrics or DOC_METRICS
    qids = list(corpus.queries)
    vecs = retriever.embedder.embed_queries([corpus.queries[q] for q in qids])
    systems = [f"size={s}" for s in retriever.sizes] + [RRF]
    scores = {name: {m: [] for m in metrics} for name in systems}

    for qid, vec in tqdm(zip(qids, vecs), total=len(qids), desc="Evaluating"):
        per_size = retriever.search_all_sizes(vec, per_size_k)
        doc_lists = {s: dedupe_preserving_order([r["doc_id"] for r in rows]) for s, rows in per_size.items()}
        runs = {f"size={s}": docs for s, docs in doc_lists.items()}
        runs[RRF] = [d for d, _ in reciprocal_rank_fusion({str(s): v for s, v in doc_lists.items()}, k=rrf_k)]
        for name, ranked in runs.items():
            for m, fn in metrics.items():
                scores[name][m].append(fn(ranked, corpus.qrels[qid]))

    return {name: {m: np.array(v) for m, v in ms.items()} for name, ms in scores.items()}


def paired_permutation_test(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 0) -> float:
    """Two-sided p-value for mean(a - b) != 0 by randomly flipping the sign of each paired difference."""
    diff = a - b
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(diff)))
    return float(((np.abs((signs * diff).mean(axis=1)) >= observed).sum() + 1) / (n + 1))


def print_report(results: dict[str, dict[str, np.ndarray]], fused: tuple[str, ...] = (RRF,)) -> None:
    """Print means, then compare each fused system against the best single size per metric."""
    metrics = list(next(iter(results.values())))
    width = max(18, max(len(n) for n in results) + 2)
    print(f"\n{'system':<{width}}" + "".join(f"{m:>10}" for m in metrics))
    print("-" * (width + 10 * len(metrics)))
    for name, vals in results.items():
        print(f"{name:<{width}}" + "".join(f"{vals[m].mean():>10.4f}" for m in metrics))

    singles = [n for n in results if n.startswith("size=")]
    for f in fused:
        print(f"\n{f} vs best single size (paired permutation test, 10k samples):")
        for m in metrics:
            best = max(singles, key=lambda n: results[n][m].mean())
            d = results[f][m].mean() - results[best][m].mean()
            p = paired_permutation_test(results[f][m], results[best][m])
            print(f"  {m:<8} best={best:<9} delta={d:+.4f}  p={p:.3f}{'  *' if p < 0.05 else ''}")

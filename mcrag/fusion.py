"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[Hashable]],
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[Hashable, float]]:
    """Fuse several ranked lists into one.

    score(item) = sum over lists L of  w_L / (k + rank_L(item)),  rank starting at 1.

    Items missing from a list simply get no contribution from it. Only the
    ranks matter, so lists from different chunk sizes (whose raw similarity
    scores are not comparable) can be combined safely.
    """
    scores: dict[Hashable, float] = defaultdict(float)
    for name, items in ranked_lists.items():
        w = 1.0 if weights is None else weights.get(name, 1.0)
        for rank, item in enumerate(items, start=1):
            scores[item] += w / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def dedupe_preserving_order(items: Sequence[Hashable]) -> list[Hashable]:
    seen: set[Hashable] = set()
    return [x for x in items if not (x in seen or seen.add(x))]

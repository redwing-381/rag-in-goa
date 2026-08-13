"""Reciprocal rank fusion.

Fuse by *rank*, not by score. Cosine similarity sits in [-1, 1] while BM25 is
unbounded and corpus-dependent, so any weighted score sum silently lets BM25
dominate. RRF only needs the ordering, which is the one thing both legs agree on
how to express.

    score(d) = sum over legs of 1 / (k + rank(d))

k=60 is the constant from the original TREC work; larger k flattens the curve so
deep results matter more, smaller k sharpens the preference for rank 1.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[int, float]]:
    """Fuse ranked label lists into one list of (label, score), best first."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must be the same length")

    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, label in enumerate(ranking):
            scores[label] = scores.get(label, 0.0) + weight / (k + rank + 1)

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

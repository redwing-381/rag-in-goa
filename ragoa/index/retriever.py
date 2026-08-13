"""Hybrid retriever with a hard deadline and graceful degradation.

The brief asks for the whole retrieval path under 200ms. Rather than hoping the
stages happen to fit, the deadline is enforced: each optional stage checks the
remaining budget against its own reserve and skips itself if it would overrun,
recording the skip in the trace. So under load the system loses accuracy in a
declared order instead of blowing the latency target.

Degradation order, cheapest signal sacrificed first:
    1. reranking      (largest cost, biggest accuracy contribution)
    2. lexical leg    (helps rare tokens; dense alone still answers)
    3. lower ef_search on the dense leg

Every skip lands in `trace.degradations`, so the latency report can state how
often the shipped numbers were achieved with the full pipeline versus a degraded
one. Hiding that would make the P50 meaningless.
"""

from __future__ import annotations

import numpy as np

from ragoa.config import Settings
from ragoa.index.dense import DenseIndex
from ragoa.index.fusion import reciprocal_rank_fusion
from ragoa.index.rerank import Reranker
from ragoa.index.sparse import SparseIndex
from ragoa.index.store import ChunkStore
from ragoa.schemas import Deadline, RetrievedChunk, Trace
from ragoa.telemetry.timing import timed


class HybridRetriever:
    def __init__(
        self,
        store: ChunkStore,
        dense: DenseIndex,
        encoder,
        sparse: SparseIndex | None = None,
        reranker: Reranker | None = None,
        settings: Settings | None = None,
    ):
        from ragoa.config import settings as default_settings

        self.store = store
        self.dense = dense
        self.encoder = encoder
        self.sparse = sparse
        self.reranker = reranker
        self.settings = settings or default_settings

    def retrieve(
        self,
        query: str,
        deadline: Deadline,
        trace: Trace,
        top_k: int | None = None,
        candidates: int | None = None,
    ) -> list[RetrievedChunk]:
        cfg = self.settings
        top_k = top_k or cfg.top_k
        candidates = candidates or cfg.dense_candidates

        with timed(trace, "embed_query"):
            vector = self.encoder.encode_query(query)

        # Dense leg. If the budget is already thin, cut ef rather than skip:
        # a dense result is the one thing we cannot do without.
        ef = cfg.hnsw_ef_search
        if deadline.remaining_ms < cfg.sparse_reserve_ms:
            ef = max(top_k, cfg.hnsw_ef_search // 2)
            trace.degradations.append("dense_ef_reduced")

        with timed(trace, "dense_search", ef=ef, k=candidates):
            dense_labels, dense_scores = self.dense.search(vector, candidates, ef=ef)

        rankings: list[list[int]] = [[int(x) for x in dense_labels]]
        weights = [1.0]
        dense_by_label = {
            int(label): float(score)
            for label, score in zip(dense_labels, dense_scores, strict=True)
        }
        sparse_by_label: dict[int, float] = {}

        # Lexical leg, only if there is room for it plus a rerank.
        if self.sparse is not None:
            if deadline.exceeded(cfg.sparse_reserve_ms):
                trace.degradations.append("sparse_skipped")
            else:
                with timed(trace, "sparse_search", k=cfg.sparse_candidates):
                    sparse_labels, sparse_scores = self.sparse.search(
                        query, cfg.sparse_candidates
                    )
                rankings.append([int(x) for x in sparse_labels])
                weights.append(1.0)
                sparse_by_label = {
                    int(label): float(score)
                    for label, score in zip(sparse_labels, sparse_scores, strict=True)
                }

        with timed(trace, "fusion", legs=len(rankings)):
            fused = reciprocal_rank_fusion(rankings, k=cfg.rrf_k, weights=weights)

        results = [
            RetrievedChunk(
                chunk=self.store.to_chunk(label),
                dense_score=dense_by_label.get(label),
                sparse_score=sparse_by_label.get(label),
                fused_score=score,
            )
            for label, score in fused[: max(cfg.rerank_candidates, top_k)]
        ]

        # Rerank, the expensive stage. Skipped whenever the reserve is gone.
        if self.reranker is not None and results:
            if deadline.exceeded(cfg.rerank_reserve_ms):
                trace.degradations.append("rerank_skipped")
            else:
                subset = results[: cfg.rerank_candidates]
                with timed(trace, "rerank", candidates=len(subset),
                           max_length=cfg.rerank_max_length):
                    scores = self.reranker.score(query, [r.chunk.text for r in subset])
                for item, score in zip(subset, scores, strict=True):
                    item.rerank_score = float(score)
                subset.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)
                results = subset + results[cfg.rerank_candidates:]

        results = results[:top_k]

        # Expand to the wider context span for generation. Cheap: a memmap slice.
        with timed(trace, "expand_context", k=len(results)):
            for item in results:
                label = self.store.index_of(item.chunk.chunk_id)
                if label is not None:
                    item.expanded_text = self.store.context(label)

        trace.retrieval_ms = deadline.elapsed_ms
        return results

    def top_score(self, results: list[RetrievedChunk]) -> float:
        """Best dense similarity, used by the out-of-domain guardrail.

        Deliberately the dense score and not the fused or rerank score: RRF
        scores are ~1/60 regardless of relevance, so they carry no information
        about whether anything relevant was found at all.
        """
        scores = [r.dense_score for r in results if r.dense_score is not None]
        return max(scores) if scores else 0.0


def normalize_scores(values: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1], for display only. Never fuse on these."""
    if values.size == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.ones_like(values)
    return (values - low) / (high - low)

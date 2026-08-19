"""Judge-format retrieval latency bench.

Same table the evaluation script prints: embed / search / total with
avg, p50, p95, p99, then PASS/FAIL against the retrieval budget.

    python benchmark.py
    python benchmark.py 400
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

from ragoa.config import settings
from ragoa.factory import load_retriever
from ragoa.schemas import Deadline, Trace

LATENCY_BUDGET_MS = settings.retrieval_budget_ms
WARMUP = 10


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def load_queries(n: int) -> list[str]:
    path = settings.corpus_dir / "queries.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}; run python -m ragoa.data.docbuilder")

    queries: list[str] = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2000, columns=["lang", "eng_query"]):
        for row in batch.to_pylist():
            if row["lang"] != "hi":
                continue
            if row["eng_query"]:
                queries.append(row["eng_query"])
            if len(queries) >= n:
                return queries
    return queries


def span_ms(trace: Trace, name: str) -> float:
    return sum(s.duration_ms for s in trace.spans if s.name == name)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    index_dir = Path(settings.index_dir)
    encoder = "onnx" if settings.onnx_model("embedder").exists() else "st"

    print("Warming up (model load + first inference)...")
    retriever, manifest = load_retriever(
        index_dir, encoder_kind=encoder, use_rerank=False, use_sparse=True, device="cpu",
    )
    queries = load_queries(n + WARMUP)
    if len(queries) < WARMUP + 1:
        raise SystemExit(f"only {len(queries)} queries available")

    for i, query in enumerate(queries[:WARMUP]):
        retriever.retrieve(query, Deadline(LATENCY_BUDGET_MS), Trace(request_id=f"w{i}"))

    total_ms, embed_ms, search_ms = [], [], []
    measured = queries[WARMUP:WARMUP + n]
    for i, query in enumerate(measured):
        trace = Trace(request_id=str(i))
        t0 = time.perf_counter()
        retriever.retrieve(query, Deadline(LATENCY_BUDGET_MS), trace)
        total = (time.perf_counter() - t0) * 1000
        embed = span_ms(trace, "embed_query")
        total_ms.append(total)
        embed_ms.append(embed)
        search_ms.append(max(total - embed, 0.0))

    print(f"\nRan {len(measured)} queries")
    print(f"index {manifest.get('n_docs', 0):,} docs / {manifest.get('n_chunks', 0):,} chunks"
          f"  encoder={encoder}")
    print()
    print(f"{'stage':<12}{'avg':>8}{'p50':>8}{'p95':>8}{'p99':>8}   (ms)")
    for name, values in [("embed", embed_ms), ("search", search_ms), ("total", total_ms)]:
        print(
            f"{name:<12}"
            f"{statistics.mean(values):>8.2f}"
            f"{percentile(values, 50):>8.2f}"
            f"{percentile(values, 95):>8.2f}"
            f"{percentile(values, 99):>8.2f}"
        )

    p95_total = percentile(total_ms, 95)
    print(f"\nLatency budget: {LATENCY_BUDGET_MS:.0f}ms | p95 total: {p95_total:.2f}ms")
    if p95_total <= LATENCY_BUDGET_MS:
        print("PASS: within budget")
    else:
        print("FAIL: over budget -- see README 'Tuning latency' section")
        sys.exit(1)


if __name__ == "__main__":
    main()

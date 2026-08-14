"""Latency benchmark: P50 / P70 / P100 per stage and end to end.

Reports P95 and P99 alongside the three the brief asks for, because P100 on its
own is a single worst outlier - one page fault or one GC pause and it stops
describing the system. P50 with P95 and P100 tells the honest story.

Also counts how often each degradation fired. A P50 achieved by silently skipping
the reranker is a different claim from one achieved with the full pipeline, so the
report states which.

    python bench/latency.py --index-dir data/index-test --encoder hash --queries 300
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from ragoa.config import settings
from ragoa.factory import load_retriever
from ragoa.index.retriever import HybridRetriever
from ragoa.schemas import Deadline, Trace


def percentiles(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {}
    ordered = sorted(samples)

    def at(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]

    return {
        "p50": statistics.median(ordered),
        "p70": at(0.70),
        "p95": at(0.95),
        "p99": at(0.99),
        "p100": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def load_queries(n: int, lang: str | None) -> list[str]:
    path = settings.corpus_dir / "queries.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}; run python -m ragoa.data.docbuilder")

    queries: list[str] = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2000, columns=["lang", "eng_query"]):
        for row in batch.to_pylist():
            if lang and row["lang"] != lang:
                continue
            if row["eng_query"]:
                queries.append(row["eng_query"])
            if len(queries) >= n:
                return queries
    return queries


def build_retriever(index_dir: Path, encoder_kind: str, use_rerank: bool,
                    use_sparse: bool) -> tuple[HybridRetriever, dict]:
    """Built via the shared factory, so this measures the same stack the API serves."""
    retriever, manifest = load_retriever(
        index_dir, encoder_kind=encoder_kind, use_rerank=use_rerank,
        use_sparse=use_sparse, device="cpu",
    )
    return retriever, manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default=str(settings.index_dir))
    ap.add_argument("--encoder", choices=["st", "onnx", "hash"], default="st")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-sparse", action="store_true")
    ap.add_argument("--budget-ms", type=float, default=settings.retrieval_budget_ms)
    ap.add_argument("--out", default="docs/LATENCY.md")
    args = ap.parse_args()

    retriever, manifest = build_retriever(Path(args.index_dir), args.encoder,
                                          not args.no_rerank, not args.no_sparse)
    queries = load_queries(args.queries + args.warmup, args.lang)
    if len(queries) < args.warmup + 10:
        raise SystemExit(f"only {len(queries)} queries available")

    # Warm up: first call pays lazy init, page faults and allocator growth. That
    # cost is real but it is a cold-start number, not a steady-state one.
    cold_ms = None
    for i, query in enumerate(queries[: args.warmup]):
        t0 = time.perf_counter()
        retriever.retrieve(query, Deadline(args.budget_ms), Trace(request_id=f"w{i}"))
        if i == 0:
            cold_ms = (time.perf_counter() - t0) * 1000

    stage_samples: dict[str, list[float]] = {}
    totals: list[float] = []
    degradations: Counter[str] = Counter()
    over_budget = 0

    for i, query in enumerate(queries[args.warmup:]):
        trace = Trace(request_id=str(i))
        deadline = Deadline(args.budget_ms)
        t0 = time.perf_counter()
        retriever.retrieve(query, deadline, trace)
        total = (time.perf_counter() - t0) * 1000

        totals.append(total)
        if total > args.budget_ms:
            over_budget += 1
        for span in trace.spans:
            stage_samples.setdefault(span.name, []).append(span.duration_ms)
        degradations.update(trace.degradations)

    order = ["embed_query", "dense_search", "sparse_search", "fusion", "rerank",
             "expand_context"]
    stages = [s for s in order if s in stage_samples]
    stages += [s for s in stage_samples if s not in order]

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit()
    # The configuration goes *into* the report, not just the console. A percentile
    # table with no record of which encoder produced it is unreadable later, and
    # the hash encoder's numbers must never be mistaken for real ones.
    emit(f"index            : {manifest['n_chunks']:,} chunks from "
         f"{manifest.get('n_docs', 0):,} docs, strategy={manifest['strategy']}")
    emit(f"encoder          : {args.encoder} ({manifest.get('embed_model')})")
    if args.encoder == "hash":
        emit("                   WARNING: HashEncoder is not semantic. These are "
             "plumbing timings only, not retrieval quality.")
    emit(f"lexical leg      : {'on' if retriever.sparse is not None else 'off'}")
    emit(f"reranker         : "
         f"{settings.rerank_model if retriever.reranker is not None else 'off'}")
    emit(f"queries measured : {len(totals):,} (after {args.warmup} warmup)")
    emit(f"budget           : {args.budget_ms:.0f} ms")
    if cold_ms:
        emit(f"cold first call  : {cold_ms:,.1f} ms (excluded from percentiles)")
    emit()
    header = f"{'stage':<16} {'p50':>8} {'p70':>8} {'p95':>8} {'p99':>8} {'p100':>9}"
    emit(header)
    emit("-" * len(header))

    for stage in stages:
        p = percentiles(stage_samples[stage])
        emit(f"{stage:<16} {p['p50']:>8.2f} {p['p70']:>8.2f} {p['p95']:>8.2f} "
             f"{p['p99']:>8.2f} {p['p100']:>9.2f}")

    p = percentiles(totals)
    emit("-" * len(header))
    emit(f"{'TOTAL':<16} {p['p50']:>8.2f} {p['p70']:>8.2f} {p['p95']:>8.2f} "
         f"{p['p99']:>8.2f} {p['p100']:>9.2f}")
    emit()
    verdict = "PASS" if p["p100"] <= args.budget_ms else (
        "PASS at p95" if p["p95"] <= args.budget_ms else "FAIL")
    emit(f"under {args.budget_ms:.0f} ms: p50 {'yes' if p['p50'] <= args.budget_ms else 'no'}, "
         f"p95 {'yes' if p['p95'] <= args.budget_ms else 'no'}, "
         f"p100 {'yes' if p['p100'] <= args.budget_ms else 'no'}  -> {verdict}")
    emit(f"over budget      : {over_budget}/{len(totals)} requests")
    emit(f"degradations     : "
         f"{dict(degradations) if degradations else 'none (full pipeline)'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Latency\n\n```\n" + "\n".join(lines).strip() + "\n```\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

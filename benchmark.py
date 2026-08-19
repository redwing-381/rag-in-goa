"""Judge-format retrieval latency bench.

Same table the evaluation script prints: embed / search / total with
avg, p50, p95, p99, then PASS/FAIL against the retrieval budget.

    python benchmark.py
    python benchmark.py 400
    python benchmark.py 400 --url https://rag-in-goa.fly.dev
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

from ragoa.config import settings

LATENCY_BUDGET_MS = settings.retrieval_budget_ms
WARMUP = 10
RETRIEVAL_SPANS = (
    "embed_query",
    "dense_search",
    "sparse_search",
    "fusion",
    "rerank",
    "expand_context",
)


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


def span_ms(spans: list[dict], name: str) -> float:
    return sum(s["duration_ms"] for s in spans if s["name"] == name)


def print_table(rows: list[tuple[str, list[float]]]) -> None:
    print(f"{'stage':<16}{'avg':>8}{'p50':>8}{'p95':>8}{'p99':>8}   (ms)")
    for name, values in rows:
        if not values:
            continue
        print(
            f"{name:<16}"
            f"{statistics.mean(values):>8.2f}"
            f"{percentile(values, 50):>8.2f}"
            f"{percentile(values, 95):>8.2f}"
            f"{percentile(values, 99):>8.2f}"
        )


def run_local(queries: list[str]) -> dict[str, list[float]]:
    from ragoa.factory import load_retriever
    from ragoa.schemas import Deadline, Trace

    index_dir = Path(settings.index_dir)
    encoder = "onnx" if settings.onnx_model("embedder").exists() else "st"
    print("Warming up (model load + first inference)...")
    retriever, manifest = load_retriever(
        index_dir, encoder_kind=encoder, use_rerank=False, use_sparse=True, device="cpu",
    )
    print(
        f"index {manifest.get('n_docs', 0):,} docs / {manifest.get('n_chunks', 0):,} chunks"
        f"  encoder={encoder}  target=local"
    )

    for i, query in enumerate(queries[:WARMUP]):
        retriever.retrieve(query, Deadline(LATENCY_BUDGET_MS), Trace(request_id=f"w{i}"))

    buckets: dict[str, list[float]] = {
        "embed": [], "search": [], "total": [],
        "dense_search": [], "sparse_search": [],
    }
    measured = queries[WARMUP:]
    for i, query in enumerate(measured):
        trace = Trace(request_id=str(i))
        t0 = time.perf_counter()
        retriever.retrieve(query, Deadline(LATENCY_BUDGET_MS), trace)
        total = (time.perf_counter() - t0) * 1000
        spans = [s.model_dump() for s in trace.spans]
        embed = span_ms(spans, "embed_query")
        buckets["embed"].append(embed)
        buckets["search"].append(max(total - embed, 0.0))
        buckets["total"].append(total)
        buckets["dense_search"].append(span_ms(spans, "dense_search"))
        buckets["sparse_search"].append(span_ms(spans, "sparse_search"))
    return buckets


def post_ask(base: str, query: str, timeout_s: float = 60.0) -> tuple[dict, float]:
    payload = json.dumps({"query": query, "top_k": 3, "lang": "en"}).encode()
    req = urllib.request.Request(
        f"{base}/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode())
    rtt = (time.perf_counter() - t0) * 1000
    return body, rtt


def run_live(queries: list[str], url: str) -> dict[str, list[float]]:
    base = url.rstrip("/")
    print(f"Warming up live {base} ...")
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=180) as resp:
            health = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise SystemExit(f"live health failed: {exc}") from exc
    print(
        f"index {health.get('docs', 0):,} docs / {health.get('chunks', 0):,} chunks"
        f"  encoder={health.get('encoder')}  target={base}"
        f"  ready={health.get('ready')}"
    )

    for i, query in enumerate(queries[:WARMUP]):
        post_ask(base, query)
        print(f"  warmup {i + 1}/{WARMUP}", flush=True)

    buckets: dict[str, list[float]] = {
        "embed": [], "search": [], "total": [],
        "dense_search": [], "sparse_search": [],
        "client_rtt": [], "llm": [],
    }
    measured = queries[WARMUP:]
    errors = 0
    for i, query in enumerate(measured):
        try:
            body, rtt = post_ask(base, query)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            errors += 1
            print(f"  skip {i}: {exc}", flush=True)
            continue
        trace = body.get("trace") or {}
        spans = trace.get("spans") or []
        embed = span_ms(spans, "embed_query")
        retrieval = float(trace.get("retrieval_ms") or 0.0)
        if retrieval <= 0:
            retrieval = sum(span_ms(spans, name) for name in RETRIEVAL_SPANS)
        buckets["embed"].append(embed)
        buckets["search"].append(max(retrieval - embed, 0.0))
        buckets["total"].append(retrieval)
        buckets["dense_search"].append(span_ms(spans, "dense_search"))
        buckets["sparse_search"].append(span_ms(spans, "sparse_search"))
        buckets["client_rtt"].append(rtt)
        buckets["llm"].append(span_ms(spans, "llm"))
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  {i + 1}/{len(measured)}  retrieval={retrieval:.1f}ms  rtt={rtt:.0f}ms",
                  flush=True)
    if errors:
        print(f"errors: {errors}/{len(measured)}")
    return buckets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=50)
    ap.add_argument("--url", default="", help="Live API base, e.g. https://rag-in-goa.fly.dev")
    args = ap.parse_args()

    queries = load_queries(args.n + WARMUP)
    if len(queries) < WARMUP + 1:
        raise SystemExit(f"only {len(queries)} queries available")

    if args.url:
        buckets = run_live(queries, args.url)
        target = args.url.rstrip("/")
    else:
        buckets = run_local(queries)
        target = "local"

    n = len(buckets["total"])
    print(f"\nRan {n} queries  target={target}")
    print()
    rows = [
        ("embed", buckets["embed"]),
        ("search", buckets["search"]),
        ("total", buckets["total"]),
        ("dense_search", buckets["dense_search"]),
        ("sparse_search", buckets["sparse_search"]),
    ]
    if buckets.get("llm"):
        rows.extend([
            ("llm", buckets["llm"]),
            ("client_rtt", buckets["client_rtt"]),
        ])
    print_table(rows)

    p95_total = percentile(buckets["total"], 95) if buckets["total"] else float("inf")
    print(f"\nLatency budget: {LATENCY_BUDGET_MS:.0f}ms | p95 total: {p95_total:.2f}ms")
    if p95_total <= LATENCY_BUDGET_MS:
        print("PASS: within budget")
    else:
        print("FAIL: over budget -- see README 'Tuning latency' section")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Measure the numbers that decide our corpus scale and latency budget.

Three questions this answers with real data, not estimates:
  1. Corpus encode throughput -> how long a full-validation index build takes.
  2. Single short-query encode latency -> the biggest fixed cost inside the
     200ms retrieval ceiling.
  3. Cross-encoder rerank latency vs candidate count and chunk length -> whether
     a "greater chunk size + small K + strong rerank" design actually fits.

Sized for an 8GB M2 Air, which is the real constraint here. Large batches of
long sequences on MPS will push this machine into swap and the process ends up
in `stuck` state rather than failing cleanly, so batches stay small, the MPS
cache is dropped between phases, and RSS is printed after each phase. If you see
RSS climbing past ~3GB, lower --max-batch.

    python scripts/bench_embed.py                  # mps, default sizes
    python scripts/bench_embed.py --device cpu     # CPU comparison
    python scripts/bench_embed.py --quick          # fast sanity pass
"""

from __future__ import annotations

import argparse
import gc
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SLIM = Path("data/slim/hi.parquet")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ~4 chars per token, so 1800 chars is roughly the 450-token chunk we plan to ship.
CHUNK_CHARS = 1800
FULL_VALIDATION_QUERIES = 97_941


def say(message: str = "") -> None:
    """Print immediately: this script is watched while it runs, and block
    buffering makes a slow phase indistinguishable from a hang."""
    print(message, flush=True)


def rss_mb() -> float:
    # macOS reports maxrss in bytes, Linux in kilobytes.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def release(device: str) -> None:
    """Drop cached device memory between phases so peaks do not compound."""
    gc.collect()
    if device == "mps":
        try:
            import torch

            torch.mps.empty_cache()
        except (ImportError, AttributeError):
            pass


def load_texts(n_chunks: int) -> tuple[list[str], list[str]]:
    """Build realistic chunks and queries from the slim shard."""
    pf = pq.ParquetFile(SLIM)
    chunks: list[str] = []
    queries: list[str] = []

    for batch in pf.iter_batches(batch_size=500,
                                 columns=["eng_query", "passages_eng"]):
        for row in batch.to_pylist():
            doc = " ".join(row["passages_eng"])
            for start in range(0, len(doc), CHUNK_CHARS):
                piece = doc[start:start + CHUNK_CHARS]
                if len(piece) > 200:
                    chunks.append(piece)
            if row["eng_query"]:
                queries.append(row["eng_query"])
            if len(chunks) >= n_chunks:
                return chunks[:n_chunks], queries
    return chunks[:n_chunks], queries


class TorchBackend:
    """SentenceTransformer, used for the offline build where MPS is available."""

    label = "torch"

    def __init__(self, device: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(EMBED_MODEL, device=device)
        self.model.max_seq_length = 512

    def encode(self, texts: list[str], batch_size: int = 32):
        return self.model.encode(texts, batch_size=batch_size,
                                 show_progress_bar=False,
                                 normalize_embeddings=True)


class OnnxBackend:
    """Quantised ONNX, which is what the CPU-only serving path actually runs."""

    label = "onnx-int8"

    def __init__(self, threads: int = 4):
        from ragoa.config import settings as cfg
        from ragoa.embed.encoder import OnnxEncoder

        path = cfg.onnx_model("embedder")
        if not path.exists():
            raise SystemExit(f"missing {path}; run scripts/export_onnx.py first")
        self.encoder = OnnxEncoder(str(path), str(cfg.onnx_embedder_dir),
                                   threads=threads)

    def encode(self, texts: list[str], batch_size: int = 32):
        # Batched explicitly: OnnxEncoder pads to the longest item in the call, so
        # feeding it everything at once would pad short chunks to the longest one.
        out = [self.encoder.encode(texts[i:i + batch_size])
               for i in range(0, len(texts), batch_size)]
        return np.vstack(out) if out else np.zeros((0, self.encoder.dim))


def bench_corpus(model, chunks: list[str], batch_size: int) -> float:
    t0 = time.perf_counter()
    model.encode(chunks, batch_size=batch_size)
    elapsed = time.perf_counter() - t0
    return len(chunks) / elapsed


def bench_query(model, queries: list[str], runs: int) -> dict[str, float]:
    # Warm up so we measure steady state, which is what the server sees.
    for q in queries[:8]:
        model.encode([q])

    samples = []
    for i in range(runs):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        model.encode([q])
        samples.append((time.perf_counter() - t0) * 1000)

    samples.sort()
    return {
        "p50": statistics.median(samples),
        "p70": samples[int(0.70 * (len(samples) - 1))],
        "p95": samples[int(0.95 * (len(samples) - 1))],
        "p100": samples[-1],
    }


def bench_rerank(device: str, queries: list[str], chunks: list[str],
                 candidate_counts: tuple[int, ...], runs: int) -> None:
    from sentence_transformers import CrossEncoder

    say(f"\n--- reranker: {RERANK_MODEL} on {device} ---")
    say("  (p50 must stay well under the 200ms total, so <120ms is the bar)")
    ce = CrossEncoder(RERANK_MODEL, device=device, max_length=512)

    for n_cand in candidate_counts:
        for max_len in (256, 512):
            ce.max_length = max_len
            pairs = [(queries[0], c) for c in chunks[:n_cand]]
            ce.predict(pairs, batch_size=n_cand, show_progress_bar=False)  # warm

            samples = []
            for i in range(runs):
                pairs = [(queries[i % len(queries)], c) for c in chunks[:n_cand]]
                t0 = time.perf_counter()
                ce.predict(pairs, batch_size=n_cand, show_progress_bar=False)
                samples.append((time.perf_counter() - t0) * 1000)

            p50 = statistics.median(samples)
            verdict = "fits" if p50 < 120 else "TOO SLOW"
            say(f"  {n_cand:>2} candidates @ max_len {max_len:>3}: "
                f"p50 {p50:>7.1f} ms   {verdict}")
        release(device)

    say(f"  peak RSS after rerank: {rss_mb():,.0f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"],
                    help="torch backend only; ONNX Runtime here is CPU")
    ap.add_argument("--n-chunks", type=int, default=1200)
    ap.add_argument("--query-runs", type=int, default=60)
    ap.add_argument("--max-batch", type=int, default=32,
                    help="largest encode batch to try; 32 is the 8GB ceiling")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-rerank", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.n_chunks = 400
        args.query_runs = 25
        args.max_batch = 16

    if args.backend == "onnx":
        args.device = "cpu"
        say(f"loading ONNX int8 embedder on cpu, {args.threads} threads ...")
        model = OnnxBackend(threads=args.threads)
    else:
        say(f"loading {EMBED_MODEL} on {args.device} ...")
        model = TorchBackend(args.device)
    say(f"  loaded ({model.label}), RSS {rss_mb():,.0f} MB")

    chunks, queries = load_texts(args.n_chunks)
    avg_chars = sum(len(c) for c in chunks) / len(chunks)
    say(f"{len(chunks):,} chunks, avg {avg_chars:,.0f} chars (~{avg_chars / 4:,.0f} tokens); "
        f"{len(queries):,} queries")

    say(f"\n--- corpus encode: {model.label} on {args.device} ---")
    batch_sizes = [b for b in (8, 16, 32, 64, 128) if b <= args.max_batch]
    best = 0.0
    best_batch = batch_sizes[0]
    for batch_size in batch_sizes:
        rate = bench_corpus(model, chunks, batch_size)
        if rate > best:
            best, best_batch = rate, batch_size
        say(f"  batch {batch_size:>3}: {rate:>8,.1f} chunks/s   RSS {rss_mb():,.0f} MB")
        release(args.device)

    say(f"  best: batch {best_batch} at {best:,.1f} chunks/s")

    # Extrapolate the full-validation build from the measured best rate.
    for chunks_per_doc in (1.7, 2.0):
        total = int(FULL_VALIDATION_QUERIES * chunks_per_doc)
        minutes = total / best / 60
        say(f"  full validation @ {chunks_per_doc} chunks/doc = {total:,} vectors "
            f"-> {minutes:,.1f} min build, {total * 384 * 4 / 1e6:,.0f} MB raw fp32")

    say(f"\n--- single query encode (batch 1): {model.label} on {args.device} ---")
    q = bench_query(model, queries, args.query_runs)
    say(f"  p50 {q['p50']:.1f} ms | p70 {q['p70']:.1f} ms | "
        f"p95 {q['p95']:.1f} ms | p100 {q['p100']:.1f} ms")
    say(f"  budget check: {'OK' if q['p50'] < 20 else 'consider ONNX int8'} "
        f"(target <20ms of the 200ms ceiling)")

    # Hand the embedder's memory back before loading a second model.
    del model
    release(args.device)

    if args.skip_rerank:
        return
    candidates = (8, 16) if args.quick else (8, 12, 16, 24, 32)
    runs = 3 if args.quick else 7
    bench_rerank(args.device, queries, chunks, candidates, runs)


if __name__ == "__main__":
    main()

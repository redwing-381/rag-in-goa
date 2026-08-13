"""Decide the reranker backend on both axes at once: speed and ranking quality.

The reranker is the only stage that can consume the entire 200ms budget, and the
deploy target is CPU-only, so this is the load-bearing decision in the retrieval
path. Torch, ONNX fp32 and ONNX int8 are measured on the same candidate sets in
the same process.

Quality is measured as gold-passage rank on realistic candidate sets - one
passage that genuinely contains the answer plus distractors drawn from other
documents - because that is the reranker's actual job. Rank correlation against
torch on arbitrary text is the wrong test: when most candidates are irrelevant
their scores bunch together, so the correlation mostly reports noise in ranks
nobody will ever read. What matters is whether the answer-bearing passage still
comes first.

    python scripts/bench_rerank_backends.py
    python scripts/bench_rerank_backends.py --queries 40 --candidates 8 12 16
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CORPUS = Path("data/corpus/docs.parquet")
ONNX_DIR = Path("data/onnx/reranker")
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Case:
    """One query with a candidate list whose first entry is the gold passage."""

    __slots__ = ("query", "candidates", "gold_index")

    def __init__(self, query: str, candidates: list[str], gold_index: int):
        self.query = query
        self.candidates = candidates
        self.gold_index = gold_index


def passages_of(row: dict) -> list[str]:
    text = row["text"]
    return [text[s:e] for s, e in zip(row["passage_starts"], row["passage_ends"],
                                      strict=True)]


def gold_passage_of(row: dict) -> str | None:
    """The passage that covers the first gold span."""
    if not row["gold_starts"]:
        return None
    gold_start, gold_end = row["gold_starts"][0], row["gold_ends"][0]
    for start, end in zip(row["passage_starts"], row["passage_ends"], strict=True):
        if start < gold_end and gold_start < end:
            return row["text"][start:end]
    return None


def build_cases(n_queries: int, n_candidates: int, seed: int = 17) -> list[Case]:
    rows = pq.read_table(CORPUS).to_pylist()
    answerable = [r for r in rows if r["answerable"] and r["gold_starts"]
                  and r["eng_query"]]

    rng = random.Random(seed)
    # Distractor pool from every document, so distractors are real passages that a
    # retriever could plausibly have surfaced.
    pool: list[str] = []
    for row in rows:
        pool.extend(p for p in passages_of(row) if len(p) > 200)

    cases: list[Case] = []
    for row in rng.sample(answerable, min(n_queries, len(answerable))):
        gold = gold_passage_of(row)
        if not gold or len(gold) < 100:
            continue
        own = set(passages_of(row))
        distractors: list[str] = []
        while len(distractors) < n_candidates - 1:
            pick = rng.choice(pool)
            if pick not in own:
                distractors.append(pick)

        candidates = [gold, *distractors]
        order = list(range(len(candidates)))
        rng.shuffle(order)
        shuffled = [candidates[i] for i in order]
        cases.append(Case(row["eng_query"], shuffled, order.index(0)))
    return cases


def make_backend(name: str, max_length: int):
    from ragoa.index.rerank import CrossEncoderReranker, OnnxReranker

    if name == "torch":
        return CrossEncoderReranker(RERANK_MODEL, max_length=max_length, device="cpu")
    filename = "model.onnx" if name == "onnx-fp32" else "model_int8.onnx"
    path = ONNX_DIR / filename
    if not path.exists():
        return None
    return OnnxReranker(str(path), str(ONNX_DIR), max_length=max_length)


def evaluate(backend, cases: list[Case]) -> dict:
    """Latency plus gold rank, from one pass over the cases."""
    # Warm up: first call pays session and tokenizer setup.
    backend.score(cases[0].query, cases[0].candidates)

    latencies: list[float] = []
    gold_ranks: list[int] = []
    all_scores: list[np.ndarray] = []

    for case in cases:
        t0 = time.perf_counter()
        scores = backend.score(case.query, case.candidates)
        latencies.append((time.perf_counter() - t0) * 1000)
        # Rank of the gold passage, 1 = best.
        order = np.argsort(-scores)
        gold_ranks.append(int(np.where(order == case.gold_index)[0][0]) + 1)
        all_scores.append(scores)

    latencies.sort()
    return {
        "p50": statistics.median(latencies),
        "p95": latencies[int(0.95 * (len(latencies) - 1))],
        "p100": latencies[-1],
        "gold_at_1": sum(r == 1 for r in gold_ranks) / len(gold_ranks),
        "gold_at_3": sum(r <= 3 for r in gold_ranks) / len(gold_ranks),
        "mean_gold_rank": statistics.mean(gold_ranks),
        "scores": all_scores,
    }


def top1_agreement(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    return sum(int(np.argmax(x)) == int(np.argmax(y))
               for x, y in zip(a, b, strict=True)) / len(a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--candidates", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--max-lengths", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--budget-ms", type=float, default=110.0,
                    help="rerank budget: 200ms total minus the other stages")
    args = ap.parse_args()

    names = ["torch", "onnx-fp32", "onnx-int8"]

    for n_cand in args.candidates:
        cases = build_cases(args.queries, n_cand)
        for max_len in args.max_lengths:
            print(f"\n=== {len(cases)} queries, {n_cand} candidates, "
                  f"max_len {max_len} ===", flush=True)
            print(f"{'backend':<11} {'p50':>8} {'p95':>8} "
                  f"{'gold@1':>7} {'gold@3':>7} {'vs torch':>9}  verdict")

            baseline: list[np.ndarray] | None = None
            for name in names:
                backend = make_backend(name, max_len)
                if backend is None:
                    print(f"{name:<11} not exported; run scripts/export_onnx.py")
                    continue

                result = evaluate(backend, cases)
                if baseline is None:
                    baseline = result["scores"]
                    agreement = "-"
                else:
                    agreement = f"{top1_agreement(baseline, result['scores']):.0%}"

                fits = result["p50"] <= args.budget_ms
                verdict = "fits" if fits else "over budget"
                print(f"{name:<11} {result['p50']:>7.1f}ms {result['p95']:>7.1f}ms "
                      f"{result['gold_at_1']:>7.1%} {result['gold_at_3']:>7.1%} "
                      f"{agreement:>9}  {verdict}", flush=True)

    print(f"\nbudget line: rerank p50 must stay under {args.budget_ms:.0f}ms")
    print("gold@1 is the quality bar; agreement with torch only explains changes.")


if __name__ == "__main__":
    main()

"""Validate every chunking strategy and report the metrics that need no model.

Two invariants are checked hard, because the entire evaluation rests on them:
  1. Offsets are exact - `doc.text[char_start:char_end]` must equal the chunk
     text (metadata-aware is exempt, since it deliberately prefixes the embedded
     string while leaving offsets pointing at the untouched document).
  2. Chunks tile the document - every gold span must be reachable, i.e. overlapped
     by at least one chunk. A strategy that silently drops text would cap its own
     recall and the offset check alone would not catch it.

It also reports **gold containment**: the share of gold passages that fit
entirely inside a single chunk. That is the metric that actually separates these
strategies, and it needs no embedding model. A gold passage split across two
chunks means neither one holds the whole answer, so retrieval has to be lucky.

    python scripts/verify_chunking.py --docs 2000
"""

from __future__ import annotations

import argparse
import time

from ragoa.chunking.base import Chunker
from ragoa.chunking.registry import deterministic_sweep
from ragoa.data.docbuilder import iter_documents
from ragoa.schemas import ChunkingStrategy, PseudoDocument


def evaluate(chunker: Chunker, docs: list[PseudoDocument]) -> dict[str, float]:
    n_chunks = 0
    total_chars = 0
    gold_total = 0
    gold_reachable = 0
    gold_contained = 0
    ctx_contained = 0
    fragments = 0
    offset_errors = 0

    exempt = chunker.strategy is ChunkingStrategy.METADATA_AWARE

    t0 = time.perf_counter()
    for doc in docs:
        chunks = chunker.chunk(doc)
        n_chunks += len(chunks)

        for chunk in chunks:
            total_chars += chunk.char_end - chunk.char_start
            if not exempt and doc.text[chunk.char_start:chunk.char_end] != chunk.text:
                offset_errors += 1

        for span in doc.gold_spans:
            gold_total += 1
            overlapping = [c for c in chunks if c.overlaps(span)]
            if overlapping:
                gold_reachable += 1
            fragments += len(overlapping)
            if any(c.char_start <= span[0] and c.char_end >= span[1] for c in overlapping):
                gold_contained += 1
            # Small-to-big strategies retrieve on a child but generate from a
            # wider span, so judging them on the indexed span alone understates
            # what the LLM actually gets to read.
            for c in overlapping:
                ctx_start, ctx_end = c.context_span()
                if ctx_start <= span[0] and ctx_end >= span[1]:
                    ctx_contained += 1
                    break

    elapsed = time.perf_counter() - t0
    n_docs = max(len(docs), 1)

    return {
        "chunks_per_doc": n_chunks / n_docs,
        "total_chunks": n_chunks,
        "avg_chunk_chars": total_chars / max(n_chunks, 1),
        "reachable_pct": 100.0 * gold_reachable / max(gold_total, 1),
        "contained_pct": 100.0 * gold_contained / max(gold_total, 1),
        "ctx_contained_pct": 100.0 * ctx_contained / max(gold_total, 1),
        "fragments_per_gold": fragments / max(gold_total, 1),
        "offset_errors": offset_errors,
        "docs_per_s": n_docs / max(elapsed, 1e-9),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=2000)
    args = ap.parse_args()

    docs = [d for d in iter_documents("hi", limit=args.docs)]
    n_gold = sum(len(d.gold_spans) for d in docs)
    print(f"{len(docs):,} pseudo-documents, {n_gold:,} gold spans, "
          f"avg {sum(len(d.text) for d in docs) / len(docs):,.0f} chars/doc\n")

    header = (f"{'strategy':<52} {'chunks':>8} {'/doc':>6} {'avg c':>7} "
              f"{'reach%':>7} {'contain%':>9} {'ctx%':>7} {'frag':>6} {'docs/s':>8} {'err':>5}")
    print(header)
    print("-" * len(header))

    rows = []
    failures = 0
    for chunker in deterministic_sweep():
        m = evaluate(chunker, docs)
        rows.append((chunker.name, m))
        failures += m["offset_errors"]
        print(f"{chunker.name:<52} {m['total_chunks']:>8,.0f} {m['chunks_per_doc']:>6.2f} "
              f"{m['avg_chunk_chars']:>7,.0f} {m['reachable_pct']:>7.2f} "
              f"{m['contained_pct']:>9.2f} {m['ctx_contained_pct']:>7.2f} "
              f"{m['fragments_per_gold']:>6.2f} "
              f"{m['docs_per_s']:>8,.0f} {m['offset_errors']:>5,}")

    print()
    best = max(rows, key=lambda r: r[1]["contained_pct"])
    print(f"best indexed-span containment : {best[0]} at {best[1]['contained_pct']:.2f}%")
    best_ctx = max(rows, key=lambda r: r[1]["ctx_contained_pct"])
    print(f"best context-span containment : {best_ctx[0]} at "
          f"{best_ctx[1]['ctx_contained_pct']:.2f}%")
    leanest = min(rows, key=lambda r: r[1]["chunks_per_doc"])
    print(f"fewest vectors        : {leanest[0]} at {leanest[1]['chunks_per_doc']:.2f} chunks/doc")

    unreachable = [name for name, m in rows if m["reachable_pct"] < 99.999]
    if unreachable:
        print(f"\nFAIL: strategies dropping gold text: {unreachable}")
    if failures:
        print(f"FAIL: {failures:,} offset mismatches")
    if not unreachable and not failures:
        print("\nall invariants hold: offsets exact, gold fully reachable")

    # Extrapolate index size to the full validation split.
    print("\nfull-validation index projection (97,941 docs, 384-dim float32):")
    for name, m in sorted(rows, key=lambda r: r[1]["chunks_per_doc"]):
        vectors = int(97_941 * m["chunks_per_doc"])
        print(f"  {name:<52} {vectors:>9,} vectors  {vectors * 384 * 4 / 1e6:>7,.0f} MB")


if __name__ == "__main__":
    main()

"""Build the chunk store, dense index and lexical index in one streaming pass.

Nothing accumulates: documents stream out of parquet, chunks stream into the
on-disk store, and vectors stream into HNSW batch by batch. Peak RSS is reported
so the 8 GB budget is verified rather than assumed.

    # validate the plumbing with no model weights at all
    python scripts/build_index.py --docs 20000 --encoder hash --no-sparse

    # the real build
    python scripts/build_index.py --encoder st
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ragoa.chunking.registry import default_chunker
from ragoa.config import settings
from ragoa.data.docbuilder import iter_documents
from ragoa.embed.encoder import HashEncoder, SentenceTransformerEncoder
from ragoa.index.dense import DenseIndex
from ragoa.index.sparse import SparseIndex
from ragoa.index.store import ChunkStore, ChunkStoreBuilder
from ragoa.telemetry.rss import peak_rss_mb


def build_store(index_dir: Path, source_lang: str, limit: int | None) -> tuple[int, int]:
    chunker = default_chunker()
    builder = ChunkStoreBuilder(index_dir)
    n_docs = 0
    t0 = time.perf_counter()

    for doc in iter_documents(source_lang, limit):
        builder.add(doc, chunker.chunk(doc))
        n_docs += 1
        if n_docs % 20_000 == 0:
            print(f"  store: {n_docs:,} docs, {len(builder.chunk_ids):,} chunks, "
                  f"peak {peak_rss_mb():,.0f} MB", flush=True)

    n_chunks = builder.close()
    print(f"store   : {n_docs:,} docs -> {n_chunks:,} chunks "
          f"({chunker.name}) in {time.perf_counter() - t0:.1f}s")
    return n_docs, n_chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=None, help="limit documents (default: all)")
    ap.add_argument("--source-lang", default="hi")
    ap.add_argument("--encoder", choices=["st", "hash"], default="st",
                    help="'hash' needs no weights and is for plumbing tests only")
    ap.add_argument("--device", default=None, help="mps or cpu (default: auto)")
    ap.add_argument("--no-sparse", action="store_true")
    ap.add_argument("--index-dir", default=None)
    args = ap.parse_args()

    index_dir = Path(args.index_dir) if args.index_dir else settings.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    n_docs, n_chunks = build_store(index_dir, args.source_lang, args.docs)

    store = ChunkStore(index_dir)
    print(f"store   : offset arrays resident = {store.resident_mb():,.1f} MB "
          f"(text stays memory-mapped)")

    if args.encoder == "hash":
        encoder = HashEncoder(dim=settings.embed_dim, query_prefix=settings.query_prefix)
        print("encoder : HashEncoder (NOT semantic - plumbing test only)")
    else:
        encoder = SentenceTransformerEncoder(
            settings.embed_model, device=args.device,
            dim=settings.embed_dim, query_prefix=settings.query_prefix,
        )
        print(f"encoder : {settings.embed_model} on {encoder.device}")

    dense = DenseIndex(dim=settings.embed_dim, m=settings.hnsw_m,
                       ef_construction=settings.hnsw_ef_construction,
                       ef_search=settings.hnsw_ef_search)
    t0 = time.perf_counter()
    inserted = dense.build(store.iter_texts(batch_size=512), capacity=n_chunks,
                           encoder=encoder, batch_size=settings.embed_batch_size)
    elapsed = time.perf_counter() - t0
    dense.save(settings.hnsw_path if index_dir == settings.index_dir
               else index_dir / "dense.hnsw")
    print(f"dense   : {inserted:,} vectors in {elapsed:,.1f}s "
          f"({inserted / max(elapsed, 1e-9):,.0f} chunks/s), "
          f"~{dense.resident_mb():,.0f} MB resident")

    sparse_stats = None
    if not args.no_sparse:
        sparse = SparseIndex()
        t0 = time.perf_counter()
        texts = [t for batch in store.iter_texts(batch_size=2048) for t in batch]
        sparse_stats = sparse.build(texts)
        del texts
        sparse.save(index_dir / "bm25")
        print(f"sparse  : {n_chunks:,} docs in {time.perf_counter() - t0:,.1f}s, "
              f"peak after index {sparse_stats['rss_after_index_mb']:,.0f} MB")

    manifest = {
        "n_docs": n_docs,
        "n_chunks": n_chunks,
        "strategy": store.strategy.value,
        "encoder": args.encoder,
        "embed_model": settings.embed_model if args.encoder == "st" else "hash",
        "embed_dim": settings.embed_dim,
        "has_sparse": not args.no_sparse,
        "build_seconds": round(time.perf_counter() - t_start, 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nmanifest: {index_dir / 'manifest.json'}")
    print(f"total   : {manifest['build_seconds']:,.1f}s, peak RSS "
          f"{manifest['peak_rss_mb']:,.0f} MB")
    on_disk = sum(p.stat().st_size for p in index_dir.rglob("*") if p.is_file())
    print(f"on disk : {on_disk / 1e6:,.0f} MB")


if __name__ == "__main__":
    main()

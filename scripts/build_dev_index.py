"""Build a small local index so the API can start without the Colab dump.

The shipped path expects data/index from ragoa-index.tar.gz. This writes a
tiny semantic index over the README sample topics so Windows / first-run
setups can boot.
"""

from __future__ import annotations

import json
import time

from ragoa.chunking.registry import default_chunker
from ragoa.config import settings
from ragoa.embed.encoder import SentenceTransformerEncoder
from ragoa.index.dense import DenseIndex
from ragoa.index.sparse import SparseIndex
from ragoa.index.store import ChunkStore, ChunkStoreBuilder
from ragoa.schemas import Language, PseudoDocument, QueryType
from ragoa.telemetry.rss import peak_rss_mb

DOCS: list[tuple[str, QueryType, str]] = [
    (
        "corporation",
        QueryType.DESCRIPTION,
        "A corporation is a legal entity that is separate from its owners. "
        "It can enter contracts, own property, and is typically owned by "
        "shareholders who elect a board of directors. Limited liability means "
        "shareholders are not personally responsible for the corporation's debts.",
    ),
    (
        "mount-fuji",
        QueryType.ENTITY,
        "Mount Fuji is a stratovolcano, an active composite volcano, on Honshu "
        "in Japan. It is Japan's highest peak at 3,776 metres. A stratovolcano "
        "is built from layers of lava, ash, and tephra. Mount Fuji last erupted "
        "in 1707 during the Hoei eruption.",
    ),
    (
        "bridget-moynahan",
        QueryType.PERSON,
        "Bridget Moynahan is an American actress. She was married to NFL "
        "quarterback Tom Brady. The two were engaged and later separated; they "
        "have a son together. Moynahan is known for roles in Blue Bloods and "
        "I, Robot.",
    ),
    (
        "kinsey",
        QueryType.PERSON,
        "Alfred Kinsey is most known for the Kinsey Reports on human sexual "
        "behavior and for founding the Kinsey Institute. His research in the "
        "1940s and 1950s surveyed thousands of people and made sex research a "
        "public scientific topic.",
    ),
    (
        "philosophy",
        QueryType.DESCRIPTION,
        "Philosophy is the study of fundamental questions about existence, "
        "knowledge, values, reason, mind, and language. The word comes from "
        "Greek philosophia, meaning love of wisdom. Major branches include "
        "metaphysics, epistemology, ethics, and logic.",
    ),
    (
        "potassium-bananas",
        QueryType.NUMERIC,
        "Bananas are a well-known dietary source of potassium. A medium banana "
        "contains about 400 to 450 milligrams of potassium. Potassium helps "
        "regulate fluid balance and muscle contractions.",
    ),
]


def main() -> None:
    index_dir = settings.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    chunker = default_chunker()
    builder = ChunkStoreBuilder(index_dir)
    t_start = time.perf_counter()

    for i, (slug, query_type, text) in enumerate(DOCS, start=1):
        doc = PseudoDocument(
            doc_id=slug,
            query_id=i,
            query_type=query_type,
            lang=Language.EN,
            text=text,
            eng_query=slug.replace("-", " "),
        )
        builder.add(doc, chunker.chunk(doc))

    n_chunks = builder.close()
    store = ChunkStore(index_dir)
    encoder = SentenceTransformerEncoder(
        settings.embed_model, device="cpu",
        dim=settings.embed_dim, query_prefix=settings.query_prefix,
    )
    dense = DenseIndex(
        dim=settings.embed_dim, m=settings.hnsw_m,
        ef_construction=settings.hnsw_ef_construction,
        ef_search=settings.hnsw_ef_search,
    )
    inserted = dense.build(
        store.iter_texts(batch_size=32), capacity=n_chunks,
        encoder=encoder, batch_size=settings.embed_batch_size,
    )
    dense.save(index_dir / "dense.hnsw")

    sparse = SparseIndex()
    texts = [t for batch in store.iter_texts(batch_size=32) for t in batch]
    sparse.build(texts)
    sparse.save(index_dir / "bm25")

    manifest = {
        "n_docs": len(DOCS),
        "n_chunks": inserted,
        "strategy": store.strategy.value,
        "encoder": "st",
        "embed_model": settings.embed_model,
        "embed_dim": settings.embed_dim,
        "has_sparse": True,
        "build_seconds": round(time.perf_counter() - t_start, 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "dev_index": True,
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {index_dir} ({inserted} chunks, {len(DOCS)} docs)", flush=True)


if __name__ == "__main__":
    main()

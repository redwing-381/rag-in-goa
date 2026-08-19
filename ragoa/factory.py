"""Assembly in one place.

The API, the benchmarks and the evaluation scripts must all run the *same*
pipeline, or the latency numbers we publish describe something the demo does not
do. Everything is constructed here so there is a single definition of "the
system".
"""

from __future__ import annotations

import json
from pathlib import Path

from ragoa.config import Settings
from ragoa.config import settings as default_settings
from ragoa.guardrails.input_gate import InputGate
from ragoa.guardrails.output_gate import OutputGate
from ragoa.harness.llm import OpenRouterLLM
from ragoa.harness.pipeline import RagPipeline
from ragoa.index.dense import DenseIndex
from ragoa.index.retriever import HybridRetriever
from ragoa.index.sparse import SparseIndex
from ragoa.index.store import ChunkStore


def read_manifest(index_dir: Path) -> dict:
    path = index_dir / "manifest.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/build_index.py first")
    return json.loads(path.read_text())


def load_encoder(kind: str, dim: int, cfg: Settings, device: str | None = None):
    """`hash` needs no weights and is for plumbing tests; `onnx` is the serving path."""
    if kind == "hash":
        from ragoa.embed.encoder import HashEncoder

        return HashEncoder(dim=dim, query_prefix=cfg.query_prefix)

    if kind == "onnx":
        onnx_path = cfg.onnx_model("embedder")
        tokenizer_dir = cfg.onnx_embedder_dir
        if onnx_path.exists() and tokenizer_dir.exists():
            from ragoa.embed.encoder import OnnxEncoder

            return OnnxEncoder(
                onnx_path=str(onnx_path),
                tokenizer_dir=str(tokenizer_dir),
                dim=dim, query_prefix=cfg.query_prefix, threads=cfg.onnx_threads,
            )
        print(
            f"onnx encoder missing at {onnx_path}; falling back to sentence-transformers",
            flush=True,
        )

    from ragoa.embed.encoder import SentenceTransformerEncoder

    return SentenceTransformerEncoder(
        cfg.embed_model, device=device, dim=dim, query_prefix=cfg.query_prefix
    )


def load_reranker(cfg: Settings):
    """Quantised ONNX when it is available, torch otherwise.

    The fallback is not a convenience: a missing export would otherwise take the
    whole service down at boot, and a slower reranker is a far better outcome than
    no service. The deadline logic already handles a reranker that is too slow by
    skipping it, so degrading here is safe.
    """
    onnx_path = cfg.onnx_model("reranker")
    if cfg.prefer_onnx and onnx_path.exists():
        from ragoa.index.rerank import OnnxReranker

        return OnnxReranker(str(onnx_path), str(cfg.onnx_reranker_dir),
                            max_length=cfg.rerank_max_length,
                            threads=cfg.onnx_threads)

    from ragoa.index.rerank import CrossEncoderReranker

    return CrossEncoderReranker(cfg.rerank_model, max_length=cfg.rerank_max_length)


def load_retriever(
    index_dir: Path,
    encoder_kind: str = "st",
    use_rerank: bool = False,
    use_sparse: bool = True,
    device: str | None = None,
    settings: Settings | None = None,
) -> tuple[HybridRetriever, dict]:
    cfg = settings or default_settings
    manifest = read_manifest(index_dir)

    store = ChunkStore(index_dir)
    dense = DenseIndex(dim=manifest["embed_dim"], m=cfg.hnsw_m,
                       ef_search=cfg.hnsw_ef_search)
    dense.load(index_dir / "dense.hnsw", size=manifest["n_chunks"])

    sparse = None
    if use_sparse and manifest.get("has_sparse"):
        sparse = SparseIndex()
        sparse.load(index_dir / "bm25", size=manifest["n_chunks"])

    reranker = load_reranker(cfg) if use_rerank else None

    encoder = load_encoder(encoder_kind, manifest["embed_dim"], cfg, device)
    retriever = HybridRetriever(store, dense, encoder, sparse=sparse,
                                reranker=reranker, settings=cfg)
    return retriever, manifest


def build_pipeline(
    index_dir: Path | None = None,
    encoder_kind: str = "st",
    use_rerank: bool = False,
    use_sparse: bool = True,
    use_stt: bool = True,
    llm=None,
    settings: Settings | None = None,
) -> tuple[RagPipeline, dict]:
    cfg = settings or default_settings
    retriever, manifest = load_retriever(
        index_dir or cfg.index_dir, encoder_kind, use_rerank, use_sparse, settings=cfg
    )

    stt = None
    if use_stt and cfg.sarvam_api_key:
        from ragoa.stt.sarvam import SarvamSTT

        stt = SarvamSTT(cfg)

    translator = None
    if cfg.sarvam_api_key:
        from ragoa.stt.translate import SarvamTranslate

        translator = SarvamTranslate(cfg)

    pipeline = RagPipeline(
        retriever=retriever,
        llm=llm or OpenRouterLLM(cfg),
        input_gate=InputGate(cfg),
        # The output gate reuses the retriever's encoder for its semantic rescue
        # path, so a paraphrased-but-correct answer is not refused. No second model.
        output_gate=OutputGate(cfg, encoder=retriever.encoder),
        stt=stt,
        translator=translator,
        settings=cfg,
    )
    return pipeline, manifest

"""Export the embedder and reranker to ONNX, then quantise to int8.

This is not an optimisation we chose, it is one the budget forces. Measured on an
M2 CPU, the torch cross-encoder needs ~177ms to score 8 candidates at 256 tokens,
which alone exceeds the 200ms ceiling for the whole pipeline. The deploy target
has no GPU, so CPU is the only number that counts and int8 ONNX is the way back
under budget.

Quantisation is only worth taking if it does not move the ranking, so this
verifies against torch instead of assuming: cosine agreement for the embedder,
and Spearman plus top-1 agreement for the reranker, which is what actually
matters for a reranker's job.

    python scripts/export_onnx.py                  # both models, arm64 (local)
    python scripts/export_onnx.py --arch avx512_vnni   # for x86 deploy
    python scripts/export_onnx.py --only reranker
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUT_ROOT = Path("data/onnx")

# Held-out-ish text for the agreement check. Deliberately varied in length and
# vocabulary so quantisation error is not measured on one easy distribution.
PROBE_TEXTS = [
    "what is the capital of france",
    "The mitochondrion is the powerhouse of the cell, generating most of the "
    "chemical energy needed to power the cell's biochemical reactions.",
    "how much does a 2019 honda civic cost in bangalore",
    "Reciprocal rank fusion combines ranked lists by summing 1/(k+rank) across "
    "each list, which avoids having to calibrate scores from different retrievers "
    "onto a shared scale before combining them.",
    "symptoms of vitamin b12 deficiency in adults over 50",
    "Goa is a state on the southwestern coast of India within the Konkan region, "
    "geographically separated from the Deccan highlands by the Western Ghats.",
    "define ephemeral",
    "The train leaves at 6:45 from platform 3 and arrives the following morning.",
]


def dir_size_mb(path: Path, pattern: str) -> float:
    return sum(f.stat().st_size for f in path.glob(pattern)) / 1e6


def reranker_cases(n: int = 25) -> list[tuple[str, list[str]]]:
    """Realistic (query, candidates) sets: real queries over real passages.

    Quantisation error only matters where it changes a decision, so it has to be
    measured on the kind of input the reranker sees in production. Falls back to
    the generic probes when the corpus has not been built yet.
    """
    # Running this file directly puts scripts/ on sys.path rather than the repo
    # root, so the sibling module is not importable without this.
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

    try:
        from scripts.bench_rerank_backends import build_cases

        return [(c.query, c.candidates) for c in build_cases(n, 8)]
    except Exception as exc:
        print(f"  (corpus unavailable, verifying on generic probes instead: {exc})")
        return [("what is reciprocal rank fusion", list(PROBE_TEXTS))]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without pulling in scipy."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / denom) if denom else 1.0


def quantize(out_dir: Path, arch: str) -> Path | None:
    from optimum.onnxruntime import AutoQuantizationConfig, ORTQuantizer

    # Re-running the export leaves an int8 graph next to the fp32 one, and the
    # quantiser refuses a directory holding more than one model, so clear our own
    # earlier output and name the source explicitly.
    for stale in out_dir.glob("*quantized*.onnx"):
        stale.unlink()
    (out_dir / "model_int8.onnx").unlink(missing_ok=True)

    quantizer = ORTQuantizer.from_pretrained(out_dir, file_name="model.onnx")
    # Dynamic quantisation: weights to int8 offline, activations at runtime. No
    # calibration set needed, which is what we want for an encoder whose input
    # distribution is user speech.
    config = getattr(AutoQuantizationConfig, arch)(is_static=False, per_channel=True)
    quantizer.quantize(save_dir=out_dir, quantization_config=config)

    produced = sorted(out_dir.glob("*quantized*.onnx"))
    if not produced:
        return None
    target = out_dir / "model_int8.onnx"
    shutil.move(str(produced[0]), target)
    return target


def export_embedder(arch: str) -> bool:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    out_dir = OUT_ROOT / "embedder"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== embedder: {EMBED_MODEL} -> {out_dir} ===", flush=True)

    model = ORTModelForFeatureExtraction.from_pretrained(EMBED_MODEL, export=True)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(EMBED_MODEL).save_pretrained(out_dir)
    print(f"  fp32 exported: {dir_size_mb(out_dir, 'model.onnx'):,.0f} MB", flush=True)

    int8_path = quantize(out_dir, arch)
    if int8_path is None:
        print("  FAIL: quantiser produced no file")
        return False
    print(f"  int8 written : {int8_path.stat().st_size / 1e6:,.0f} MB", flush=True)

    # Agreement against torch. bge pools on CLS and L2-normalises, so compare the
    # vectors the retriever would actually store.
    from sentence_transformers import SentenceTransformer

    from ragoa.embed.encoder import OnnxEncoder

    reference = SentenceTransformer(EMBED_MODEL, device="cpu").encode(
        PROBE_TEXTS, normalize_embeddings=True, show_progress_bar=False
    )

    ok = True
    for label, filename in (("fp32", "model.onnx"), ("int8", "model_int8.onnx")):
        encoder = OnnxEncoder(str(out_dir / filename), str(out_dir))
        vectors = encoder.encode(list(PROBE_TEXTS))
        cosines = np.sum(vectors * reference, axis=1)
        worst = float(cosines.min())
        # 0.99 keeps quantisation error far below the gap between a relevant and
        # an irrelevant passage, so neighbour order is preserved.
        verdict = "OK" if worst >= 0.99 else "DEGRADED"
        if worst < 0.99:
            ok = False
        print(f"  {label} vs torch: mean cos {cosines.mean():.5f}, "
              f"worst {worst:.5f}  {verdict}", flush=True)
    return ok


def export_reranker(arch: str) -> bool:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    out_dir = OUT_ROOT / "reranker"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== reranker: {RERANK_MODEL} -> {out_dir} ===", flush=True)

    model = ORTModelForSequenceClassification.from_pretrained(RERANK_MODEL, export=True)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(RERANK_MODEL).save_pretrained(out_dir)
    print(f"  fp32 exported: {dir_size_mb(out_dir, 'model.onnx'):,.0f} MB", flush=True)

    int8_path = quantize(out_dir, arch)
    if int8_path is None:
        print("  FAIL: quantiser produced no file")
        return False
    print(f"  int8 written : {int8_path.stat().st_size / 1e6:,.0f} MB", flush=True)

    from sentence_transformers import CrossEncoder

    from ragoa.index.rerank import OnnxReranker

    cases = reranker_cases()
    torch_model = CrossEncoder(RERANK_MODEL, device="cpu", max_length=256)
    reference = [
        np.asarray(torch_model.predict([(q, p) for p in passages],
                                       show_progress_bar=False), dtype=np.float32)
        for q, passages in cases
    ]

    ok = True
    for label, filename in (("fp32", "model.onnx"), ("int8", "model_int8.onnx")):
        reranker = OnnxReranker(str(out_dir / filename), str(out_dir), max_length=256)
        scores = [reranker.score(q, passages) for q, passages in cases]

        # Top-1 is the decision the retriever actually consumes. Rank correlation
        # over the whole list is reported for context but not gated on: when most
        # candidates are irrelevant their scores bunch up, so the lower ranks
        # reorder on quantisation noise without changing any served result.
        top1 = sum(int(np.argmax(r)) == int(np.argmax(s))
                   for r, s in zip(reference, scores, strict=True)) / len(cases)
        rho = statistics.mean(spearman(r, s)
                              for r, s in zip(reference, scores, strict=True))
        drift = max(float(np.abs(r - s).max())
                    for r, s in zip(reference, scores, strict=True))

        verdict = "OK" if top1 >= 0.98 else "DEGRADED"
        if top1 < 0.98:
            ok = False
        print(f"  {label} vs torch: top1 {top1:.0%} over {len(cases)} sets, "
              f"mean spearman {rho:.3f}, max score drift {drift:.3f}  {verdict}",
              flush=True)

    print("  (full speed/quality comparison: scripts/bench_rerank_backends.py)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["embedder", "reranker"])
    ap.add_argument("--arch", default="arm64",
                    choices=["arm64", "avx2", "avx512", "avx512_vnni"],
                    help="arm64 for local Apple Silicon, avx512_vnni for x86 deploy")
    args = ap.parse_args()

    print(f"quantisation target: {args.arch}")
    results = {}
    if args.only in (None, "embedder"):
        results["embedder"] = export_embedder(args.arch)
    if args.only in (None, "reranker"):
        results["reranker"] = export_reranker(args.arch)

    print()
    bad = [name for name, ok in results.items() if not ok]
    if bad:
        print(f"{', '.join(bad)} lost accuracy in quantisation; do not ship int8 for these")
        return 1
    print(f"exported and verified: {', '.join(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

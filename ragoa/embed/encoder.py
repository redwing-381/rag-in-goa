"""Encoders behind one interface.

Three implementations, for three different jobs:

  SentenceTransformerEncoder  index building, can use MPS
  OnnxEncoder                 serving; no torch resident, 2-3x faster on CPU
  HashEncoder                 tests and CI; deterministic, needs no weights

The split matters on an 8 GB machine: torch plus a model is ~600-900 MB resident,
while ONNX int8 is ~150-250 MB. The build runs once and can afford torch; the
server runs constantly and cannot.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Encoder(Protocol):
    dim: int

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return matrix / norms


def best_device() -> str:
    """CUDA first, so the index build uses the GPU when run on Colab or Kaggle.

    Checking only for MPS would silently fall back to CPU on exactly the hosted
    GPU box we go to for the build, turning a five-minute job into hours.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SentenceTransformerEncoder:
    """Torch-backed encoder for index building."""

    def __init__(self, model_name: str, device: str | None = None,
                 dim: int = 384, query_prefix: str = ""):
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = best_device()
        self.device = device
        self.dim = dim
        self.query_prefix = query_prefix
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 512

    def encode(self, texts: list[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """bge wants an instruction prefix on queries but not on passages."""
        return self.encode([self.query_prefix + text])[0]


class OnnxEncoder:
    """ONNX Runtime encoder for the serving path."""

    def __init__(self, onnx_path: str, tokenizer_dir: str,
                 dim: int = 384, query_prefix: str = "", threads: int = 4):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, options,
                                           providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.dim = dim
        self.query_prefix = query_prefix
        self._input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        encoded = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=512, return_tensors="np")
        feeds = {k: v for k, v in encoded.items() if k in self._input_names}
        outputs = self.session.run(None, feeds)
        # bge pools with CLS, which is position 0 of the token embeddings.
        hidden = outputs[0]
        return _l2_normalize(np.asarray(hidden[:, 0], dtype=np.float32))

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([self.query_prefix + text])[0]


class HashEncoder:
    """Deterministic hashing encoder. No weights, no network, no torch.

    Not semantic, so it must never be used for accuracy claims. It exists so the
    index, retriever, harness and API can all be built and latency-tested before
    real weights are available, and so CI can run without downloading anything.
    """

    def __init__(self, dim: int = 384, query_prefix: str = ""):
        self.dim = dim
        self.query_prefix = query_prefix

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[slot] += sign
        return vector

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        return _l2_normalize(np.stack([self._vector(t) for t in texts]))

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([self.query_prefix + text])[0]

"""Adapter for rag-local-eval-loop: real ragoa embeddings."""

from __future__ import annotations

import threading

import numpy as np

from ragoa.config import settings
from ragoa.factory import load_encoder, read_manifest

_encoder = None
_dim = 384
_init_lock = threading.Lock()


def get_model():
    global _encoder, _dim
    if _encoder is not None:
        return _encoder
    with _init_lock:
        if _encoder is None:
            manifest = read_manifest(settings.index_dir)
            _dim = manifest["embed_dim"]
            kind = "onnx" if settings.prefer_onnx else "st"
            _encoder = load_encoder(kind, _dim, settings)
    return _encoder


def embed_one(text: str) -> np.ndarray:
    encoder = get_model()
    if hasattr(encoder, "encode_query"):
        return np.asarray(encoder.encode_query(text), dtype=np.float32)
    return np.asarray(encoder.encode([text])[0], dtype=np.float32)


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, _dim), dtype=np.float32)
    return np.asarray(get_model().encode(texts), dtype=np.float32)

"""Semantic chunking: cut where the topic actually changes.

Embed each sentence, measure cosine distance between consecutive sentences, and
place a boundary wherever that distance exceeds a percentile threshold of the
document's own distribution. Using a per-document percentile rather than a fixed
distance matters: a tightly-focused document has uniformly low distances, and an
absolute threshold would either never cut it or shred a diverse one.

This is the most expensive strategy to index, since it needs one embedding per
sentence at build time on top of one per chunk. The chunking eval reports that
cost alongside the accuracy so the trade is visible.

Our pseudo-documents are concatenations of ~10 independently-retrieved passages,
so real topic shifts exist inside them, which is exactly what this should find.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ragoa.chunking.base import Chunker, split_sentences
from ragoa.schemas import Chunk, ChunkingStrategy, PseudoDocument


class Encoder(Protocol):
    """Minimal encoder interface, so this module never imports torch."""

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        ...


class SemanticChunker(Chunker):
    strategy = ChunkingStrategy.SEMANTIC

    def __init__(
        self,
        encoder: Encoder,
        percentile: float = 80.0,
        min_chars: int = 300,
        max_chars: int = 2400,
        buffer: int = 1,
    ):
        self.encoder = encoder
        self.percentile = percentile
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.buffer = buffer

    @property
    def name(self) -> str:
        return f"semantic[p{int(self.percentile)},{self.min_chars}-{self.max_chars}c]"

    def _windowed(self, texts: list[str]) -> list[str]:
        """Widen each sentence with its neighbours to stabilise short-sentence vectors."""
        if self.buffer <= 0:
            return texts
        out: list[str] = []
        for i in range(len(texts)):
            lo = max(0, i - self.buffer)
            hi = min(len(texts), i + self.buffer + 1)
            out.append(" ".join(texts[lo:hi]))
        return out

    def chunk(self, doc: PseudoDocument) -> list[Chunk]:
        sentences = split_sentences(doc.text)
        if len(sentences) < 3:
            return self._build(doc, [(0, len(doc.text))])

        texts = [doc.text[s:e] for s, e in sentences]
        vectors = np.asarray(
            self.encoder.encode(self._windowed(texts), normalize_embeddings=True),
            dtype=np.float32,
        )

        # Normalized vectors, so cosine distance is 1 - dot.
        distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)
        threshold = float(np.percentile(distances, self.percentile))

        spans: list[tuple[int, int]] = []
        start = sentences[0][0]

        for i, distance in enumerate(distances):
            end = sentences[i][1]
            width = end - start
            # Cut on a semantic break once the chunk is big enough to be useful,
            # or force a cut when it has grown past the ceiling.
            if (distance >= threshold and width >= self.min_chars) or width >= self.max_chars:
                spans.append((start, end))
                start = sentences[i + 1][0]

        spans.append((start, sentences[-1][1]))
        return self._build(doc, spans)

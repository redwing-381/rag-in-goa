"""Fixed-size character windows with configurable overlap.

The declared baseline. It exists to be beaten: the whole point of the chunking
lab is to show what a naive fixed split costs in Recall and MRR, so we report it
at several overlap ratios rather than pretending it does not work at all.
"""

from __future__ import annotations

from ragoa.chunking.base import Chunker
from ragoa.schemas import Chunk, ChunkingStrategy, PseudoDocument


class FixedChunker(Chunker):
    strategy = ChunkingStrategy.FIXED

    def __init__(self, size_chars: int = 1800, overlap_ratio: float = 0.0):
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError("overlap_ratio must be in [0, 1)")
        self.size_chars = size_chars
        self.overlap_ratio = overlap_ratio
        self.stride = max(1, int(size_chars * (1.0 - overlap_ratio)))

    @property
    def name(self) -> str:
        return f"fixed[{self.size_chars}c,ov={int(self.overlap_ratio * 100)}%]"

    def chunk(self, doc: PseudoDocument) -> list[Chunk]:
        spans: list[tuple[int, int]] = []
        start = 0
        length = len(doc.text)

        while start < length:
            end = min(start + self.size_chars, length)
            spans.append((start, end))
            if end == length:
                break
            start += self.stride

        return self._build(doc, spans)

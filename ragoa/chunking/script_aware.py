"""Recursive splitting that respects script-specific sentence boundaries.

A character-count splitter mangles Indic text: it will cut mid-word in
Devanagari, and it never recognises the danda (U+0964) as a sentence end, so it
treats a whole paragraph as one unbroken run. This strategy walks a separator
hierarchy from strongest boundary to weakest, only falling back to a hard cut
when a single fragment still exceeds the budget.

Our indexed corpus is English, so this mainly buys clean sentence boundaries
today. It is written script-aware because the same chunker runs over the Indic
side for the native-retrieval comparison, and because the danda separators cost
nothing to include.
"""

from __future__ import annotations

from ragoa.chunking.base import RECURSIVE_SEPARATORS, Chunker
from ragoa.schemas import Chunk, ChunkingStrategy, PseudoDocument


class ScriptAwareChunker(Chunker):
    strategy = ChunkingStrategy.SCRIPT_AWARE

    def __init__(self, size_chars: int = 1800, overlap_chars: int = 200,
                 separators: tuple[str, ...] = RECURSIVE_SEPARATORS):
        self.size_chars = size_chars
        self.overlap_chars = overlap_chars
        self.separators = separators

    @property
    def name(self) -> str:
        return f"script_aware[{self.size_chars}c,ov={self.overlap_chars}c]"

    def _split(self, text: str, base: int, depth: int) -> list[tuple[int, int]]:
        """Recursively split [base, base+len(text)) using separators[depth:]."""
        if len(text) <= self.size_chars:
            return [(base, base + len(text))]

        if depth >= len(self.separators):
            # Out of separators: hard-cut, which is the honest fallback.
            return [
                (base + i, base + min(i + self.size_chars, len(text)))
                for i in range(0, len(text), self.size_chars)
            ]

        sep = self.separators[depth]
        pieces = text.split(sep)
        if len(pieces) == 1:
            return self._split(text, base, depth + 1)

        # Greedily pack pieces up to the budget, recursing into anything still too big.
        spans: list[tuple[int, int]] = []
        buf_start = 0
        buf_end = 0
        cursor = 0

        for piece in pieces:
            piece_start = cursor
            piece_end = cursor + len(piece)
            cursor = piece_end + len(sep)

            if piece_end - buf_start > self.size_chars and buf_end > buf_start:
                spans.append((base + buf_start, base + buf_end))
                buf_start = piece_start
            elif buf_end == buf_start:
                buf_start = piece_start

            buf_end = piece_end

            if piece_end - piece_start > self.size_chars:
                # A single piece overflows: split it deeper, and flush what we held.
                if buf_start < piece_start:
                    spans.append((base + buf_start, base + piece_start))
                spans.extend(self._split(piece, base + piece_start, depth + 1))
                buf_start = buf_end = cursor

        if buf_end > buf_start:
            spans.append((base + buf_start, base + buf_end))

        return spans

    def _apply_overlap(self, spans: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
        """Extend each chunk backwards so answers straddling a boundary survive."""
        if self.overlap_chars <= 0:
            return spans
        return [
            (max(0, start - self.overlap_chars) if i > 0 else start, min(end, limit))
            for i, (start, end) in enumerate(spans)
        ]

    def chunk(self, doc: PseudoDocument) -> list[Chunk]:
        spans = self._split(doc.text, 0, 0)
        spans = [(s, e) for s, e in spans if e > s]
        spans = self._apply_overlap(spans, len(doc.text))
        return self._build(doc, spans)

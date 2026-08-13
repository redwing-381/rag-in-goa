"""Chunker protocol and shared text-splitting utilities.

Sizes are expressed in **characters**, not tokens, on purpose. A tokenizer call
per chunk is real indexing cost for no benefit here: the eval measures overlap
with gold character spans, and ~4 chars/token is a stable ratio on this corpus
(measured avg 3,156 chars / ~789 tokens per document). `estimate_tokens` exists
for reporting only.

Every chunk must carry accurate `char_start` / `char_end` offsets into its parent
document, because that is how `Chunk.overlaps` decides a retrieval hit against
the gold spans. A strategy that loses offsets cannot be evaluated.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from ragoa.schemas import Chunk, ChunkingStrategy, PseudoDocument

CHARS_PER_TOKEN = 4.0

# Sentence terminators across the scripts we handle: Latin, plus Devanagari and
# Bengali danda (U+0964) and double danda (U+0965), plus the Urdu full stop
# (U+06D4). Tamil and Marathi use the danda or Latin punctuation.
SENTENCE_END = r"[.!?\u0964\u0965\u06D4]"

# Do not split after a single capital (initials) or common abbreviations.
_ABBREV = r"(?<!\b[A-Z])(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bSt)(?<!\bvs)(?<!\betc)(?<!\bNo)"

_SENTENCE_RE = re.compile(rf"{_ABBREV}{SENTENCE_END}+[\s\u200b]+")

# Ordered separators for recursive splitting: strongest boundary first.
RECURSIVE_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    "\u0964 ",  # danda + space (Devanagari, Bengali, Marathi)
    "\u0965 ",  # double danda
    "\u06D4 ",  # Urdu full stop
    ". ",
    "! ",
    "? ",
    "; ",
    ": ",
    ", ",
    " ",
)


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def split_sentences(text: str, min_chars: int = 24) -> list[tuple[int, int]]:
    """Split into sentence spans as (start, end) offsets into `text`.

    Returns offsets rather than strings so callers can keep exact provenance.
    Fragments shorter than `min_chars` are merged forward, which stops a stray
    "Inc." or a lone number from becoming its own chunk.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0

    for match in _SENTENCE_RE.finditer(text):
        end = match.start() + 1  # keep the terminator with the sentence
        if end > cursor:
            spans.append((cursor, end))
        cursor = match.end()

    if cursor < len(text):
        spans.append((cursor, len(text)))

    # Merge runts forward so every span is a usable unit.
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and (end - start) < min_chars:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged or [(0, len(text))]


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink a span to exclude leading/trailing whitespace, preserving offsets."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


class Chunker(ABC):
    """Turns one pseudo-document into retrievable chunks.

    Implementations must be deterministic: the chunking eval compares strategies
    across runs, so identical input has to produce identical chunk boundaries.
    """

    strategy: ChunkingStrategy

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier including hyperparameters, used as the report label."""

    @abstractmethod
    def chunk(self, doc: PseudoDocument) -> list[Chunk]:
        ...

    def _build(
        self,
        doc: PseudoDocument,
        spans: list[tuple[int, int]],
        *,
        text_override: dict[int, str] | None = None,
    ) -> list[Chunk]:
        """Materialise Chunk objects from spans, dropping empties."""
        chunks: list[Chunk] = []
        for i, (start, end) in enumerate(spans):
            start, end = trim_span(doc.text, start, end)
            if end <= start:
                continue
            body = (text_override or {}).get(i) or doc.text[start:end]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{self.strategy.value}#{i}",
                    doc_id=doc.doc_id,
                    text=body,
                    char_start=start,
                    char_end=end,
                    strategy=self.strategy,
                    query_type=doc.query_type,
                    n_tokens=estimate_tokens(body),
                )
            )
        return chunks

    def chunk_many(self, docs: list[PseudoDocument]) -> list[Chunk]:
        out: list[Chunk] = []
        for doc in docs:
            out.extend(self.chunk(doc))
        return out

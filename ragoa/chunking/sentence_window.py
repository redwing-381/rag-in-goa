"""Sentence-window (small-to-big) chunking.

Index a small unit for retrieval precision, but hand the LLM a wider window for
context: the embedded text is a group of `group` sentences, while
`expand_start`/`expand_end` cover `window` sentences either side.

This decouples the two jobs a chunk has to do. A 450-token chunk generates well
and embeds poorly, because one vector has to stand for several claims. A single
sentence embeds sharply but reads as a fragment. Indexing small and expanding at
answer time gets both, at the cost of more vectors.
"""

from __future__ import annotations

from ragoa.chunking.base import Chunker, split_sentences
from ragoa.schemas import Chunk, ChunkingStrategy, PseudoDocument


class SentenceWindowChunker(Chunker):
    strategy = ChunkingStrategy.SENTENCE_WINDOW

    def __init__(self, group: int = 2, window: int = 3):
        self.group = max(1, group)
        self.window = max(0, window)

    @property
    def name(self) -> str:
        return f"sentence_window[group={self.group},win={self.window}]"

    def chunk(self, doc: PseudoDocument) -> list[Chunk]:
        sentences = split_sentences(doc.text)

        # Pair each chunk span with its window span, then drop blanks *before*
        # building, so the two lists stay index-aligned with the built chunks.
        pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for i in range(0, len(sentences), self.group):
            group_end = min(i + self.group, len(sentences))
            span = (sentences[i][0], sentences[group_end - 1][1])
            if not doc.text[span[0]:span[1]].strip():
                continue
            lo = max(0, i - self.window)
            hi = min(len(sentences), group_end + self.window)
            pairs.append((span, (sentences[lo][0], sentences[hi - 1][1])))

        chunks = self._build(doc, [span for span, _ in pairs])
        for chunk, (_, window) in zip(chunks, pairs, strict=True):
            chunk.expand_start, chunk.expand_end = window

        return chunks

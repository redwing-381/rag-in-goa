"""Parent-child hierarchical chunking.

Children are indexed; the parent is what the LLM reads. Unlike sentence-window,
the parent boundary is a fixed budget rather than a sentence count, so context
size is predictable regardless of how long the sentences are.

Because the whole pseudo-document averages ~3,156 chars, a 1,800-char parent is
usually one or two per document. That makes this strategy a direct test of the
"greater chunk size, small K" idea: retrieve on precise children, then generate
from a large parent, so a small K still yields plenty of context.
"""

from __future__ import annotations

from ragoa.chunking.base import Chunker, split_sentences
from ragoa.schemas import Chunk, ChunkingStrategy, PseudoDocument


class HierarchicalChunker(Chunker):
    strategy = ChunkingStrategy.HIERARCHICAL

    def __init__(self, parent_chars: int = 1800, child_chars: int = 400):
        if child_chars >= parent_chars:
            raise ValueError("child_chars must be smaller than parent_chars")
        self.parent_chars = parent_chars
        self.child_chars = child_chars

    @property
    def name(self) -> str:
        return f"hierarchical[parent={self.parent_chars}c,child={self.child_chars}c]"

    def _pack(self, spans: list[tuple[int, int]], budget: int) -> list[tuple[int, int]]:
        """Greedily merge consecutive spans up to a character budget."""
        packed: list[tuple[int, int]] = []
        for start, end in spans:
            if packed and end - packed[-1][0] <= budget:
                packed[-1] = (packed[-1][0], end)
            else:
                packed.append((start, end))
        return packed

    def chunk(self, doc: PseudoDocument) -> list[Chunk]:
        sentences = split_sentences(doc.text)
        parents = self._pack(sentences, self.parent_chars)
        children = self._pack(sentences, self.child_chars)

        pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for child in children:
            if not doc.text[child[0]:child[1]].strip():
                continue
            # The parent containing this child's start offset.
            parent = next(
                (p for p in parents if p[0] <= child[0] < p[1]),
                (child[0], min(child[0] + self.parent_chars, len(doc.text))),
            )
            pairs.append((child, parent))

        chunks = self._build(doc, [child for child, _ in pairs])
        for chunk, (_, parent) in zip(chunks, pairs, strict=True):
            chunk.expand_start, chunk.expand_end = parent

        return chunks

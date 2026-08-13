"""Metadata-aware chunking: wraps another strategy and enriches what gets embedded.

Two things happen here. The embedded text is prefixed with the chunk's
`query_type` (MS MARCO's own taxonomy: DESCRIPTION, NUMERIC, ENTITY, LOCATION,
PERSON), which pulls numeric-bearing chunks closer to numeric questions in
embedding space. And `query_type` stays on the Chunk as a filterable field, so
the retriever can restrict the candidate set instead of relying on the vector
alone.

The prefix only affects the embedded string. `char_start` / `char_end` still
point at the untouched document, so gold-span evaluation stays exact and the
citation shown to the user has no synthetic text in it.
"""

from __future__ import annotations

from ragoa.chunking.base import Chunker
from ragoa.schemas import Chunk, ChunkingStrategy, PseudoDocument


class MetadataAwareChunker(Chunker):
    strategy = ChunkingStrategy.METADATA_AWARE

    def __init__(self, base: Chunker, include_query_type: bool = True,
                 include_title_hint: bool = True):
        self.base = base
        self.include_query_type = include_query_type
        self.include_title_hint = include_title_hint

    @property
    def name(self) -> str:
        flags = []
        if self.include_query_type:
            flags.append("qtype")
        if self.include_title_hint:
            flags.append("hint")
        return f"metadata_aware[{self.base.name}+{'+'.join(flags) or 'none'}]"

    def _prefix(self, doc: PseudoDocument) -> str:
        parts: list[str] = []
        if self.include_query_type:
            parts.append(doc.query_type.value.lower())
        if self.include_title_hint:
            # First few words of the document act as a cheap topic label.
            head = " ".join(doc.text.split()[:8])
            if head:
                parts.append(head)
        return " | ".join(parts)

    def chunk(self, doc: PseudoDocument) -> list[Chunk]:
        chunks = self.base.chunk(doc)
        prefix = self._prefix(doc)

        for chunk in chunks:
            # Re-key so ids do not collide with the wrapped strategy's own output.
            chunk.chunk_id = chunk.chunk_id.replace(
                f"#{self.base.strategy.value}#", f"#{self.strategy.value}#"
            )
            chunk.strategy = self.strategy
            if prefix:
                chunk.text = f"[{prefix}] {chunk.text}"

        return chunks

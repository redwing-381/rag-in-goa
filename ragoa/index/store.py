"""On-disk chunk store: keeps 300+ MB of text out of RAM.

Chunk text is never stored. Every chunk is a byte range into one flat file of
concatenated document text, which we memory-map, so slicing a chunk costs a page
touch instead of a Python string allocation. On an 8 GB machine that is the
difference between ~325 MB of resident strings and ~6 MB of offset arrays.

Chunk offsets arrive as *character* indices (that is what the chunkers produce
and what gold-span evaluation needs), but the file is UTF-8 bytes. `_byte_map`
converts once at build time; storing both means retrieval slices by byte and
evaluation still compares by character.

Layout in `data/index/`:
    docs.bin    concatenated UTF-8 document text
    chunks.npz  offset arrays + chunk ids + doc ids
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from ragoa.schemas import Chunk, ChunkingStrategy, PseudoDocument, QueryType

QUERY_TYPES = list(QueryType)
_QT_TO_CODE = {qt: i for i, qt in enumerate(QUERY_TYPES)}


def _byte_map(text: str, positions: Iterable[int]) -> dict[int, int]:
    """Map character positions to byte offsets in one left-to-right pass.

    Encoding a prefix per lookup would be O(n) per chunk; this is O(n) per
    document regardless of how many chunks it produced.
    """
    wanted = sorted(set(positions))
    out: dict[int, int] = {}
    byte_pos = 0
    char_pos = 0
    idx = 0

    for char_pos, ch in enumerate(text):
        while idx < len(wanted) and wanted[idx] == char_pos:
            out[wanted[idx]] = byte_pos
            idx += 1
        byte_pos += len(ch.encode("utf-8"))

    # Any remaining positions are at or past the end of the string.
    while idx < len(wanted):
        out[wanted[idx]] = byte_pos
        idx += 1
    return out


class ChunkStoreBuilder:
    """Streams documents to disk. Never holds the corpus in memory."""

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.index_dir / "docs.bin", "wb")
        self._doc_byte_base = 0

        self.chunk_ids: list[str] = []
        self.doc_ids: list[str] = []
        self.byte_start: list[int] = []
        self.byte_end: list[int] = []
        self.ctx_byte_start: list[int] = []
        self.ctx_byte_end: list[int] = []
        self.char_start: list[int] = []
        self.char_end: list[int] = []
        self.query_type: list[int] = []
        self.strategy: str | None = None

    def add(self, doc: PseudoDocument, chunks: list[Chunk]) -> None:
        encoded = doc.text.encode("utf-8")
        self._fh.write(encoded)
        base = self._doc_byte_base
        self._doc_byte_base += len(encoded)

        if not chunks:
            return
        if self.strategy is None:
            self.strategy = chunks[0].strategy.value

        positions: set[int] = set()
        for c in chunks:
            ctx_start, ctx_end = c.context_span()
            positions.update((c.char_start, c.char_end, ctx_start, ctx_end))
        mapping = _byte_map(doc.text, positions)

        for c in chunks:
            ctx_start, ctx_end = c.context_span()
            self.chunk_ids.append(c.chunk_id)
            self.doc_ids.append(c.doc_id)
            self.byte_start.append(base + mapping[c.char_start])
            self.byte_end.append(base + mapping[c.char_end])
            self.ctx_byte_start.append(base + mapping[ctx_start])
            self.ctx_byte_end.append(base + mapping[ctx_end])
            self.char_start.append(c.char_start)
            self.char_end.append(c.char_end)
            self.query_type.append(_QT_TO_CODE.get(c.query_type, 0) if c.query_type else 0)

    def close(self) -> int:
        self._fh.close()
        np.savez(
            self.index_dir / "chunks.npz",
            byte_start=np.asarray(self.byte_start, dtype=np.int64),
            byte_end=np.asarray(self.byte_end, dtype=np.int64),
            ctx_byte_start=np.asarray(self.ctx_byte_start, dtype=np.int64),
            ctx_byte_end=np.asarray(self.ctx_byte_end, dtype=np.int64),
            char_start=np.asarray(self.char_start, dtype=np.int32),
            char_end=np.asarray(self.char_end, dtype=np.int32),
            query_type=np.asarray(self.query_type, dtype=np.int8),
        )
        (self.index_dir / "chunk_ids.json").write_text(
            json.dumps({"strategy": self.strategy,
                        "chunk_ids": self.chunk_ids,
                        "doc_ids": self.doc_ids})
        )
        return len(self.chunk_ids)


class ChunkStore:
    """Read side. Memory-maps `docs.bin`; text is served from the page cache."""

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        arrays = np.load(index_dir / "chunks.npz")
        self.byte_start = arrays["byte_start"]
        self.byte_end = arrays["byte_end"]
        self.ctx_byte_start = arrays["ctx_byte_start"]
        self.ctx_byte_end = arrays["ctx_byte_end"]
        self.char_start = arrays["char_start"]
        self.char_end = arrays["char_end"]
        self.query_type = arrays["query_type"]

        meta = json.loads((index_dir / "chunk_ids.json").read_text())
        self.strategy = ChunkingStrategy(meta["strategy"])
        self.chunk_ids: list[str] = meta["chunk_ids"]
        self.doc_ids: list[str] = meta["doc_ids"]
        self._by_id = {cid: i for i, cid in enumerate(self.chunk_ids)}

        self._mm = np.memmap(index_dir / "docs.bin", dtype=np.uint8, mode="r")

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def _slice(self, start: int, end: int) -> str:
        return self._mm[start:end].tobytes().decode("utf-8", errors="replace")

    def text(self, i: int) -> str:
        """The embedded/indexed span."""
        return self._slice(self.byte_start[i], self.byte_end[i])

    def context(self, i: int) -> str:
        """The wider span handed to the LLM (parent or sentence window)."""
        return self._slice(self.ctx_byte_start[i], self.ctx_byte_end[i])

    def index_of(self, chunk_id: str) -> int | None:
        return self._by_id.get(chunk_id)

    def to_chunk(self, i: int) -> Chunk:
        return Chunk(
            chunk_id=self.chunk_ids[i],
            doc_id=self.doc_ids[i],
            text=self.text(i),
            char_start=int(self.char_start[i]),
            char_end=int(self.char_end[i]),
            strategy=self.strategy,
            query_type=QUERY_TYPES[int(self.query_type[i])],
        )

    def iter_texts(self, batch_size: int = 512) -> Iterator[list[str]]:
        """Batched text for index building, so nothing materialises at once."""
        batch: list[str] = []
        for i in range(len(self)):
            batch.append(self.text(i))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def resident_mb(self) -> float:
        """Offset arrays only. `docs.bin` stays in the page cache, not the heap."""
        arrays = (self.byte_start, self.byte_end, self.ctx_byte_start,
                  self.ctx_byte_end, self.char_start, self.char_end, self.query_type)
        return sum(a.nbytes for a in arrays) / 1e6

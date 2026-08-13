"""The strategy sweep. One place that defines what the chunking lab compares.

`default_chunker()` is what the served pipeline uses. It stays a single call so
the shipped configuration and the benchmarked configuration cannot drift apart:
whatever wins in docs/CHUNKING.md is changed here and nowhere else.
"""

from __future__ import annotations

from ragoa.chunking.base import Chunker
from ragoa.chunking.fixed import FixedChunker
from ragoa.chunking.hierarchical import HierarchicalChunker
from ragoa.chunking.metadata_aware import MetadataAwareChunker
from ragoa.chunking.script_aware import ScriptAwareChunker
from ragoa.chunking.semantic import Encoder, SemanticChunker
from ragoa.chunking.sentence_window import SentenceWindowChunker


# Larger chunks with a small K and a strong reranker is the shipped shape, so the
# sweep centres on ~1800 chars (~450 tokens) and brackets it either side.
def deterministic_sweep() -> list[Chunker]:
    """Every strategy that needs no encoder. Safe to run anywhere."""
    return [
        # Baseline, with the overlap sweep the brief explicitly asks about.
        FixedChunker(size_chars=1800, overlap_ratio=0.0),
        FixedChunker(size_chars=1800, overlap_ratio=0.10),
        FixedChunker(size_chars=1800, overlap_ratio=0.25),
        FixedChunker(size_chars=900, overlap_ratio=0.10),
        FixedChunker(size_chars=2400, overlap_ratio=0.10),
        # Script-aware recursive, at the same sizes for a fair comparison.
        ScriptAwareChunker(size_chars=1800, overlap_chars=0),
        ScriptAwareChunker(size_chars=1800, overlap_chars=200),
        ScriptAwareChunker(size_chars=900, overlap_chars=150),
        # Small-to-big.
        SentenceWindowChunker(group=1, window=3),
        SentenceWindowChunker(group=2, window=3),
        HierarchicalChunker(parent_chars=1800, child_chars=400),
        HierarchicalChunker(parent_chars=2400, child_chars=600),
        # Metadata-aware, wrapping the strongest deterministic base.
        MetadataAwareChunker(ScriptAwareChunker(size_chars=1800, overlap_chars=200)),
    ]


def encoder_sweep(encoder: Encoder) -> list[Chunker]:
    """Strategies that need an encoder at index time."""
    return [
        SemanticChunker(encoder, percentile=70.0),
        SemanticChunker(encoder, percentile=80.0),
        SemanticChunker(encoder, percentile=90.0),
    ]


def full_sweep(encoder: Encoder | None = None) -> list[Chunker]:
    chunkers = deterministic_sweep()
    if encoder is not None:
        chunkers.extend(encoder_sweep(encoder))
    return chunkers


def default_chunker() -> Chunker:
    """The shipped strategy. Update only from a chunking-eval result."""
    return ScriptAwareChunker(size_chars=1800, overlap_chars=200)

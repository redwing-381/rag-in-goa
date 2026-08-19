"""Lexical BM25 leg, via bm25s.

Why bother when we have a dense index and a reranker: dense embeddings blur rare
strings. MS MARCO validation is full of ENTITY, NUMERIC and LOCATION queries
("struthers city school district state number") where the answer hinges on an
exact token that a 384-dim vector smears away. Lexical retrieval catches those,
and RRF lets the two legs disagree without either dominating.

Memory is the live risk on an 8 GB machine: tokenizing ~211k chunks produces tens
of millions of token ids, and bm25s holds them as Python lists during the build.
`build` therefore reports peak RSS so the cost is measured rather than assumed,
and `load(mmap=True)` keeps the serving footprint small.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ragoa.telemetry.rss import peak_rss_mb as _peak_rss_mb


class SparseIndex:
    def __init__(self, stemmer: bool = True):
        self.use_stemmer = stemmer
        self._bm25 = None
        self._stemmer = None
        self._size = 0

    def _get_stemmer(self):
        if not self.use_stemmer:
            return None
        if self._stemmer is None:
            import Stemmer

            self._stemmer = Stemmer.Stemmer("english")
        return self._stemmer

    def _tokenize(self, texts: list[str], show_progress: bool = False):
        import bm25s

        return bm25s.tokenize(texts, stopwords="en", stemmer=self._get_stemmer(),
                              show_progress=show_progress)

    def build(self, texts: list[str], show_progress: bool = False) -> dict[str, float]:
        import bm25s

        before = _peak_rss_mb()
        tokens = self._tokenize(texts, show_progress)
        after_tokenize = _peak_rss_mb()

        self._bm25 = bm25s.BM25()
        self._bm25.index(tokens, show_progress=show_progress)
        self._size = len(texts)

        return {
            "rss_before_mb": before,
            "rss_after_tokenize_mb": after_tokenize,
            "rss_after_index_mb": _peak_rss_mb(),
            "n_docs": float(self._size),
        }

    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (labels, scores). Raw BM25 scores are unbounded, so callers
        should fuse by rank rather than by score."""
        if self._bm25 is None:
            raise RuntimeError("index not built or loaded")
        tokens = self._tokenize([query])
        labels, scores = self._bm25.retrieve(tokens, k=min(k, self._size),
                                             show_progress=False)
        return np.asarray(labels[0]), np.asarray(scores[0], dtype=np.float32)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._bm25.save(str(directory))

    def load(self, directory: Path, size: int, mmap: bool = True) -> None:
        import bm25s

        self._bm25 = bm25s.BM25.load(str(directory), mmap=mmap)
        self._size = size

    def __len__(self) -> int:
        return self._size

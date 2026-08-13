"""Dense HNSW index, built by streaming so the corpus never lands in RAM at once.

hnswlib keeps its own copy of the vectors, so the resident cost is the honest one:
~324 MB for 211k chunks at 384-dim float32, plus ~27 MB of graph links. We add
vectors batch by batch and never hold a full matrix ourselves, which keeps build
peak close to that floor instead of doubling it.

`ef_search` is the accuracy/latency dial at query time and can be lowered by the
deadline logic without rebuilding anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class DenseIndex:
    def __init__(self, dim: int, m: int = 16, ef_construction: int = 200,
                 ef_search: int = 64):
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._index = None
        self._size = 0

    def _new_index(self, capacity: int):
        import hnswlib

        index = hnswlib.Index(space="cosine", dim=self.dim)
        index.init_index(max_elements=capacity, M=self.m,
                         ef_construction=self.ef_construction)
        index.set_ef(self.ef_search)
        return index

    def build(self, batches, capacity: int, encoder, batch_size: int = 32,
              progress_every: int = 20_000) -> int:
        """Encode and insert batches of text. `batches` yields lists of strings."""
        self._index = self._new_index(capacity)
        inserted = 0

        for texts in batches:
            vectors = encoder.encode(texts, batch_size=batch_size)
            labels = np.arange(inserted, inserted + len(texts), dtype=np.int64)
            self._index.add_items(vectors, labels)
            inserted += len(texts)
            if progress_every and inserted % progress_every < len(texts):
                print(f"  dense: {inserted:,}/{capacity:,} vectors", flush=True)

        self._size = inserted
        return inserted

    def search(self, vector: np.ndarray, k: int, ef: int | None = None):
        """Return (labels, similarities). hnswlib gives cosine *distance*."""
        if self._index is None:
            raise RuntimeError("index not built or loaded")
        if ef is not None:
            self._index.set_ef(max(ef, k))

        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        labels, distances = self._index.knn_query(query, k=min(k, self._size))
        return labels[0], 1.0 - distances[0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._index.save_index(str(path))

    def load(self, path: Path, size: int) -> None:
        import hnswlib

        self._index = hnswlib.Index(space="cosine", dim=self.dim)
        self._index.load_index(str(path), max_elements=size)
        self._index.set_ef(self.ef_search)
        self._size = size

    def __len__(self) -> int:
        return self._size

    def resident_mb(self) -> float:
        vectors = self._size * self.dim * 4 / 1e6
        links = self._size * self.m * 4 * 2 / 1e6
        return vectors + links

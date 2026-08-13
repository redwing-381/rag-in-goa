"""Cross-encoder reranking, kept on a short leash.

This is the one compute-bound stage that can eat the whole 200ms budget on CPU: a
cross-encoder scores every (query, chunk) pair jointly, so cost scales with
candidates x tokens, not with corpus size. Three levers keep it affordable, all
exposed here rather than buried: how many candidates we score, how many tokens we
truncate to, and whether we run at all.

`ms-marco-MiniLM-L-6-v2` is chosen because it was trained on MS MARCO, which is
literally this corpus, so it is both the fastest and the best-matched option.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> np.ndarray:
        ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 256,
                 device: str | None = None, batch_size: int = 16):
        from sentence_transformers import CrossEncoder

        if device is None:
            device = "cpu"  # serving is CPU; MPS is for the offline build only
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        self.max_length = max_length
        self.batch_size = batch_size

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs, batch_size=self.batch_size,
                                    show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32)


class OnnxReranker:
    """ONNX Runtime cross-encoder: the only variant that fits the budget on CPU.

    Measured on an M2 CPU, the torch path needs ~177ms for 8 candidates at 256
    tokens, which overruns the whole 200ms pipeline ceiling on its own. Scoring
    happens in a single batched session run so the candidates share one pass.
    """

    def __init__(self, onnx_path: str, tokenizer_dir: str, max_length: int = 256,
                 threads: int = 4):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, options,
                                            providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.max_length = max_length
        self._input_names = {i.name for i in self.session.get_inputs()}

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)

        encoded = self.tokenizer(
            [query] * len(texts), texts,
            padding=True, truncation=True, max_length=self.max_length,
            return_tensors="np",
        )
        feeds = {k: v for k, v in encoded.items() if k in self._input_names}
        logits = self.session.run(None, feeds)[0]
        # This checkpoint has a single regression head, so the score is the lone
        # logit; squeezing keeps the output shape consistent with the torch path.
        return np.asarray(logits, dtype=np.float32).reshape(-1)


class NullReranker:
    """No-op, used when the deadline says there is no time for reranking."""

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        return np.zeros(len(texts), dtype=np.float32)

"""Central configuration. Reads .env, so no module hardcodes a key or a path.

Defaults are tuned for an 8 GB machine: text lives on disk and is memory-mapped,
vectors are the only thing we deliberately keep resident, and the reranker sees a
small candidate set because it is the one compute-bound stage that can eat the
whole latency budget on CPU.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- credentials -------------------------------------------------------
    openrouter_api_key: str = ""
    sarvam_api_key: str = ""

    # --- paths -------------------------------------------------------------
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    slim_dir: Path = Path("data/slim")
    corpus_dir: Path = Path("data/corpus")
    index_dir: Path = Path("data/index")
    onnx_dir: Path = Path("data/onnx")

    # --- models ------------------------------------------------------------
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    # bge asks for this prefix on queries but not on passages; omitting it costs
    # a few points of recall.
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embed_batch_size: int = 16  # measured best on MPS; 32+ pushed 8 GB into swap
    rerank_max_length: int = 256  # truncation is the cheapest rerank speedup
    # Serving prefers quantised ONNX. On CPU the torch reranker costs ~124ms p50
    # at 16 candidates, which does not leave room for the rest of the pipeline;
    # int8 does the same work in ~75ms at 12 candidates with identical top-1
    # picks. Falls back to torch automatically when the export is missing.
    prefer_onnx: bool = True
    onnx_threads: int = 4

    # --- retrieval ---------------------------------------------------------
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64
    dense_candidates: int = 30
    sparse_candidates: int = 30
    rerank_candidates: int = 12
    top_k: int = 3
    rrf_k: int = 60

    # --- latency budget ----------------------------------------------------
    # A reserve is "how much budget this stage needs to finish", so each one is
    # set from the measured p95 of the stage plus everything that must still run
    # after it. Setting a reserve below the stage's real cost is worse than having
    # no deadline at all: the check passes, the stage starts, and the budget is
    # blown anyway. Numbers from scripts/bench_rerank_backends.py on CPU.
    retrieval_budget_ms: float = 200.0
    # onnx-int8 rerank at 12 candidates / 256 tokens: p50 75ms, p95 115ms.
    rerank_reserve_ms: float = 120.0
    # Sparse leg plus the rerank that follows it, so skipping sparse is decided
    # with the rerank still affordable.
    sparse_reserve_ms: float = 140.0

    # --- guardrail thresholds (calibrated by bench/guardrail_eval.py) ------
    ood_score_threshold: float = 0.45
    groundedness_threshold: float = 0.35
    min_transcript_chars: int = 3
    min_language_probability: float = 0.30
    # Below this, a generated answer is treated as a failure rather than shown.
    # The extractive fallback used to return 0.2 and a random passage; that
    # looked like an answer and was worse than a named refusal.
    min_answer_confidence: float = 0.35

    # --- llm ---------------------------------------------------------------
    llm_model: str = "openai/gpt-oss-120b"
    llm_providers: list[str] = Field(default_factory=lambda: ["cerebras", "groq"])
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_max_tokens: int = 400
    llm_temperature: float = 0.1
    llm_timeout_s: float = 20.0
    llm_max_retries: int = 2

    # --- stt ---------------------------------------------------------------
    sarvam_stt_url: str = "https://api.sarvam.ai/speech-to-text"
    sarvam_stt_model: str = "saaras:v3"
    sarvam_stt_mode: str = "translate"  # Indic speech -> English text in one call
    sarvam_timeout_s: float = 20.0
    sarvam_tts_url: str = "https://api.sarvam.ai/text-to-speech"
    sarvam_tts_model: str = "bulbul:v3"
    # Ishita is Sarvam's recommended conversational female voice for en-IN / ta-IN.
    sarvam_tts_speaker: str = "ishita"
    sarvam_tts_pace: float = 0.9
    sarvam_tts_temperature: float = 0.6

    @property
    def docs_bin(self) -> Path:
        return self.index_dir / "docs.bin"

    @property
    def chunks_npz(self) -> Path:
        return self.index_dir / "chunks.npz"

    @property
    def hnsw_path(self) -> Path:
        return self.index_dir / "dense.hnsw"

    @property
    def bm25_dir(self) -> Path:
        return self.index_dir / "bm25"

    @property
    def onnx_embedder_dir(self) -> Path:
        return self.onnx_dir / "embedder"

    @property
    def onnx_reranker_dir(self) -> Path:
        return self.onnx_dir / "reranker"

    def onnx_model(self, which: str, quantized: bool = True) -> Path:
        """Path to an exported graph. `which` is "embedder" or "reranker"."""
        directory = self.onnx_dir / which
        return directory / ("model_int8.onnx" if quantized else "model.onnx")


settings = Settings()

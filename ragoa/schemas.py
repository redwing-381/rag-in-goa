"""Frozen contracts for the whole system.

Every module, benchmark and API route codes against these models so the three
workstreams (data/retrieval, serving/harness, voice/UI/guardrails) can be built
in parallel without waiting on each other. Changing a field here is a
cross-team decision, not a local one.
"""

from __future__ import annotations

import time
from enum import StrEnum

from pydantic import BaseModel, Field


class Language(StrEnum):
    """Languages we accept as speech input. Retrieval always happens in English."""

    EN = "en"
    HI = "hi"
    BN = "bn"
    TA = "ta"
    MR = "mr"


class QueryType(StrEnum):
    """MS MARCO's own query taxonomy, carried through as chunk metadata."""

    DESCRIPTION = "DESCRIPTION"
    NUMERIC = "NUMERIC"
    ENTITY = "ENTITY"
    LOCATION = "LOCATION"
    PERSON = "PERSON"


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    SCRIPT_AWARE = "script_aware"
    SEMANTIC = "semantic"
    SENTENCE_WINDOW = "sentence_window"
    HIERARCHICAL = "hierarchical"
    METADATA_AWARE = "metadata_aware"


class RefusalReason(StrEnum):
    """Why the system declined. Every refusal must name one of these."""

    UNSAFE_INPUT = "unsafe_input"
    PROMPT_INJECTION = "prompt_injection"
    UNINTELLIGIBLE_AUDIO = "unintelligible_audio"
    OUT_OF_DOMAIN = "out_of_domain"
    NOT_GROUNDED = "not_grounded"
    INVALID_CITATIONS = "invalid_citations"
    PROVIDER_FAILURE = "provider_failure"


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


class Passage(BaseModel):
    """One MS MARCO candidate passage, with its aligned translation.

    `text` is always English (what we embed and search). `translated` is the
    same passage in `lang`, used to cite back in the speaker's language.
    """

    passage_idx: int
    text: str
    translated: str | None = None
    is_selected: bool = False


class PseudoDocument(BaseModel):
    """The ~10 candidate passages of one query, concatenated.

    MS MARCO ships pre-chunked at ~60 words, which would make chunking a no-op
    variable. Reconstituting a topical document (~2.6 KB) is what lets us
    measure chunking strategies against each other. `gold_spans` are character
    offsets into `text` of the `is_selected` passages, and are the ground truth
    for retrieval eval.
    """

    doc_id: str
    query_id: int
    query_type: QueryType
    lang: Language
    text: str
    gold_spans: list[tuple[int, int]] = Field(default_factory=list)
    passage_offsets: list[tuple[int, int]] = Field(default_factory=list)
    eng_query: str
    translated_query: str | None = None
    eng_answer: str | None = None
    translated_answer: str | None = None


class Chunk(BaseModel):
    """A retrievable unit.

    `char_start` / `char_end` are what get embedded and what the gold-span
    overlap test uses. `expand_start` / `expand_end` are the wider span handed to
    the LLM for context, which is how small-to-big and hierarchical strategies
    index a sharp unit while generating from a readable one.
    """

    chunk_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int
    strategy: ChunkingStrategy
    query_type: QueryType | None = None
    n_tokens: int | None = None
    expand_start: int | None = None
    expand_end: int | None = None

    def overlaps(self, span: tuple[int, int]) -> bool:
        """True if this chunk covers any part of a gold span (the eval hit test)."""
        return self.char_start < span[1] and span[0] < self.char_end

    def context_span(self) -> tuple[int, int]:
        """Span to use when building the generation context."""
        start = self.expand_start if self.expand_start is not None else self.char_start
        end = self.expand_end if self.expand_end is not None else self.char_end
        return start, end


class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    expanded_text: str | None = None

    @property
    def score(self) -> float:
        """Best available score, most-trustworthy first."""
        for value in (self.rerank_score, self.fused_score, self.dense_score, self.sparse_score):
            if value is not None:
                return value
        return 0.0


# --------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------


class Span(BaseModel):
    name: str
    duration_ms: float
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class Trace(BaseModel):
    """Per-request timing, returned to the client and rendered in the UI.

    This is the same data that backs docs/LATENCY.md, so the demo and the
    benchmark report cannot drift apart.
    """

    request_id: str
    spans: list[Span] = Field(default_factory=list)
    degradations: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    retrieval_ms: float = 0.0
    llm_ttft_ms: float | None = None
    total_ms: float = 0.0
    cache_hit: bool = False

    def add(self, name: str, duration_ms: float, **metadata) -> None:
        self.spans.append(Span(name=name, duration_ms=duration_ms, metadata=metadata))


class Deadline:
    """Wall-clock budget threaded through the pipeline.

    Stages call `remaining_ms` / `exceeded` to decide whether to run optional
    work. This is what enforces the 200ms retrieval ceiling by dropping the
    reranker and then the sparse leg rather than by blowing the budget.
    """

    def __init__(self, budget_ms: float):
        self.budget_ms = budget_ms
        self.started = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000.0

    @property
    def remaining_ms(self) -> float:
        return self.budget_ms - self.elapsed_ms

    def exceeded(self, reserve_ms: float = 0.0) -> bool:
        return self.remaining_ms <= reserve_ms


# --------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------


class GuardrailVerdict(BaseModel):
    allowed: bool
    reason: RefusalReason | None = None
    detail: str | None = None
    score: float | None = None


class GroundednessReport(BaseModel):
    grounded: bool
    score: float
    unsupported_sentences: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


class STTResult(BaseModel):
    """Sarvam saaras:v3 output. With mode=translate, `text` is English."""

    text: str
    detected_language: str | None = None
    language_probability: float | None = None
    duration_ms: float | None = None


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    translated_text: str | None = None
    score: float


class AskRequest(BaseModel):
    query: str
    lang: Language = Language.EN
    top_k: int = Field(default=3, ge=1, le=20)
    candidates: int = Field(default=30, ge=1, le=200)
    budget_ms: float = Field(default=200.0, gt=0)
    strategy: ChunkingStrategy | None = None
    stream: bool = False


class AnswerPayload(BaseModel):
    """The LLM's schema-constrained output. Nothing else is accepted."""

    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    answer_language: Language = Language.EN
    refusal_reason: RefusalReason | None = None


class AskResponse(BaseModel):
    answer: str
    refused: bool = False
    refusal_reason: RefusalReason | None = None
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    transcript: str | None = None
    answer_language: Language = Language.EN
    groundedness: GroundednessReport | None = None
    trace: Trace

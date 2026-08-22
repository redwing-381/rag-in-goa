"""Adapter for rag-local-eval-loop: production guardrails + OpenRouter."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ragoa.config import settings
from ragoa.factory import build_pipeline
from ragoa.harness.answer_policy import (
    finalize_answer,
    refusal_message,
    top_retrieval_score,
)
from ragoa.schemas import Chunk, ChunkingStrategy, Language, RefusalReason, RetrievedChunk

_pipeline = None
_init_lock = threading.Lock()


@dataclass
class GeneratedAnswer:
    text: str
    grounded: bool
    generation_ms: float
    model: str


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _init_lock:
        if _pipeline is None:
            _pipeline, _ = build_pipeline(use_stt=False)
    return _pipeline


def _to_retrieved(results) -> list[RetrievedChunk]:
    retrieved: list[RetrievedChunk] = []
    for i, ctx in enumerate(results):
        score = float(getattr(ctx, "score", 1.0))
        chunk = Chunk(
            chunk_id=f"eval#{i}",
            doc_id=getattr(ctx, "source", f"eval-{i}"),
            text=ctx.text,
            char_start=0,
            char_end=len(ctx.text),
            strategy=ChunkingStrategy.SCRIPT_AWARE,
        )
        retrieved.append(
            RetrievedChunk(
                chunk=chunk,
                dense_score=score,
                fused_score=score,
                expanded_text=ctx.text,
            )
        )
    return retrieved


def _decline(reason: RefusalReason, model: str) -> GeneratedAnswer:
    return GeneratedAnswer(
        text=refusal_message(reason, Language.EN),
        grounded=False,
        generation_ms=0.0,
        model=model,
    )


def generate_answer(query: str, results) -> GeneratedAnswer:
    pipeline = _get_pipeline()
    model = settings.llm_model

    retrieved = _to_retrieved(results)
    if not retrieved:
        return _decline(RefusalReason.NOT_GROUNDED, model)

    domain = pipeline.input_gate.check_domain(top_retrieval_score(retrieved))
    if not domain.allowed:
        return _decline(RefusalReason.OUT_OF_DOMAIN, model)

    payload, ms = pipeline.llm.answer(query, retrieved, Language.EN)
    response = finalize_answer(
        payload,
        retrieved,
        Language.EN,
        settings=pipeline.settings,
        input_gate=pipeline.input_gate,
        output_gate=pipeline.output_gate,
        check_domain=False,
    )
    return GeneratedAnswer(
        text=response.answer,
        grounded=not response.refused,
        generation_ms=ms,
        model=model,
    )

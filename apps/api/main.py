"""FastAPI service.

The index and models load once at startup and stay warm, because the 200ms target
is a steady-state claim: a request that pays lazy model initialisation is measuring
cold start, not retrieval. `lifespan` also runs a real query through the full path
before the service reports ready, so the first user request is never the one that
warms the caches.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ragoa.config import settings
from ragoa.factory import build_pipeline
from ragoa.schemas import AskRequest, AskResponse, Language, Trace

STATE: dict = {}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.lower() in ("1", "true", "yes", "on")


@asynccontextmanager
async def lifespan(app: FastAPI):
    index_dir = Path(os.getenv("RAGOA_INDEX_DIR", str(settings.index_dir)))
    encoder_kind = os.getenv("RAGOA_ENCODER", "st")

    pipeline, manifest = build_pipeline(
        index_dir=index_dir,
        encoder_kind=encoder_kind,
        use_rerank=_env_flag("RAGOA_RERANK", True),
        use_sparse=_env_flag("RAGOA_SPARSE", True),
        # Explicit rather than inferred from whether a key happens to be present,
        # so tests can exercise the unconfigured path without depending on the
        # developer's .env - and never reach the real provider.
        use_stt=_env_flag("RAGOA_STT", True),
    )
    STATE["pipeline"] = pipeline
    STATE["manifest"] = manifest

    # Warm up: touch every stage so nothing initialises lazily on a user request.
    warm = pipeline.ask(AskRequest(query="what is a corporation", top_k=3))
    STATE["warmup_ms"] = warm.trace.total_ms
    STATE["ready"] = True
    print(f"ready: {manifest['n_chunks']:,} chunks, encoder={encoder_kind}, "
          f"warmup {warm.trace.total_ms:.0f} ms", flush=True)

    yield
    STATE.clear()


app = FastAPI(title="RAG in Goa", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("RAGOA_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pipeline():
    pipeline = STATE.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="index still loading")
    return pipeline


@app.get("/health")
def health() -> dict:
    manifest = STATE.get("manifest") or {}
    return {
        "ready": bool(STATE.get("ready")),
        "chunks": manifest.get("n_chunks"),
        "docs": manifest.get("n_docs"),
        "strategy": manifest.get("strategy"),
        "encoder": manifest.get("encoder"),
        "has_sparse": manifest.get("has_sparse"),
        "warmup_ms": STATE.get("warmup_ms"),
        "retrieval_budget_ms": settings.retrieval_budget_ms,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Text in, grounded answer out. Used by benchmarks and by the text fallback."""
    return get_pipeline().ask(request)


@app.post("/ask/audio", response_model=AskResponse)
async def ask_audio(
    file: UploadFile = File(..., description="16 kHz mono WAV, at most 30 seconds"),
    lang: Language = Form(Language.EN),
    top_k: int = Form(settings.top_k),
) -> AskResponse:
    """Speech in, grounded answer out, answered in the language that was spoken."""
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="empty audio upload")

    pipeline = get_pipeline()
    if pipeline.stt is None:
        raise HTTPException(
            status_code=503,
            detail="speech-to-text is not configured; set SARVAM_API_KEY",
        )
    return pipeline.ask_audio(
        audio, language=lang, filename=file.filename or "audio.wav", top_k=top_k
    )


@app.get("/config")
def config() -> dict:
    """Non-secret settings, so the UI can display the active configuration."""
    return {
        "embed_model": settings.embed_model,
        "rerank_model": settings.rerank_model,
        "llm_model": settings.llm_model,
        "llm_providers": settings.llm_providers,
        "stt_model": settings.sarvam_stt_model,
        "stt_mode": settings.sarvam_stt_mode,
        "top_k": settings.top_k,
        "dense_candidates": settings.dense_candidates,
        "rerank_candidates": settings.rerank_candidates,
        "retrieval_budget_ms": settings.retrieval_budget_ms,
        "ood_score_threshold": settings.ood_score_threshold,
        "groundedness_threshold": settings.groundedness_threshold,
        "languages": [lang.value for lang in Language],
    }


@app.get("/trace/example", response_model=Trace)
def trace_example() -> Trace:
    """A real trace for one query, so the UI can be built before the demo runs."""
    return get_pipeline().ask(AskRequest(query="what is a corporation")).trace

"""Eval-loop config surface (optional contract in rag-local-eval-loop)."""

from dotenv import load_dotenv

load_dotenv()

from ragoa.config import settings

GENERATION_BACKEND = "openrouter"
EMBEDDING_BACKEND = "local"
GENERATION_MODEL = settings.llm_model
LATENCY_BUDGET_MS = settings.retrieval_budget_ms

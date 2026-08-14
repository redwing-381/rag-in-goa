"""Output-side guardrails: citation validation, then groundedness.

Two separate failures, checked in order of cost.

Citation validation is exact and nearly free: every chunk id the model cites must
be one we actually retrieved this turn. A model that cites a plausible-looking id
it invented is hallucinating in the most detectable way possible, and no semantic
check is needed to catch it.

Groundedness is fuzzy, so it is tiered. Content-word overlap between each answer
sentence and its cited text handles the clear cases for free; only sentences in the
ambiguous middle get escalated to an embedding comparison, which costs a forward
pass. Most answers never trigger the expensive path.

The threshold is calibrated in bench/guardrail_eval.py rather than guessed, and
the tradeoff is explicit: too low and we pass through invention, too high and we
refuse answers that are correct but worded differently from the source.
"""

from __future__ import annotations

import re

import numpy as np

from ragoa.config import Settings
from ragoa.schemas import (
    AnswerPayload,
    GroundednessReport,
    RetrievedChunk,
)

_WORD_RE = re.compile(r"[a-z0-9]+")

# Function words carry no evidence, so counting them inflates overlap. A stopword
# hit would let "the of and in" score as grounded against any passage.
_STOPWORDS = frozenset("""
a an the this that these those there here and or but if then than so because as
of in on at to from by for with without within into onto about over under
is are was were be been being am do does did done doing have has had having
can could will would shall should may might must
it its it's they them their he she his her you your i we our us my me
what which who whom whose when where why how
not no nor only just also very more most much many some any each every
""".split())

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u0964\u0965\u06D4])\s+")


def content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower())
            if w not in _STOPWORDS and len(w) > 2}


def overlap_ratio(claim: str, evidence: str) -> float:
    """Share of the claim's content words that appear in the evidence.

    Asymmetric on purpose: we care whether the answer is supported by the source,
    not whether the source is covered by the answer. A two-line answer drawn from
    a long passage should score 1.0.
    """
    claim_words = content_words(claim)
    if not claim_words:
        return 1.0  # nothing substantive asserted, so nothing to contradict
    return len(claim_words & content_words(evidence)) / len(claim_words)


def split_claims(answer: str, min_chars: int = 12) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer.strip())
            if len(s.strip()) >= min_chars]


class OutputGate:
    def __init__(self, settings: Settings | None = None, encoder=None):
        from ragoa.config import settings as default_settings

        self.settings = settings or default_settings
        # Optional; when absent the gate is overlap-only and says so in the report.
        self.encoder = encoder

    def validate_citations(
        self, payload: AnswerPayload, retrieved: list[RetrievedChunk]
    ) -> list[str]:
        """Cited ids that were not in this turn's retrieved set."""
        available = {r.chunk.chunk_id for r in retrieved}
        return [cid for cid in payload.citations if cid not in available]

    def _evidence_text(
        self, payload: AnswerPayload, retrieved: list[RetrievedChunk]
    ) -> str:
        """Text the answer claims to rest on.

        Prefers the cited chunks. Falls back to everything retrieved when the
        model cited nothing, so an uncited-but-correct answer is judged on the
        evidence it was actually shown rather than failed outright.
        """
        cited = {cid for cid in payload.citations}
        chosen = [r for r in retrieved if r.chunk.chunk_id in cited] or retrieved
        return " ".join((r.expanded_text or r.chunk.text) for r in chosen)

    def check_groundedness(
        self, payload: AnswerPayload, retrieved: list[RetrievedChunk]
    ) -> GroundednessReport:
        cfg = self.settings
        invalid = self.validate_citations(payload, retrieved)
        evidence = self._evidence_text(payload, retrieved)
        claims = split_claims(payload.answer)

        if not claims or not evidence:
            return GroundednessReport(
                grounded=not invalid, score=1.0 if not invalid else 0.0,
                invalid_citations=invalid,
            )

        scores: list[float] = []
        unsupported: list[str] = []
        ambiguous: list[tuple[int, str]] = []

        for i, claim in enumerate(claims):
            ratio = overlap_ratio(claim, evidence)
            scores.append(ratio)
            # Clear pass and clear fail are decided for free; only the band in
            # between is worth an embedding call.
            if ratio < cfg.groundedness_threshold:
                if self.encoder is not None and ratio >= cfg.groundedness_threshold * 0.5:
                    ambiguous.append((i, claim))
                else:
                    unsupported.append(claim)

        if ambiguous and self.encoder is not None:
            rescued = self._semantic_rescue([c for _, c in ambiguous], evidence)
            for (i, claim), similarity in zip(ambiguous, rescued, strict=True):
                if similarity >= 0.60:
                    scores[i] = max(scores[i], similarity)
                else:
                    unsupported.append(claim)

        score = float(np.mean(scores)) if scores else 0.0
        grounded = not invalid and not unsupported and score >= cfg.groundedness_threshold

        return GroundednessReport(
            grounded=grounded, score=score,
            unsupported_sentences=unsupported, invalid_citations=invalid,
        )

    def _semantic_rescue(self, claims: list[str], evidence: str) -> list[float]:
        """Cosine similarity of each claim against the evidence.

        Catches correct answers that paraphrase rather than quote, which is the
        main false-refusal mode of pure lexical overlap.
        """
        try:
            vectors = self.encoder.encode(claims + [evidence])
        except Exception:
            return [0.0] * len(claims)
        evidence_vector = vectors[-1]
        return [float(np.dot(v, evidence_vector)) for v in vectors[:-1]]

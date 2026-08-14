"""Input-side guardrails: refuse before spending any retrieval or LLM budget.

Ordered cheapest-first, so a bad request is rejected in microseconds rather than
after an LLM call. Every check returns a named `RefusalReason`, because "the
system declined" is only useful to a user if it says which rule fired.

Scope, stated honestly: these are heuristics, not classifiers. They catch the
obvious cases (empty or garbled transcripts, instruction-override attempts,
requests for harmful procedures) and they will miss adversarial phrasing. The
load-bearing guardrail against wrong answers is not here - it is the
retrieval-score out-of-domain check plus the groundedness gate on the output,
both of which are grounded in evidence rather than pattern matching.
"""

from __future__ import annotations

import re
import unicodedata

from ragoa.config import Settings
from ragoa.schemas import GuardrailVerdict, RefusalReason, STTResult

# Attempts to override the system prompt or exfiltrate it. Sarvam's translate mode
# delivers English, so English patterns cover Indic speech too.
_INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instruction|prompt|rule|direction)",
    r"\bdisregard\s+(all\s+|the\s+)?(previous|prior|above)\b",
    r"\b(system|developer)\s+prompt\b",
    r"\b(reveal|show|print|repeat|output)\s+(me\s+)?(your|the)\s+(instruction|prompt|rule|system)",
    r"\byou\s+are\s+now\s+(a|an|the)\b",
    r"\bact\s+as\s+(if\s+you|a\s+jailbroken|an?\s+unrestricted)\b",
    r"\b(pretend|imagine)\s+(that\s+)?(you|there)\s+(are|is)\s+no\s+(rule|restriction|filter)",
    r"\bdeveloper\s+mode\b",
    r"\bDAN\b",
    r"</?(system|assistant|user)>",
]

# Requests for harmful procedures. Deliberately about *instructions for harm*
# rather than a slur list: a keyword blocklist punishes people for quoting or
# asking about a topic, while a procedural request is unambiguous.
_UNSAFE_PATTERNS = [
    r"\bhow\s+(to|do\s+i|can\s+i)\s+(make|build|synthesi[sz]e|construct)\s+"
    r"(a\s+)?(bomb|explosive|ied|napalm|nerve\s+agent|meth|fentanyl|ricin)",
    r"\bhow\s+(to|do\s+i|can\s+i)\s+(kill|murder|poison)\s+(a\s+|my\s+)?(person|someone|him|her|them|my)",
    r"\b(how\s+to|best\s+way\s+to)\s+(commit\s+suicide|kill\s+myself|end\s+my\s+life)",
    r"\bhow\s+(to|do\s+i)\s+(hack|ddos|breach)\s+(into\s+)?(a\s+)?(bank|hospital|government|someone)",
    r"\b(child|minor|underage)\s+(porn|sexual|explicit)",
    r"\bhow\s+to\s+(buy|obtain)\s+(illegal\s+)?(firearms?|guns?)\s+(without|illegally|no\s+background)",
    # Break-in / theft how-tos. "what is a lock" still passes; the verb is the signal.
    r"\bhow\s+(to|do\s+i|can\s+i)\s+pick\s+(a\s+)?locks?\b",
    r"\b(pick|picking)\s+(a\s+)?lock\b.{0,40}\b(break\s+in|burgle|burglary|steal)\b",
    r"\bhow\s+(to|do\s+i|can\s+i)\s+(break\s+into|break\s+in(\s+to)?)\b",
    r"\bhow\s+(to|do\s+i|can\s+i)\s+(hotwire|steal)\s+(a\s+)?(car|vehicle|bike)\b",
    r"\bhow\s+(to|do\s+i|can\s+i)\s+(disable|bypass)\s+(a\s+)?(alarm|security\s+system|cctv)\b",
]

_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
_UNSAFE_RE = [re.compile(p, re.IGNORECASE) for p in _UNSAFE_PATTERNS]

# Sarvam sometimes emits a single repeated syllable on silence or pure noise.
_REPEAT_RE = re.compile(r"(.{1,4}?)\1{5,}")


def _printable_ratio(text: str) -> float:
    """Share of characters that are letters, digits, marks, or spaces.

    Catches transcripts that are mostly control characters or replacement
    glyphs, which is what a failed decode looks like.
    """
    if not text:
        return 0.0
    good = sum(
        1 for ch in text
        if ch.isspace() or unicodedata.category(ch)[0] in ("L", "N", "M", "P")
    )
    return good / len(text)


class InputGate:
    def __init__(self, settings: Settings | None = None):
        from ragoa.config import settings as default_settings

        self.settings = settings or default_settings

    def check(self, text: str, stt: STTResult | None = None) -> GuardrailVerdict:
        cfg = self.settings
        stripped = (text or "").strip()

        if len(stripped) < cfg.min_transcript_chars:
            return GuardrailVerdict(
                allowed=False, reason=RefusalReason.UNINTELLIGIBLE_AUDIO,
                detail="transcript empty or too short to be a question",
            )

        if _printable_ratio(stripped) < 0.80:
            return GuardrailVerdict(
                allowed=False, reason=RefusalReason.UNINTELLIGIBLE_AUDIO,
                detail="transcript is mostly non-textual characters",
                score=_printable_ratio(stripped),
            )

        if _REPEAT_RE.search(stripped):
            return GuardrailVerdict(
                allowed=False, reason=RefusalReason.UNINTELLIGIBLE_AUDIO,
                detail="transcript is a repeated fragment, typical of silence or noise",
            )

        # Trust Sarvam's own confidence when it reports low language probability.
        if stt is not None and stt.language_probability is not None:
            if stt.language_probability < cfg.min_language_probability:
                return GuardrailVerdict(
                    allowed=False, reason=RefusalReason.UNINTELLIGIBLE_AUDIO,
                    detail=f"speech recognition confidence too low "
                           f"({stt.language_probability:.2f})",
                    score=stt.language_probability,
                )

        for pattern in _UNSAFE_RE:
            if pattern.search(stripped):
                return GuardrailVerdict(
                    allowed=False, reason=RefusalReason.UNSAFE_INPUT,
                    detail="request appears to seek instructions for causing harm",
                )

        for pattern in _INJECTION_RE:
            if pattern.search(stripped):
                return GuardrailVerdict(
                    allowed=False, reason=RefusalReason.PROMPT_INJECTION,
                    detail="request attempts to override system instructions",
                )

        return GuardrailVerdict(allowed=True)

    def check_domain(self, top_score: float) -> GuardrailVerdict:
        """Out-of-domain check on the retrieval score.

        This is the guardrail that actually decides "I do not know this", and it
        runs on evidence: if the best dense similarity across the whole corpus is
        below threshold, nothing relevant exists and answering would mean
        inventing. The threshold is calibrated in bench/guardrail_eval.py against
        the 44,046 queries this dataset labels 'No Answer Present.', so it is
        tuned on real unanswerable questions rather than picked by feel.
        """
        threshold = self.settings.ood_score_threshold
        if top_score < threshold:
            return GuardrailVerdict(
                allowed=False, reason=RefusalReason.OUT_OF_DOMAIN,
                detail=f"best match scored {top_score:.3f}, below {threshold:.2f}",
                score=top_score,
            )
        return GuardrailVerdict(allowed=True, score=top_score)

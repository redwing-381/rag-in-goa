"""OpenRouter client with provider pinning, retries, and a circuit breaker.

This is the "harness" part rather than a raw prompt call: structured JSON output
enforced by schema, a repair retry when parsing fails, backoff with jitter, a
breaker that stops hammering a provider that is down, and a final extractive
fallback so a demo never dies on someone else's outage.

Three OpenRouter specifics that are easy to get wrong, all handled here:

  * `require_parameters: true` is mandatory alongside `response_format`. Without
    it, if no upstream provider supports structured output, OpenRouter silently
    ignores the schema and returns free text - the failure is invisible.
  * `openai/gpt-oss-120b` is served by both Cerebras and Groq, so ordering those
    two gives a real failover path on one model.
  * The generation endpoint reports true upstream latency after the fact, which is
    how we separate model time from proxy overhead in the latency report.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from ragoa.config import Settings
from ragoa.schemas import AnswerPayload, Language, RefusalReason, RetrievedChunk

ANSWER_SCHEMA = {
    "name": "grounded_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer", "citations", "confidence", "answer_language"],
        "properties": {
            "answer": {
                "type": "string",
                "description": "Answer using only the provided context. If the "
                               "context does not contain the answer, say so.",
            },
            "citations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Bracketed passage numbers that support the answer, as "
                               "strings, e.g. [\"1\", \"3\"]. Use only numbers shown "
                               "in the context.",
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "answer_language": {
                "type": "string",
                "enum": [lang.value for lang in Language],
            },
        },
    },
}

SYSTEM_PROMPT = """You answer questions using only the numbered context passages provided.

Rules:
- Use only facts present in the context. Never add outside knowledge.
- If the passages do not directly answer the question, reply with a brief statement \
that you do not have that information, set citations to [], and confidence below 0.5. \
Do not guess from loosely related text.
- Cite the bracketed number of every passage you used, e.g. ["1", "3"].
- Write the entire answer in the language named by answer_language. For hi, bn, \
ta, or mr that means the native script (Devanagari, Bengali, Tamil), not English. \
Latin letters are allowed only for names of people, places, or brands.
- Be concise: two or three sentences at most."""

# Unicode blocks for the Indic languages we answer in. English is handled as
# "mostly Latin letters" rather than a block, because mixed punctuation is fine.
_SCRIPT_RANGES: dict[Language, tuple[int, int]] = {
    Language.HI: (0x0900, 0x097F),
    Language.MR: (0x0900, 0x097F),
    Language.BN: (0x0980, 0x09FF),
    Language.TA: (0x0B80, 0x0BFF),
}


def script_ratio(text: str, language: Language) -> float:
    """Share of letters that belong to `language`'s writing system.

    Used to catch the common failure where the model sets answer_language=ta
    but writes English. Proper nouns in Latin are expected, so the threshold
    that callers apply is well below 1.0.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 1.0
    if language is Language.EN:
        return sum(1 for ch in letters if ch.isascii()) / len(letters)
    start, end = _SCRIPT_RANGES[language]
    return sum(1 for ch in letters if start <= ord(ch) <= end) / len(letters)


def language_matches(text: str, language: Language, minimum: float = 0.25) -> bool:
    return script_ratio(text, language) >= minimum


class LLMError(RuntimeError):
    pass


@dataclass
class CircuitBreaker:
    """Stops calling a provider that is failing, then lets one request test it.

    Without this, a provider outage turns every request into a full retry ladder,
    so latency degrades for everyone instead of failing fast to the fallback.
    """

    threshold: int = 3
    cooldown_s: float = 30.0
    failures: int = 0
    opened_at: float | None = field(default=None)

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.time() - self.opened_at >= self.cooldown_s:
            # Half-open: allow one probe through rather than staying dark forever.
            self.opened_at = None
            self.failures = 0
            return False
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = time.time()


def build_context(retrieved: list[RetrievedChunk], max_chars: int = 6000) -> str:
    """Numbered context block.

    Passages are labelled with small ordinals and the internal chunk id is kept out
    of the prompt on purpose. Asking a model to echo an opaque id invites it to
    approximate one, and since citation validation is exact, a single mistyped
    character turns a correct answer into a refusal. A one- or two-digit number is
    a token space it cannot get wrong, and `resolve_citations` maps it back.
    """
    parts: list[str] = []
    used = 0
    for i, item in enumerate(retrieved, start=1):
        text = item.expanded_text or item.chunk.text
        if used + len(text) > max_chars:
            text = text[: max(0, max_chars - used)]
        if not text:
            break
        parts.append(f"[{i}]\n{text}")
        used += len(text)
    return "\n\n".join(parts)


_ORDINAL_RE = re.compile(r"\d+")
_BRACKET_CITE = re.compile(r"\[(\d+)\]")


def citations_from_prose(text: str) -> list[str]:
    """Passage numbers the model mentioned inline, e.g. [1] or [3]."""
    return list(dict.fromkeys(_BRACKET_CITE.findall(text or "")))


def resolve_citations(
    citations: list[str], retrieved: list[RetrievedChunk]
) -> list[str]:
    """Turn the model's passage numbers into chunk ids.

    Anything that cannot be resolved is passed through untouched rather than
    dropped, so the output gate still sees - and reports - a genuinely invented
    citation. Silently discarding the unresolvable would disable the cheapest
    hallucination check we have.
    """
    by_ordinal = {str(i): item.chunk.chunk_id for i, item in enumerate(retrieved, start=1)}
    known = {item.chunk.chunk_id for item in retrieved}

    resolved: list[str] = []
    for raw in citations:
        candidate = raw.strip()
        if candidate in known:
            resolved.append(candidate)
            continue
        # Accepts "1", "[1]", "passage 1" and similar; the model is asked for a
        # bare number but should not be punished for decorating it.
        match = _ORDINAL_RE.search(candidate)
        if match and match.group() in by_ordinal:
            resolved.append(by_ordinal[match.group()])
            continue
        resolved.append(candidate)

    # Deduplicate while preserving order: models often cite the same passage twice.
    seen: set[str] = set()
    return [c for c in resolved if not (c in seen or seen.add(c))]


class OpenRouterLLM:
    def __init__(self, settings: Settings | None = None, client=None):
        from ragoa.config import settings as default_settings

        self.settings = settings or default_settings
        self.breaker = CircuitBreaker()
        self._client = client
        self.last_generation_id: str | None = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            if not self.settings.openrouter_api_key:
                raise LLMError("OPENROUTER_API_KEY is not set")
            self._client = OpenAI(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.openrouter_api_key,
                timeout=self.settings.llm_timeout_s,
            )
        return self._client

    def _extra_body(self) -> dict:
        return {
            "provider": {
                "order": list(self.settings.llm_providers),
                "allow_fallbacks": True,
                # Fail loudly instead of silently dropping response_format.
                "require_parameters": True,
            }
        }

    def _messages(
        self,
        query: str,
        context: str,
        language: Language,
        history: list | None = None,
        *,
        prose: bool = False,
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history or []:
            role = getattr(turn, "role", None) or turn.get("role")
            text = getattr(turn, "text", None) or turn.get("text")
            if role in ("user", "assistant") and text:
                messages.append({"role": role, "content": str(text)})
        user = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"answer_language must be: {language.value}"
        )
        if prose:
            user += (
                "\n\nReply as plain prose, not JSON. Cite supporting passages "
                "as [1] or [2]. Two or three sentences."
            )
        messages.append({"role": "user", "content": user})
        return messages

    def answer(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
        language: Language = Language.EN,
        history: list | None = None,
    ) -> tuple[AnswerPayload, float]:
        """Return (payload, time_to_response_ms). Falls back rather than raising."""
        if self.breaker.is_open:
            return self._extractive_fallback(query, retrieved, language,
                                             "provider circuit open"), 0.0

        context = build_context(retrieved)
        messages = self._messages(query, context, language, history)
        started = time.perf_counter()
        last_error: Exception | None = None
        last_payload: AnswerPayload | None = None

        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=messages,
                    max_tokens=self.settings.llm_max_tokens,
                    temperature=self.settings.llm_temperature,
                    response_format={"type": "json_schema", "json_schema": ANSWER_SCHEMA},
                    extra_body=self._extra_body(),
                )
                self.last_generation_id = getattr(response, "id", None)
                content = response.choices[0].message.content or ""
                payload = self._parse(content)
                if payload is not None:
                    payload.citations = resolve_citations(payload.citations, retrieved)
                    last_payload = payload
                    if language_matches(payload.answer, language):
                        payload.answer_language = language
                        self.breaker.record_success()
                        return payload, (time.perf_counter() - started) * 1000.0
                    # Usable facts, wrong script: one rewrite, not a fallback.
                    messages = messages + [
                        {"role": "assistant", "content": content[:2000]},
                        {"role": "user",
                         "content": f"Rewrite 'answer' in {language.value} script. "
                                    "Keep the same facts and citations. JSON only."},
                    ]
                    last_error = LLMError("answer language mismatch")
                    continue

                # Valid HTTP response, unusable body: ask once for a repair before
                # burning another full generation.
                messages = messages + [
                    {"role": "assistant", "content": content[:2000]},
                    {"role": "user",
                     "content": "That was not valid JSON matching the schema. "
                                "Reply with only the JSON object."},
                ]
                last_error = LLMError("schema parse failed")
            except Exception as exc:  # provider errors, timeouts, rate limits
                last_error = exc
                self.breaker.record_failure()

            if attempt < self.settings.llm_max_retries:
                # Jitter matters: without it, concurrent retries synchronise and
                # re-hammer a recovering provider in lockstep.
                time.sleep(0.3 * (2 ** attempt) * (0.5 + random.random()))

        if last_payload is not None:
            # Keep the facts; do not claim they are in a language they are not.
            if not language_matches(last_payload.answer, language):
                last_payload.answer_language = Language.EN
            self.breaker.record_success()
            return last_payload, (time.perf_counter() - started) * 1000.0

        return self._extractive_fallback(
            query, retrieved, language, f"llm unavailable: {last_error}"
        ), (time.perf_counter() - started) * 1000.0

    def stream(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
        language: Language = Language.EN,
        history: list | None = None,
    ) -> Iterator[str]:
        """Stream raw text deltas, for time-to-first-token in the demo.

        OpenRouter injects `: OPENROUTER PROCESSING` keep-alive comments and
        reports mid-stream errors as a data event with HTTP 200, so callers must
        not assume every chunk carries a choice.
        """
        context = build_context(retrieved)
        stream = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=self._messages(query, context, language, history, prose=True),
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
            extra_body={
                "provider": {
                    "order": list(self.settings.llm_providers),
                    "allow_fallbacks": True,
                }
            },
            stream=True,
        )
        for event in stream:
            if not getattr(event, "choices", None):
                continue
            delta = event.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def upstream_stats(self, generation_id: str | None = None,
                       attempts: int = 4, delay_s: float = 1.2) -> dict:
        """True upstream latency, via OpenRouter's generation endpoint.

        Lets the latency report separate model time from proxy overhead instead of
        attributing all of it to the model.

        The record is written asynchronously and 404s for roughly the first two
        seconds after the response returns, so this polls. Never call it on the
        request path - it is for offline reporting only.
        """
        gid = generation_id or self.last_generation_id
        if not gid:
            return {}

        import httpx

        for attempt in range(attempts):
            if attempt:
                time.sleep(delay_s)
            try:
                response = httpx.get(
                    f"{self.settings.llm_base_url}/generation",
                    params={"id": gid},
                    headers={"Authorization": f"Bearer {self.settings.openrouter_api_key}"},
                    timeout=10.0,
                )
            except Exception:
                continue
            if response.status_code == 404:
                continue  # not written yet
            if response.status_code != 200:
                return {}
            data = response.json().get("data", {})
            return {
                "provider": data.get("provider_name"),
                "latency_ms": data.get("latency"),
                "generation_ms": data.get("generation_time"),
                "tokens_completion": data.get("native_tokens_completion"),
                "cost": data.get("total_cost"),
            }
        return {}

    @staticmethod
    def _parse(content: str) -> AnswerPayload | None:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{"):] if "{" in text else text
        try:
            return AnswerPayload.model_validate_json(text)
        except Exception:
            pass
        # Some providers wrap the object in prose; take the outermost braces.
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return AnswerPayload.model_validate(json.loads(text[start:end + 1]))
            except Exception:
                return None
        return None

    @staticmethod
    def _extractive_fallback(
        query: str, retrieved: list[RetrievedChunk], language: Language, why: str
    ) -> AnswerPayload:
        """Refuse rather than quote a passage the model never endorsed.

        Quoting the top hit used to look like an answer (confidence 0.2, a
        relevant-looking excerpt) and produced off-topic text when retrieval
        was only loosely related. A named provider failure is honest.
        """
        return AnswerPayload(
            answer="The answering model is unavailable right now.",
            citations=[],
            confidence=0.0,
            answer_language=language,
            refusal_reason=RefusalReason.PROVIDER_FAILURE,
        )

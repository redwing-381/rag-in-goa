"""The harness: one ordered, instrumented path from speech to grounded answer.

Structured rather than a single prompt call. Each stage has a typed input and
output, its own span in the trace, and a defined failure behaviour. Three gates
can end the request early, and each returns a named reason:

    input gate      unsafe, injection, or unintelligible audio
    domain gate     nothing in the corpus is close enough to answer from
    output gate     the answer is not supported by what we retrieved

The retrieval budget and the LLM budget are deliberately separate. Retrieval is
ours to control and is held under 200ms; generation is a network call to someone
else's GPU, so it is measured and reported on its own rather than folded into a
number we would then have to explain away.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Iterator

from ragoa.config import Settings
from ragoa.guardrails.input_gate import InputGate
from ragoa.guardrails.output_gate import OutputGate, content_words
from ragoa.harness.llm import OpenRouterLLM, citations_from_prose, resolve_citations
from ragoa.index.retriever import HybridRetriever
from ragoa.schemas import (
    AnswerPayload,
    AskRequest,
    AskResponse,
    Citation,
    Deadline,
    GuardrailVerdict,
    Language,
    RefusalReason,
    RetrievedChunk,
    STTResult,
    Trace,
)
from ragoa.telemetry.timing import timed

# Refusals are shown in the language the user spoke. A system that declines in a
# language the user does not read has not really communicated the refusal.
REFUSAL_TEXT: dict[RefusalReason, dict[Language, str]] = {
    RefusalReason.OUT_OF_DOMAIN: {
        Language.EN: "I do not have information about that in my knowledge base.",
        Language.HI: "मेरे ज्ञान भंडार में इसकी जानकारी नहीं है।",
        Language.BN: "আমার জ্ঞানভাণ্ডারে এই বিষয়ে তথ্য নেই।",
        Language.TA: "என் தகவல் தளத்தில் அதற்கான விவரம் இல்லை.",
        Language.MR: "माझ्या ज्ञानसंग्रहात याबद्दल माहिती नाही.",
    },
    RefusalReason.NOT_GROUNDED: {
        Language.EN: "I found related passages but none of them actually support an "
                     "answer, so I would rather not guess.",
        Language.HI: "मुझे संबंधित अंश मिले, लेकिन उनमें उत्तर नहीं है, इसलिए मैं अनुमान नहीं लगाऊँगा।",
        Language.BN: "সম্পর্কিত অংশ পেয়েছি, কিন্তু সেগুলিতে উত্তর নেই, তাই অনুমান করব না।",
        Language.TA: "தொடர்புடைய பகுதிகள் கிடைத்தன, ஆனால் அவற்றில் பதில் இல்லை.",
        Language.MR: "संबंधित उतारे मिळाले, पण त्यात उत्तर नाही, म्हणून अंदाज लावणार नाही.",
    },
    RefusalReason.UNSAFE_INPUT: {
        Language.EN: "I cannot help with that request.",
        Language.HI: "मैं इस अनुरोध में सहायता नहीं कर सकता।",
        Language.BN: "আমি এই অনুরোধে সহায়তা করতে পারব না।",
        Language.TA: "இந்தக் கோரிக்கையில் உதவ முடியாது.",
        Language.MR: "या विनंतीत मी मदत करू शकत नाही.",
    },
    RefusalReason.PROMPT_INJECTION: {
        Language.EN: "That request tries to change my instructions, so I have ignored it. "
                     "Ask me a question about the documents instead.",
        Language.HI: "यह अनुरोध मेरे निर्देश बदलने का प्रयास करता है, इसलिए मैंने इसे अनदेखा किया।",
        Language.BN: "এই অনুরোধ আমার নির্দেশ বদলাতে চায়, তাই এটি উপেক্ষা করেছি।",
        Language.TA: "இந்தக் கோரிக்கை என் வழிமுறைகளை மாற்ற முயல்கிறது, எனவே புறக்கணித்தேன்.",
        Language.MR: "ही विनंती माझ्या सूचना बदलण्याचा प्रयत्न करते, म्हणून दुर्लक्ष केले.",
    },
    RefusalReason.UNINTELLIGIBLE_AUDIO: {
        Language.EN: "I could not make out a question in that audio. Please try again.",
        Language.HI: "मुझे उस ऑडियो में कोई प्रश्न समझ नहीं आया। कृपया फिर कोशिश करें।",
        Language.BN: "সেই অডিওতে কোনো প্রশ্ন বুঝতে পারিনি। আবার চেষ্টা করুন।",
        Language.TA: "அந்த ஒலியில் கேள்வி புரியவில்லை. மீண்டும் முயற்சிக்கவும்.",
        Language.MR: "त्या ऑडिओमध्ये प्रश्न समजला नाही. कृपया पुन्हा प्रयत्न करा.",
    },
    RefusalReason.INVALID_CITATIONS: {
        Language.EN: "The draft answer cited sources that do not exist, so I discarded it.",
        Language.HI: "मसौदा उत्तर ने ऐसे स्रोत बताए जो मौजूद नहीं हैं, इसलिए मैंने उसे हटा दिया।",
        Language.BN: "খসড়া উত্তরে অস্তিত্বহীন সূত্র উল্লেখ ছিল, তাই বাতিল করেছি।",
        Language.TA: "வரைவு பதிலில் இல்லாத ஆதாரங்கள் இருந்தன, எனவே நிராகரித்தேன்.",
        Language.MR: "मसुदा उत्तरात अस्तित्वात नसलेले स्रोत होते, म्हणून ते रद्द केले.",
    },
    RefusalReason.PROVIDER_FAILURE: {
        Language.EN: "The answering model is unavailable right now.",
        Language.HI: "उत्तर देने वाला मॉडल अभी उपलब्ध नहीं है।",
        Language.BN: "উত্তরদাতা মডেল এখন উপলব্ধ নয়।",
        Language.TA: "பதிலளிக்கும் மாதிரி இப்போது கிடைக்கவில்லை.",
        Language.MR: "उत्तर देणारे मॉडेल आता उपलब्ध नाही.",
    },
}


# Model declining a harmful request in prose, rather than via refusal_reason.
# Mapped to UNSAFE_INPUT so the UI does not report it as "not grounded".
_SAFETY_SELF_REFUSAL = re.compile(
    r"\b(i (cannot|can't|won't|will not) (help|assist) with|"
    r"i (cannot|can't|won't|will not) (provide|give) "
    r"(instructions|guidance|advice) (for|on|about))\b",
    re.IGNORECASE,
)


def looks_like_safety_refusal(text: str) -> bool:
    return bool(_SAFETY_SELF_REFUSAL.search(text or ""))


# The system prompt tells the model to say this when the passages do not answer.
# That sentence is not a claim about the world, so running it through the
# groundedness gate produces "unsupported: I do not have that information."
_UNANSWERABLE = re.compile(
    r"\b(i (do not|don't) have (that |this |the )?information|"
    r"i (do not|don't) know|"
    r"(the )?(provided )?context does not|"
    r"none of the (passages|context)|"
    r"no (relevant )?information (in|about))\b",
    re.IGNORECASE,
)


def looks_like_unanswerable(text: str) -> bool:
    return bool(_UNANSWERABLE.search(text or ""))


def evidence_overlaps_query(query: str, retrieved: list[RetrievedChunk]) -> bool:
    """True when the retrieved text shares at least two content words with the query.

    Used to decide whether a model 'I do not know' is worth one retry: if the
    passages never mentioned the subject, retrying will not help.
    """
    wanted = content_words(query)
    if not wanted:
        return False
    evidence = " ".join((item.expanded_text or item.chunk.text) for item in retrieved)
    need = 2 if len(wanted) >= 2 else 1
    return len(wanted & content_words(evidence)) >= need


def refusal_message(reason: RefusalReason, language: Language) -> str:
    per_language = REFUSAL_TEXT.get(reason, {})
    return per_language.get(language) or per_language.get(Language.EN) or "I cannot answer that."


class RagPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: OpenRouterLLM | None = None,
        input_gate: InputGate | None = None,
        output_gate: OutputGate | None = None,
        stt=None,
        settings: Settings | None = None,
    ):
        from ragoa.config import settings as default_settings

        self.settings = settings or default_settings
        self.retriever = retriever
        self.llm = llm or OpenRouterLLM(self.settings)
        self.input_gate = input_gate or InputGate(self.settings)
        self.output_gate = output_gate or OutputGate(self.settings)
        self.stt = stt

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _refuse(reason: RefusalReason, language: Language, trace: Trace,
                detail: str | None = None,
                citations: list[Citation] | None = None) -> AskResponse:
        trace.total_ms = max(trace.total_ms, sum(s.duration_ms for s in trace.spans))
        return AskResponse(
            answer=refusal_message(reason, language),
            refused=True,
            refusal_reason=reason,
            citations=citations or [],
            confidence=0.0,
            answer_language=language,
            trace=trace,
        )

    @staticmethod
    def _citations(payload: AnswerPayload, retrieved: list[RetrievedChunk]) -> list[Citation]:
        cited = set(payload.citations)
        chosen = [r for r in retrieved if r.chunk.chunk_id in cited] or retrieved[:1]
        return [
            Citation(
                chunk_id=r.chunk.chunk_id,
                doc_id=r.chunk.doc_id,
                text=r.chunk.text,
                score=r.score,
            )
            for r in chosen
        ]

    # -- entry points -----------------------------------------------------

    def transcribe(
        self,
        audio: bytes,
        filename: str,
        trace: Trace,
        language: Language | None = None,
    ) -> STTResult:
        if self.stt is None:
            raise RuntimeError("no speech-to-text client configured")
        language_code = None
        if language is not None:
            language_code = {
                Language.EN: "en-IN",
                Language.HI: "hi-IN",
                Language.BN: "bn-IN",
                Language.TA: "ta-IN",
                Language.MR: "mr-IN",
            }.get(language)
        with timed(trace, "stt"):
            return self.stt.transcribe(
                audio, filename=filename, language_code=language_code,
            )

    def ask_audio(self, audio: bytes, language: Language = Language.EN,
                  filename: str = "audio.wav", **kwargs) -> AskResponse:
        """Speech in, grounded answer out.

        Sarvam runs in translate mode, so `transcript` is English while `language`
        stays the language the user spoke and drives the answer language.
        """
        final = None
        for event in self.ask_audio_iter(audio, language=language, filename=filename, **kwargs):
            if event["type"] == "final":
                final = event["response"]
        assert final is not None
        return final

    def ask(self, request: AskRequest, trace: Trace | None = None,
            stt: STTResult | None = None) -> AskResponse:
        final = None
        for event in self.ask_iter(request, trace=trace, stt=stt, stream=False):
            if event["type"] == "final":
                final = event["response"]
        assert final is not None
        return final

    def ask_audio_iter(
        self,
        audio: bytes,
        language: Language = Language.EN,
        filename: str = "audio.wav",
        **kwargs,
    ) -> Iterator[dict]:
        """Yield transcript, then the same events as `ask_iter`."""
        trace = Trace(request_id=uuid.uuid4().hex[:12])
        try:
            stt = self.transcribe(audio, filename, trace, language=language)
        except Exception as exc:
            trace.add("stt_error", 0.0, error=str(exc)[:120])
            yield {"type": "stage", "name": "stt_error", "duration_ms": 0.0}
            yield {
                "type": "final",
                "response": self._refuse(
                    RefusalReason.UNINTELLIGIBLE_AUDIO, language, trace, detail=str(exc)
                ),
            }
            return

        yield {"type": "stage", "name": "stt",
               "duration_ms": next((s.duration_ms for s in trace.spans if s.name == "stt"), 0.0)}
        yield {"type": "transcript", "text": stt.text}
        request = AskRequest(query=stt.text, lang=language, **kwargs)
        for event in self.ask_iter(request, trace=trace, stt=stt, stream=True):
            if event["type"] == "final":
                event["response"].transcript = stt.text
            yield event

    def ask_iter(
        self,
        request: AskRequest,
        trace: Trace | None = None,
        stt: STTResult | None = None,
        *,
        stream: bool = True,
    ) -> Iterator[dict]:
        """Yield `stage`, `token`, and a terminal `final` event for one turn."""
        trace = trace or Trace(request_id=uuid.uuid4().hex[:12])
        language = request.lang
        seen = 0

        def flush_stages() -> list[dict]:
            nonlocal seen
            events = [
                {"type": "stage", "name": span.name, "duration_ms": span.duration_ms}
                for span in trace.spans[seen:]
            ]
            seen = len(trace.spans)
            return events

        # Gate 1: cheap input checks, before spending any budget.
        with timed(trace, "input_gate"):
            verdict: GuardrailVerdict = self.input_gate.check(request.query, stt)
        yield from flush_stages()
        if not verdict.allowed:
            trace.add("refused", 0.0, reason=verdict.reason.value if verdict.reason else "")
            yield from flush_stages()
            yield {
                "type": "final",
                "response": self._refuse(
                    verdict.reason or RefusalReason.UNSAFE_INPUT,
                    language, trace, verdict.detail,
                ),
            }
            return

        # Retrieval, under its own hard deadline. This turn's query only.
        deadline = Deadline(request.budget_ms)
        retrieved = self.retriever.retrieve(
            request.query, deadline, trace,
            top_k=request.top_k, candidates=request.candidates,
        )
        yield from flush_stages()

        # Gate 2: is anything in the corpus actually close enough to answer from?
        with timed(trace, "domain_gate"):
            top_score = self.retriever.top_score(retrieved)
            domain = self.input_gate.check_domain(top_score)
        yield from flush_stages()
        if not domain.allowed:
            trace.add("refused", 0.0, reason=RefusalReason.OUT_OF_DOMAIN.value,
                      top_score=top_score)
            yield from flush_stages()
            yield {
                "type": "final",
                "response": self._refuse(
                    RefusalReason.OUT_OF_DOMAIN, language, trace, domain.detail
                ),
            }
            return

        payload, llm_ms = yield from self._generate(
            request, retrieved, language, trace, stream=stream,
        )
        trace.llm_ttft_ms = llm_ms
        yield from flush_stages()

        response = self._after_generation(payload, retrieved, language, trace)
        yield from flush_stages()
        yield {"type": "final", "response": response}

    def _generate(
        self,
        request: AskRequest,
        retrieved: list[RetrievedChunk],
        language: Language,
        trace: Trace,
        *,
        stream: bool,
    ) -> Iterator[dict]:
        """Yield token events; return (payload, llm_ms) via StopIteration.value."""
        history = request.history
        started = time.perf_counter()
        payload: AnswerPayload | None = None
        collected: list[str] = []

        if stream:
            stream_fn = getattr(self.llm, "stream", None)
            if stream_fn is not None:
                try:
                    for delta in stream_fn(
                        request.query, retrieved, language, history=history
                    ):
                        if delta:
                            collected.append(delta)
                            yield {"type": "token", "text": delta}
                except Exception:
                    collected = []

        if collected:
            full = "".join(collected)
            parsed = None
            parse = getattr(self.llm, "_parse", None)
            if parse is not None:
                parsed = parse(full)
            if parsed is not None:
                payload = parsed
            else:
                payload = AnswerPayload(
                    answer=full.strip(),
                    citations=citations_from_prose(full),
                    confidence=0.7,
                    answer_language=language,
                )
            payload.citations = resolve_citations(payload.citations, retrieved)
            llm_ms = (time.perf_counter() - started) * 1000.0
            trace.add("llm", llm_ms)
        else:
            with timed(trace, "llm"):
                payload, llm_ms = self.llm.answer(
                    request.query, retrieved, language, history=history
                )
                if payload.answer:
                    yield {"type": "token", "text": payload.answer}

        # One retry when the model claims the passages are empty but they
        # actually mention the question. Cheap compared to a false refusal
        # on a question the index can answer (Kinsey, Fuji, etc.).
        if (
            looks_like_unanswerable(payload.answer)
            and payload.refusal_reason is None
            and evidence_overlaps_query(request.query, retrieved)
        ):
            retried, retry_ms = self.llm.answer(
                request.query, retrieved, language, history=history
            )
            llm_ms += retry_ms
            if not looks_like_unanswerable(retried.answer):
                payload = retried
                trace.add("llm_retry", retry_ms, reason="unanswerable_with_evidence")
                yield {"type": "token", "text": payload.answer}

        return payload, llm_ms

    def _after_generation(
        self,
        payload: AnswerPayload,
        retrieved: list[RetrievedChunk],
        language: Language,
        trace: Trace,
    ) -> AskResponse:
        # The model may decline on its own; respect that rather than overriding it.
        if payload.refusal_reason is not None:
            return self._refuse(payload.refusal_reason, language, trace)

        if looks_like_safety_refusal(payload.answer):
            trace.add("refused", 0.0, reason=RefusalReason.UNSAFE_INPUT.value,
                      via="model_self_refusal")
            return self._refuse(RefusalReason.UNSAFE_INPUT, language, trace)

        if looks_like_unanswerable(payload.answer):
            trace.add("refused", 0.0, reason=RefusalReason.NOT_GROUNDED.value,
                      via="model_unanswerable")
            return self._refuse(RefusalReason.NOT_GROUNDED, language, trace)

        if payload.confidence < self.settings.min_answer_confidence:
            reason = (
                RefusalReason.PROVIDER_FAILURE
                if payload.confidence <= 0.2
                else RefusalReason.NOT_GROUNDED
            )
            trace.add("refused", 0.0, reason=reason.value,
                      confidence=round(payload.confidence, 3))
            return self._refuse(reason, language, trace)

        # Gate 3: is the answer supported by what we retrieved?
        with timed(trace, "output_gate"):
            report = self.output_gate.check_groundedness(payload, retrieved)

        if report.invalid_citations:
            trace.add("refused", 0.0, reason=RefusalReason.INVALID_CITATIONS.value,
                      invalid=len(report.invalid_citations))
            response = self._refuse(RefusalReason.INVALID_CITATIONS, language, trace)
            response.groundedness = report
            return response

        if not report.grounded:
            trace.add("refused", 0.0, reason=RefusalReason.NOT_GROUNDED.value,
                      score=round(report.score, 3))
            response = self._refuse(RefusalReason.NOT_GROUNDED, language, trace)
            response.groundedness = report
            return response

        trace.total_ms = sum(s.duration_ms for s in trace.spans)
        return AskResponse(
            answer=payload.answer,
            refused=False,
            citations=self._citations(payload, retrieved),
            confidence=payload.confidence,
            answer_language=payload.answer_language,
            groundedness=report,
            trace=trace,
        )

"""Shared post-LLM policy: when to show an answer vs refuse.

Both the live pipeline and the eval-loop adapter call this so reliability
numbers reflect the same guardrails users see in production.
"""

from __future__ import annotations

import re

from ragoa.config import Settings
from ragoa.guardrails.input_gate import InputGate
from ragoa.guardrails.output_gate import OutputGate
from ragoa.schemas import (
    AnswerPayload,
    AskResponse,
    Citation,
    Language,
    RefusalReason,
    RetrievedChunk,
    Trace,
)

# Model declining a harmful request in prose, rather than via refusal_reason.
_SAFETY_SELF_REFUSAL = re.compile(
    r"\b(i (cannot|can't|won't|will not) (help|assist) with|"
    r"i (cannot|can't|won't|will not) (provide|give) "
    r"(instructions|guidance|advice) (for|on|about))\b",
    re.IGNORECASE,
)

# The system prompt tells the model to say this when the passages do not answer.
_UNANSWERABLE = re.compile(
    r"\b(i (do not|don't) have (that |this |the )?information|"
    r"i (do not|don't) know|"
    r"(the )?(provided )?(documents?|passages?|context) (do not|don't) "
    r"(contain|include|mention|cover|address)|"
    r"(the )?(provided )?context does not|"
    r"none of the (passages|context|documents)|"
    r"no (relevant )?information (in|about|found)|"
    r"does not (specify|state|mention|include|provide|contain|cover|address)|"
    r"information does not (specify|include|contain|mention)|"
    r"i'?m sorry,? but (the )?(provided )?(information|context|passages?)|"
    r"cannot (be )?answer(ed)? from (the )?(provided )?(context|passages|documents)|"
    r"not (mentioned|stated|described|provided) in (the )?(context|passages|documents))\b",
    re.IGNORECASE,
)

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


def looks_like_safety_refusal(text: str) -> bool:
    return bool(_SAFETY_SELF_REFUSAL.search(text or ""))


def looks_like_unanswerable(text: str) -> bool:
    return bool(_UNANSWERABLE.search(text or ""))


def refusal_message(reason: RefusalReason, language: Language) -> str:
    per_language = REFUSAL_TEXT.get(reason, {})
    return per_language.get(language) or per_language.get(Language.EN) or "I cannot answer that."


def top_retrieval_score(retrieved: list[RetrievedChunk]) -> float:
    """Best dense similarity for the domain gate."""
    scores = [r.dense_score for r in retrieved if r.dense_score is not None]
    if scores:
        return max(scores)
    return max((r.score for r in retrieved), default=0.0)


def _refuse(
    reason: RefusalReason,
    language: Language,
    trace: Trace | None,
    *,
    citations: list[Citation] | None = None,
) -> AskResponse:
    if trace is not None:
        trace.add("refused", 0.0, reason=reason.value)
    return AskResponse(
        answer=refusal_message(reason, language),
        refused=True,
        refusal_reason=reason,
        citations=citations or [],
        confidence=0.0,
        answer_language=language,
        trace=trace or Trace(request_id="eval"),
    )


def _citations(payload: AnswerPayload, retrieved: list[RetrievedChunk]) -> list[Citation]:
    cited = set(payload.citations)
    chosen = [r for r in retrieved if r.chunk.chunk_id in cited] or retrieved[:3]
    return [
        Citation(
            chunk_id=r.chunk.chunk_id,
            doc_id=r.chunk.doc_id,
            text=r.chunk.text,
            score=r.score,
        )
        for r in chosen
    ]


def finalize_answer(
    payload: AnswerPayload,
    retrieved: list[RetrievedChunk],
    language: Language,
    *,
    settings: Settings,
    input_gate: InputGate | None = None,
    output_gate: OutputGate | None = None,
    trace: Trace | None = None,
    check_domain: bool = True,
) -> AskResponse:
    """Apply domain, confidence, and groundedness policy after the LLM returns."""
    gate_in = input_gate or InputGate(settings)
    gate_out = output_gate or OutputGate(settings)

    if check_domain:
        domain = gate_in.check_domain(top_retrieval_score(retrieved))
        if not domain.allowed:
            return _refuse(RefusalReason.OUT_OF_DOMAIN, language, trace)

    if payload.refusal_reason is not None:
        return _refuse(payload.refusal_reason, language, trace)

    if looks_like_safety_refusal(payload.answer):
        return _refuse(RefusalReason.UNSAFE_INPUT, language, trace)

    if looks_like_unanswerable(payload.answer):
        return _refuse(RefusalReason.NOT_GROUNDED, language, trace)

    if payload.confidence < settings.min_answer_confidence:
        reason = (
            RefusalReason.PROVIDER_FAILURE
            if payload.confidence <= 0.2
            else RefusalReason.NOT_GROUNDED
        )
        return _refuse(reason, language, trace)

    report = gate_out.check_groundedness(payload, retrieved)

    if report.invalid_citations:
        response = _refuse(RefusalReason.INVALID_CITATIONS, language, trace)
        response.groundedness = report
        return response

    if not report.grounded:
        response = _refuse(RefusalReason.NOT_GROUNDED, language, trace)
        response.groundedness = report
        return response

    if trace is not None:
        trace.total_ms = sum(span.duration_ms for span in trace.spans)

    return AskResponse(
        answer=payload.answer,
        refused=False,
        citations=_citations(payload, retrieved),
        confidence=payload.confidence,
        answer_language=payload.answer_language,
        groundedness=report,
        trace=trace or Trace(request_id="eval"),
    )

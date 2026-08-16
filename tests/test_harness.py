from __future__ import annotations

import time

from ragoa.harness.llm import (
    CircuitBreaker,
    OpenRouterLLM,
    build_context,
    citations_from_prose,
    language_matches,
    resolve_citations,
    script_ratio,
)
from ragoa.index.fusion import reciprocal_rank_fusion
from ragoa.schemas import (
    Chunk,
    ChunkingStrategy,
    Deadline,
    Language,
    RefusalReason,
    RetrievedChunk,
    Trace,
)


def make_retrieved(chunk_id: str, text: str, expanded: str | None = None):
    return RetrievedChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id="d1", text=text, char_start=0,
                    char_end=len(text), strategy=ChunkingStrategy.SCRIPT_AWARE),
        expanded_text=expanded,
    )


class TestFusion:
    def test_agreement_beats_single_leg_top_hit(self):
        # 7 is second on both legs; 1 and 2 are first on only one each.
        fused = reciprocal_rank_fusion([[1, 7, 3], [2, 7, 4]])
        assert fused[0][0] == 7

    def test_ranks_not_scores(self):
        # Identical rankings must give identical fused order regardless of any
        # underlying score magnitudes, which RRF never sees.
        assert [label for label, _ in reciprocal_rank_fusion([[5, 6], [5, 6]])] == [5, 6]

    def test_weights_respected(self):
        fused = reciprocal_rank_fusion([[1], [2]], weights=[0.1, 10.0])
        assert fused[0][0] == 2

    def test_empty_input(self):
        assert reciprocal_rank_fusion([[], []]) == []


class TestDeadline:
    def test_remaining_shrinks(self):
        deadline = Deadline(50.0)
        time.sleep(0.01)
        assert deadline.remaining_ms < 50.0
        assert deadline.elapsed_ms >= 10.0

    def test_exceeded_respects_reserve(self):
        deadline = Deadline(20.0)
        # 500ms of reserve cannot fit in a 20ms budget, so optional work is skipped.
        assert deadline.exceeded(reserve_ms=500.0)
        assert not deadline.exceeded(reserve_ms=0.0)


class TestCircuitBreaker:
    def test_opens_after_threshold(self):
        breaker = CircuitBreaker(threshold=3, cooldown_s=60)
        for _ in range(2):
            breaker.record_failure()
        assert not breaker.is_open
        breaker.record_failure()
        assert breaker.is_open

    def test_success_resets(self):
        breaker = CircuitBreaker(threshold=2)
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        assert not breaker.is_open

    def test_half_open_after_cooldown(self):
        breaker = CircuitBreaker(threshold=1, cooldown_s=0.05)
        breaker.record_failure()
        assert breaker.is_open
        time.sleep(0.06)
        assert not breaker.is_open  # one probe allowed through


class TestContext:
    def test_passages_numbered_and_ids_withheld(self):
        context = build_context([make_retrieved("d1#s#0", "Alpha."),
                                 make_retrieved("d1#s#1", "Beta.")])
        assert "[1]" in context
        assert "[2]" in context
        # Opaque ids stay out of the prompt: a model asked to echo one will
        # approximate it, and citation validation is an exact match.
        assert "d1#s#0" not in context

    def test_prefers_expanded_text(self):
        context = build_context([make_retrieved("d1#s#0", "short", "much longer window")])
        assert "much longer window" in context

    def test_respects_char_budget(self):
        context = build_context([make_retrieved(f"c{i}", "x" * 1000) for i in range(20)],
                                max_chars=2500)
        assert len(context) < 4000


class TestCitationResolution:
    """The model cites passage numbers; the gate validates chunk ids. This is the
    seam between them, and a bug here shows up as a refused correct answer."""

    def setup_method(self):
        self.retrieved = [make_retrieved("d1#s#0", "Alpha."),
                          make_retrieved("d1#s#1", "Beta."),
                          make_retrieved("d1#s#2", "Gamma.")]

    def test_inline_brackets_from_streamed_prose(self):
        assert citations_from_prose("Fuji is a volcano [1] in Japan [3].") == ["1", "3"]

    def test_ordinals_become_chunk_ids(self):
        assert resolve_citations(["1", "3"], self.retrieved) == ["d1#s#0", "d1#s#2"]

    def test_decorated_ordinals_accepted(self):
        # Models wrap the number even when told not to; that is not invention.
        assert resolve_citations(["[2]", "passage 1"], self.retrieved) == [
            "d1#s#1", "d1#s#0",
        ]

    def test_real_chunk_ids_pass_through(self):
        assert resolve_citations(["d1#s#1"], self.retrieved) == ["d1#s#1"]

    def test_unresolvable_citation_survives_for_the_gate_to_catch(self):
        # Dropping this would hide a hallucination instead of reporting it.
        assert resolve_citations(["1102432"], self.retrieved) == ["1102432"]

    def test_out_of_range_ordinal_is_not_invented_into_a_hit(self):
        assert resolve_citations(["9"], self.retrieved) == ["9"]

    def test_duplicates_collapse(self):
        assert resolve_citations(["1", "[1]", "d1#s#0"], self.retrieved) == ["d1#s#0"]

    def test_empty(self):
        assert resolve_citations([], self.retrieved) == []


class TestParsing:
    def test_plain_json(self):
        payload = OpenRouterLLM._parse(
            '{"answer":"A corporation is an entity.","citations":["c1"],'
            '"confidence":0.9,"answer_language":"en"}'
        )
        assert payload is not None and payload.citations == ["c1"]

    def test_fenced_json(self):
        payload = OpenRouterLLM._parse(
            '```json\n{"answer":"Yes.","citations":[],"confidence":0.5,'
            '"answer_language":"hi"}\n```'
        )
        assert payload is not None and payload.answer_language is Language.HI

    def test_json_wrapped_in_prose(self):
        payload = OpenRouterLLM._parse(
            'Here you go: {"answer":"Yes.","citations":[],"confidence":0.5,'
            '"answer_language":"en"} hope that helps'
        )
        assert payload is not None and payload.answer == "Yes."

    def test_garbage_returns_none(self):
        assert OpenRouterLLM._parse("I cannot produce JSON right now") is None


class TestExtractiveFallback:
    def test_refuses_even_when_passages_exist(self):
        retrieved = [make_retrieved("d1#s#0", "Potassium is a mineral. It is in bananas.")]
        payload = OpenRouterLLM._extractive_fallback("q", retrieved, Language.EN, "down")
        assert payload.refusal_reason is RefusalReason.PROVIDER_FAILURE
        assert payload.confidence == 0.0
        assert payload.citations == []

    def test_refuses_when_nothing_retrieved(self):
        payload = OpenRouterLLM._extractive_fallback("q", [], Language.EN, "down")
        assert payload.refusal_reason is RefusalReason.PROVIDER_FAILURE
        assert payload.confidence == 0.0

    def test_open_breaker_short_circuits_without_network(self):
        # No API key and no client: proves the breaker path never calls out.
        llm = OpenRouterLLM()
        llm.breaker.record_failure()
        llm.breaker.record_failure()
        llm.breaker.record_failure()
        retrieved = [make_retrieved("d1#s#0", "Alpha beta gamma.")]
        payload, elapsed = llm.answer("q", retrieved, Language.EN)
        assert elapsed == 0.0
        assert payload.confidence < 0.5


class TestLanguageScript:
    def test_tamil_text_matches_ta(self):
        assert language_matches("மௌண்ட் ஃபுஜி ஒரு எரிமலை ஆகும்.", Language.TA)

    def test_english_does_not_match_ta(self):
        assert not language_matches(
            "Bridget Moynahan is married to Andrew Frankel.", Language.TA
        )
        assert script_ratio(
            "Bridget Moynahan is married to Andrew Frankel.", Language.TA
        ) < 0.1

    def test_english_matches_en(self):
        assert language_matches("Mt. Fuji is a composite volcano.", Language.EN)


class TestTrace:
    def test_spans_accumulate(self):
        trace = Trace(request_id="t1")
        trace.add("embed_query", 8.2, model="bge-small")
        trace.add("dense_search", 0.3, ef=64)
        assert [s.name for s in trace.spans] == ["embed_query", "dense_search"]
        assert trace.spans[0].metadata["model"] == "bge-small"

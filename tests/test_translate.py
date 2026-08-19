from __future__ import annotations

from ragoa.config import Settings
from ragoa.harness.pipeline import looks_indic
from ragoa.schemas import Language
from ragoa.stt.translate import SOURCE_CODES, SarvamTranslate, TranslateError


class FakeResponse:
    def __init__(self, status: int, payload: dict | None = None, text: str = ""):
        self.status_code = status
        self._payload = payload or {}
        self.text = text or str(payload)

    def json(self) -> dict:
        return self._payload


class FakeClient:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.response


def test_looks_indic_on_sample_taps():
    assert looks_indic("माउंट फ़ूजी किस प्रकार का पहाड़ है?")
    assert looks_indic("দর্শনের সংজ্ঞা কী?")
    assert looks_indic("என்ன வகையான மலை எம்டி ஃபுஜி?")
    assert looks_indic("किन्से कशासाठी सर्वाधिक प्रसिद्ध आहेत?")
    assert not looks_indic("what type of mountain is Mount Fuji")


def test_source_codes_cover_the_ui():
    assert set(SOURCE_CODES) == set(Language)


def test_to_english_posts_hi_in():
    client = FakeClient(FakeResponse(200, {"translated_text": "what type of mountain is Mount Fuji"}))
    translator = SarvamTranslate(Settings(sarvam_api_key="test-key"), client=client)

    english = translator.to_english("माउंट फ़ूजी किस प्रकार का पहाड़ है?", Language.HI)

    assert english == "what type of mountain is Mount Fuji"
    body = client.calls[0]["json"]
    assert body["source_language_code"] == "hi-IN"
    assert body["target_language_code"] == "en-IN"
    assert client.calls[0]["headers"]["api-subscription-key"] == "test-key"


def test_english_source_uses_auto():
    client = FakeClient(FakeResponse(200, {"translated_text": "hello"}))
    translator = SarvamTranslate(Settings(sarvam_api_key="test-key"), client=client)
    translator.to_english("नमस्ते", Language.EN)
    assert client.calls[0]["json"]["source_language_code"] == "auto"


def test_http_error_is_named():
    client = FakeClient(FakeResponse(422, text="bad lang"))
    translator = SarvamTranslate(Settings(sarvam_api_key="test-key"), client=client)
    try:
        translator.to_english("माउंट", Language.HI)
    except TranslateError as exc:
        assert "422" in str(exc)
    else:
        raise AssertionError("expected TranslateError")


def test_empty_translation_is_named():
    client = FakeClient(FakeResponse(200, {"translated_text": "  "}))
    translator = SarvamTranslate(Settings(sarvam_api_key="test-key"), client=client)
    try:
        translator.to_english("माउंट", Language.HI)
    except TranslateError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected TranslateError")

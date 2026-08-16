from __future__ import annotations

import base64

from ragoa.config import Settings
from ragoa.schemas import Language
from ragoa.tts.sarvam import LANGUAGE_CODES, SarvamTTS, TTSError


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


def test_synthesize_decodes_wav():
    wav = b"RIFF-fake-wav"
    client = FakeClient(FakeResponse(200, {"audios": [base64.b64encode(wav).decode()]}))
    tts = SarvamTTS(Settings(sarvam_api_key="test-key"), client=client)

    audio = tts.synthesize("Mount Fuji is a stratovolcano.", Language.EN)

    assert audio == wav
    body = client.calls[0]["json"]
    assert body["language_code"] == "en-IN"
    assert body["model"] == "bulbul:v3"
    assert "pitch" not in body
    assert client.calls[0]["headers"]["api-subscription-key"] == "test-key"


def test_language_codes_cover_the_ui():
    assert set(LANGUAGE_CODES) == set(Language)


def test_http_error_is_named():
    client = FakeClient(FakeResponse(422, text="bad speaker"))
    tts = SarvamTTS(Settings(sarvam_api_key="test-key"), client=client)
    try:
        tts.synthesize("hello", Language.TA)
    except TTSError as exc:
        assert "422" in str(exc)
    else:
        raise AssertionError("expected TTSError")


def test_empty_audio_list_is_named():
    client = FakeClient(FakeResponse(200, {"audios": []}))
    tts = SarvamTTS(Settings(sarvam_api_key="test-key"), client=client)
    try:
        tts.synthesize("hello", Language.HI)
    except TTSError as exc:
        assert "no audio" in str(exc)
    else:
        raise AssertionError("expected TTSError")

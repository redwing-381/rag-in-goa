"""Sarvam Bulbul text-to-speech client.

The browser voice is a fallback. This is the speak-back path the demo should use:
same `api-subscription-key` as STT, REST convert, one WAV per phrase.

`bulbul:v3` rejects pitch/loudness. Pace and temperature are the knobs that work.
"""

from __future__ import annotations

import base64

import httpx

from ragoa.config import Settings
from ragoa.schemas import Language

MAX_CHARS = 2500

LANGUAGE_CODES: dict[Language, str] = {
    Language.EN: "en-IN",
    Language.HI: "hi-IN",
    Language.BN: "bn-IN",
    Language.TA: "ta-IN",
    Language.MR: "mr-IN",
}


class TTSError(RuntimeError):
    pass


class SarvamTTS:
    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
    ):
        from ragoa.config import settings as default_settings

        self.settings = settings or default_settings
        self._client = client or httpx.Client(timeout=self.settings.sarvam_timeout_s)

    def _headers(self) -> dict[str, str]:
        if not self.settings.sarvam_api_key:
            raise TTSError("SARVAM_API_KEY is not set")
        return {
            "api-subscription-key": self.settings.sarvam_api_key,
            "Content-Type": "application/json",
        }

    def synthesize(self, text: str, language: Language) -> bytes:
        """Return a WAV body for `text` in `language`."""
        spoken = (text or "").strip()
        if not spoken:
            raise TTSError("nothing to speak")
        if len(spoken) > MAX_CHARS:
            spoken = spoken[:MAX_CHARS]

        response = self._client.post(
            self.settings.sarvam_tts_url,
            headers=self._headers(),
            json={
                "text": spoken,
                "language_code": LANGUAGE_CODES[language],
                "model": self.settings.sarvam_tts_model,
                "speaker": self.settings.sarvam_tts_speaker,
                "pace": self.settings.sarvam_tts_pace,
                "temperature": self.settings.sarvam_tts_temperature,
                "speech_sample_rate": 24000,
                "output_audio_codec": "wav",
            },
        )
        if response.status_code >= 400:
            raise TTSError(
                f"Sarvam TTS rejected the request ({response.status_code}): "
                f"{response.text[:300]}"
            )
        payload = response.json()
        audios = payload.get("audios") or []
        if not audios:
            raise TTSError("Sarvam TTS returned no audio")
        return base64.b64decode(audios[0])

    def close(self) -> None:
        self._client.close()

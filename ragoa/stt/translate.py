"""Sarvam text translate: typed Indic → English for retrieval.

Speech already arrives in English because STT runs `saaras:v3` in translate
mode. Tapped or typed questions do not. The index and the embedder are
English-only, so a Hindi sample would otherwise retrieve nothing useful and
the model would honestly say it does not know.

Auth matches STT/TTS: `api-subscription-key`, not Bearer.
"""

from __future__ import annotations

import httpx

from ragoa.config import Settings
from ragoa.schemas import Language

SOURCE_CODES: dict[Language, str] = {
    Language.EN: "en-IN",
    Language.HI: "hi-IN",
    Language.BN: "bn-IN",
    Language.TA: "ta-IN",
    Language.MR: "mr-IN",
}


class TranslateError(RuntimeError):
    pass


class SarvamTranslate:
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
            raise TranslateError("SARVAM_API_KEY is not set")
        return {
            "api-subscription-key": self.settings.sarvam_api_key,
            "Content-Type": "application/json",
        }

    def to_english(self, text: str, source: Language) -> str:
        """Translate `text` to English. `source` is a hint; EN uses auto-detect."""
        payload = {
            "input": text[:1000],
            "source_language_code": (
                "auto" if source is Language.EN else SOURCE_CODES[source]
            ),
            "target_language_code": "en-IN",
        }
        response = self._client.post(
            self.settings.sarvam_translate_url,
            headers=self._headers(),
            json=payload,
        )
        if response.status_code >= 400:
            raise TranslateError(
                f"Sarvam translate HTTP {response.status_code}: {response.text[:200]}"
            )
        body = response.json()
        translated = (body.get("translated_text") or "").strip()
        if not translated:
            raise TranslateError("Sarvam translate returned empty text")
        return translated

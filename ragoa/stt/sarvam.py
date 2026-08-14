"""Sarvam speech-to-text client.

Uses `saaras:v3` with `mode="translate"`, which turns Indic speech directly into
English text in a single call. That is the cross-lingual bridge the whole design
rests on: because the transcript arrives in English, the vector index is
English-only and needs no multilingual embedder, while the user still speaks
Hindi, Bengali, Tamil or Marathi.

Two API details that are easy to get wrong: auth is the `api-subscription-key`
header, not `Authorization: Bearer`, and the sync endpoint rejects audio longer
than 30 seconds with a 422. We check duration client-side so the user gets a clear
message instead of a raw API error.
"""

from __future__ import annotations

import io
import time
import wave

import httpx

from ragoa.config import Settings
from ragoa.schemas import STTResult

MAX_SECONDS = 30.0


class STTError(RuntimeError):
    pass


def wav_duration_seconds(audio: bytes) -> float | None:
    """Duration of a WAV payload, or None if it is not parseable WAV."""
    try:
        with wave.open(io.BytesIO(audio), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            return frames / float(rate) if rate else None
    except (wave.Error, EOFError):
        return None


class SarvamSTT:
    def __init__(self, settings: Settings | None = None,
                 client: httpx.Client | None = None):
        from ragoa.config import settings as default_settings

        self.settings = settings or default_settings
        self._client = client or httpx.Client(timeout=self.settings.sarvam_timeout_s)

    def _headers(self) -> dict[str, str]:
        if not self.settings.sarvam_api_key:
            raise STTError("SARVAM_API_KEY is not set")
        return {"api-subscription-key": self.settings.sarvam_api_key}

    def transcribe(
        self,
        audio: bytes,
        filename: str = "audio.wav",
        language_code: str | None = None,
        mode: str | None = None,
        max_retries: int = 2,
    ) -> STTResult:
        cfg = self.settings

        duration = wav_duration_seconds(audio)
        if duration is not None and duration > MAX_SECONDS:
            raise STTError(
                f"audio is {duration:.1f}s; the sync endpoint accepts at most "
                f"{MAX_SECONDS:.0f}s. Record a shorter question."
            )

        data = {"model": cfg.sarvam_stt_model, "mode": mode or cfg.sarvam_stt_mode}
        if language_code:
            data["language_code"] = language_code

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            started = time.perf_counter()
            try:
                response = self._client.post(
                    cfg.sarvam_stt_url,
                    headers=self._headers(),
                    data=data,
                    files={"file": (filename, audio, "audio/wav")},
                )
            except httpx.RequestError as exc:
                last_error = exc
                self._backoff(attempt, max_retries)
                continue

            # 429 and 5xx are worth retrying; 4xx otherwise is our mistake.
            if response.status_code == 429 or response.status_code >= 500:
                last_error = STTError(
                    f"Sarvam returned {response.status_code}: {response.text[:200]}"
                )
                self._backoff(attempt, max_retries, response)
                continue

            if response.status_code >= 400:
                raise STTError(
                    f"Sarvam rejected the request ({response.status_code}): "
                    f"{response.text[:300]}"
                )

            payload = response.json()
            return STTResult(
                text=(payload.get("transcript") or "").strip(),
                detected_language=payload.get("language_code"),
                language_probability=payload.get("language_probability"),
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

        raise STTError(f"speech-to-text failed after {max_retries + 1} attempts: {last_error}")

    def _backoff(self, attempt: int, max_retries: int,
                 response: httpx.Response | None = None) -> None:
        if attempt >= max_retries:
            return
        # Honour Retry-After when the server sends one; it knows better than we do.
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    time.sleep(min(float(retry_after), 5.0))
                    return
                except ValueError:
                    pass
        time.sleep(0.4 * (2 ** attempt))

    def close(self) -> None:
        self._client.close()

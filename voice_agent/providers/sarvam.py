"""Sarvam AI providers: Saaras for speech-to-text, Bulbul for text-to-speech.

Saaras is the right STT here because it is trained on Indian codemix -- a
Tenglish sentence like "meeting ni reschedule cheyyandi tomorrow" comes back as
one coherent transcript instead of two half-broken monolingual guesses.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Dict, List

import httpx

from ..config import Settings
from ..models import AudioClip, Language, Transcription
from .base import ProviderError, SpeechToText, TextToSpeech, register_stt, register_tts

logger = logging.getLogger(__name__)

#: Bulbul rejects longer payloads, so text is split on sentence boundaries.
TTS_CHAR_LIMIT = 480


def _retrying_post(
    url: str,
    settings: Settings,
    *,
    label: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """POST with exponential backoff on 429s and 5xx responses."""
    last_error: Exception | None = None

    for attempt in range(1, settings.max_retries + 1):
        try:
            response = httpx.post(url, timeout=settings.request_timeout, **kwargs)
            if response.status_code == 429 or response.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"retryable status {response.status_code}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            last_error = exc
            if attempt == settings.max_retries:
                break
            backoff = 2 ** (attempt - 1)
            logger.warning(
                "%s failed (attempt %d/%d): %s -- retrying in %ss",
                label,
                attempt,
                settings.max_retries,
                exc,
                backoff,
            )
            time.sleep(backoff)

    raise ProviderError(
        f"{label} failed after {settings.max_retries} attempts: {last_error}"
    ) from last_error


class SarvamSTT(SpeechToText):
    """Speech-to-text via Sarvam's Saaras speech-to-text-translate endpoint."""

    name = "sarvam"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.api_key = settings.require(
            settings.sarvam_api_key, "SARVAM_API_KEY", "sarvam"
        )
        self.endpoint = (
            f"{settings.sarvam_base_url.rstrip('/')}/speech-to-text-translate"
        )

    def transcribe(self, audio: bytes, filename: str = "audio.wav") -> Transcription:
        if not audio:
            raise ProviderError("Cannot transcribe empty audio.")

        payload = _retrying_post(
            self.endpoint,
            self.settings,
            label="Sarvam STT",
            headers={"api-subscription-key": self.api_key},
            files={"file": (filename, io.BytesIO(audio), "audio/wav")},
            data={"model": self.settings.sarvam_stt_model},
        )
        return self.parse(payload)

    @staticmethod
    def parse(payload: Dict[str, Any]) -> Transcription:
        """Normalise a Saaras response into a :class:`Transcription`."""
        text = (payload.get("transcript") or "").strip()
        return Transcription(
            text=text,
            language=Language.coerce(payload.get("language_code")),
            confidence=float(payload.get("confidence", 1.0) or 1.0),
            duration=float(payload.get("duration_seconds", 0.0) or 0.0),
        )


class SarvamTTS(TextToSpeech):
    """Text-to-speech via Sarvam's Bulbul endpoint."""

    name = "sarvam"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.api_key = settings.require(
            settings.sarvam_api_key, "SARVAM_API_KEY", "sarvam"
        )
        self.endpoint = f"{settings.sarvam_base_url.rstrip('/')}/text-to-speech"

    def synthesize(self, text: str, language: Language) -> AudioClip:
        text = (text or "").strip()
        if not text:
            raise ProviderError("Cannot synthesize empty text.")

        # Bulbul speaks one language at a time; codemix is voiced in Telugu,
        # which reads Latin-script English words acceptably.
        target = Language.TELUGU if language is Language.MIXED else language

        chunks = self.split_text(text)
        audio = bytearray()
        for chunk in chunks:
            payload = _retrying_post(
                self.endpoint,
                self.settings,
                label="Sarvam TTS",
                headers={
                    "api-subscription-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": [chunk],
                    "target_language_code": target.value,
                    "speaker": self.settings.sarvam_speaker,
                    "model": self.settings.sarvam_tts_model,
                    "pace": self.settings.tts_pace,
                    "speech_sample_rate": self.settings.tts_sample_rate,
                },
            )
            audio.extend(self.decode(payload))

        return AudioClip(
            data=bytes(audio),
            sample_rate=self.settings.tts_sample_rate,
            format="wav",
        )

    @staticmethod
    def decode(payload: Dict[str, Any]) -> bytes:
        """Extract base64 WAV bytes from a Bulbul response."""
        audios = payload.get("audios") or []
        if not audios:
            raise ProviderError("Sarvam TTS returned no audio.")
        try:
            return base64.b64decode(audios[0])
        except (ValueError, TypeError) as exc:
            raise ProviderError(f"Sarvam TTS returned undecodable audio: {exc}") from exc

    @staticmethod
    def split_text(text: str, limit: int = TTS_CHAR_LIMIT) -> List[str]:
        """Split on sentence boundaries, keeping each piece under ``limit``."""
        text = text.strip()
        if len(text) <= limit:
            return [text]

        chunks: List[str] = []
        current = ""
        # Telugu uses the danda as well as the ASCII full stop.
        for sentence in _split_sentences(text):
            if not current:
                current = sentence
            elif len(current) + len(sentence) + 1 <= limit:
                current = f"{current} {sentence}"
            else:
                chunks.append(current)
                current = sentence

            while len(current) > limit:
                chunks.append(current[:limit])
                current = current[limit:]

        if current.strip():
            chunks.append(current.strip())
        return chunks


def _split_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    buffer = ""
    for char in text:
        buffer += char
        if char in ".!?।":
            stripped = buffer.strip()
            if stripped:
                sentences.append(stripped)
            buffer = ""
    if buffer.strip():
        sentences.append(buffer.strip())
    return sentences or [text]


register_stt("sarvam", SarvamSTT)
register_tts("sarvam", SarvamTTS)

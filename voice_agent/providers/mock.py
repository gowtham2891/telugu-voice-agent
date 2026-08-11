"""Credential-free providers so the whole agent runs offline.

The mock STT picks a sample utterance deterministically from the audio bytes,
the mock LLM answers from intent-aware templates, and the mock TTS emits a real
(if uninteresting) WAV file -- so download, playback and duration all behave
exactly as they would with live providers.
"""

from __future__ import annotations

import hashlib
import io
import math
import struct
import wave
from typing import List, Tuple

from ..config import Settings
from ..models import AudioClip, Conversation, Language, Transcription
from .base import (
    LanguageModel,
    ProviderError,
    SpeechToText,
    TextToSpeech,
    register_llm,
    register_stt,
    register_tts,
)

#: Realistic Tenglish utterances, in the shape the agent actually receives.
SAMPLE_UTTERANCES: List[Tuple[str, Language]] = [
    ("Repu meeting ni 4 PM ki reschedule cheyyandi", Language.MIXED),
    ("What's the weather like in Hyderabad today?", Language.ENGLISH),
    ("నాకు ఈ రోజు టాస్క్ లిస్ట్ చెప్పండి", Language.TELUGU),
    ("Bank balance ela check cheyyali?", Language.MIXED),
    ("Set a reminder for tomorrow morning at 9", Language.ENGLISH),
    ("Naaku oka joke cheppu", Language.MIXED),
]

TEMPLATES = {
    "greeting": "Namaskaram! Nenu mee voice assistant. Meeku ela help cheyyagalanu?",
    "reminder": "Sare, reminder set chesanu: '{subject}'. Time ki notify chestanu.",
    "schedule": "Meeting ni '{subject}' ki reschedule chesanu. Attendees ki update pampanu.",
    "weather": "Ee roju Hyderabad lo 32 degrees, konchem clouds unnayi. Evening ki chinna vaana radhu.",
    "task_list": "Ee roju mee list lo mudu pathakalu unnayi: standup, code review, and the client call at 5.",
    "joke": "Programmer ki beach ki velthe enduku bore kodutundi? Endukante akkada anni 'C' matrame!",
    "farewell": "Dhanyavadalu! Malli kaavali ante pilavandi.",
    "general": "Ardham ayyindi. '{subject}' gurinchi meeku inka em kavali?",
    "unknown": "Sorry, adi sariga vinipinchaledu. Malli konchem clear ga cheppagalara?",
}


class MockSTT(SpeechToText):
    """Returns a sample utterance chosen deterministically from the audio."""

    name = "mock"

    def transcribe(self, audio: bytes, filename: str = "audio.wav") -> Transcription:
        if not audio:
            raise ProviderError("Cannot transcribe empty audio.")

        digest = hashlib.sha256(audio).digest()
        text, language = SAMPLE_UTTERANCES[digest[0] % len(SAMPLE_UTTERANCES)]
        return Transcription(
            text=text,
            language=language,
            confidence=0.95,
            duration=round(len(audio) / 32000.0, 2),
        )


class MockLLM(LanguageModel):
    """Intent-aware templated replies. Deterministic, offline, no API key."""

    name = "mock"

    def reply(
        self, conversation: Conversation, system_prompt: str, language: Language
    ) -> str:
        from ..intents import classify  # local import avoids a circular import

        user_text = conversation.last_user_text()
        if not user_text:
            return TEMPLATES["unknown"]

        intent = classify(user_text)
        template = TEMPLATES.get(intent.name, TEMPLATES["general"])
        subject = intent.slots.get("subject") or self._subject_of(user_text)
        return template.format(subject=subject)

    @staticmethod
    def _subject_of(text: str, max_words: int = 8) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text.strip()
        return " ".join(words[:max_words]) + "..."


class MockTTS(TextToSpeech):
    """Emits a valid WAV whose length scales with the text, so UIs behave."""

    name = "mock"

    def synthesize(self, text: str, language: Language) -> AudioClip:
        text = (text or "").strip()
        if not text:
            raise ProviderError("Cannot synthesize empty text.")

        sample_rate = self.settings.tts_sample_rate
        # Roughly 2.5 words per second of speech, clamped to something sane.
        seconds = min(max(len(text.split()) / 2.5, 0.5), 30.0)
        frame_count = int(sample_rate * seconds)

        # A quiet 220 Hz tone: audible, obviously synthetic, and a real WAV.
        frames = b"".join(
            struct.pack(
                "<h", int(2000 * math.sin(2 * math.pi * 220 * i / sample_rate))
            )
            for i in range(frame_count)
        )

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(frames)

        return AudioClip(
            data=buffer.getvalue(), sample_rate=sample_rate, format="wav"
        )


register_stt("mock", MockSTT)
register_llm("mock", MockLLM)
register_tts("mock", MockTTS)

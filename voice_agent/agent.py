"""The voice agent loop: audio in -> transcript -> intent -> reply -> speech."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

from .config import Settings, get_settings
from .intents import classify, detect_language
from .models import (
    AgentResponse,
    AudioClip,
    Conversation,
    Intent,
    Language,
    Transcription,
)
from .providers.base import (
    LanguageModel,
    ProviderError,
    SpeechToText,
    TextToSpeech,
    get_llm,
    get_stt,
    get_tts,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]

SYSTEM_PROMPT = """\
You are a helpful voice assistant for users in India who speak Telugu, English, \
or a natural mix of both (Tenglish).

Rules:
1. Reply in the SAME language mix the user used. If they codemix, you codemix.
2. Keep replies short -- 1 to 3 sentences. This is spoken aloud, not read.
3. Use plain words. No markdown, no bullet points, no emoji, no code blocks.
4. Write numbers, dates and times the way a person would say them.
5. If you did not understand, say so briefly and ask them to repeat.
"""

LANGUAGE_HINT = {
    Language.TELUGU: "The user spoke Telugu. Reply in Telugu.",
    Language.ENGLISH: "The user spoke English. Reply in English.",
    Language.MIXED: (
        "The user codemixed Telugu and English. Reply the same way, using "
        "romanised Telugu with English technical words."
    ),
}


def _noop(stage: str, message: str) -> None:
    logger.info("[%s] %s", stage, message)


class VoiceAgent:
    """Holds conversation state and runs one turn at a time.

    Providers are resolved once at construction, so a bad provider name fails
    immediately rather than after the user has already spoken.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        progress: Optional[ProgressCallback] = None,
        conversation: Optional[Conversation] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.progress = progress or _noop
        self.conversation = conversation or Conversation(
            max_turns=self.settings.max_history_turns
        )

        self.stt: SpeechToText = get_stt(self.settings)
        self.llm: LanguageModel = get_llm(self.settings)
        self.tts: TextToSpeech = get_tts(self.settings)

    # -- turns --------------------------------------------------------------

    def listen(self, audio: bytes, filename: str = "audio.wav") -> Transcription:
        self.progress("stt", f"Transcribing {len(audio)} bytes via {self.stt.name}")
        transcription = self.stt.transcribe(audio, filename=filename)
        self.progress(
            "stt",
            f"Heard: {transcription.text!r} ({transcription.language.label})",
        )
        return transcription

    def think(self, transcription: Transcription) -> tuple[str, Intent]:
        intent = classify(transcription.text)
        self.progress(
            "intent", f"{intent.name} (confidence {intent.confidence:.2f})"
        )

        self.conversation.add_user(
            transcription.text, language=transcription.language, intent=intent
        )

        prompt = f"{SYSTEM_PROMPT}\n{LANGUAGE_HINT[transcription.language]}"
        self.progress("llm", f"Generating a reply via {self.llm.name}")
        reply = self.llm.reply(self.conversation, prompt, transcription.language)
        self.progress("llm", f"Reply: {reply!r}")
        return reply, intent

    def speak(self, text: str, language: Language) -> Optional[AudioClip]:
        if not self.settings.enable_tts:
            self.progress("tts", "Skipped (ENABLE_TTS is off)")
            return None
        self.progress("tts", f"Synthesizing via {self.tts.name}")
        clip = self.tts.synthesize(text, language)
        self.progress("tts", f"Produced {clip.size_bytes} bytes of audio")
        return clip

    def respond(self, audio: bytes, filename: str = "audio.wav") -> AgentResponse:
        """Run one full turn: speech in, speech out."""
        started = time.monotonic()
        timings: dict[str, float] = {}

        stage_started = time.monotonic()
        transcription = self.listen(audio, filename=filename)
        timings["stt_ms"] = (time.monotonic() - stage_started) * 1000

        if transcription.is_empty:
            raise ProviderError("Transcription came back empty -- nothing to answer.")

        stage_started = time.monotonic()
        reply, intent = self.think(transcription)
        timings["llm_ms"] = (time.monotonic() - stage_started) * 1000

        stage_started = time.monotonic()
        clip = self.speak(reply, transcription.language)
        timings["tts_ms"] = (time.monotonic() - stage_started) * 1000

        elapsed_ms = (time.monotonic() - started) * 1000
        self.conversation.add_assistant(
            reply, language=transcription.language, latency_ms=elapsed_ms
        )
        self.progress("done", f"Turn completed in {elapsed_ms:.0f}ms")

        return AgentResponse(
            transcription=transcription,
            intent=intent,
            reply_text=reply,
            reply_language=transcription.language,
            audio=clip,
            latency_ms=elapsed_ms,
            stage_timings=timings,
        )

    def respond_to_text(self, text: str) -> AgentResponse:
        """Text-only turn -- useful for testing and for typed input in the UI."""
        if not text.strip():
            raise ProviderError("Cannot respond to empty text.")

        started = time.monotonic()
        transcription = Transcription(
            text=text.strip(), language=detect_language(text), confidence=1.0
        )
        reply, intent = self.think(transcription)
        clip = self.speak(reply, transcription.language)
        elapsed_ms = (time.monotonic() - started) * 1000

        self.conversation.add_assistant(
            reply, language=transcription.language, latency_ms=elapsed_ms
        )
        return AgentResponse(
            transcription=transcription,
            intent=intent,
            reply_text=reply,
            reply_language=transcription.language,
            audio=clip,
            latency_ms=elapsed_ms,
        )

    # -- helpers ------------------------------------------------------------

    def respond_to_file(self, path: Path) -> AgentResponse:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return self.respond(path.read_bytes(), filename=path.name)

    def reset(self) -> None:
        self.conversation.clear()
        self.progress("reset", "Conversation history cleared")

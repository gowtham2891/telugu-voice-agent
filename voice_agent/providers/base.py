"""Provider interfaces and registries for the three pipeline stages.

Speech-to-text, language model and text-to-speech each get their own registry,
so a live STT can be paired with a mock LLM (or any other combination) without
touching the agent code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List

from ..config import Settings
from ..models import AudioClip, Conversation, Language, Transcription


class SpeechToText(ABC):
    """Converts audio bytes into text."""

    name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def transcribe(self, audio: bytes, filename: str = "audio.wav") -> Transcription:
        """Transcribe ``audio``; raise :class:`ProviderError` on failure."""


class LanguageModel(ABC):
    """Generates the assistant's reply from conversation context."""

    name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def reply(
        self,
        conversation: Conversation,
        system_prompt: str,
        language: Language,
    ) -> str:
        """Produce the next assistant message."""


class TextToSpeech(ABC):
    """Renders text as speech audio."""

    name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def synthesize(self, text: str, language: Language) -> AudioClip:
        """Speak ``text`` in ``language``."""


class ProviderError(RuntimeError):
    """Raised when a provider fails in a way the agent cannot recover from."""


_STT: Dict[str, Callable[[Settings], SpeechToText]] = {}
_LLM: Dict[str, Callable[[Settings], LanguageModel]] = {}
_TTS: Dict[str, Callable[[Settings], TextToSpeech]] = {}


def register_stt(name: str, factory: Callable[[Settings], SpeechToText]) -> None:
    _STT[name] = factory


def register_llm(name: str, factory: Callable[[Settings], LanguageModel]) -> None:
    _LLM[name] = factory


def register_tts(name: str, factory: Callable[[Settings], TextToSpeech]) -> None:
    _TTS[name] = factory


def available_stt() -> List[str]:
    return sorted(_STT)


def available_llm() -> List[str]:
    return sorted(_LLM)


def available_tts() -> List[str]:
    return sorted(_TTS)


def _resolve(registry: Dict[str, Callable], name: str, kind: str, settings: Settings):
    try:
        factory = registry[name]
    except KeyError:
        raise ValueError(
            f"Unknown {kind} provider '{name}'. Available: {', '.join(sorted(registry))}"
        ) from None
    return factory(settings)


def get_stt(settings: Settings) -> SpeechToText:
    return _resolve(_STT, settings.stt, "STT", settings)


def get_llm(settings: Settings) -> LanguageModel:
    return _resolve(_LLM, settings.llm, "LLM", settings)


def get_tts(settings: Settings) -> TextToSpeech:
    return _resolve(_TTS, settings.tts, "TTS", settings)

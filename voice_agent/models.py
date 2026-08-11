"""Domain models for the voice agent: turns, transcripts, audio and intents."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Language(str, Enum):
    """Languages the agent speaks. ``MIXED`` is Tenglish codemix."""

    TELUGU = "te-IN"
    ENGLISH = "en-IN"
    MIXED = "mixed"

    @property
    def label(self) -> str:
        return {"te-IN": "Telugu", "en-IN": "English", "mixed": "Telugu + English"}[
            self.value
        ]

    @classmethod
    def coerce(cls, value: Optional[str]) -> "Language":
        """Map assorted provider language codes onto our three buckets."""
        if not value:
            return cls.MIXED
        normalized = value.strip().lower()
        if normalized.startswith("te"):
            return cls.TELUGU
        if normalized.startswith("en"):
            return cls.ENGLISH
        return cls.MIXED


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AudioClip(BaseModel):
    """Raw audio plus enough metadata to write a valid WAV file."""

    data: bytes = Field(..., description="Encoded audio bytes")
    sample_rate: int = Field(default=22050, gt=0)
    format: str = Field(default="wav")

    model_config = {"arbitrary_types_allowed": True}

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    @property
    def is_empty(self) -> bool:
        return not self.data


class Transcription(BaseModel):
    """What the STT provider heard."""

    text: str = Field(default="")
    language: Language = Field(default=Language.MIXED)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    duration: float = Field(default=0.0, ge=0.0)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class Intent(BaseModel):
    """A classified user intent plus whatever slots were filled."""

    name: str = Field(default="general")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    slots: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.5


class Turn(BaseModel):
    """One message in the conversation."""

    role: Role
    text: str
    language: Language = Field(default=Language.MIXED)
    intent: Optional[Intent] = None
    latency_ms: float = Field(default=0.0, ge=0.0)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("turn text must not be blank")
        return value


class Conversation(BaseModel):
    """Rolling conversation state with a bounded history window."""

    turns: List[Turn] = Field(default_factory=list)
    max_turns: int = Field(default=20, gt=0)

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)
        # Trim oldest turns in pairs so the history never starts mid-exchange.
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def add_user(self, text: str, language: Language = Language.MIXED,
                 intent: Optional[Intent] = None) -> Turn:
        turn = Turn(role=Role.USER, text=text, language=language, intent=intent)
        self.add(turn)
        return turn

    def add_assistant(self, text: str, language: Language = Language.MIXED,
                      latency_ms: float = 0.0) -> Turn:
        turn = Turn(
            role=Role.ASSISTANT, text=text, language=language, latency_ms=latency_ms
        )
        self.add(turn)
        return turn

    def to_messages(self) -> List[Dict[str, str]]:
        """Render as OpenAI-style ``{role, content}`` dicts for the LLM."""
        return [{"role": turn.role.value, "content": turn.text} for turn in self.turns]

    def last_user_text(self) -> str:
        for turn in reversed(self.turns):
            if turn.role is Role.USER:
                return turn.text
        return ""

    def clear(self) -> None:
        self.turns.clear()

    @property
    def turn_count(self) -> int:
        return len(self.turns)


class AgentResponse(BaseModel):
    """Everything one turn of the agent produced."""

    transcription: Transcription
    intent: Intent
    reply_text: str
    reply_language: Language = Field(default=Language.MIXED)
    audio: Optional[AudioClip] = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    stage_timings: Dict[str, float] = Field(default_factory=dict)

    @property
    def has_audio(self) -> bool:
        return self.audio is not None and not self.audio.is_empty

"""Telugu/English Voice AI Agent.

A real-time voice agent for codemixed Tenglish speech: Sarvam Saaras handles
speech input, an LLM processes intent, and Sarvam Bulbul speaks the reply.

Importing this package registers every built-in provider.
"""

from __future__ import annotations

from .agent import SYSTEM_PROMPT, VoiceAgent
from .config import ConfigError, Settings, get_settings
from .health import (
    CheckResult,
    MissingCredential,
    check_settings,
    credentials_ready,
    missing_credentials,
)
from .intents import classify, is_codemixed
from .models import (
    AgentResponse,
    AudioClip,
    Conversation,
    Intent,
    Language,
    Role,
    Transcription,
    Turn,
)
from .providers.base import ProviderError

# Importing the provider modules populates the registries.
from .providers import mock as _mock  # noqa: F401
from .providers import sarvam as _sarvam  # noqa: F401
from .providers import llm as _llm  # noqa: F401

__version__ = "1.0.0"

__all__ = [
    "AgentResponse",
    "AudioClip",
    "CheckResult",
    "ConfigError",
    "Conversation",
    "Intent",
    "Language",
    "MissingCredential",
    "ProviderError",
    "Role",
    "SYSTEM_PROMPT",
    "Settings",
    "Transcription",
    "Turn",
    "VoiceAgent",
    "check_settings",
    "classify",
    "credentials_ready",
    "get_settings",
    "is_codemixed",
    "missing_credentials",
    "__version__",
]

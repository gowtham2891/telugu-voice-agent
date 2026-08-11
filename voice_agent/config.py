"""Environment-driven configuration for the voice agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

MOCK_PROVIDERS = {"mock"}


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


class ConfigError(RuntimeError):
    """Raised when a live provider is selected without its credentials."""


@dataclass
class Settings:
    """Resolved runtime settings for one agent session."""

    # --- provider selection -------------------------------------------------
    stt: str = field(default_factory=lambda: _env("STT_PROVIDER", "mock"))
    llm: str = field(default_factory=lambda: _env("LLM_PROVIDER", "mock"))
    tts: str = field(default_factory=lambda: _env("TTS_PROVIDER", "mock"))

    # --- Sarvam AI ----------------------------------------------------------
    sarvam_api_key: str = field(default_factory=lambda: _env("SARVAM_API_KEY"))
    sarvam_base_url: str = field(
        default_factory=lambda: _env("SARVAM_BASE_URL", "https://api.sarvam.ai")
    )
    sarvam_stt_model: str = field(
        default_factory=lambda: _env("SARVAM_STT_MODEL", "saaras:v2.5")
    )
    sarvam_tts_model: str = field(
        default_factory=lambda: _env("SARVAM_TTS_MODEL", "bulbul:v2")
    )
    sarvam_speaker: str = field(default_factory=lambda: _env("SARVAM_SPEAKER", "anushka"))
    tts_pace: float = field(default_factory=lambda: _env_float("TTS_PACE", 1.0))
    tts_sample_rate: int = field(
        default_factory=lambda: _env_int("TTS_SAMPLE_RATE", 22050)
    )

    # --- LLM ----------------------------------------------------------------
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    gemini_model: str = field(
        default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.0-flash")
    )
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.7))
    max_tokens: int = field(default_factory=lambda: _env_int("MAX_TOKENS", 300))

    # --- behaviour ----------------------------------------------------------
    default_language: str = field(
        default_factory=lambda: _env("DEFAULT_LANGUAGE", "mixed")
    )
    max_history_turns: int = field(
        default_factory=lambda: _env_int("MAX_HISTORY_TURNS", 20)
    )
    request_timeout: float = field(
        default_factory=lambda: _env_float("REQUEST_TIMEOUT", 60.0)
    )
    max_retries: int = field(default_factory=lambda: _env_int("MAX_RETRIES", 3))
    enable_tts: bool = field(
        default_factory=lambda: _env("ENABLE_TTS", "true").lower()
        in {"1", "true", "yes", "on"}
    )
    output_dir: str = field(default_factory=lambda: _env("OUTPUT_DIR", "output"))

    @property
    def is_mock(self) -> bool:
        return (
            self.stt in MOCK_PROVIDERS
            and self.llm in MOCK_PROVIDERS
            and self.tts in MOCK_PROVIDERS
        )

    def require(self, value: str, name: str, provider: str) -> str:
        if not value:
            raise ConfigError(
                f"{name} is required for the '{provider}' provider. "
                f"Set it in your .env file or switch to the 'mock' provider."
            )
        return value

    def describe(self) -> str:
        mode = "mock (no credentials needed)" if self.is_mock else "live"
        return f"mode={mode} stt={self.stt} llm={self.llm} tts={self.tts}"


_settings: Optional[Settings] = None


def get_settings(refresh: bool = False) -> Settings:
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings

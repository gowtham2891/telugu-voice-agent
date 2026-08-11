"""Live credential checks.

Each check makes the smallest authenticated request a provider supports and
classifies the response, so someone pasting a key into the UI finds out whether
it works *before* spending a full transcription run discovering that it doesn't.

Checks are single-attempt with a short timeout: this is a "does this key work
right now" probe, not a resilient production call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

#: Short on purpose -- a credential probe should fail fast, not hang a UI.
CHECK_TIMEOUT = 20.0

OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"


@dataclass
class CheckResult:
    """The verdict on one credential."""

    provider: str
    ok: bool
    message: str

    def __str__(self) -> str:
        return f"{'OK' if self.ok else 'FAILED'} {self.provider}: {self.message}"


def _missing(provider: str) -> CheckResult:
    return CheckResult(provider, False, "No key provided.")


def _network_error(provider: str, exc: Exception) -> CheckResult:
    return CheckResult(provider, False, f"Could not reach the API: {exc}")


def _error_message(response: httpx.Response) -> str:
    """Pull a human-readable reason out of a JSON error body."""
    try:
        payload: Dict[str, Any] = response.json()
    except ValueError:
        return response.text[:200].strip() or f"HTTP {response.status_code}"

    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "").strip() or f"HTTP {response.status_code}"
    if isinstance(error, str):
        return error
    for key in ("message", "detail", "error_message"):
        if payload.get(key):
            return str(payload[key])
    return f"HTTP {response.status_code}"


def check_sarvam(
    api_key: str,
    base_url: str = "https://api.sarvam.ai",
    model: str = "bulbul:v2",
    speaker: str = "anushka",
    timeout: float = CHECK_TIMEOUT,
) -> CheckResult:
    """Verify a Sarvam key by synthesizing a single word.

    Sarvam exposes no unauthenticated ping. Text-to-speech is used rather than
    STT because the same key covers both, and a one-word synthesis is the
    cheapest request that proves the key reaches Bulbul.
    """
    provider = "Sarvam AI"
    if not (api_key or "").strip():
        return _missing(provider)

    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/text-to-speech",
            headers={
                "api-subscription-key": api_key.strip(),
                "Content-Type": "application/json",
            },
            json={
                "inputs": ["test"],
                "target_language_code": "te-IN",
                "speaker": speaker,
                "model": model,
            },
            timeout=timeout,
        )
    except httpx.TransportError as exc:
        return _network_error(provider, exc)

    if response.status_code in (200, 201):
        return CheckResult(provider, True, "Key works — speech synthesis succeeded.")
    if response.status_code in (401, 403):
        return CheckResult(provider, False, f"Key rejected: {_error_message(response)}")
    if response.status_code == 429:
        return CheckResult(
            provider, False, "Rate limited — the key is valid but throttled."
        )
    return CheckResult(
        provider, False, f"HTTP {response.status_code}: {_error_message(response)}"
    )


def check_openai(
    api_key: str, model: str = "", timeout: float = CHECK_TIMEOUT
) -> CheckResult:
    """Verify an OpenAI key by listing models — consumes no tokens."""
    provider = "OpenAI"
    if not (api_key or "").strip():
        return _missing(provider)

    try:
        response = httpx.get(
            OPENAI_MODELS_URL,
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            timeout=timeout,
        )
    except httpx.TransportError as exc:
        return _network_error(provider, exc)

    if response.status_code == 200:
        names = _openai_model_names(response)
        if model and model not in names:
            return CheckResult(
                provider, True,
                f"Key works, but '{model}' is not in the {len(names)} models this "
                f"key can reach. Pick another model or check your plan.",
            )
        return CheckResult(provider, True, f"Key works — {len(names)} models available.")
    if response.status_code == 401:
        return CheckResult(provider, False, f"Key rejected: {_error_message(response)}")
    if response.status_code == 429:
        return CheckResult(
            provider, False,
            "Rate limited or out of quota — check billing on the OpenAI dashboard.",
        )
    return CheckResult(
        provider, False, f"HTTP {response.status_code}: {_error_message(response)}"
    )


def _openai_model_names(response: httpx.Response) -> List[str]:
    try:
        payload = response.json()
    except ValueError:
        return []
    return [
        str(entry.get("id", ""))
        for entry in payload.get("data", [])
        if isinstance(entry, dict)
    ]


def check_gemini(
    api_key: str, model: str = "", timeout: float = CHECK_TIMEOUT
) -> CheckResult:
    """Verify a Gemini key by listing models — consumes no tokens."""
    provider = "Gemini"
    if not (api_key or "").strip():
        return _missing(provider)

    try:
        response = httpx.get(
            GEMINI_MODELS_URL, params={"key": api_key.strip()}, timeout=timeout
        )
    except httpx.TransportError as exc:
        return _network_error(provider, exc)

    if response.status_code == 200:
        names = _gemini_model_names(response)
        if model and not any(model.split("/")[-1].lower() == n.lower() for n in names):
            return CheckResult(
                provider, True,
                f"Key works, but '{model}' was not among the {len(names)} available "
                f"models.",
            )
        return CheckResult(provider, True, f"Key works — {len(names)} models available.")
    if response.status_code in (400, 401, 403):
        return CheckResult(provider, False, f"Key rejected: {_error_message(response)}")
    if response.status_code == 429:
        return CheckResult(
            provider, False, "Rate limited — the key is valid but throttled."
        )
    return CheckResult(
        provider, False, f"HTTP {response.status_code}: {_error_message(response)}"
    )


def _gemini_model_names(response: httpx.Response) -> List[str]:
    try:
        payload = response.json()
    except ValueError:
        return []
    return [
        str(entry.get("name", "")).split("/")[-1]
        for entry in payload.get("models", [])
        if isinstance(entry, dict)
    ]


def check_settings(settings, only: Optional[str] = None) -> List[CheckResult]:
    """Check every credential the current provider selection actually needs.

    STT and TTS share one Sarvam key, so it is probed once and reported once
    rather than making the same request twice.
    """
    results: List[CheckResult] = []
    timeout = min(settings.request_timeout, CHECK_TIMEOUT)

    speech_providers = {settings.stt, settings.tts}
    if only in (None, "speech"):
        if speech_providers == {"mock"}:
            results.append(
                CheckResult("Speech (mock)", True, "No credentials required.")
            )
        elif "sarvam" in speech_providers:
            results.append(
                check_sarvam(
                    settings.sarvam_api_key,
                    settings.sarvam_base_url,
                    settings.sarvam_tts_model,
                    settings.sarvam_speaker,
                    timeout=timeout,
                )
            )

    if only in (None, "llm"):
        if settings.llm == "mock":
            results.append(CheckResult("LLM (mock)", True, "No credentials required."))
        elif settings.llm == "openai":
            results.append(
                check_openai(settings.openai_api_key, settings.openai_model, timeout=timeout)
            )
        elif settings.llm == "gemini":
            results.append(
                check_gemini(settings.gemini_api_key, settings.gemini_model, timeout=timeout)
            )

    return results

"""LLM providers for reply generation: OpenAI and Gemini.

SDKs are imported lazily so the base install stays small and the offline test
suite never needs either package present.
"""

from __future__ import annotations

import logging

from ..config import Settings
from ..models import Conversation, Language
from .base import LanguageModel, ProviderError, register_llm

logger = logging.getLogger(__name__)


class OpenAIModel(LanguageModel):
    """Chat completions through the official OpenAI SDK."""

    name = "openai"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.api_key = settings.require(
            settings.openai_api_key, "OPENAI_API_KEY", "openai"
        )
        self._client = None

    def _build_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ProviderError(
                    "openai is not installed. Run: pip install 'telugu-voice-agent[openai]'"
                ) from exc
            self._client = OpenAI(
                api_key=self.api_key, timeout=self.settings.request_timeout
            )
        return self._client

    def reply(
        self, conversation: Conversation, system_prompt: str, language: Language
    ) -> str:
        client = self._build_client()
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation.to_messages())

        try:
            response = client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - SDK raises many error types
            raise ProviderError(f"OpenAI request failed: {exc}") from exc

        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise ProviderError("OpenAI returned an empty reply.")
        return text


class GeminiModel(LanguageModel):
    """Reply generation through the Google Gemini SDK."""

    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.api_key = settings.require(
            settings.gemini_api_key, "GEMINI_API_KEY", "gemini"
        )
        self._client = None

    def _build_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ProviderError(
                    "google-genai is not installed. "
                    "Run: pip install 'telugu-voice-agent[gemini]'"
                ) from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def reply(
        self, conversation: Conversation, system_prompt: str, language: Language
    ) -> str:
        client = self._build_client()

        try:
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ProviderError("google-genai is not installed.") from exc

        contents = [
            types.Content(
                role="user" if message["role"] == "user" else "model",
                parts=[types.Part(text=message["content"])],
            )
            for message in conversation.to_messages()
        ]

        try:
            response = client.models.generate_content(
                model=self.settings.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.settings.temperature,
                    max_output_tokens=self.settings.max_tokens,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - SDK raises many error types
            raise ProviderError(f"Gemini request failed: {exc}") from exc

        text = (getattr(response, "text", "") or "").strip()
        if not text:
            raise ProviderError("Gemini returned an empty reply.")
        return text


register_llm("openai", OpenAIModel)
register_llm("gemini", GeminiModel)

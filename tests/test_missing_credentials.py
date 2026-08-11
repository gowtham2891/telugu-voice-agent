"""Tests for the missing-credential detector. Never touches the network."""

from __future__ import annotations

import httpx
import pytest

from voice_agent.config import Settings
from voice_agent.health import (
    MissingCredential,
    credentials_ready,
    missing_credentials,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """The detector must be pure inspection: any request here is a bug."""

    def guard(*args, **kwargs):
        raise AssertionError("missing_credentials must not make network calls")

    monkeypatch.setattr(httpx, "get", guard)
    monkeypatch.setattr(httpx, "post", guard)


class TestMissingCredential:
    def test_str_includes_the_variable_name(self):
        item = MissingCredential("A Key", "A_KEY", "the thing")
        assert "A_KEY" in str(item)
        assert "the thing" in str(item)


class TestDetector:
    def test_mock_mode_needs_nothing(self):
        settings = Settings(stt="mock", llm="mock", tts="mock")
        assert missing_credentials(settings) == []
        assert credentials_ready(settings)

    def test_sarvam_reported_once_even_when_both_stages_use_it(self):
        settings = Settings(stt="sarvam", tts="sarvam", llm="mock", sarvam_api_key="")
        gaps = missing_credentials(settings)
        assert [item.env_var for item in gaps] == ["SARVAM_API_KEY"]

    def test_gap_names_both_stages_that_need_it(self):
        settings = Settings(stt="sarvam", tts="sarvam", llm="mock", sarvam_api_key="")
        needed_for = missing_credentials(settings)[0].needed_for
        assert "speech-to-text" in needed_for and "text-to-speech" in needed_for

    def test_gap_names_only_the_stage_using_sarvam(self):
        settings = Settings(stt="mock", tts="sarvam", llm="mock", sarvam_api_key="")
        needed_for = missing_credentials(settings)[0].needed_for
        assert needed_for == "text-to-speech"

    def test_sarvam_with_a_key_is_satisfied(self):
        settings = Settings(stt="sarvam", tts="sarvam", llm="mock", sarvam_api_key="k")
        assert credentials_ready(settings)

    @pytest.mark.parametrize("llm,var", [("openai", "OPENAI_API_KEY"),
                                         ("gemini", "GEMINI_API_KEY")])
    def test_each_llm_reports_its_own_key(self, llm, var):
        settings = Settings(stt="mock", tts="mock", llm=llm)
        assert [i.env_var for i in missing_credentials(settings)] == [var]

    def test_every_gap_says_where_to_get_the_key(self):
        settings = Settings(stt="sarvam", tts="sarvam", llm="openai")
        assert all(item.get_it_at for item in missing_credentials(settings))

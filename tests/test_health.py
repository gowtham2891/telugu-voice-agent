"""Tests for the credential-check module. No network is ever touched."""

from __future__ import annotations

import httpx
import pytest

from voice_agent.config import Settings
from voice_agent.health import (
    CheckResult,
    check_gemini,
    check_openai,
    check_sarvam,
    check_settings,
)


def fake_response(status: int, payload=None, text: str = "") -> httpx.Response:
    request = httpx.Request("POST", "https://example.com")
    if payload is not None:
        return httpx.Response(status, json=payload, request=request)
    return httpx.Response(status, text=text, request=request)


@pytest.fixture
def no_network(monkeypatch):
    def guard(*args, **kwargs):
        raise AssertionError("unexpected network call")

    monkeypatch.setattr(httpx, "get", guard)
    monkeypatch.setattr(httpx, "post", guard)


class TestCheckResult:
    def test_str_ok(self):
        assert str(CheckResult("X", True, "fine")) == "OK X: fine"

    def test_str_failed(self):
        assert str(CheckResult("X", False, "bad")) == "FAILED X: bad"


class TestSarvamCheck:
    def test_blank_key_makes_no_request(self, no_network):
        assert not check_sarvam("  ").ok

    def test_success(self, monkeypatch):
        monkeypatch.setattr(
            httpx, "post", lambda *a, **k: fake_response(200, {"audios": ["ZmFrZQ=="]})
        )
        result = check_sarvam("valid-key")
        assert result.ok
        assert "synthesis" in result.message

    def test_probes_the_tts_endpoint(self, monkeypatch):
        """This app needs the key to reach Bulbul, so TTS is what gets probed."""
        seen = {}

        def capture(url, headers=None, json=None, timeout=None):
            seen["url"] = url
            seen["headers"] = headers or {}
            seen["json"] = json or {}
            return fake_response(200, {"audios": []})

        monkeypatch.setattr(httpx, "post", capture)
        check_sarvam("  my-key  ")
        assert seen["url"].endswith("/text-to-speech")
        assert seen["headers"]["api-subscription-key"] == "my-key"
        assert seen["json"]["inputs"] == ["test"]

    @pytest.mark.parametrize("status", [401, 403])
    def test_rejected_key(self, monkeypatch, status):
        monkeypatch.setattr(
            httpx, "post",
            lambda *a, **k: fake_response(status, {"error": {"message": "invalid"}}),
        )
        assert not check_sarvam("bad-key").ok

    def test_rate_limited(self, monkeypatch):
        monkeypatch.setattr(httpx, "post", lambda *a, **k: fake_response(429, {}))
        result = check_sarvam("valid-key")
        assert not result.ok
        assert "throttled" in result.message

    def test_network_failure(self, monkeypatch):
        def boom(*args, **kwargs):
            raise httpx.ReadTimeout("timed out")

        monkeypatch.setattr(httpx, "post", boom)
        result = check_sarvam("valid-key")
        assert not result.ok
        assert "Could not reach" in result.message


class TestOpenAICheck:
    def test_blank_key(self, no_network):
        assert not check_openai("").ok

    def test_success(self, monkeypatch):
        payload = {"data": [{"id": "gpt-4o-mini"}]}
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(200, payload))
        assert check_openai("valid-key").ok

    def test_bearer_header(self, monkeypatch):
        seen = {}

        def capture(url, headers=None, timeout=None):
            seen.update(headers or {})
            return fake_response(200, {"data": []})

        monkeypatch.setattr(httpx, "get", capture)
        check_openai("secret")
        assert seen["Authorization"] == "Bearer secret"

    def test_unauthorized(self, monkeypatch):
        monkeypatch.setattr(
            httpx, "get",
            lambda *a, **k: fake_response(401, {"error": {"message": "bad key"}}),
        )
        assert not check_openai("bad").ok


class TestGeminiCheck:
    def test_blank_key(self, no_network):
        assert not check_gemini("").ok

    def test_success(self, monkeypatch):
        payload = {"models": [{"name": "models/gemini-2.0-flash"}]}
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(200, payload))
        assert check_gemini("valid-key").ok

    def test_rejected(self, monkeypatch):
        monkeypatch.setattr(
            httpx, "get",
            lambda *a, **k: fake_response(400, {"error": {"message": "not valid"}}),
        )
        assert not check_gemini("bad").ok


class TestCheckSettings:
    def test_mock_mode_needs_no_network(self, no_network):
        results = check_settings(Settings(stt="mock", llm="mock", tts="mock"))
        assert len(results) == 2
        assert all(result.ok for result in results)

    def test_one_sarvam_check_covers_stt_and_tts(self, monkeypatch):
        """The same key serves both stages, so it must be probed only once."""
        calls = {"n": 0}

        def capture(*args, **kwargs):
            calls["n"] += 1
            return fake_response(200, {"audios": []})

        monkeypatch.setattr(httpx, "post", capture)
        results = check_settings(
            Settings(stt="sarvam", tts="sarvam", llm="mock", sarvam_api_key="k")
        )
        assert calls["n"] == 1
        assert sum(1 for r in results if r.provider == "Sarvam AI") == 1

    def test_sarvam_checked_when_only_tts_uses_it(self, monkeypatch):
        monkeypatch.setattr(httpx, "post", lambda *a, **k: fake_response(200, {"audios": []}))
        results = check_settings(
            Settings(stt="mock", tts="sarvam", llm="mock", sarvam_api_key="k")
        )
        assert any(r.provider == "Sarvam AI" for r in results)

    def test_llm_is_checked(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(200, {"data": []}))
        results = check_settings(
            Settings(stt="mock", tts="mock", llm="openai", openai_api_key="k")
        )
        assert any(r.provider == "OpenAI" for r in results)

    def test_only_filter(self, no_network):
        results = check_settings(
            Settings(stt="mock", llm="mock", tts="mock"), only="llm"
        )
        assert len(results) == 1

    def test_missing_key_fails(self, no_network):
        results = check_settings(
            Settings(stt="sarvam", tts="sarvam", llm="mock", sarvam_api_key="")
        )
        assert not results[0].ok

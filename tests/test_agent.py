"""Tests for models, providers and the agent loop."""

from __future__ import annotations

import base64
import io
import wave

import httpx
import pytest
from pydantic import ValidationError

from voice_agent.agent import VoiceAgent
from voice_agent.config import ConfigError, Settings
from voice_agent.models import (
    AudioClip,
    Conversation,
    Intent,
    Language,
    Role,
    Transcription,
    Turn,
)
from voice_agent.providers.base import (
    ProviderError,
    available_llm,
    available_stt,
    available_tts,
    get_llm,
    get_stt,
    get_tts,
)
from voice_agent.providers.mock import MockLLM, MockSTT, MockTTS
from voice_agent.providers.sarvam import SarvamSTT, SarvamTTS


@pytest.fixture
def settings() -> Settings:
    return Settings(stt="mock", llm="mock", tts="mock")


@pytest.fixture
def audio_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x01" * 8000)
    return buffer.getvalue()


class TestLanguage:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("te-IN", Language.TELUGU),
            ("te", Language.TELUGU),
            ("en-IN", Language.ENGLISH),
            ("en", Language.ENGLISH),
            ("hi-IN", Language.MIXED),
            (None, Language.MIXED),
            ("", Language.MIXED),
        ],
    )
    def test_coerce(self, raw, expected):
        assert Language.coerce(raw) is expected

    def test_labels_are_human_readable(self):
        assert Language.TELUGU.label == "Telugu"
        assert Language.MIXED.label == "Telugu + English"


class TestConversation:
    def test_add_user_and_assistant(self):
        conversation = Conversation()
        conversation.add_user("Hello")
        conversation.add_assistant("Namaskaram")
        assert conversation.turn_count == 2
        assert conversation.turns[0].role is Role.USER

    def test_history_is_bounded(self):
        conversation = Conversation(max_turns=4)
        for index in range(10):
            conversation.add_user(f"message {index}")
        assert conversation.turn_count == 4
        assert conversation.turns[-1].text == "message 9"

    def test_to_messages_shape(self):
        conversation = Conversation()
        conversation.add_user("Hi")
        conversation.add_assistant("Hello")
        assert conversation.to_messages() == [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

    def test_last_user_text_skips_assistant_turns(self):
        conversation = Conversation()
        conversation.add_user("first")
        conversation.add_assistant("reply")
        conversation.add_user("second")
        conversation.add_assistant("reply two")
        assert conversation.last_user_text() == "second"

    def test_last_user_text_on_empty_history(self):
        assert Conversation().last_user_text() == ""

    def test_clear(self):
        conversation = Conversation()
        conversation.add_user("Hi")
        conversation.clear()
        assert conversation.turn_count == 0

    def test_blank_turn_is_rejected(self):
        with pytest.raises(ValidationError):
            Turn(role=Role.USER, text="   ")


class TestAudioClip:
    def test_reports_size(self):
        assert AudioClip(data=b"1234").size_bytes == 4

    def test_empty_detection(self):
        assert AudioClip(data=b"").is_empty
        assert not AudioClip(data=b"x").is_empty

    def test_sample_rate_must_be_positive(self):
        with pytest.raises(ValidationError):
            AudioClip(data=b"x", sample_rate=0)


class TestRegistries:
    def test_mock_is_registered_for_every_stage(self):
        assert "mock" in available_stt()
        assert "mock" in available_llm()
        assert "mock" in available_tts()

    def test_sarvam_is_registered_for_speech(self):
        assert "sarvam" in available_stt()
        assert "sarvam" in available_tts()

    def test_llm_providers(self):
        assert {"mock", "openai", "gemini"} <= set(available_llm())

    def test_resolution(self, settings):
        assert isinstance(get_stt(settings), MockSTT)
        assert isinstance(get_llm(settings), MockLLM)
        assert isinstance(get_tts(settings), MockTTS)

    @pytest.mark.parametrize("field,kind", [("stt", "STT"), ("llm", "LLM"), ("tts", "TTS")])
    def test_unknown_provider_raises(self, settings, field, kind):
        setattr(settings, field, "nonexistent")
        getter = {"stt": get_stt, "llm": get_llm, "tts": get_tts}[field]
        with pytest.raises(ValueError, match=f"Unknown {kind} provider"):
            getter(settings)


class TestMockProviders:
    def test_stt_is_deterministic(self, settings, audio_bytes):
        provider = MockSTT(settings)
        assert provider.transcribe(audio_bytes) == provider.transcribe(audio_bytes)

    def test_stt_rejects_empty_audio(self, settings):
        with pytest.raises(ProviderError, match="empty audio"):
            MockSTT(settings).transcribe(b"")

    def test_stt_returns_a_known_sample(self, settings, audio_bytes):
        from voice_agent.providers.mock import SAMPLE_UTTERANCES

        result = MockSTT(settings).transcribe(audio_bytes)
        assert result.text in {text for text, _ in SAMPLE_UTTERANCES}

    def test_llm_answers_by_intent(self, settings):
        conversation = Conversation()
        conversation.add_user("Naaku oka joke cheppu")
        reply = MockLLM(settings).reply(conversation, "system", Language.MIXED)
        assert "Programmer" in reply

    def test_llm_handles_empty_history(self, settings):
        reply = MockLLM(settings).reply(Conversation(), "system", Language.MIXED)
        assert reply

    def test_tts_emits_a_valid_wav(self, settings):
        clip = MockTTS(settings).synthesize("Namaskaram andi", Language.MIXED)
        with wave.open(io.BytesIO(clip.data)) as handle:
            assert handle.getnchannels() == 1
            assert handle.getframerate() == settings.tts_sample_rate
            assert handle.getnframes() > 0

    def test_tts_length_scales_with_text(self, settings):
        provider = MockTTS(settings)
        short = provider.synthesize("Hi", Language.ENGLISH)
        long = provider.synthesize(" ".join(["word"] * 40), Language.ENGLISH)
        assert long.size_bytes > short.size_bytes

    def test_tts_rejects_empty_text(self, settings):
        with pytest.raises(ProviderError, match="empty text"):
            MockTTS(settings).synthesize("  ", Language.MIXED)


class TestSarvamProviders:
    def test_stt_requires_a_key(self):
        with pytest.raises(ConfigError, match="SARVAM_API_KEY"):
            SarvamSTT(Settings(stt="sarvam", sarvam_api_key=""))

    def test_tts_requires_a_key(self):
        with pytest.raises(ConfigError, match="SARVAM_API_KEY"):
            SarvamTTS(Settings(tts="sarvam", sarvam_api_key=""))

    def test_stt_parses_a_response(self):
        result = SarvamSTT.parse(
            {
                "transcript": "  Repu meeting ki వెళ్దాం  ",
                "language_code": "te-IN",
                "duration_seconds": 3.5,
            }
        )
        assert result.text == "Repu meeting ki వెళ్దాం"
        assert result.language is Language.TELUGU
        assert result.duration == 3.5

    def test_stt_parses_an_empty_response(self):
        result = SarvamSTT.parse({})
        assert result.is_empty
        assert result.language is Language.MIXED

    def test_stt_rejects_empty_audio(self):
        provider = SarvamSTT(Settings(sarvam_api_key="test-key"))
        with pytest.raises(ProviderError, match="empty audio"):
            provider.transcribe(b"")

    def test_tts_decodes_base64_audio(self):
        payload = {"audios": [base64.b64encode(b"RIFFfake").decode()]}
        assert SarvamTTS.decode(payload) == b"RIFFfake"

    def test_tts_decode_rejects_an_empty_payload(self):
        with pytest.raises(ProviderError, match="no audio"):
            SarvamTTS.decode({"audios": []})

    def test_tts_decode_rejects_bad_base64(self):
        with pytest.raises(ProviderError, match="undecodable"):
            SarvamTTS.decode({"audios": ["not!valid!base64!!"]})

    def test_text_splitting_respects_the_limit(self):
        text = " ".join(f"Sentence number {i} here." for i in range(80))
        chunks = SarvamTTS.split_text(text, limit=200)
        assert len(chunks) > 1
        assert all(len(chunk) <= 200 for chunk in chunks)

    def test_short_text_is_one_chunk(self):
        assert SarvamTTS.split_text("Namaskaram") == ["Namaskaram"]

    def test_a_single_giant_sentence_is_hard_split(self):
        chunks = SarvamTTS.split_text("x" * 1000, limit=100)
        assert len(chunks) >= 10
        assert all(len(chunk) <= 100 for chunk in chunks)

    def test_retries_then_raises(self, monkeypatch):
        provider = SarvamSTT(Settings(sarvam_api_key="test-key", max_retries=2))
        calls = {"n": 0}

        def boom(*args, **kwargs):
            calls["n"] += 1
            raise httpx.ConnectError("network down")

        monkeypatch.setattr(httpx, "post", boom)
        monkeypatch.setattr("time.sleep", lambda _: None)

        with pytest.raises(ProviderError, match="after 2 attempts"):
            provider.transcribe(b"audio-bytes")
        assert calls["n"] == 2


class TestVoiceAgent:
    def test_full_turn_from_audio(self, settings, audio_bytes):
        response = VoiceAgent(settings=settings).respond(audio_bytes)
        assert response.transcription.text
        assert response.reply_text
        assert response.has_audio
        assert response.latency_ms >= 0
        assert set(response.stage_timings) == {"stt_ms", "llm_ms", "tts_ms"}

    def test_text_turn(self, settings):
        response = VoiceAgent(settings=settings).respond_to_text("Namaskaram!")
        assert response.intent.name == "greeting"
        assert response.reply_text

    def test_conversation_accumulates(self, settings):
        agent = VoiceAgent(settings=settings)
        agent.respond_to_text("Hello")
        agent.respond_to_text("Tell me a joke")
        assert agent.conversation.turn_count == 4

    def test_reset_clears_history(self, settings):
        agent = VoiceAgent(settings=settings)
        agent.respond_to_text("Hello")
        agent.reset()
        assert agent.conversation.turn_count == 0

    def test_empty_text_is_rejected(self, settings):
        with pytest.raises(ProviderError, match="empty text"):
            VoiceAgent(settings=settings).respond_to_text("   ")

    def test_tts_can_be_disabled(self, settings):
        settings.enable_tts = False
        response = VoiceAgent(settings=settings).respond_to_text("Hello")
        assert response.audio is None
        assert not response.has_audio

    def test_progress_reports_every_stage(self, settings, audio_bytes):
        seen = []
        VoiceAgent(
            settings=settings, progress=lambda stage, msg: seen.append(stage)
        ).respond(audio_bytes)
        assert {"stt", "intent", "llm", "tts", "done"} <= set(seen)

    def test_language_is_carried_through(self, settings):
        response = VoiceAgent(settings=settings).respond_to_text(
            "Repu meeting ni reschedule cheyyandi"
        )
        assert response.reply_language is Language.MIXED

    def test_pure_english_is_detected(self, settings):
        response = VoiceAgent(settings=settings).respond_to_text(
            "Please schedule a meeting"
        )
        assert response.reply_language is Language.ENGLISH

    def test_telugu_script_is_detected(self, settings):
        response = VoiceAgent(settings=settings).respond_to_text("నమస్కారం")
        assert response.reply_language is Language.TELUGU

    def test_bad_provider_fails_at_construction(self, settings):
        settings.llm = "nonexistent"
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            VoiceAgent(settings=settings)

    def test_respond_to_missing_file(self, settings, tmp_path):
        with pytest.raises(FileNotFoundError):
            VoiceAgent(settings=settings).respond_to_file(tmp_path / "nope.wav")

    def test_respond_to_file(self, settings, tmp_path, audio_bytes):
        path = tmp_path / "clip.wav"
        path.write_bytes(audio_bytes)
        response = VoiceAgent(settings=settings).respond_to_file(path)
        assert response.reply_text

    def test_history_survives_across_turns_in_order(self, settings):
        agent = VoiceAgent(settings=settings)
        agent.respond_to_text("Hello")
        agent.respond_to_text("Tell me a joke")
        roles = [turn.role for turn in agent.conversation.turns]
        assert roles == [Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT]


class TestSettings:
    def test_mock_mode_needs_all_three(self):
        assert Settings(stt="mock", llm="mock", tts="mock").is_mock
        assert not Settings(stt="sarvam", llm="mock", tts="mock").is_mock

    def test_require_rejects_blanks(self):
        with pytest.raises(ConfigError, match="MY_KEY is required"):
            Settings().require("", "MY_KEY", "provider")

    def test_describe_names_each_stage(self):
        described = Settings(stt="mock", llm="mock", tts="mock").describe()
        assert "stt=mock" in described and "tts=mock" in described

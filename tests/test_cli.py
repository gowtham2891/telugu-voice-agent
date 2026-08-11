"""CLI tests via Click's runner."""

from __future__ import annotations

import io
import wave

import pytest
from click.testing import CliRunner

from voice_agent import __version__
from voice_agent.cli import SYMBOLS, cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "clip.wav"
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x01" * 8000)
    path.write_bytes(buffer.getvalue())
    return path


class TestCli:
    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "voice agent" in result.output.lower()

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_providers(self, runner):
        result = runner.invoke(cli, ["providers"])
        assert result.exit_code == 0
        assert "sarvam" in result.output

    def test_say(self, runner):
        result = runner.invoke(cli, ["say", "Namaskaram!"])
        assert result.exit_code == 0
        assert "Assistant" in result.output

    def test_say_saves_audio(self, runner, tmp_path):
        target = tmp_path / "reply.wav"
        result = runner.invoke(
            cli, ["say", "Tell me a joke", "--save-audio", str(target)]
        )
        assert result.exit_code == 0
        assert target.exists()
        with wave.open(str(target)) as handle:
            assert handle.getnframes() > 0

    def test_say_with_a_bad_provider_exits_1(self, runner):
        result = runner.invoke(cli, ["say", "Hello", "--llm", "nonexistent"])
        assert result.exit_code == 1
        assert "Unknown LLM provider" in result.output

    def test_listen(self, runner, audio_file):
        result = runner.invoke(cli, ["listen", str(audio_file)])
        assert result.exit_code == 0
        assert "You said" in result.output

    def test_listen_on_a_missing_file_exits_1(self, runner, tmp_path):
        result = runner.invoke(cli, ["listen", str(tmp_path / "nope.wav")])
        assert result.exit_code == 1

    def test_classify(self, runner):
        result = runner.invoke(
            cli, ["classify", "Repu meeting ni 4 PM ki reschedule cheyyandi"]
        )
        assert result.exit_code == 0
        assert "schedule" in result.output

    def test_demo_runs_the_whole_script(self, runner):
        result = runner.invoke(cli, ["demo"])
        assert result.exit_code == 0, result.output
        assert "Conversation length: 10 turns" in result.output

    def test_chat_exits_cleanly(self, runner):
        result = runner.invoke(cli, ["chat"], input="Hello\nexit\n")
        assert result.exit_code == 0
        assert "Bye" in result.output

    def test_chat_reset_command(self, runner):
        result = runner.invoke(cli, ["chat"], input="Hello\nreset\nexit\n")
        assert result.exit_code == 0
        assert "History cleared" in result.output


class TestSymbols:
    def test_every_symbol_survives_the_active_stdout(self):
        """Guards the Windows cp1252 crash on glyphs and Telugu script."""
        import sys

        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        for value in SYMBOLS.values():
            value.encode(encoding)

    def test_telugu_text_survives_the_active_stdout(self):
        import sys

        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        "నమస్కారం".encode(encoding, errors="replace")

"""Tests for codemix detection and intent classification."""

from __future__ import annotations

import pytest

from voice_agent.intents import (
    classify,
    detect_language,
    detect_script_mix,
    extract_subject,
    extract_time,
    is_codemixed,
)
from voice_agent.models import Language


class TestScriptDetection:
    def test_pure_telugu_script(self):
        assert detect_script_mix("నమస్కారం") == (True, False)

    def test_pure_latin(self):
        assert detect_script_mix("hello there") == (False, True)

    def test_both_scripts(self):
        assert detect_script_mix("Hello నమస్కారం") == (True, True)


class TestCodemixDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "Repu meeting ni 4 PM ki reschedule cheyyandi",
            "Bank balance ela check cheyyali",
            "Naaku oka joke cheppu",
            "Hello నమస్కారం",
        ],
    )
    def test_detects_codemix(self, text):
        assert is_codemixed(text)

    @pytest.mark.parametrize(
        "text",
        [
            "What is the weather like today",
            "Please schedule a meeting for tomorrow",
            "నాకు ఈ రోజు టాస్క్ లిస్ట్ చెప్పండి",
        ],
    )
    def test_monolingual_is_not_codemix(self, text):
        assert not is_codemixed(text)

    def test_romanised_telugu_alone_is_not_codemix(self):
        # Every token is a Telugu marker -- no English content mixed in.
        assert not is_codemixed("nenu meeku")

    def test_empty_string(self):
        assert not is_codemixed("")


class TestTimeExtraction:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Meeting at 4 PM", "4 PM"),
            ("Let's meet at 10:30 am", "10:30 am"),
            ("Remind me tomorrow", "tomorrow"),
            ("Repu ki set cheyyandi", "repu"),
            ("Next monday works", "next monday"),
            ("See you in the morning", "morning"),
        ],
    )
    def test_finds_time_expressions(self, text, expected):
        assert extract_time(text).lower() == expected.lower()

    def test_returns_blank_when_absent(self):
        assert extract_time("Tell me a joke") == ""


class TestSubjectExtraction:
    def test_strips_leading_filler(self):
        assert extract_subject("Please schedule the call") == "schedule the call"

    def test_truncates_long_input(self):
        subject = extract_subject(" ".join(f"word{i}" for i in range(30)))
        assert subject.endswith("...")
        assert len(subject.split()) <= 11

    def test_strips_trailing_punctuation(self):
        assert extract_subject("Tell me a joke!") == "Tell me a joke"


class TestClassification:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Hello there", "greeting"),
            ("Namaskaram!", "greeting"),
            ("నమస్కారం", "greeting"),
            ("Thanks, bye!", "farewell"),
            ("Remind me at 5", "reminder"),
            ("Repu meeting ni reschedule cheyyandi", "schedule"),
            ("What's the weather today?", "weather"),
            ("Show me my task list", "task_list"),
            ("Naaku oka joke cheppu", "joke"),
        ],
    )
    def test_classifies_known_intents(self, text, expected):
        assert classify(text).name == expected

    def test_unknown_text_falls_back_to_general(self):
        intent = classify("purple monkey dishwasher")
        assert intent.name == "general"
        assert intent.confidence < 0.5

    def test_empty_text_is_unknown(self):
        intent = classify("   ")
        assert intent.name == "unknown"
        assert intent.confidence == 0.0

    def test_slots_carry_time_and_subject(self):
        intent = classify("Remind me tomorrow about the invoice")
        assert intent.slots["time"] == "tomorrow"
        assert "invoice" in intent.slots["subject"]

    def test_slots_flag_codemix(self):
        assert classify("Repu meeting ni reschedule cheyyandi").slots["codemixed"]
        assert not classify("Schedule the meeting").slots["codemixed"]

    def test_confident_intents_are_marked_confident(self):
        assert classify("Remind me at 5 PM").is_confident

    def test_is_deterministic(self):
        text = "Repu meeting ni 4 PM ki reschedule cheyyandi"
        assert classify(text) == classify(text)


class TestLanguageDetection:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Hello, how are you?", Language.ENGLISH),
            ("Please schedule a meeting", Language.ENGLISH),
            ("నాకు ఈ రోజు టాస్క్ లిస్ట్ చెప్పండి", Language.TELUGU),
            ("Namaskaram!", Language.TELUGU),
            ("nenu meeku", Language.TELUGU),
            ("Repu meeting ni 4 PM ki reschedule cheyyandi", Language.MIXED),
            ("Bank balance ela check cheyyali", Language.MIXED),
            ("Hello నమస్కారం", Language.MIXED),
            ("", Language.MIXED),
        ],
    )
    def test_detects_language(self, text, expected):
        assert detect_language(text) is expected

    def test_codemix_agrees_with_detection(self):
        text = "Repu meeting ni reschedule cheyyandi"
        assert is_codemixed(text) is (detect_language(text) is Language.MIXED)

"""Codemix-aware intent classification.

Rule-based on purpose: intent routing has to work on the *first* turn, before
any LLM round-trip, and it has to work for Tenglish where a single sentence
carries Telugu grammar with English nouns ("meeting ni reschedule cheyyandi").
Keyword sets therefore cover Telugu script, romanised Telugu, and English.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .models import Intent

#: intent -> keyword variants across Telugu script, romanised Telugu and English.
KEYWORDS: Dict[str, List[str]] = {
    "greeting": [
        "hello", "hi ", "hey", "good morning", "good evening",
        "namaskaram", "namaste", "నమస్కారం", "బాగున్నారా",
    ],
    "farewell": [
        "bye", "goodbye", "see you", "thanks", "thank you",
        "dhanyavadalu", "selavu", "వీడ్కోలు", "ధన్యవాదాలు",
    ],
    "reminder": [
        "remind", "reminder", "alarm", "gurthu chey", "gurtu chey",
        "గుర్తు చేయి", "రిమైండర్",
    ],
    "schedule": [
        "meeting", "schedule", "reschedule", "appointment", "calendar",
        "postpone", "మీటింగ్", "సమావేశం",
    ],
    "weather": [
        "weather", "temperature", "rain", "forecast",
        "vaatavaranam", "vaana", "వాతావరణం", "వాన",
    ],
    "task_list": [
        "task", "todo", "to do", "task list", "pathakalu", "pani",
        "టాస్క్", "పని",
    ],
    "joke": ["joke", "funny", "laugh", "joke cheppu", "navvu", "జోక్"],
}

#: Rough time expressions worth capturing as a slot.
TIME_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(\d{1,2}\s*(?::|\.)\s*\d{2}\s*(?:am|pm)?)\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,2}\s*(?:am|pm))\b", re.IGNORECASE),
    re.compile(r"\b(tomorrow|today|tonight|repu|ee roju|ee rojU|nedu)\b", re.IGNORECASE),
    re.compile(r"\b(next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday))\b", re.IGNORECASE),
    re.compile(r"\b(morning|evening|afternoon|night)\b", re.IGNORECASE),
]

_TELUGU_RANGE = re.compile(r"[ఀ-౿]")
_LATIN_RANGE = re.compile(r"[A-Za-z]")

#: Romanised Telugu words -- the tell for Tenglish written in Latin script.
ROMAN_TELUGU_MARKERS = {
    "ni", "ki", "lo", "tho", "cheyyandi", "cheyyali", "cheppu", "cheppandi",
    "kavali", "unnayi", "undi", "ela", "emi", "ento", "enti", "chey",
    "chesanu", "chestanu", "meeku", "naaku", "nenu", "mee", "repu", "roju",
    "sare", "malli", "konchem", "namaskaram", "dhanyavadalu", "baagunnara",
    "ledu", "avunu", "ekkada", "eppudu", "andi", "oka",
}

_PUNCTUATION = ".,!?;:'\"()"


def detect_script_mix(text: str) -> Tuple[bool, bool]:
    """Return ``(has_telugu_script, has_latin_script)`` for ``text``."""
    return bool(_TELUGU_RANGE.search(text)), bool(_LATIN_RANGE.search(text))


def _words(text: str) -> set:
    return {word.strip(_PUNCTUATION).lower() for word in text.split()} - {""}


def detect_language(text: str) -> "Language":
    """Classify an utterance as Telugu, English, or codemixed Tenglish.

    Two scripts in one string is unambiguous codemix. Latin-only text is the
    harder case: romanised Telugu ("repu meeting ni reschedule cheyyandi")
    looks like English to a naive detector, so it is decided on function-word
    markers rather than on the script alone.
    """
    from .models import Language

    if not text or not text.strip():
        return Language.MIXED

    has_telugu, has_latin = detect_script_mix(text)
    if has_telugu and has_latin:
        return Language.MIXED
    if has_telugu:
        return Language.TELUGU

    words = _words(text)
    if not words & ROMAN_TELUGU_MARKERS:
        return Language.ENGLISH
    # Markers plus other Latin words means the two languages are mixed;
    # markers alone means it is Telugu that merely happens to be romanised.
    return Language.MIXED if words - ROMAN_TELUGU_MARKERS else Language.TELUGU


def is_codemixed(text: str) -> bool:
    """True when the utterance mixes Telugu and English in either script.

    Note the asymmetry with :func:`detect_language`, which returns ``MIXED``
    for empty input as a safe default for the agent's language hint. Nothing
    is not codemixed, so the empty case is answered directly here.
    """
    from .models import Language

    if not text or not text.strip():
        return False
    return detect_language(text) is Language.MIXED


def extract_time(text: str) -> str:
    for pattern in TIME_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return ""


def extract_subject(text: str, max_words: int = 10) -> str:
    """Trim filler off the front and cap the length, for use in replies."""
    cleaned = re.sub(
        r"^(please|plz|hey|hi|hello|ok|okay|sare)\b[\s,]*", "", text.strip(),
        flags=re.IGNORECASE,
    )
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned.strip(" .!?")
    return " ".join(words[:max_words]).strip(" .!?") + "..."


def classify(text: str) -> Intent:
    """Classify ``text`` into an intent with a confidence and filled slots.

    Confidence is the share of matched keywords relative to the best-scoring
    intent, floored so a single strong keyword still counts as confident.
    """
    if not text or not text.strip():
        return Intent(name="unknown", confidence=0.0)

    lowered = f" {text.lower()} "
    scores: Dict[str, int] = {}

    for intent_name, keywords in KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
        if hits:
            scores[intent_name] = hits

    slots: Dict[str, str] = {}
    time_slot = extract_time(text)
    if time_slot:
        slots["time"] = time_slot
    slots["subject"] = extract_subject(text)
    slots["codemixed"] = is_codemixed(text)

    if not scores:
        return Intent(name="general", confidence=0.3, slots=slots)

    best_name = max(scores, key=lambda name: (scores[name], -len(name)))
    best_score = scores[best_name]
    total = sum(scores.values())
    confidence = min(0.5 + 0.5 * (best_score / total), 1.0)

    return Intent(name=best_name, confidence=round(confidence, 2), slots=slots)

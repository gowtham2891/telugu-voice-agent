"""Streamlit UI for the Telugu/English voice agent.

    streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st

from voice_agent import VoiceAgent, __version__, get_settings
from voice_agent.config import ConfigError
from voice_agent.intents import classify, is_codemixed
from voice_agent.models import Conversation, Role
from voice_agent.providers.base import (
    ProviderError,
    available_llm,
    available_stt,
    available_tts,
)

st.set_page_config(page_title="Telugu Voice Agent", page_icon="🎤", layout="wide")


def _load_secrets_into_env() -> None:
    """Bridge Streamlit Cloud secrets into the environment.

    Configuration is read from environment variables, so secrets added in the
    Streamlit Cloud dashboard are copied across before settings are resolved.
    Existing environment variables win, so a local .env still takes precedence.
    """
    try:
        items = list(st.secrets.items())
    except Exception:  # noqa: BLE001 - no secrets file is the normal local case
        return
    for key, value in items:
        if isinstance(value, (str, int, float, bool)) and key not in os.environ:
            os.environ[key] = str(value)


_load_secrets_into_env()

SAMPLES = [
    "Namaskaram!",
    "Repu meeting ni 4 PM ki reschedule cheyyandi",
    "What's the weather like in Hyderabad today?",
    "నాకు ఈ రోజు టాస్క్ లిస్ట్ చెప్పండి",
    "Naaku oka joke cheppu",
]


def build_settings():
    """Render the sidebar and return the resolved settings."""
    settings = get_settings(refresh=True)

    with st.sidebar:
        st.title("🎤 Voice Agent")
        st.caption(f"v{__version__} · Telugu + English")
        st.divider()

        st.subheader("Providers")

        def pick(label: str, options, current: str, help_text: str) -> str:
            index = options.index(current) if current in options else 0
            return st.selectbox(label, options, index=index, help=help_text)

        settings.stt = pick(
            "Speech-to-text", available_stt(), settings.stt,
            "`mock` returns sample utterances without an API key.",
        )
        settings.llm = pick(
            "Language model", available_llm(), settings.llm,
            "`mock` answers from intent-aware templates.",
        )
        settings.tts = pick(
            "Text-to-speech", available_tts(), settings.tts,
            "`mock` emits a real WAV tone so playback works.",
        )

        st.subheader("Behaviour")
        settings.enable_tts = st.toggle("Speak replies", value=settings.enable_tts)
        settings.temperature = st.slider(
            "Temperature", 0.0, 1.5, float(settings.temperature), 0.1
        )

        st.divider()
        if settings.is_mock:
            st.success("Mock mode — no credentials required.")
        else:
            st.info("Live mode — reading credentials from .env")

        if st.button("Clear conversation", width="stretch"):
            st.session_state.pop("conversation", None)
            st.rerun()

    return settings


def get_agent(settings) -> VoiceAgent:
    """Rebuild the agent when providers change, preserving history."""
    if "conversation" not in st.session_state:
        st.session_state.conversation = Conversation(
            max_turns=settings.max_history_turns
        )

    signature = (settings.stt, settings.llm, settings.tts)
    if st.session_state.get("provider_signature") != signature:
        st.session_state.provider_signature = signature
        st.session_state.agent = VoiceAgent(
            settings=settings, conversation=st.session_state.conversation
        )

    agent = st.session_state.agent
    agent.settings = settings
    return agent


def render_history(conversation: Conversation) -> None:
    for turn in conversation.turns:
        role = "user" if turn.role is Role.USER else "assistant"
        with st.chat_message(role):
            st.write(turn.text)
            if turn.role is Role.USER and turn.intent:
                st.caption(
                    f"intent: {turn.intent.name} "
                    f"({turn.intent.confidence:.2f}) · {turn.language.label}"
                )


def main() -> None:
    settings = build_settings()

    st.title("Telugu / English Voice Agent")
    st.caption(
        "Speak or type in Telugu, English, or a natural mix of both — "
        "the agent detects the language, routes the intent, and answers in kind."
    )

    try:
        agent = get_agent(settings)
    except (ConfigError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    chat_tab, audio_tab, intent_tab = st.tabs(
        ["💬 Conversation", "🎙️ Audio input", "🔍 Intent inspector"]
    )

    with chat_tab:
        st.caption("Try one of these:")
        columns = st.columns(len(SAMPLES))
        pending = None
        for column, sample in zip(columns, SAMPLES):
            if column.button(sample[:18] + "…" if len(sample) > 18 else sample,
                             width="stretch"):
                pending = sample

        render_history(agent.conversation)

        typed = st.chat_input("Type in Telugu, English or Tenglish…")
        message = pending or typed

        if message:
            with st.chat_message("user"):
                st.write(message)
            try:
                with st.spinner("Thinking…"):
                    response = agent.respond_to_text(message)
            except ProviderError as exc:
                st.error(str(exc))
                st.stop()

            with st.chat_message("assistant"):
                st.write(response.reply_text)
                st.caption(
                    f"intent: {response.intent.name} "
                    f"({response.intent.confidence:.2f}) · "
                    f"{response.reply_language.label} · "
                    f"{response.latency_ms:.0f}ms"
                )
                if response.has_audio:
                    st.audio(response.audio.data, format="audio/wav")

    with audio_tab:
        st.markdown("Record a clip or upload one, then let the agent answer it.")
        recorded = None
        if hasattr(st, "audio_input"):
            recorded = st.audio_input("Record")
        uploaded = st.file_uploader("…or upload a WAV/MP3", type=["wav", "mp3", "m4a"])

        source = recorded or uploaded
        if source is not None and st.button("Send to agent", type="primary"):
            try:
                with st.spinner("Listening…"):
                    response = agent.respond(
                        source.getvalue(), filename=getattr(source, "name", "audio.wav")
                    )
            except ProviderError as exc:
                st.error(str(exc))
                st.stop()

            st.success(f"Answered in {response.latency_ms:.0f}ms")
            st.markdown(f"**You said:** {response.transcription.text}")
            st.markdown(f"**Agent:** {response.reply_text}")
            if response.has_audio:
                st.audio(response.audio.data, format="audio/wav")

            if response.stage_timings:
                st.bar_chart(response.stage_timings)

    with intent_tab:
        st.markdown(
            "The intent router is rule-based so it works on the first turn, "
            "before any LLM round-trip — and it understands romanised Telugu."
        )
        probe = st.text_input(
            "Utterance", value="Repu meeting ni 4 PM ki reschedule cheyyandi"
        )
        if probe:
            intent = classify(probe)
            col1, col2, col3 = st.columns(3)
            col1.metric("Intent", intent.name)
            col2.metric("Confidence", f"{intent.confidence:.2f}")
            col3.metric("Codemixed", "yes" if is_codemixed(probe) else "no")
            st.json(intent.slots)


if __name__ == "__main__":
    main()

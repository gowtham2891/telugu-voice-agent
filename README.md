# Telugu / English Voice AI Agent

[![Live demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gowtham2891-telugu-voice-agent.streamlit.app)
[![CI](https://github.com/gowtham2891/telugu-voice-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/gowtham2891/telugu-voice-agent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**▶ Try it live — [gowtham2891-telugu-voice-agent.streamlit.app](https://gowtham2891-telugu-voice-agent.streamlit.app)** · runs in mock mode, no credentials needed.

A real-time voice agent that actually handles how people in Hyderabad speak:

> **"Repu meeting ni 4 PM ki reschedule cheyyandi"**

That sentence is Telugu grammar, English nouns, written in Latin script. Most
voice stacks either force it through an English model (which hears nonsense) or
a Telugu model (which drops every English word). This one treats codemix as the
default case, not the exception.

---

## The loop

```
  🎤 speech
      │
      ▼
┌──────────────┐  Sarvam Saaras — one transcript for codemixed audio
│    Listen    │
└──────┬───────┘
       ▼
┌──────────────┐  Rule-based router, runs before any LLM call
│    Intent    │  → greeting · reminder · schedule · weather · task_list · joke
└──────┬───────┘
       ▼
┌──────────────┐  GPT-4o / Gemini, prompted to answer in the user's own mix
│    Think     │
└──────┬───────┘
       ▼
┌──────────────┐  Sarvam Bulbul — long text split on sentence boundaries
│    Speak     │
└──────┬───────┘
       ▼
  🔊 audio reply
```

Intent classification is deliberately **not** an LLM call. It has to work on the
first turn with zero latency budget, and it has to understand romanised Telugu
function words (`ni`, `ki`, `cheyyandi`, `naaku`) that no English tokenizer
treats as meaningful.

---

## Quickstart

```bash
git clone https://github.com/gowtham2891/telugu-voice-agent.git
cd telugu-voice-agent
pip install -e ".[dev,ui]"

# Scripted 5-turn conversation, fully offline
voice-agent demo
```

```
You: Repu meeting ni 4 PM ki reschedule cheyyandi
Agent: Meeting ni 'Repu meeting ni 4 PM ki reschedule cheyyandi' ki reschedule
       chesanu. Attendees ki update pampanu.
intent=schedule (0.88) · Telugu + English · 78ms
```

### Web UI

```bash
streamlit run app.py
```

Three tabs: a chat conversation with audio playback, a microphone/upload tab
that runs the full speech-to-speech loop, and an **intent inspector** that shows
exactly how any utterance is classified and which slots were filled.

### CLI

```bash
# One-shot message, save the spoken reply
voice-agent say "Naaku oka joke cheppu" --save-audio reply.wav

# Interactive conversation with history
voice-agent chat

# Answer a recording end to end
voice-agent listen question.wav --stt sarvam --llm openai --tts sarvam

# Inspect the router without running the agent
voice-agent classify "Repu meeting ni 4 PM ki reschedule cheyyandi"
```

---

## Language detection

The hard case is Latin-script Telugu, which looks like English to a naive
detector. Detection resolves in three steps:

| Input | Detected as |
| --- | --- |
| `Please schedule a meeting` | English |
| `నాకు ఈ రోజు టాస్క్ లిస్ట్ చెప్పండి` | Telugu |
| `Namaskaram!` | Telugu (romanised) |
| `Repu meeting ni 4 PM ki reschedule cheyyandi` | **Codemix** |
| `Hello నమస్కారం` | **Codemix** |

Two scripts in one string is unambiguous. For Latin-only text, the decision
rests on romanised Telugu function-word markers: markers *plus* other words
means codemix; markers alone means Telugu that merely happens to be romanised.
The detected language then drives the system prompt, so the agent replies in the
same register the user chose.

---

## Mock mode

Every stage ships a credential-free implementation, selected by default:

- **`mock` STT** — picks a sample utterance deterministically from the audio
  bytes, so the same clip always transcribes the same way.
- **`mock` LLM** — intent-aware templated replies. Offline, instant, deterministic.
- **`mock` TTS** — emits a **real WAV file** whose length scales with the text, so
  playback, download and duration all behave exactly as with live providers.

This is what lets `voice-agent demo` work on a fresh clone and keeps the
123-test suite asserting on real behaviour rather than stubs.

---

## Configuration

Copy `.env.example` to `.env` and fill in only what you need.

| Variable | Purpose | Default |
| --- | --- | --- |
| `STT_PROVIDER` | `mock` or `sarvam` | `mock` |
| `LLM_PROVIDER` | `mock`, `openai` or `gemini` | `mock` |
| `TTS_PROVIDER` | `mock` or `sarvam` | `mock` |
| `SARVAM_API_KEY` | Sarvam AI key (STT + TTS) | — |
| `SARVAM_SPEAKER` | Bulbul voice | `anushka` |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` | LLM credentials | — |
| `MAX_HISTORY_TURNS` | Conversation window | `20` |
| `ENABLE_TTS` | Speak replies at all | `true` |
| `MAX_RETRIES` | Retries on transient provider errors | `3` |

```bash
pip install -e ".[openai]"   # or ".[gemini]"
```

---

## Design notes

**Three independent registries.** STT, LLM and TTS each resolve by name, so a
live Sarvam transcriber can be paired with a mock LLM while you debug — no code
changes, just env vars. Unknown provider names fail at construction, before the
user has spoken.

**Bounded conversation memory.** History is capped at `MAX_HISTORY_TURNS` and
trimmed from the oldest end, so long sessions can't grow the prompt without limit.

**TTS chunking.** Bulbul caps input length, so replies are split on sentence
boundaries — including the Telugu danda `।` — and the resulting audio is
concatenated. A single over-long sentence is hard-split rather than dropped.

**Retries with backoff.** Sarvam calls retry on 429s and 5xx with exponential
backoff, then raise a typed `ProviderError` the CLI turns into exit code 2.

**Console encoding.** Windows terminals default to cp1252, which raises
`UnicodeEncodeError` on the first Telugu character printed. The CLI reconfigures
stdout to UTF-8 and falls back to ASCII symbols when it can't — regression-tested,
because this crashes the conversation mid-turn otherwise.

---

## Project structure

```
voice_agent/
├── models.py         # Language, Turn, Conversation, AgentResponse
├── intents.py        # Codemix detection + rule-based intent routing
├── agent.py          # The listen → think → speak loop
├── config.py         # Env-driven settings
├── cli.py            # chat · say · listen · classify · demo · providers
└── providers/
    ├── base.py       # Three ABCs, three registries
    ├── sarvam.py     # Saaras STT + Bulbul TTS
    ├── llm.py        # OpenAI + Gemini
    └── mock.py       # Offline implementations of all three
app.py                # Streamlit UI
tests/                # 123 tests, all offline
```

---

## Testing

```bash
pytest -v
pytest --cov=voice_agent --cov-report=term-missing
```

Covers language detection across scripts, intent classification and slot
filling, conversation-window trimming, Sarvam response parsing and base64 audio
decoding, TTS text splitting, retry-then-fail behaviour, WAV validity of
synthesized audio, and every CLI exit code.

---

## Roadmap

- [ ] Streaming STT for true barge-in instead of turn-by-turn
- [ ] Voice activity detection for hands-free conversation
- [ ] Tool calling (real calendar and reminder backends behind the intents)
- [ ] Hindi and Tamil codemix using the same marker-based detector

---

## Deploy your own

Ready for [Streamlit Community Cloud](https://share.streamlit.io): free, and it
redeploys on every push to `main`.

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **Create app** → this repo, branch `main`, main file `app.py`.
3. Under **Advanced settings**, choose Python **3.11**.
4. Set the custom subdomain to `gowtham2891-telugu-voice-agent` so it matches the link above.
5. Deploy — the first build takes a couple of minutes.

No secrets are needed for the demo. To switch it to the live providers, open
**Settings → Secrets** in the Streamlit dashboard and paste:

```toml
STT_PROVIDER = "sarvam"
LLM_PROVIDER = "openai"
TTS_PROVIDER = "sarvam"
SARVAM_API_KEY = "your-sarvam-key"
OPENAI_API_KEY = "your-openai-key"
```

`app.py` copies those secrets into the environment before settings are resolved,
so the exact same configuration works locally through `.env` and in the cloud
through the dashboard — no code changes either way.

---

## License

MIT — see [LICENSE](LICENSE).

**Ganesh Gowtham Dupati** · [GitHub](https://github.com/gowtham2891) · gowthamdupati28@gmail.com

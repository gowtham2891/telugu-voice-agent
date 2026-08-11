"""Command line interface.

    voice-agent chat                      # interactive typed conversation
    voice-agent say "Repu meeting ki reschedule cheyyandi"
    voice-agent listen recording.wav
    voice-agent providers
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .agent import VoiceAgent
from .config import ConfigError, get_settings
from .health import check_settings, missing_credentials
from .intents import classify, is_codemixed
from .models import AgentResponse
from .providers.base import (
    ProviderError,
    available_llm,
    available_stt,
    available_tts,
)


def _init_stdio() -> bool:
    """Make stdout UTF-8 where possible; report whether glyphs are safe.

    This matters more here than in most CLIs: replies contain Telugu script,
    and a cp1252 Windows console raises UnicodeEncodeError on every one of
    them. Reconfiguring to UTF-8 with ``errors="replace"`` keeps output legible
    instead of crashing the conversation mid-turn.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - platform specific
                pass

    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "▸✓✗•".encode(encoding)
        return True
    except (LookupError, UnicodeEncodeError):  # pragma: no cover - platform specific
        return False


UNICODE_OK = _init_stdio()

SYMBOLS = {
    "arrow": "▸" if UNICODE_OK else ">",
    "bullet": "•" if UNICODE_OK else "-",
    "check": "✓" if UNICODE_OK else "+",
    "cross": "✗" if UNICODE_OK else "x",
    "mic": "🎤" if UNICODE_OK else "[mic]",
    "speaker": "🔊" if UNICODE_OK else "[spk]",
}

console = Console()

STAGE_STYLE = {
    "stt": "cyan",
    "intent": "magenta",
    "llm": "blue",
    "tts": "yellow",
    "done": "green",
    "reset": "dim",
}


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
    )


def _progress(stage: str, message: str) -> None:
    style = STAGE_STYLE.get(stage, "white")
    console.print(f"[{style}]{SYMBOLS['arrow']} {stage:<7}[/{style}] {message}")


def _quiet(stage: str, message: str) -> None:
    """Progress sink for interactive mode, where the transcript is the output."""


def _render_response(response: AgentResponse, show_meta: bool = True) -> None:
    console.print()
    console.print(
        Panel(
            response.reply_text,
            title=f"{SYMBOLS['speaker']} Assistant",
            border_style="green",
        )
    )
    if show_meta:
        console.print(
            f"[dim]intent={response.intent.name} "
            f"({response.intent.confidence:.2f}) · "
            f"language={response.reply_language.label} · "
            f"{response.latency_ms:.0f}ms"
            + (f" · audio={response.audio.size_bytes}B" if response.has_audio else "")
            + "[/dim]"
        )


def _build_agent(overrides: dict, progress) -> VoiceAgent:
    settings = get_settings(refresh=True)
    for key, value in overrides.items():
        if value:
            setattr(settings, key, value)
    # chat, say and listen all build the agent here, so one guard covers them.
    _require_credentials(settings, "voice-agent")
    return VoiceAgent(settings=settings, progress=progress)


def _require_credentials(settings, command: str) -> None:
    """Stop before the run when a selected provider has no key.

    Exits 1 with the variable name and where to get it, rather than letting the
    failure surface as a ConfigError part way through the pipeline.
    """
    gaps = missing_credentials(settings)
    if not gaps:
        return

    console.print("[red]Missing credentials for this configuration:[/red]")
    for item in gaps:
        where = f" [dim]({item.get_it_at})[/dim]" if item.get_it_at else ""
        console.print(
            f"  {SYMBOLS['cross']} [bold]{item.env_var}[/bold] "
            f"- needed for {item.needed_for}{where}"
        )
    console.print(
        f"\nSet it in your .env file, then re-run [bold]{command} check[/bold] "
        f"to verify. Or switch the provider back to 'mock' to run offline."
    )
    raise SystemExit(1)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="voice-agent")
def cli() -> None:
    """A Telugu/English voice agent for codemixed speech."""


@cli.command()
@click.option("--stt", default=None, help="Override STT_PROVIDER.")
@click.option("--llm", default=None, help="Override LLM_PROVIDER.")
@click.option("--tts", default=None, help="Override TTS_PROVIDER.")
@click.option("--no-tts", is_flag=True, help="Skip speech synthesis.")
@click.option("-v", "--verbose", is_flag=True, help="Show pipeline stages.")
def chat(stt, llm, tts, no_tts, verbose) -> None:
    """Start an interactive typed conversation. Type 'exit' to quit."""
    _configure_logging(verbose)
    try:
        agent = _build_agent(
            {"stt": stt, "llm": llm, "tts": tts},
            _progress if verbose else _quiet,
        )
    except (ConfigError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    if no_tts:
        agent.settings.enable_tts = False

    console.print(
        Panel(
            "Type in Telugu, English, or a mix of both.\n"
            "Commands: [bold]reset[/bold] to clear history, "
            "[bold]exit[/bold] to quit.",
            title=f"{SYMBOLS['mic']} Voice Agent",
            border_style="blue",
        )
    )
    console.print(f"[dim]{agent.settings.describe()}[/dim]\n")

    while True:
        try:
            text = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not text:
            continue
        if text.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Bye.[/dim]")
            break
        if text.lower() == "reset":
            agent.reset()
            console.print("[dim]History cleared.[/dim]\n")
            continue

        try:
            response = agent.respond_to_text(text)
        except ProviderError as exc:
            console.print(f"[red]Provider error:[/red] {exc}\n")
            continue

        _render_response(response, show_meta=verbose)
        console.print()


@cli.command()
@click.argument("text")
@click.option("--stt", default=None, help="Override STT_PROVIDER.")
@click.option("--llm", default=None, help="Override LLM_PROVIDER.")
@click.option("--tts", default=None, help="Override TTS_PROVIDER.")
@click.option(
    "--save-audio",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the synthesized reply to this WAV path.",
)
@click.option("-v", "--verbose", is_flag=True, help="Show pipeline stages.")
def say(text, stt, llm, tts, save_audio, verbose) -> None:
    """Send one TEXT message and print the reply."""
    _configure_logging(verbose)
    try:
        agent = _build_agent(
            {"stt": stt, "llm": llm, "tts": tts}, _progress if verbose else _quiet
        )
        response = agent.respond_to_text(text)
    except (ConfigError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except ProviderError as exc:
        console.print(f"[red]Provider error:[/red] {exc}")
        sys.exit(2)

    _render_response(response)

    if save_audio and response.has_audio:
        save_audio.parent.mkdir(parents=True, exist_ok=True)
        save_audio.write_bytes(response.audio.data)
        console.print(f"[dim]Audio written to {save_audio}[/dim]")


@cli.command()
@click.argument("audio", type=click.Path(path_type=Path))
@click.option("--stt", default=None, help="Override STT_PROVIDER.")
@click.option("--llm", default=None, help="Override LLM_PROVIDER.")
@click.option("--tts", default=None, help="Override TTS_PROVIDER.")
@click.option(
    "--save-audio",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the synthesized reply to this WAV path.",
)
@click.option("-v", "--verbose", is_flag=True, help="Show pipeline stages.")
def listen(audio, stt, llm, tts, save_audio, verbose) -> None:
    """Transcribe AUDIO, answer it, and speak the reply."""
    _configure_logging(verbose)
    try:
        agent = _build_agent(
            {"stt": stt, "llm": llm, "tts": tts}, _progress
        )
        response = agent.respond_to_file(audio)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except (ConfigError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except ProviderError as exc:
        console.print(f"[red]Provider error:[/red] {exc}")
        sys.exit(2)

    console.print()
    console.print(
        Panel(
            response.transcription.text,
            title=f"{SYMBOLS['mic']} You said",
            border_style="cyan",
        )
    )
    _render_response(response)

    if save_audio and response.has_audio:
        save_audio.parent.mkdir(parents=True, exist_ok=True)
        save_audio.write_bytes(response.audio.data)
        console.print(f"[dim]Audio written to {save_audio}[/dim]")


@cli.command(name="classify")
@click.argument("text")
def classify_command(text) -> None:
    """Show the detected intent, slots and codemix verdict for TEXT."""
    intent = classify(text)

    table = Table(title="Intent", header_style="bold")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("intent", intent.name)
    table.add_row("confidence", f"{intent.confidence:.2f}")
    table.add_row("codemixed", str(is_codemixed(text)))
    for key, value in intent.slots.items():
        if key != "codemixed":
            table.add_row(f"slot:{key}", str(value))
    console.print(table)


@cli.command()
def providers() -> None:
    """List available providers and the current configuration."""
    settings = get_settings(refresh=True)

    table = Table(title="Providers", header_style="bold")
    table.add_column("Stage", style="cyan")
    table.add_column("Available")
    table.add_column("Selected", style="green")
    table.add_row("stt", ", ".join(available_stt()), settings.stt)
    table.add_row("llm", ", ".join(available_llm()), settings.llm)
    table.add_row("tts", ", ".join(available_tts()), settings.tts)
    console.print(table)
    console.print(f"\n[dim]{settings.describe()}[/dim]")


@cli.command()
def check() -> None:
    """Verify the configured API keys actually work."""
    settings = get_settings(refresh=True)
    console.print(f"[dim]{settings.describe()}[/dim]")
    console.print()

    results = check_settings(settings)
    if not results:
        console.print("[yellow]Nothing to check for this configuration.[/yellow]")
        return

    for result in results:
        icon = (
            f"[green]{SYMBOLS['check']}[/green]"
            if result.ok
            else f"[red]{SYMBOLS['cross']}[/red]"
        )
        console.print(f"  {icon} [bold]{result.provider}[/bold]: {result.message}")

    failed = [result for result in results if not result.ok]
    console.print()
    if failed:
        console.print(f"[red]{len(failed)} check(s) failed.[/red]")
        sys.exit(2)
    console.print("[green]All checks passed.[/green]")

@cli.command()
def demo() -> None:
    """Run a scripted multi-turn conversation -- no API keys needed."""
    _configure_logging(False)
    settings = get_settings(refresh=True)
    settings.stt = settings.llm = settings.tts = "mock"
    agent = VoiceAgent(settings=settings, progress=_quiet)

    script = [
        "Namaskaram!",
        "Repu meeting ni 4 PM ki reschedule cheyyandi",
        "What's the weather like today?",
        "Naaku oka joke cheppu",
        "Thanks, bye!",
    ]

    console.print(
        Panel(
            "Scripted conversation using the offline mock providers.",
            title=f"{SYMBOLS['mic']} Demo",
            border_style="blue",
        )
    )

    for line in script:
        console.print(f"\n[bold cyan]You:[/bold cyan] {line}")
        response = agent.respond_to_text(line)
        console.print(f"[bold green]Agent:[/bold green] {response.reply_text}")
        console.print(
            f"[dim]intent={response.intent.name} "
            f"({response.intent.confidence:.2f}) · "
            f"{response.reply_language.label} · "
            f"{response.latency_ms:.0f}ms[/dim]"
        )

    console.print(
        f"\n[dim]Conversation length: {agent.conversation.turn_count} turns[/dim]"
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()

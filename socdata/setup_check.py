"""setup_check.py — API key management (no R dependency)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

SETUP_DIR = Path.home() / ".socdata"
ENV_FILE = SETUP_DIR / ".env"

console = Console()


def _load_env() -> None:
    """Load env vars from ~/.socdata/.env then project .env."""
    load_dotenv(ENV_FILE, override=False)
    load_dotenv(override=False)


def _ensure_api_key(force: bool = False) -> str:
    """Return API key, prompting and saving if absent or force=True."""
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if key and key.startswith("sk-ant") and not force:
        return key

    console.print(
        Panel(
            "[bold yellow]Anthropic API key not found.[/]\n\n"
            "Get one at [link=https://console.anthropic.com]console.anthropic.com[/link]\n"
            "It will be saved to [dim]~/.socdata/.env[/]",
            title="API Key Required",
            border_style="yellow",
        )
    )
    while True:
        key = Prompt.ask("[bold]Paste your API key[/]", password=True).strip()
        if key.startswith("sk-ant"):
            break
        console.print("[red]That doesn't look like an Anthropic key (should start with sk-ant-).[/]")

    SETUP_DIR.mkdir(parents=True, exist_ok=True)
    ENV_FILE.touch(exist_ok=True)
    set_key(str(ENV_FILE), "ANTHROPIC_API_KEY", key)
    os.environ["ANTHROPIC_API_KEY"] = key
    console.print("[green]API key saved.[/]")
    return key


def run_setup(force: bool = False) -> bool:
    """Run setup checks. Returns True if ready to proceed."""
    _load_env()

    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or force:
        try:
            _ensure_api_key(force=force)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Setup cancelled.[/]")
            return False

    return True

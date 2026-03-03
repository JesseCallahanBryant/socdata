"""cli.py — Click entry point, banner, and startup sequence."""

from __future__ import annotations

import sys

import click

from socdata.display import console, print_banner
from socdata.setup_check import run_setup


@click.command()
@click.option(
    "--model", "-m",
    default=None,
    help="Claude model to use (default: claude-opus-4-6).",
    envvar="SOCDATA_MODEL",
)
@click.option(
    "--setup", "force_setup",
    is_flag=True,
    default=False,
    help="Re-run setup (reset API key).",
)
@click.version_option(package_name="socdata")
def main(model: str | None, force_setup: bool) -> None:
    """socdata — Multi-Dataset Sociology Research Assistant.

    Guides you through the full research lifecycle: developing research
    questions, formalizing hypotheses, selecting statistical methods, running
    survey-weighted analyses, and interpreting results.

    Supports GSS, ANES, Census/ACS, WVS, and IPUMS datasets.
    """
    print_banner()

    ready = run_setup(force=force_setup)
    if not ready:
        console.print("[red]Setup incomplete. Please resolve the issues above and try again.[/]")
        sys.exit(1)

    # Import datasets to trigger self-registration
    import socdata.datasets  # noqa: F401

    from socdata.session import Session

    session = Session(model=model)
    try:
        session.run()
    except Exception as exc:
        console.print_exception(show_locals=False)
        console.print(f"\n[red]Unexpected error: {exc}[/]")
        sys.exit(1)

"""display.py — Rich terminal formatting helpers for socdata."""

from __future__ import annotations

from contextlib import contextmanager

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console()


@contextmanager
def elapsed_progress(description: str = "Working..."):
    """Indeterminate progress bar with elapsed time. Yields update(desc) callable."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(pulse_style="cyan"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(description, total=None)

        def update(desc: str) -> None:
            progress.update(task, description=desc)

        yield update

BANNER = r"""
 ____              ____        _
/ ___|  ___   ___ |  _ \  __ _| |_ __ _
\___ \ / _ \ / __|| | | |/ _` | __/ _` |
 ___) | (_) | (__ | |_| | (_| | || (_| |
|____/ \___/ \___||____/ \__,_|\__\__,_|
"""

STAGE_COLORS = {
    "topic_exploration": "bright_cyan",
    "variable_discovery": "bright_green",
    "hypothesis_formalization": "bright_yellow",
    "method_selection": "bright_magenta",
    "analysis_execution": "bright_blue",
    "interpretation": "bright_white",
}

STAGE_LABELS = {
    "topic_exploration": "Stage 1 — Topic Exploration",
    "variable_discovery": "Stage 2 — Variable Discovery",
    "hypothesis_formalization": "Stage 3 — Hypothesis Formalization",
    "method_selection": "Stage 4 — Method Selection",
    "analysis_execution": "Stage 5 — Analysis Execution",
    "interpretation": "Stage 6 — Interpretation",
}


def print_banner() -> None:
    console.print(BANNER, style="bold cyan", highlight=False)
    console.print(
        "  [dim]Multi-Dataset Sociology Research Assistant[/]  •  "
        "Claude AI  •  [dim]type /help for commands[/]\n"
    )


def print_stage_header(stage: str) -> None:
    label = STAGE_LABELS.get(stage, stage.replace("_", " ").title())
    color = STAGE_COLORS.get(stage, "white")
    bar = "─" * (len(label) + 4)
    console.print(f"\n[{color}]{bar}[/]")
    console.print(f"[bold {color}]  {label}  [/]")
    console.print(f"[{color}]{bar}[/]\n")


def print_claude_response(text: str, stage: str = "") -> None:
    color = STAGE_COLORS.get(stage, "cyan")
    console.print(
        Panel(
            Markdown(text),
            title="[bold]socdata[/]",
            border_style=color,
            padding=(0, 1),
        )
    )


def print_variable_table(variables: dict) -> None:
    """Print Rich table of variable metadata from inspect_variables result."""
    from socdata.datasets.base import VariableInfo

    table = Table(
        title="Variable Confirmation",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Variable", style="bold", min_width=10)
    table.add_column("Label", min_width=30)
    table.add_column("Years Available", min_width=16)
    table.add_column("N (non-missing)", justify="right", min_width=14)
    table.add_column("Type", min_width=10)

    for var_name, info in variables.items():
        if isinstance(info, VariableInfo):
            found = info.found
            label = info.label
            years = info.years_available
            n_obs = info.n_valid
            dtype = info.type
        else:
            found = info.get("found", True)
            label = info.get("label", "")
            years = info.get("years_available", [])
            n_obs = info.get("n_valid", 0)
            dtype = info.get("type", "unknown")

        if not found:
            table.add_row(
                f"[red]{var_name}[/]",
                "[red]NOT FOUND[/]",
                "—", "—", "—",
            )
            continue

        if len(years) > 6:
            yr_str = f"{years[0]}–{years[-1]} ({len(years)} waves)"
        elif years:
            yr_str = ", ".join(str(y) for y in years)
        else:
            yr_str = "—"

        table.add_row(
            var_name,
            (label or var_name)[:55],
            yr_str,
            f"{n_obs:,}",
            dtype,
        )

    console.print(table)


def print_coefficient_table(result: dict) -> None:
    """Print regression coefficient table from analysis output."""
    method = result.get("method", "")
    n = result.get("n", "?")
    years = result.get("years_used", [])
    yr_str = (
        f"{years[0]}–{years[-1]}" if len(years) > 1
        else str(years[0]) if years
        else "all"
    )

    is_logistic = method in ("survey_logistic", "survey_ordinal_logistic")
    is_chisq = method == "survey_chisq"

    title = (
        f"Results: {method.replace('_', ' ').title()}  •  n={n:,}  •  years={yr_str}"
        if isinstance(n, int)
        else f"Results: {method}"
    )

    if is_chisq:
        _print_chisq_table(result, title)
        return

    coefs = result.get("coefficients", [])
    if not coefs:
        console.print("[yellow]No coefficients returned.[/]")
        return

    table = Table(title=title, show_header=True, header_style="bold magenta", border_style="dim")
    table.add_column("Term", min_width=20)

    if is_logistic:
        table.add_column("Odds Ratio", justify="right", min_width=10)
    else:
        table.add_column("Estimate", justify="right", min_width=10)

    table.add_column("95% CI", min_width=16)
    table.add_column("Std.Err", justify="right", min_width=8)
    table.add_column("p-value", justify="right", min_width=8)
    table.add_column("Sig", min_width=4)

    for row in coefs:
        term = str(row.get("term", ""))
        se = f"{row.get('std.error', 0):.3f}"
        pval = row.get("p.value", 1.0)
        pstr = f"{pval:.4f}" if pval >= 0.0001 else "<.0001"

        sig = ""
        if pval < 0.001:
            sig = "[bold red]***[/]"
        elif pval < 0.01:
            sig = "[red]**[/]"
        elif pval < 0.05:
            sig = "[yellow]*[/]"

        if is_logistic:
            or_val = row.get("odds_ratio", row.get("estimate", 0))
            lo = row.get("or_lo", "")
            hi = row.get("or_hi", "")
            ci_str = f"[{lo:.2f}, {hi:.2f}]" if lo != "" else "—"
            table.add_row(term, f"{or_val:.3f}", ci_str, se, pstr, sig)
        else:
            est = row.get("estimate", 0)
            lo = row.get("conf.low", "")
            hi = row.get("conf.high", "")
            ci_str = f"[{lo:.2f}, {hi:.2f}]" if lo != "" else "—"
            table.add_row(term, f"{est:.3f}", ci_str, se, pstr, sig)

    console.print(table)

    if "r_squared" in result:
        console.print(f"  [dim]R² = {result['r_squared']:.4f}[/]")
    if "thresholds" in result:
        thresh = result["thresholds"]
        if thresh:
            thr_str = "  |  ".join(
                f"{t['threshold']}: {t['estimate']:.3f}" for t in thresh
            )
            console.print(f"  [dim]Thresholds: {thr_str}[/]")

    console.print()


def _print_chisq_table(result: dict, title: str) -> None:
    stat = result.get("statistic", "?")
    df = result.get("df", "?")
    pval = result.get("p_value", 1.0)
    pstr = f"{pval:.4f}" if pval >= 0.0001 else "<.0001"
    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))

    console.print(
        Panel(
            f"χ²({df}) = {stat:.3f}, p = {pstr} {sig}\n"
            f"DV: {result.get('dv', '')}  ×  IV: {result.get('iv', '')}",
            title=title,
            border_style="magenta",
        )
    )

    pct_data = result.get("crosstab_pct", [])
    if pct_data:
        console.print("\n[dim]Column percentages:[/]")
        tbl = Table(show_header=True, header_style="bold", border_style="dim")
        rows_by_iv: dict[str, dict] = {}
        dv_key = result.get("dv", "")
        iv_key = result.get("iv", "")
        for rec in pct_data:
            dv_val = str(rec.get(dv_key, ""))
            iv_val = str(rec.get(iv_key, ""))
            freq = rec.get("Freq", 0)
            rows_by_iv.setdefault(iv_val, {})[dv_val] = freq

        dv_vals = sorted({dv for iv_dict in rows_by_iv.values() for dv in iv_dict})
        tbl.add_column(iv_key)
        for dv_v in dv_vals:
            tbl.add_column(str(dv_v), justify="right")

        for iv_val, dv_dict in sorted(rows_by_iv.items()):
            row_cells = [iv_val] + [f"{dv_dict.get(dv_v, 0):.1f}%" for dv_v in dv_vals]
            tbl.add_row(*row_cells)

        console.print(tbl)
    console.print()


def print_dataset_table(providers: list) -> None:
    """Print table of available datasets."""
    table = Table(
        title="Available Datasets",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Name", style="bold", min_width=10)
    table.add_column("Description", min_width=40)
    table.add_column("URL", style="dim")

    for p in providers:
        table.add_row(p.name.upper(), p.description, p.url)

    console.print(table)


def print_error(message: str) -> None:
    console.print(Panel(f"[red]{message}[/]", title="Error", border_style="red"))


def print_info(message: str) -> None:
    console.print(f"[dim]→[/] {message}")


def print_help() -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Command", style="bold cyan")
    table.add_column("Description")

    commands = [
        ("/help", "Show this help message"),
        ("/stage", "Show current workflow stage"),
        ("/context", "Display current research design (DV, IVs, method, etc.)"),
        ("/datasets", "List available datasets"),
        ("/dataset <name>", "Select a dataset (e.g., /dataset GSS)"),
        ("/search <query>", "Search variables in current dataset"),
        ("/reset", "Clear research context and start a new question"),
        ("/export", "Save conversation + results to a timestamped Markdown file"),
        ("/quit", "Exit socdata"),
    ]
    for cmd, desc in commands:
        table.add_row(cmd, desc)

    console.print(
        Panel(table, title="[bold]Available Commands[/]", border_style="cyan")
    )

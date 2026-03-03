"""tool.py — CLI subcommands for direct GSS data analysis by Claude Code.

Registered as `socdata-tool` entry point. Reads cached parquet/metadata
so Claude Code can search, inspect, describe, and analyze GSS variables
without the inner REPL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — prefer project data/ dir, fall back to ~/.socdata/cache/
# ---------------------------------------------------------------------------

_PROJECT_DATA = Path(__file__).resolve().parent.parent / "data"
_HOME_CACHE = Path.home() / ".socdata" / "cache"


def _resolve(filename: str) -> Path:
    """Return the first existing path for *filename*, preferring project data/."""
    proj = _PROJECT_DATA / filename
    if proj.exists():
        return proj
    return _HOME_CACHE / filename


META_PATH = _resolve("gss_cumulative_meta.json")
PARQUET_PATH = _resolve("gss_cumulative.parquet")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_meta() -> dict:
    if not META_PATH.exists():
        click.echo(f"Metadata not found at {META_PATH}. Run `socdata` first to download GSS.", err=True)
        sys.exit(1)
    with open(META_PATH) as f:
        return json.load(f)


def _load_parquet(columns: list[str] | None = None) -> pd.DataFrame:
    if not PARQUET_PATH.exists():
        click.echo(f"Parquet not found at {PARQUET_PATH}. Run `socdata` first to download GSS.", err=True)
        sys.exit(1)
    if columns:
        return pd.read_parquet(PARQUET_PATH, columns=columns)
    return pd.read_parquet(PARQUET_PATH)


def _col(name: str) -> str:
    """Normalize variable name to lowercase (parquet + metadata convention)."""
    return name.lower()


def _display(name: str) -> str:
    """Uppercase for display to user."""
    return name.upper()


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def cli():
    """socdata-tool: Direct GSS data access for Claude Code."""
    pass


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=30, help="Max results to show.")
def search(query: str, limit: int):
    """Search variable names and labels for QUERY."""
    meta = _load_meta()
    var_labels: dict[str, str] = meta["variable_labels"]
    q = query.lower()

    matches = []
    for var, label in var_labels.items():
        if q in var.lower() or (label and q in label.lower()):
            matches.append((_display(var), label or ""))

    if not matches:
        click.echo(f"No variables matching '{query}'.")
        return

    matches.sort(key=lambda x: x[0])
    for var, label in matches[:limit]:
        click.echo(f"{var:<16} {label}")

    if len(matches) > limit:
        click.echo(f"\n... and {len(matches) - limit} more. Use --limit to see more.")


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("variables", nargs=-1, required=True)
def inspect(variables: tuple[str, ...]):
    """Show detailed metadata for one or more variables (JSON output)."""
    meta = _load_meta()
    var_labels = meta["variable_labels"]
    value_labels = meta["value_labels"]

    # Load only the requested columns from parquet (lowercase)
    col_vars = list(dict.fromkeys([_col(v) for v in variables] + ["year"]))
    try:
        df = _load_parquet(columns=col_vars)
    except Exception as e:
        click.echo(f"Error loading columns: {e}", err=True)
        sys.exit(1)

    result = {}
    for var in variables:
        vc = _col(var)
        vd = _display(var)
        info: dict = {"variable": vd}

        # Label
        info["label"] = var_labels.get(vc, "")

        # Value labels (categories)
        vlabs = value_labels.get(vc, {})
        # Filter out IAP/missing codes
        skip_codes = {"i", "d", "j", "m", "n", "p", "r", "s", "u", "x", "y", "z"}
        categories = {k: v for k, v in vlabs.items() if k.lower() not in skip_codes}
        if categories:
            info["categories"] = categories
            info["type"] = "categorical"
        else:
            info["type"] = "continuous"

        # N and years from parquet
        if vc in df.columns:
            col = df[vc]
            info["n_total"] = int(len(col))
            info["n_valid"] = int(col.notna().sum())
            # Years where variable is present (non-null)
            valid_mask = col.notna()
            years_present = sorted(df.loc[valid_mask, "year"].dropna().unique().astype(int).tolist())
            info["years"] = years_present

        result[vd] = info

    click.echo(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("variable")
@click.option("--years", "-y", type=str, default="", help="Comma-separated years, e.g. 2018,2021")
def describe(variable: str, years: str):
    """Weighted frequency table (categorical) or summary stats (continuous)."""
    year_list = [int(y.strip()) for y in years.split(",") if y.strip()] if years else []

    meta = _load_meta()
    var_labels = meta["variable_labels"]
    value_labels = meta["value_labels"]

    vc = _col(variable)
    vd = _display(variable)

    cols = [vc, "year", "wtssps"]
    try:
        df = _load_parquet(columns=cols)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Filter years
    if year_list:
        df = df[df["year"].isin(year_list)]

    # Drop rows where variable or weight is missing
    df = df.dropna(subset=[vc])
    df = df[df["wtssps"].notna() & (df["wtssps"] > 0)]

    if len(df) == 0:
        click.echo(f"No valid data for {vd}" + (f" in years {year_list}" if year_list else ""))
        return

    label = var_labels.get(vc, "")
    click.echo(f"{vd}: {label}")
    click.echo(f"N = {len(df)}" + (f"  |  Years: {sorted(year_list)}" if year_list else ""))
    click.echo()

    # Check if categorical: use value labels but also check if data has many more
    # unique values than labels (e.g. AGE has 1 label for "89 or older" but 70+ values)
    vlabs = value_labels.get(vc, {})
    skip_codes = {"i", "d", "j", "m", "n", "p", "r", "s", "u", "x", "y", "z"}
    categories = {k: v for k, v in vlabs.items() if k.lower() not in skip_codes}
    n_unique = df[vc].nunique()
    is_categorical = len(categories) >= 2 and n_unique <= len(categories) * 3

    if is_categorical:
        # Weighted frequency table
        weights = df["wtssps"].values
        total_weight = weights.sum()

        rows = []
        for code, lbl in sorted(categories.items(), key=lambda x: x[0]):
            try:
                code_num = float(code)
                mask = df[vc] == code_num
            except ValueError:
                mask = df[vc].astype(str) == code
            n = int(mask.sum())
            w = float(weights[mask].sum())
            pct = (w / total_weight * 100) if total_weight > 0 else 0
            if n > 0:
                rows.append((code, lbl, n, pct))

        click.echo(f"{'Code':<6} {'Label':<40} {'N':>8} {'Wt%':>8}")
        click.echo("-" * 64)
        for code, lbl, n, pct in rows:
            click.echo(f"{code:<6} {lbl[:40]:<40} {n:>8,} {pct:>7.1f}%")
    else:
        # Summary stats (weighted)
        vals = df[vc].astype(float).values
        weights = df["wtssps"].values
        wt_mean = np.average(vals, weights=weights)
        wt_var = np.average((vals - wt_mean) ** 2, weights=weights)
        wt_std = np.sqrt(wt_var)

        click.echo(f"  Mean (weighted):  {wt_mean:.3f}")
        click.echo(f"  Std (weighted):   {wt_std:.3f}")
        click.echo(f"  Min:              {vals.min():.1f}")
        click.echo(f"  Max:              {vals.max():.1f}")
        click.echo(f"  Median:           {np.median(vals):.1f}")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--dv", required=True, help="Dependent variable.")
@click.option("--ivs", required=True, type=str, help="Comma-separated independent variable(s).")
@click.option("--controls", "-c", type=str, default="", help="Comma-separated control variable(s).")
@click.option("--method", required=True, type=click.Choice(["ols", "logistic", "ordinal", "chisq"]))
@click.option("--years", "-y", type=str, default="", help="Comma-separated years, e.g. 2018,2021")
@click.option("--weight", default="wtssps", help="Weight variable (default: wtssps).")
def analyze(dv: str, ivs: str, controls: str, method: str, years: str, weight: str):
    """Run a statistical analysis via socdata.stats.engine."""
    from socdata.context import ResearchContext
    from socdata.stats.engine import run_analysis

    year_list = [int(y.strip()) for y in years.split(",") if y.strip()] if years else []
    ivs_list = [v.strip() for v in ivs.split(",") if v.strip()]
    controls_list = [v.strip() for v in controls.split(",") if v.strip()]

    dv_c = _col(dv)
    ivs_c = [_col(v) for v in ivs_list]
    controls_c = [_col(v) for v in controls_list]
    weight_c = _col(weight)

    # Determine which columns to load (all lowercase)
    all_vars = [dv_c] + ivs_c + controls_c + [weight_c, "year"]
    all_vars = list(dict.fromkeys(all_vars))  # dedupe preserving order

    try:
        df = _load_parquet(columns=all_vars)
    except Exception as e:
        click.echo(json.dumps({"ok": False, "error": str(e)}))
        sys.exit(1)

    # Filter years
    if year_list:
        df = df[df["year"].isin(year_list)]

    ctx = ResearchContext(
        dataset="gss",
        dv=dv_c,
        ivs=ivs_c,
        controls=controls_c,
        selected_method=method,
        years=year_list,
        weight_var=weight_c,
    )

    try:
        result = run_analysis(ctx, df)
    except Exception as e:
        click.echo(json.dumps({"ok": False, "error": str(e)}, indent=2))
        sys.exit(1)

    click.echo(json.dumps(result.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()

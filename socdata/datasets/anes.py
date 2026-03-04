"""anes.py — American National Election Studies provider (1948–2024)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from socdata.datasets.base import DatasetProvider, VariableInfo
from socdata.datasets.cache import (
    CACHE_DIR,
    has_cache,
    load_cache,
    save_cache,
)
from socdata.datasets.registry import register

ANES_CACHE_KEY = "anes_timeseries"
ANES_META_PATH = CACHE_DIR / "anes_timeseries_meta.json"

# Where the user should place their downloaded CSV
ANES_DATA_DIR = CACHE_DIR / "ANES Timeseries"
ANES_CSV_GLOB = "anes_timeseries_cdf_csv_*.csv"
ANES_DOWNLOAD_URL = "https://electionstudies.org/data-center/"

# Embedded variable labels for key ANES variables (no machine-readable label
# file ships with the CSV — only PDFs).
_VARIABLE_LABELS: dict[str, str] = {
    # Survey administration
    "VCF0004": "Year of study",
    "VCF0006": "Study respondent number",
    "VCF0006A": "Case ID (unique across years)",
    # Weights
    "VCF0009X": "Weight: pre-election full sample",
    "VCF0009Y": "Weight: post-election full sample",
    "VCF0009Z": "Weight: post-stratification",
    # Demographics
    "VCF0101": "Respondent age",
    "VCF0102": "Age group (6 categories)",
    "VCF0104": "Gender of respondent",
    "VCF0105A": "Race: 4-category summary",
    "VCF0105B": "Race: 7-category detailed",
    "VCF0106": "Race-ethnicity: Hispanic + race",
    "VCF0110": "Education: 4-category summary",
    "VCF0114": "Family income percentile",
    "VCF0115": "Family income quintile",
    "VCF0116": "Family income group",
    "VCF0127": "Union membership in household",
    "VCF0128": "Region (4-category: NE, NC, S, W)",
    "VCF0130": "Region (South vs non-South)",
    "VCF0140": "Religious preference (major categories)",
    "VCF0142": "Church attendance frequency",
    "VCF0143": "Born-again Christian",
    "VCF0148A": "Marital status",
    "VCF0150": "Employment status",
    "VCF0151": "Occupation of respondent",
    # Party identification & ideology
    "VCF0301": "Party identification: 7-point scale",
    "VCF0302": "Party identification: 3-category summary",
    "VCF0303": "Party identification strength",
    "VCF0305": "Party identification: leaner classification",
    "VCF0803": "Liberal-conservative self-placement (7-point)",
    "VCF0804": "Liberal-conservative: 3-category summary",
    # Vote choice & turnout
    "VCF0702": "Did respondent vote in national election",
    "VCF0703": "Registered to vote",
    "VCF0704": "Presidential vote: 2-party",
    "VCF0704A": "Presidential vote: including 3rd party",
    "VCF0706": "House vote: party",
    "VCF0707": "Senate vote: party",
    "VCF0714": "Intended vote: pre-election",
    # Feeling thermometers (0–100)
    "VCF0201": "Thermometer: Democratic presidential candidate",
    "VCF0202": "Thermometer: Republican presidential candidate",
    "VCF0204": "Thermometer: Democratic Party",
    "VCF0206": "Thermometer: Republican Party",
    "VCF0207": "Thermometer: Liberals",
    "VCF0208": "Thermometer: Labor unions",
    "VCF0209": "Thermometer: Conservatives",
    "VCF0210": "Thermometer: Big business",
    "VCF0211": "Thermometer: Military",
    "VCF0217": "Thermometer: Blacks",
    "VCF0218": "Thermometer: Whites",
    "VCF0224": "Thermometer: Congress",
    "VCF0228": "Thermometer: Federal government",
    # Political engagement & trust
    "VCF0310": "Interest in political campaigns",
    "VCF0311": "Interest in elections",
    "VCF0312": "Follow government and public affairs",
    "VCF0604": "Trust in government (4-point)",
    "VCF0605": "Government run for benefit of all or few",
    "VCF0606": "Government waste tax money",
    "VCF0609": "Can trust people in general",
    "VCF0613": "Does government pay attention to people",
    "VCF0614": "How much does voting matter",
    "VCF0615": "Does politics seem too complicated",
    "VCF0616": "Public officials care what people think",
    "VCF0624": "How much can people be trusted",
    # Presidential approval
    "VCF0450": "Presidential approval (4-category)",
    # Policy issues
    "VCF0806": "Government services vs spending scale (7-point)",
    "VCF0809": "Government guarantee jobs vs let each person get ahead (7-point)",
    "VCF0830": "Government help minorities vs self-help (7-point)",
    "VCF0834": "Defense spending: increase vs decrease (7-point)",
    "VCF0837": "Government vs private health insurance (7-point)",
    "VCF0838": "Urban unrest: solve problems vs force (7-point)",
    "VCF0839": "Rights of accused (7-point)",
    "VCF0851": "Abortion: 4-category position",
    "VCF0852": "Abortion: self-placement 4-point scale",
    "VCF0853": "Equal role for women (7-point scale)",
    "VCF0867A": "Gun control: favor or oppose",
    "VCF0876": "Death penalty: favor or oppose",
    "VCF0878": "Legalize marijuana",
    "VCF0879": "School prayer",
    "VCF0886": "Government spending on environment",
    "VCF0889": "Gay rights: job discrimination",
    "VCF0894": "Immigration levels: increase/decrease/keep same",
}


def _load_meta() -> dict[str, str]:
    """Load variable labels from cached JSON sidecar."""
    if ANES_META_PATH.exists():
        meta = json.loads(ANES_META_PATH.read_text(encoding="utf-8"))
        return meta.get("variable_labels", {})
    return {}


def _save_meta(variable_labels: dict[str, str]) -> None:
    """Save metadata as JSON alongside the parquet cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ANES_META_PATH.write_text(
        json.dumps({"variable_labels": variable_labels}, ensure_ascii=False),
        encoding="utf-8",
    )


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the ANES CSV data.

    - Drop the VERSION metadata column
    - Uppercase all column names
    - Replace single-space missing values with NaN
    - Convert string-digit columns to numeric
    """
    df.columns = [c.strip().upper() for c in df.columns]

    # Drop VERSION metadata column if present
    if "VERSION" in df.columns:
        df = df.drop(columns=["VERSION"])

    # Replace single space (ANES missing convention) with NaN, then coerce to numeric
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].replace(" ", np.nan)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


class ANESProvider(DatasetProvider):
    name = "anes"
    display_name = "American National Election Studies"
    description = "ANES Time Series Cumulative (1948–2024), ~73K respondents"
    url = "https://electionstudies.org"
    default_weight = "VCF0009Z"
    default_psu = ""
    default_strata = ""

    def download(self, years: list[int] | None = None) -> pd.DataFrame:
        if has_cache(ANES_CACHE_KEY):
            df = load_cache(ANES_CACHE_KEY)
            vlabels = _load_meta()
            # Merge embedded labels (embedded wins for keys it covers)
            merged = {**vlabels, **_VARIABLE_LABELS}
            df.attrs["variable_labels"] = merged
        else:
            # Glob for the versioned CSV
            csv_files = sorted(ANES_DATA_DIR.glob(ANES_CSV_GLOB))
            if not csv_files:
                raise RuntimeError(
                    f"ANES Time Series CSV not found.\n\n"
                    f"Please download the CSV from:\n"
                    f"  {ANES_DOWNLOAD_URL}\n\n"
                    f"Place the CSV file at:\n"
                    f"  {ANES_DATA_DIR / ANES_CSV_GLOB}\n\n"
                    f"Run socdata again and it will be cached automatically."
                )

            from rich.console import Console

            console = Console()
            console.print(
                "[bold]Loading ANES Time Series CSV...[/bold] (this may take a minute)"
            )

            df = pd.read_csv(csv_files[-1], low_memory=False)
            df = _clean_data(df)

            # Build labels: embedded labels (no machine-readable label file)
            vlabels = dict(_VARIABLE_LABELS)
            df.attrs["variable_labels"] = vlabels

            # Cache as parquet + meta JSON
            save_cache(ANES_CACHE_KEY, df)
            _save_meta(vlabels)
            console.print(
                f"[green]Cached {len(df):,} respondents × {len(df.columns):,} variables "
                f"as parquet.[/green]"
            )

        # Filter by years if requested (VCF0004 = year of study)
        if years and "VCF0004" in df.columns:
            df = df[df["VCF0004"].isin(years)].copy()

        return df

    def list_variables(self, df: pd.DataFrame) -> list[str]:
        return list(df.columns)

    def inspect_variables(
        self, df: pd.DataFrame, var_names: list[str]
    ) -> dict[str, VariableInfo]:
        labels = df.attrs.get("variable_labels", {})
        results = {}
        for var in var_names:
            vu = var.upper()
            if vu not in df.columns:
                results[vu] = VariableInfo(name=vu, found=False)
                continue
            col = df[vu]
            n_valid = int(col.notna().sum())
            nunique = col.dropna().nunique()
            vtype = (
                "binary" if nunique == 2
                else "ordinal" if nunique <= 7
                else "continuous" if pd.api.types.is_numeric_dtype(col)
                else "categorical"
            )

            # Year availability (which election years contain this variable)
            year_info = ""
            if "VCF0004" in df.columns:
                years_present = sorted(
                    int(y) for y in df.loc[col.notna(), "VCF0004"].dropna().unique()
                )
                if years_present:
                    first, last = years_present[0], years_present[-1]
                    year_info = f"Years: {first}–{last} ({len(years_present)} elections)"

            # Build label with year info appended
            label = labels.get(vu, "")
            if year_info:
                label = f"{label}  [{year_info}]" if label else year_info

            results[vu] = VariableInfo(
                name=vu,
                label=label,
                type=vtype,
                n_valid=n_valid,
                n_missing=int(col.isna().sum()),
                found=True,
            )
        return results

    def system_prompt_appendix(self) -> str:
        return """
**ANES-specific facts:**
- American National Election Studies Time Series Cumulative File (1948–2024)
- ~73,745 respondents across 39 election-year studies
- Year variable: VCF0004 (election year, e.g. 1972, 2020, 2024)
- Weight variables:
  - VCF0009Z: post-stratification weight (default, recommended)
  - VCF0009X: pre-election full sample weight
  - VCF0009Y: post-election full sample weight
- Variables use VCF prefix with 4-digit codes
- Key variables:
  - Party ID: VCF0301 (7-point), VCF0302 (3-cat), VCF0303 (strength)
  - Ideology: VCF0803 (7-point lib-con), VCF0804 (3-cat)
  - Vote: VCF0704 (presidential 2-party), VCF0704A (incl 3rd party), VCF0706 (House)
  - Turnout: VCF0702 (voted), VCF0703 (registered)
  - Demographics: VCF0101 (age), VCF0104 (gender), VCF0105A (race 4-cat),
    VCF0110 (education 4-cat), VCF0114 (income percentile), VCF0128 (region)
  - Trust: VCF0604 (trust govt), VCF0605 (govt for all/few), VCF0606 (waste)
  - Thermometers (0–100): VCF0201/0202 (Dem/Rep candidate), VCF0204/0206 (parties)
  - Issues: VCF0806 (govt services), VCF0809 (jobs), VCF0830 (minorities),
    VCF0851 (abortion), VCF0853 (women's role), VCF0867A (guns)
  - Approval: VCF0450 (presidential approval)
- Missing data: blank/NaN (original CSV uses single space for missing; recoded to NaN)
- Common scales: 7-point (1–7), feeling thermometers (0–100), 3- or 4-category
- Not all variables are available in all years; early years (1948–1960s) have fewer items
"""


register(ANESProvider())

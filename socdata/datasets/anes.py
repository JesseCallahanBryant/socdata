"""anes.py — American National Election Studies provider."""

from __future__ import annotations

import pandas as pd
import pyreadstat

from socdata.datasets.base import DatasetProvider, VariableInfo
from socdata.datasets.cache import (
    CACHE_DIR,
    download_file,
    has_cache,
    load_cache,
    save_cache,
)
from socdata.datasets.registry import register

ANES_CACHE_KEY = "anes_cumulative"

# ANES cumulative data file (Stata format) — user must download manually
# due to ANES terms of use. We check cache only.
ANES_MANUAL_URL = "https://electionstudies.org/data-center/"


class ANESProvider(DatasetProvider):
    name = "anes"
    display_name = "American National Election Studies"
    description = "ANES Time Series Cumulative (1948–2020)"
    url = "https://electionstudies.org"
    default_weight = "VCF0009Z"
    default_psu = ""
    default_strata = ""

    def download(self, years: list[int] | None = None) -> pd.DataFrame:
        if has_cache(ANES_CACHE_KEY):
            df = load_cache(ANES_CACHE_KEY)
        else:
            raise RuntimeError(
                f"ANES data not found in cache. Due to ANES terms of use, "
                f"you must download the cumulative Stata file manually from:\n"
                f"  {ANES_MANUAL_URL}\n\n"
                f"Then place the .dta file at:\n"
                f"  {CACHE_DIR / 'anes_cumulative.dta'}\n\n"
                f"Run socdata again and it will convert and cache it."
            )

        df.columns = [c.upper() for c in df.columns]

        if years and "VCF0004" in df.columns:
            df = df[df["VCF0004"].isin(years)].reset_index(drop=True)

        return df

    def _convert_from_dta(self) -> pd.DataFrame:
        """Convert a manually placed .dta file to cached parquet."""
        dta_path = CACHE_DIR / "anes_cumulative.dta"
        if not dta_path.exists():
            raise FileNotFoundError(f"Place ANES .dta file at: {dta_path}")

        df, meta = pyreadstat.read_dta(str(dta_path))
        df.attrs["variable_labels"] = dict(
            zip(meta.column_names, meta.column_labels)
        )
        df.attrs["value_labels"] = meta.variable_value_labels
        save_cache(ANES_CACHE_KEY, df)
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
            results[vu] = VariableInfo(
                name=vu,
                label=labels.get(vu, labels.get(var.lower(), "")),
                type=vtype,
                n_valid=n_valid,
                n_missing=int(col.isna().sum()),
                found=True,
            )
        return results

    def system_prompt_appendix(self) -> str:
        return """
**ANES-specific facts:**
- American National Election Studies Time Series Cumulative File
- Covers presidential election years 1948–2020
- Year variable: VCF0004
- Weight variable: VCF0009Z (post-stratification weight)
- Variables use VCF prefix (e.g., VCF0301 = party identification 7-point)
- Key variables: VCF0301 (party ID), VCF0803 (liberal-conservative), VCF0704 (presidential vote)
"""


register(ANESProvider())

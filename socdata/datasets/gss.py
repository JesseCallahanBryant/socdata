"""gss.py — General Social Survey dataset provider."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyreadstat

from socdata.datasets.base import DatasetProvider, VariableInfo
from socdata.datasets.cache import (
    cache_path,
    download_file,
    has_cache,
    load_cache,
    save_cache,
    CACHE_DIR,
)
from socdata.datasets.registry import register

# NORC distributes the cumulative GSS as a Stata file
GSS_STATA_URL = "https://gss.norc.org/documents/stata/GSS_stata.zip"
GSS_CACHE_KEY = "gss_cumulative"
GSS_META_PATH = CACHE_DIR / "gss_cumulative_meta.json"


def _load_meta() -> tuple[dict, dict]:
    """Load variable labels and value labels from cached JSON."""
    if GSS_META_PATH.exists():
        meta = json.loads(GSS_META_PATH.read_text(encoding="utf-8"))
        return meta.get("variable_labels", {}), meta.get("value_labels", {})
    return {}, {}


def _save_meta(variable_labels: dict, value_labels: dict) -> None:
    """Save metadata as JSON alongside the parquet cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Convert value_labels keys to strings for JSON serialization
    vl_serializable = {}
    for var, mapping in value_labels.items():
        vl_serializable[var] = {str(k): str(v) for k, v in mapping.items()}

    GSS_META_PATH.write_text(
        json.dumps(
            {"variable_labels": variable_labels, "value_labels": vl_serializable},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class GSSProvider(DatasetProvider):
    name = "gss"
    display_name = "General Social Survey"
    description = "NORC cumulative GSS (1972–2022), ~6,900 variables"
    url = "https://gss.norc.org"
    default_weight = "WTSSPS"
    default_psu = "VPSU"
    default_strata = "VSTRAT"

    def download(self, years: list[int] | None = None) -> pd.DataFrame:
        if has_cache(GSS_CACHE_KEY):
            df = load_cache(GSS_CACHE_KEY)
            vlabels, val_labels = _load_meta()
            df.attrs["variable_labels"] = vlabels
            df.attrs["value_labels"] = val_labels
        else:
            df = self._download_and_convert()

        # Uppercase all column names for consistency
        df.columns = [c.upper() for c in df.columns]

        if years:
            if "YEAR" in df.columns:
                df = df[df["YEAR"].isin(years)].reset_index(drop=True)

        return df

    def _download_and_convert(self) -> pd.DataFrame:
        """Download Stata file from NORC, convert to Parquet cache."""
        import zipfile
        import tempfile

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = CACHE_DIR / "GSS_stata.zip"

        # Download zip (skip if already downloaded)
        if not zip_path.exists():
            download_file(GSS_STATA_URL, zip_path, label="GSS Stata file")

        # Extract .dta from zip
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, "r") as zf:
                dta_files = [n for n in zf.namelist() if n.endswith(".dta")]
                if not dta_files:
                    raise RuntimeError("No .dta file found in GSS zip archive")
                zf.extract(dta_files[0], tmpdir)
                dta_path = Path(tmpdir) / dta_files[0]

                # Read with pyreadstat (latin1 for non-UTF8 value labels in GSS)
                df, meta = pyreadstat.read_dta(str(dta_path), encoding="latin1")

        # Build metadata
        variable_labels = dict(zip(meta.column_names, meta.column_labels))
        value_labels = meta.variable_value_labels

        df.attrs["variable_labels"] = variable_labels
        df.attrs["value_labels"] = value_labels

        # Cache parquet + metadata
        save_cache(GSS_CACHE_KEY, df)
        _save_meta(variable_labels, value_labels)

        # Clean up zip
        zip_path.unlink(missing_ok=True)

        return df

    def list_variables(self, df: pd.DataFrame) -> list[str]:
        return list(df.columns)

    def inspect_variables(
        self, df: pd.DataFrame, var_names: list[str]
    ) -> dict[str, VariableInfo]:
        labels = df.attrs.get("variable_labels", {})
        value_labels = df.attrs.get("value_labels", {})
        results = {}

        for var in var_names:
            var_upper = var.upper()
            if var_upper not in df.columns:
                results[var_upper] = VariableInfo(name=var_upper, found=False)
                continue

            col = df[var_upper]
            n_valid = int(col.notna().sum())
            n_missing = int(col.isna().sum())
            # Try both upper and lower case keys for labels
            label = labels.get(var_upper, labels.get(var.lower(), "")) or ""

            # Determine type
            nunique = col.dropna().nunique()
            cats = {}
            # Check both cases for value labels
            vl_key = var_upper if var_upper in value_labels else var.lower()
            if vl_key in value_labels:
                cats = {str(k): str(v) for k, v in value_labels[vl_key].items()}

            if nunique == 2:
                vtype = "binary"
            elif nunique <= 7 or cats:
                vtype = "ordinal" if nunique <= 7 else "categorical"
            elif pd.api.types.is_numeric_dtype(col):
                vtype = "continuous"
            else:
                vtype = "categorical"

            # Years available
            years_avail = []
            if "YEAR" in df.columns:
                mask = col.notna()
                years_avail = sorted(
                    df.loc[mask, "YEAR"].dropna().unique().astype(int).tolist()
                )

            results[var_upper] = VariableInfo(
                name=var_upper,
                label=label,
                type=vtype,
                n_valid=n_valid,
                n_missing=n_missing,
                categories=cats,
                years_available=years_avail,
                found=True,
            )

        return results

    def search_variables(
        self, df: pd.DataFrame, query: str
    ) -> list[VariableInfo]:
        query_lower = query.lower()
        labels = df.attrs.get("variable_labels", {})
        results = []
        for col in df.columns:
            label = labels.get(col, labels.get(col.lower(), "")) or ""
            if query_lower in col.lower() or query_lower in label.lower():
                results.append(VariableInfo(name=col, label=label))
            if len(results) >= 50:
                break
        return results

    def system_prompt_appendix(self) -> str:
        return """
**GSS-specific facts:**
- Complex stratified multi-stage probability sample. Always use survey weights.
- Primary weight variable: WTSSPS (2004–present); fallback: WTSSALL (pre-2004 or pooled)
- PSU = VPSU, stratum = VSTRAT
- 1972–2022, approximately 6,900 variables
- Variables are not asked every year

**Common GSS variables:**
Demographics: EDUC (years of education), AGE, SEX (1=Male, 2=Female), RACE, INCOME, RINCOME, RELIG, ATTEND (religious attendance), MARITAL, CHILDS, REGION, SRCBELT

Attitudes/values: POLVIEWS (1=Extremely liberal to 7=Extremely conservative), PARTYID, TRUST (generalized trust), HAPPY, HEALTH, CLASS (subjective social class)

Science/knowledge: CONSCI (confidence in scientific community, 1=A great deal, 2=Only some, 3=Hardly any), EVOLVED, EARTHSUN, CONDRIFT

Work/economy: SATJOB, JOBSEC, WRKSTAT, PRESTG10 (occupational prestige), HRS1 (hours worked)
"""


# Self-register
register(GSSProvider())

"""wvs.py — World Values Survey provider."""

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

WVS_CACHE_KEY = "wvs_wave7"
WVS_STATA_URL = (
    "https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp"
)


class WVSProvider(DatasetProvider):
    name = "wvs"
    display_name = "World Values Survey"
    description = "WVS Wave 7 (2017–2022), 90+ countries"
    url = "https://www.worldvaluessurvey.org"
    default_weight = "W_WEIGHT"
    default_psu = ""
    default_strata = ""

    def download(self, years: list[int] | None = None) -> pd.DataFrame:
        if has_cache(WVS_CACHE_KEY):
            df = load_cache(WVS_CACHE_KEY)
        else:
            # WVS requires manual download due to registration
            dta_path = CACHE_DIR / "wvs_wave7.dta"
            if dta_path.exists():
                df, meta = pyreadstat.read_dta(str(dta_path))
                df.attrs["variable_labels"] = dict(
                    zip(meta.column_names, meta.column_labels)
                )
                df.attrs["value_labels"] = meta.variable_value_labels
                save_cache(WVS_CACHE_KEY, df)
            else:
                raise RuntimeError(
                    f"WVS data not found. Please download the Wave 7 Stata file from:\n"
                    f"  {WVS_STATA_URL}\n\n"
                    f"Place the .dta file at:\n"
                    f"  {dta_path}\n\n"
                    f"Run socdata again and it will be cached automatically."
                )

        df.columns = [c.upper() for c in df.columns]
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
                else "ordinal" if nunique <= 10
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
**WVS-specific facts:**
- World Values Survey Wave 7 (2017–2022), 90+ countries
- Weight variable: W_WEIGHT
- Country variable: B_COUNTRY_ALPHA (3-letter ISO code)
- Variables use Q-prefix (e.g., Q1 = importance of family, Q57 = life satisfaction)
- Key themes: social values, trust, religion, politics, economic attitudes
- 10-point scales common (1–10), with country-specific samples
"""


register(WVSProvider())

"""ipums.py — IPUMS microdata provider (CPS, ACS)."""

from __future__ import annotations

import pandas as pd

from socdata.datasets.base import DatasetProvider, VariableInfo
from socdata.datasets.cache import CACHE_DIR, has_cache, load_cache, save_cache
from socdata.datasets.registry import register

IPUMS_CACHE_KEY = "ipums_cps"


class IPUMSProvider(DatasetProvider):
    name = "ipums"
    display_name = "IPUMS Microdata"
    description = "IPUMS CPS/ACS microdata (requires ipumspy + API key)"
    url = "https://www.ipums.org"
    default_weight = "WTFINL"
    default_psu = ""
    default_strata = ""

    def download(self, years: list[int] | None = None) -> pd.DataFrame:
        try:
            import ipumspy  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "IPUMS support requires extra dependencies.\n"
                "Install with: pip install 'socdata[ipums]'"
            )

        import os

        api_key = os.getenv("IPUMS_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "IPUMS API key not found. Set IPUMS_API_KEY environment variable.\n"
                "Get one at: https://account.ipums.org/api_keys"
            )

        # Check for manually placed extract
        cache_key = IPUMS_CACHE_KEY
        if has_cache(cache_key):
            return load_cache(cache_key)

        dta_path = CACHE_DIR / "ipums_cps.dta"
        if dta_path.exists():
            import pyreadstat

            df, meta = pyreadstat.read_dta(str(dta_path))
            df.attrs["variable_labels"] = dict(
                zip(meta.column_names, meta.column_labels)
            )
            save_cache(cache_key, df)
            return df

        raise RuntimeError(
            f"IPUMS data not found. Please create an extract at:\n"
            f"  https://cps.ipums.org/cps/\n\n"
            f"Download the Stata file and place it at:\n"
            f"  {dta_path}\n\n"
            f"Run socdata again and it will be cached automatically."
        )

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
**IPUMS-specific facts:**
- IPUMS harmonized microdata (CPS, ACS, Census)
- Weight variable: WTFINL (CPS final weight), PERWT (ACS person weight)
- Key variables: AGE, SEX, RACE, EDUC, EMPSTAT, INCWAGE, OCC, IND
- Variables are harmonized across years for consistent coding
- IPUMS requires registration and extract creation at ipums.org
"""


register(IPUMSProvider())

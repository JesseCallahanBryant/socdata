"""census.py — Census/ACS provider via Census API."""

from __future__ import annotations

import pandas as pd

from socdata.datasets.base import DatasetProvider, VariableInfo
from socdata.datasets.cache import has_cache, load_cache, save_cache
from socdata.datasets.registry import register

CENSUS_CACHE_KEY = "census_acs"


class CensusProvider(DatasetProvider):
    name = "census"
    display_name = "U.S. Census / ACS"
    description = "American Community Survey via Census API (requires API key)"
    url = "https://data.census.gov"
    default_weight = "PWGTP"
    default_psu = ""
    default_strata = ""

    def download(self, years: list[int] | None = None) -> pd.DataFrame:
        try:
            from census import Census
            import us
        except ImportError:
            raise RuntimeError(
                "Census support requires extra dependencies.\n"
                "Install with: pip install 'socdata[census]'"
            )

        import os

        api_key = os.getenv("CENSUS_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "Census API key not found. Set CENSUS_API_KEY environment variable.\n"
                "Get a free key at: https://api.census.gov/data/key_signup.html"
            )

        year = (years or [2022])[0]
        cache_key = f"{CENSUS_CACHE_KEY}_{year}"

        if has_cache(cache_key):
            return load_cache(cache_key)

        c = Census(api_key, year=year)
        # Pull a standard set of ACS 1-year PUMS variables
        data = c.acs1.get(
            ("NAME", "B01001_001E", "B19013_001E", "B15003_022E"),
            {"for": "state:*"},
        )
        df = pd.DataFrame(data)
        save_cache(cache_key, df)
        return df

    def list_variables(self, df: pd.DataFrame) -> list[str]:
        return list(df.columns)

    def inspect_variables(
        self, df: pd.DataFrame, var_names: list[str]
    ) -> dict[str, VariableInfo]:
        results = {}
        for var in var_names:
            if var not in df.columns:
                results[var] = VariableInfo(name=var, found=False)
                continue
            col = df[var]
            results[var] = VariableInfo(
                name=var,
                label=var,
                type="continuous" if pd.api.types.is_numeric_dtype(col) else "categorical",
                n_valid=int(col.notna().sum()),
                n_missing=int(col.isna().sum()),
                found=True,
            )
        return results

    def system_prompt_appendix(self) -> str:
        return """
**Census/ACS-specific facts:**
- American Community Survey accessed via Census API
- Requires CENSUS_API_KEY environment variable
- Data available at various geographic levels (state, county, tract)
- Weight variable for PUMS: PWGTP (person weight)
- Common tables: B01001 (sex by age), B19013 (median household income), B15003 (educational attainment)
"""


register(CensusProvider())

"""chisq.py — Survey-weighted chi-square test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from socdata.stats.results import AnalysisResult
from socdata.stats.weights import prepare_analysis_df


def run_chisq(
    df: pd.DataFrame,
    dv: str,
    ivs: list[str],
    controls: list[str],
    weight_var: str = "",
    years_used: list[int] | None = None,
) -> AnalysisResult:
    """Run weighted chi-square test of independence (DV x first IV)."""
    iv = ivs[0] if ivs else ""
    if not iv:
        return AnalysisResult(ok=False, error="No IV specified for chi-square test", method="chisq")

    # controls are ignored for bivariate chi-square
    try:
        subset, wt_col = prepare_analysis_df(df, dv, [iv], [], weight_var)
    except ValueError as e:
        return AnalysisResult(ok=False, error=str(e), method="chisq")

    dv_col = subset[dv]
    iv_col = subset[iv]
    weights = subset[wt_col].values if wt_col else np.ones(len(subset))

    # Build weighted crosstab
    crosstab = pd.crosstab(
        dv_col, iv_col, values=weights, aggfunc="sum"
    ).fillna(0)

    # Chi-square test on the weighted contingency table
    chi2, p_value, dof, expected = sp_stats.chi2_contingency(crosstab.values)

    # Column percentages for display
    col_totals = crosstab.sum(axis=0)
    pct_table = crosstab.div(col_totals, axis=1) * 100

    crosstab_pct = []
    for dv_val in pct_table.index:
        for iv_val in pct_table.columns:
            crosstab_pct.append({
                dv: str(dv_val),
                iv: str(iv_val),
                "Freq": round(float(pct_table.loc[dv_val, iv_val]), 1),
            })

    return AnalysisResult(
        ok=True,
        method="survey_chisq",
        n=len(subset),
        years_used=years_used or [],
        weight_var=weight_var,
        chi2_statistic=chi2,
        chi2_df=dof,
        chi2_p_value=p_value,
        crosstab_pct=crosstab_pct,
        dv=dv,
        iv=iv,
    )

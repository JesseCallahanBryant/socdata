"""ols.py — Survey-weighted OLS regression via statsmodels WLS."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats

from socdata.stats.results import AnalysisResult, CoefficientRow
from socdata.stats.weights import prepare_analysis_df


def run_ols(
    df: pd.DataFrame,
    dv: str,
    ivs: list[str],
    controls: list[str],
    weight_var: str = "",
    years_used: list[int] | None = None,
) -> AnalysisResult:
    """Run survey-weighted OLS (WLS with robust SEs)."""
    try:
        subset, wt_col = prepare_analysis_df(df, dv, ivs, controls, weight_var)
    except ValueError as e:
        return AnalysisResult(ok=False, error=str(e), method="ols")

    y = subset[dv].astype(float)
    X = pd.get_dummies(subset[ivs + controls], drop_first=True, dtype=float)
    X = sm.add_constant(X)

    weights = subset[wt_col].values if wt_col else np.ones(len(y))

    model = sm.WLS(y, X, weights=weights)
    result = model.fit(cov_type="HC1")

    coefs = []
    for term in result.params.index:
        ci = result.conf_int().loc[term]
        coefs.append(CoefficientRow(
            term=term,
            estimate=result.params[term],
            std_error=result.bse[term],
            statistic=result.tvalues[term],
            p_value=result.pvalues[term],
            conf_low=ci[0],
            conf_high=ci[1],
        ))

    return AnalysisResult(
        ok=True,
        method="ols",
        n=int(result.nobs),
        years_used=years_used or [],
        weight_var=weight_var,
        coefficients=coefs,
        r_squared=result.rsquared,
    )

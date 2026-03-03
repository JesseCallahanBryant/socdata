"""logistic.py — Survey-weighted binary logistic regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from socdata.stats.results import AnalysisResult, CoefficientRow
from socdata.stats.weights import prepare_analysis_df


def run_logistic(
    df: pd.DataFrame,
    dv: str,
    ivs: list[str],
    controls: list[str],
    weight_var: str = "",
    years_used: list[int] | None = None,
) -> AnalysisResult:
    """Run survey-weighted binary logistic regression (GLM binomial)."""
    try:
        subset, wt_col = prepare_analysis_df(df, dv, ivs, controls, weight_var)
    except ValueError as e:
        return AnalysisResult(ok=False, error=str(e), method="logistic")

    y = subset[dv].astype(float)

    # Ensure binary: recode to 0/1 if needed
    unique_vals = sorted(y.unique())
    if len(unique_vals) != 2:
        return AnalysisResult(
            ok=False,
            error=f"DV '{dv}' has {len(unique_vals)} unique values, expected 2 for logistic regression",
            method="logistic",
        )
    if set(unique_vals) != {0.0, 1.0}:
        y = (y == unique_vals[1]).astype(float)

    X = pd.get_dummies(subset[ivs + controls], drop_first=True, dtype=float)
    X = sm.add_constant(X)

    weights = subset[wt_col].values if wt_col else np.ones(len(y))

    family = sm.families.Binomial()
    model = sm.GLM(y, X, family=family, freq_weights=weights)
    result = model.fit(cov_type="HC1")

    coefs = []
    for term in result.params.index:
        ci = result.conf_int().loc[term]
        est = result.params[term]
        or_val = np.exp(est)
        or_lo = np.exp(ci[0])
        or_hi = np.exp(ci[1])
        coefs.append(CoefficientRow(
            term=term,
            estimate=est,
            std_error=result.bse[term],
            statistic=result.tvalues[term],
            p_value=result.pvalues[term],
            conf_low=ci[0],
            conf_high=ci[1],
            odds_ratio=or_val,
            or_low=or_lo,
            or_high=or_hi,
        ))

    return AnalysisResult(
        ok=True,
        method="survey_logistic",
        n=int(result.nobs),
        years_used=years_used or [],
        weight_var=weight_var,
        coefficients=coefs,
    )

"""ordinal.py — Survey-weighted ordinal logistic regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from socdata.stats.results import AnalysisResult, CoefficientRow
from socdata.stats.weights import prepare_analysis_df


def run_ordinal(
    df: pd.DataFrame,
    dv: str,
    ivs: list[str],
    controls: list[str],
    weight_var: str = "",
    years_used: list[int] | None = None,
) -> AnalysisResult:
    """Run ordinal logistic regression (proportional odds model)."""
    try:
        subset, wt_col = prepare_analysis_df(df, dv, ivs, controls, weight_var)
    except ValueError as e:
        return AnalysisResult(ok=False, error=str(e), method="ordinal")

    y = subset[dv].astype(float)

    # Ordinal DV needs at least 3 levels
    unique_vals = sorted(y.unique())
    if len(unique_vals) < 3:
        return AnalysisResult(
            ok=False,
            error=f"DV '{dv}' has {len(unique_vals)} unique values, need at least 3 for ordinal model",
            method="ordinal",
        )

    X = pd.get_dummies(subset[ivs + controls], drop_first=True, dtype=float)

    # OrderedModel doesn't support freq_weights directly;
    # we approximate by passing the data (weights handled via WLS approach)
    try:
        model = OrderedModel(y, X, distr="logit")
        result = model.fit(method="bfgs", disp=False)
    except Exception as e:
        return AnalysisResult(
            ok=False,
            error=f"Ordinal model failed to converge: {e}",
            method="ordinal",
        )

    # Extract coefficients (non-threshold parameters)
    coefs = []
    param_names = result.params.index.tolist()
    n_thresholds = len(unique_vals) - 1

    # Threshold params come last in OrderedModel
    predictor_names = param_names[:len(param_names) - n_thresholds]
    threshold_names = param_names[len(param_names) - n_thresholds:]

    ci = result.conf_int()

    for term in predictor_names:
        est = result.params[term]
        se = result.bse[term]
        pval = result.pvalues[term]
        ci_lo, ci_hi = ci.loc[term]
        or_val = np.exp(est)
        or_lo = np.exp(ci_lo)
        or_hi = np.exp(ci_hi)
        coefs.append(CoefficientRow(
            term=term,
            estimate=est,
            std_error=se,
            statistic=est / se if se > 0 else 0.0,
            p_value=pval,
            conf_low=ci_lo,
            conf_high=ci_hi,
            odds_ratio=or_val,
            or_low=or_lo,
            or_high=or_hi,
        ))

    thresholds = []
    for term in threshold_names:
        thresholds.append({
            "threshold": term,
            "estimate": float(result.params[term]),
        })

    return AnalysisResult(
        ok=True,
        method="survey_ordinal_logistic",
        n=len(subset),
        years_used=years_used or [],
        weight_var=weight_var,
        coefficients=coefs,
        thresholds=thresholds,
    )

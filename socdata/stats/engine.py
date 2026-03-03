"""engine.py — Dispatcher: run_analysis(context, df) → AnalysisResult."""

from __future__ import annotations

import pandas as pd

from socdata.context import ResearchContext
from socdata.stats.results import AnalysisResult
from socdata.stats.ols import run_ols
from socdata.stats.logistic import run_logistic
from socdata.stats.ordinal import run_ordinal
from socdata.stats.chisq import run_chisq


def run_analysis(context: ResearchContext, df: pd.DataFrame) -> AnalysisResult:
    """Dispatch to the appropriate analysis based on context.selected_method."""
    method = context.selected_method.lower()

    kwargs = dict(
        df=df,
        dv=context.dv,
        ivs=context.ivs,
        controls=context.controls,
        weight_var=context.weight_var,
        years_used=context.years,
    )

    if method == "ols":
        return run_ols(**kwargs)
    elif method == "logistic":
        return run_logistic(**kwargs)
    elif method == "ordinal":
        return run_ordinal(**kwargs)
    elif method == "chisq":
        return run_chisq(**kwargs)
    else:
        return AnalysisResult(
            ok=False,
            error=f"Unknown method '{method}'. Choose from: ols, logistic, ordinal, chisq",
            method=method,
        )

"""weights.py — Survey weight preparation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_analysis_df(
    df: pd.DataFrame,
    dv: str,
    ivs: list[str],
    controls: list[str],
    weight_var: str = "",
) -> tuple[pd.DataFrame, str]:
    """Prepare a DataFrame for analysis: select columns, drop NAs, normalize weights.

    Returns:
        (cleaned_df, weight_column_name) — weight_column_name is "" if no weights.
    """
    all_vars = [dv] + ivs + controls
    if weight_var and weight_var in df.columns:
        all_vars.append(weight_var)

    # Select only needed columns
    missing_cols = [v for v in all_vars if v not in df.columns]
    if missing_cols:
        raise ValueError(f"Variables not found in data: {missing_cols}")

    subset = df[all_vars].copy()

    # Drop rows with missing values on analysis variables
    analysis_vars = [dv] + ivs + controls
    subset = subset.dropna(subset=analysis_vars).reset_index(drop=True)

    if len(subset) == 0:
        raise ValueError("No complete cases after dropping missing values")

    # Normalize weights to have mean = 1
    wt_col = ""
    if weight_var and weight_var in subset.columns:
        wt_col = weight_var
        w = subset[wt_col].astype(float)
        # Drop rows with zero/negative/missing weights
        valid_wt = w > 0
        subset = subset[valid_wt].reset_index(drop=True)
        w = subset[wt_col].astype(float)
        subset[wt_col] = w / w.mean()

    return subset, wt_col

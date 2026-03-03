"""results.py — AnalysisResult dataclass for all statistical outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CoefficientRow:
    """A single row of a coefficient table."""

    term: str
    estimate: float = 0.0
    std_error: float = 0.0
    statistic: float = 0.0
    p_value: float = 1.0
    conf_low: float = 0.0
    conf_high: float = 0.0
    odds_ratio: float | None = None
    or_low: float | None = None
    or_high: float | None = None

    def to_dict(self) -> dict:
        d = {
            "term": self.term,
            "estimate": self.estimate,
            "std.error": self.std_error,
            "statistic": self.statistic,
            "p.value": self.p_value,
            "conf.low": self.conf_low,
            "conf.high": self.conf_high,
        }
        if self.odds_ratio is not None:
            d["odds_ratio"] = self.odds_ratio
            d["or_lo"] = self.or_low
            d["or_hi"] = self.or_high
        return d


@dataclass
class AnalysisResult:
    """Unified result from any statistical analysis."""

    ok: bool = True
    method: str = ""
    n: int = 0
    years_used: list[int] = field(default_factory=list)
    weight_var: str = ""
    coefficients: list[CoefficientRow] = field(default_factory=list)
    r_squared: float | None = None
    thresholds: list[dict] = field(default_factory=list)  # ordinal model thresholds

    # Chi-square specific
    chi2_statistic: float | None = None
    chi2_df: int | None = None
    chi2_p_value: float | None = None
    crosstab_pct: list[dict] | None = None
    dv: str = ""
    iv: str = ""

    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ok": self.ok,
            "method": self.method,
            "n": self.n,
            "years_used": self.years_used,
            "weight_var": self.weight_var,
        }
        if self.error:
            d["error"] = True
            d["message"] = self.error
            return d

        d["coefficients"] = [c.to_dict() for c in self.coefficients]

        if self.r_squared is not None:
            d["r_squared"] = self.r_squared
        if self.thresholds:
            d["thresholds"] = self.thresholds
        if self.chi2_statistic is not None:
            d["statistic"] = self.chi2_statistic
            d["df"] = self.chi2_df
            d["p_value"] = self.chi2_p_value
            d["dv"] = self.dv
            d["iv"] = self.iv
            if self.crosstab_pct:
                d["crosstab_pct"] = self.crosstab_pct

        return d

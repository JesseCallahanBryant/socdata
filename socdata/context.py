"""context.py — ResearchContext dataclass tracking the active research design."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ResearchContext:
    """Tracks the active research design during a session."""

    dataset: str = ""              # e.g. "gss", "anes", "wvs"
    dv: str = ""
    ivs: list[str] = field(default_factory=list)
    controls: list[str] = field(default_factory=list)
    dv_type: str = ""              # "binary", "ordinal", "continuous", "categorical"
    hypothesis: str = ""
    selected_method: str = ""      # "logistic", "ordinal", "ols", "chisq"
    years: list[int] = field(default_factory=list)
    weight_var: str = ""
    psu_var: str = ""
    strata_var: str = ""

    def is_ready_for_analysis(self) -> bool:
        return bool(self.dv and self.ivs and self.selected_method and self.dataset)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "dv": self.dv,
            "ivs": self.ivs,
            "controls": self.controls,
            "dv_type": self.dv_type,
            "hypothesis": self.hypothesis,
            "selected_method": self.selected_method,
            "years": self.years,
            "weight_var": self.weight_var,
            "psu_var": self.psu_var,
            "strata_var": self.strata_var,
        }

    def summary_text(self) -> str:
        lines = []
        if self.dataset:
            lines.append(f"Dataset: {self.dataset.upper()}")
        if self.dv:
            lines.append(f"DV: {self.dv} ({self.dv_type or 'type TBD'})")
        if self.ivs:
            lines.append(f"IVs: {', '.join(self.ivs)}")
        if self.controls:
            lines.append(f"Controls: {', '.join(self.controls)}")
        if self.years:
            lines.append(f"Years: {', '.join(str(y) for y in self.years)}")
        if self.hypothesis:
            lines.append(f"Hypothesis: {self.hypothesis}")
        if self.selected_method:
            lines.append(f"Method: {self.selected_method}")
        if self.weight_var:
            lines.append(f"Weight: {self.weight_var}")
        return "\n".join(lines) if lines else "No research context set yet."

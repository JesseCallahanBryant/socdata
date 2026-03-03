"""base.py — DatasetProvider ABC and VariableInfo dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class VariableInfo:
    """Metadata for a single dataset variable."""

    name: str
    label: str = ""
    type: str = ""             # "continuous", "binary", "ordinal", "categorical"
    n_valid: int = 0
    n_missing: int = 0
    categories: dict[Any, str] = field(default_factory=dict)  # value → label
    years_available: list[int] = field(default_factory=list)
    found: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "type": self.type,
            "n_valid": self.n_valid,
            "n_missing": self.n_missing,
            "categories": self.categories,
            "years_available": self.years_available,
            "found": self.found,
        }


class DatasetProvider(ABC):
    """Abstract base class for all dataset providers."""

    # Subclasses must set these
    name: str = ""               # short key, e.g. "gss"
    display_name: str = ""       # human-readable, e.g. "General Social Survey"
    description: str = ""        # one-liner
    url: str = ""                # official website
    default_weight: str = ""
    default_psu: str = ""
    default_strata: str = ""

    @abstractmethod
    def download(self, years: list[int] | None = None) -> pd.DataFrame:
        """Download/load the dataset, returning a DataFrame. Uses cache if available."""

    @abstractmethod
    def list_variables(self, df: pd.DataFrame) -> list[str]:
        """Return all variable names in the dataset."""

    @abstractmethod
    def inspect_variables(
        self, df: pd.DataFrame, var_names: list[str]
    ) -> dict[str, VariableInfo]:
        """Return VariableInfo for each requested variable."""

    def search_variables(
        self, df: pd.DataFrame, query: str
    ) -> list[VariableInfo]:
        """Search variables by name or label substring. Default implementation."""
        query_lower = query.lower()
        results = []
        for col in df.columns:
            if query_lower in col.lower():
                results.append(VariableInfo(name=col, label=col))
        return results[:50]

    def system_prompt_appendix(self) -> str:
        """Return dataset-specific text to append to the system prompt."""
        return ""

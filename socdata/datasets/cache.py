"""cache.py — Local file caching for downloaded datasets."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

CACHE_DIR = Path.home() / ".socdata" / "cache"


def cache_path(dataset: str, suffix: str = ".parquet") -> Path:
    """Return the cache file path for a dataset."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{dataset}{suffix}"


def has_cache(dataset: str) -> bool:
    """Check if a Parquet cache exists."""
    return cache_path(dataset).exists()


def load_cache(dataset: str) -> pd.DataFrame:
    """Load a cached Parquet file."""
    return pd.read_parquet(cache_path(dataset))


def save_cache(dataset: str, df: pd.DataFrame) -> Path:
    """Save a DataFrame to Parquet cache."""
    path = cache_path(dataset)
    df.to_parquet(path, index=False)
    return path


def download_file(url: str, dest: Path, label: str = "") -> Path:
    """Download a file from URL to dest, showing progress via Rich."""
    from rich.progress import Progress

    dest.parent.mkdir(parents=True, exist_ok=True)

    with Progress() as progress:
        task = progress.add_task(
            f"Downloading {label or dest.name}...", total=None
        )

        def _reporthook(block_num, block_size, total_size):
            if total_size > 0:
                progress.update(task, total=total_size, completed=block_num * block_size)

        urllib.request.urlretrieve(url, str(dest), reporthook=_reporthook)

    return dest

#!/usr/bin/env python3
"""Convert official NORC GSS SAS release to parquet + metadata JSON.

Reads:
  data/GSS Data/gss7224_r2.sas7bdat  (2.4GB cumulative SAS dataset)

Value labels are carried forward from the existing meta.json (the SAS format
catalog formats.sas7bcat uses a format pyreadstat can't parse, but the labels
are the same across GSS releases). New variables from the SAS file that aren't
in the old meta get variable labels but empty value labels.

Writes:
  data/gss_cumulative.parquet
  data/gss_cumulative_meta.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pyreadstat

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data"
SAS_DIR = DATA_DIR / "GSS Data"

SAS_FILE = SAS_DIR / "gss7224_r2.sas7bdat"

OUT_PARQUET = DATA_DIR / "gss_cumulative.parquet"
OUT_META = DATA_DIR / "gss_cumulative_meta.json"
OLD_META = DATA_DIR / "gss_cumulative_meta.json"  # reuse existing value labels


def main() -> None:
    if not SAS_FILE.exists():
        raise FileNotFoundError(f"Missing: {SAS_FILE}")

    # Load existing value labels from prior Stata-based conversion
    old_value_labels: dict[str, dict[str, str]] = {}
    if OLD_META.exists():
        old = json.loads(OLD_META.read_text(encoding="utf-8"))
        old_value_labels = old.get("value_labels", {})
        print(f"Loaded {len(old_value_labels)} existing value-label mappings")

    print(f"Reading {SAS_FILE.name} ...")
    t0 = time.time()
    df, meta = pyreadstat.read_sas7bdat(str(SAS_FILE), encoding="latin1")
    elapsed = time.time() - t0
    print(f"  Read in {elapsed:.1f}s  —  {len(df):,} rows × {len(df.columns):,} columns")

    # Lowercase column names (convention used by tool.py / gss.py)
    df.columns = [c.lower() for c in df.columns]

    # Build metadata ---------------------------------------------------------
    # variable_labels: {lowercase_name: label}  (fresh from SAS metadata)
    variable_labels = {}
    for name, label in zip(meta.column_names, meta.column_labels):
        variable_labels[name.lower()] = label or ""

    # value_labels: carry forward from old meta, keyed by lowercase name
    value_labels = old_value_labels

    # Report new variables not in old metadata
    new_vars = set(df.columns) - set(old_value_labels.keys())
    if new_vars:
        print(f"  {len(new_vars)} new variables without value labels (e.g. {sorted(new_vars)[:5]})")

    # Write parquet ----------------------------------------------------------
    print(f"Writing {OUT_PARQUET.name} ...")
    df.to_parquet(OUT_PARQUET, index=False)
    pq_size = OUT_PARQUET.stat().st_size / 1024 / 1024
    print(f"  {pq_size:.1f} MB")

    # Write metadata JSON ----------------------------------------------------
    print(f"Writing {OUT_META.name} ...")
    OUT_META.write_text(
        json.dumps(
            {"variable_labels": variable_labels, "value_labels": value_labels},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    meta_size = OUT_META.stat().st_size / 1024 / 1024
    print(f"  {meta_size:.1f} MB")

    # Summary ----------------------------------------------------------------
    years = sorted(df["year"].dropna().unique().astype(int).tolist()) if "year" in df.columns else []
    print(f"\nDone. {len(df):,} rows, {len(df.columns):,} columns")
    if years:
        print(f"Years: {years[0]}–{years[-1]}  ({len(years)} waves)")


if __name__ == "__main__":
    main()

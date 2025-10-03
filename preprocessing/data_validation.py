"""
LeADS D7.4 – Minimal Data Validation (Phase 1)
MIT License

Lightweight checks to ensure inputs resemble the expected BAAC-like schema.
Phase 1 keeps dependencies minimal and behavior transparent.

Phase 2 will add: strict schema contracts, semantic constraints,
cross-table integrity, and dataset-level consistency rules.
"""

from __future__ import annotations
from typing import Iterable, Dict, List, Tuple
import pandas as pd
import numpy as np
import re

# Columns surfaced in the public context (superset-friendly; missing allowed)
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "ID_accident",
    "Date_and_hour",
    "Security_measures",
    "User_of_security_measures",
    "Place",
    "Sex",
    "Light",
    "User_category",
    "Intersection",
    "Weather_condition",
    "Collision",
    "Surface",
    "Circulation",
    "Width_of_the_roadway",
    "Width_of_the_central_bar",
    "Number_of_channels",
    "Road_category",
    "Plan",
    "Situation",
    "Year_of_birth",
    "Pedestrian_action",
    "Health_condition",
    "Point_of_shock",
    "Manuver",
    "Vehicle_category",
    # Some datasets include these sparsely:
    "Reserved_lane",
    "Infrastructure",
    "Profile",
)

_NUMERIC_SUGGESTED = (
    "Width_of_the_roadway",
    "Width_of_the_central_bar",
    "Number_of_channels",
)

_ID_CLEAN_RE = re.compile(r"[,\s]")

def validate_required_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if any required column is entirely missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_id_column(
    df: pd.DataFrame,
    id_col: str = "ID_accident",
    target_type: str = "string",
) -> pd.DataFrame:
    """
    Normalize ID values that may appear as Excel-like strings (e.g., '2,01E+11').
    - Remove commas/whitespace
    - Optionally cast to int if safe; otherwise keep as string
    """
    out = df.copy()
    if id_col not in out.columns:
        return out

    s = out[id_col].astype(str).str.strip()
    s = s.str.replace(_ID_CLEAN_RE, "", regex=True)
    if target_type == "int":
        # Best-effort integer cast; fall back to string on failure
        tmp = pd.to_numeric(s, errors="coerce")
        if tmp.notna().all():
            out[id_col] = tmp.astype("Int64")
        else:
            out[id_col] = s  # keep string if any NA after coercion
    else:
        out[id_col] = s
    return out


def coerce_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: Iterable[str] = _NUMERIC_SUGGESTED,
    errors: str = "coerce",
) -> pd.DataFrame:
    """Coerce suggested numeric columns to numeric dtype."""
    out = df.copy()
    for c in numeric_columns:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors=errors)
    return out


def basic_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a compact, human-friendly quality report.

    Metrics
    -------
    - dtype
    - non_null
    - null_pct
    - n_unique (for object columns)
    - example_values (first 3 unique)
    """
    rows: List[Dict[str, object]] = []
    for c in df.columns:
        s = df[c]
        non_null = int(s.notna().sum())
        null_pct = float(s.isna().mean() * 100)
        dtype = str(s.dtype)
        entry: Dict[str, object] = {
            "column": c,
            "dtype": dtype,
            "non_null": non_null,
            "null_pct": round(null_pct, 2),
        }
        if s.dtype == "object":
            uniq = s.dropna().unique().tolist()
            entry["n_unique"] = len(uniq)
            entry["example_values"] = uniq[:3]
        rows.append(entry)
    return pd.DataFrame(rows).sort_values(by="null_pct", ascending=False).reset_index(drop=True)

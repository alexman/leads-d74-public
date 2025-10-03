"""
LeADS D7.4 – Temporal Feature Utilities (Phase 1)
MIT License

Public, minimal, timezone-aware temporal parsing and enrichment:
- parse_datetime_column: parse ISO strings with timezone to UTC
- add_accident_time_parts: Accident_Year, Date, Time, Hour, Month
- add_temporal_features: dayofweek, weekend, part_of_day, rush_hour
- derive_age_from_year_of_birth: numeric age from 'Year_of_birth' (clipped)

Full holiday calendars, episode stitching, and cross-table temporal joins
are part of Phase 2.
"""

from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np

def parse_datetime_column(
    df: pd.DataFrame,
    source_col: str = "Date_and_hour",
    target_col: str = "dt",
    utc: bool = True,
    errors: str = "coerce",
) -> pd.DataFrame:
    """
    Parse datetime strings (ISO with timezone) into pandas datetime.
    If utc=True, convert to UTC timezone-aware dtype.
    """
    out = df.copy()
    s = pd.to_datetime(out[source_col], errors=errors, utc=utc)
    out[target_col] = s
    return out


def add_accident_time_parts(
    df: pd.DataFrame,
    dt_col: str = "dt",
    year_col: str = "Accident_Year",
) -> pd.DataFrame:
    """
    Derive standard time parts used in the EDA notebook.
    Adds: Accident_Year, Date, Time, Hour, Month.
    """
    out = df.copy()
    s = out[dt_col]
    out[year_col] = s.dt.year
    out["Date"] = s.dt.date
    out["Time"] = s.dt.time
    out["Hour"] = s.dt.hour
    out["Month"] = s.dt.month
    return out


def _part_of_day(hour: float) -> str:
    if pd.isna(hour):
        return "unknown"
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def add_temporal_features(
    df: pd.DataFrame,
    dt_col: str = "dt",
    prefix: str = "t_",
) -> pd.DataFrame:
    """
    Add lightweight temporal features for EDA and simple baselines:
    - t_dayofweek (Mon=0..Sun=6)
    - t_weekend (0/1)
    - t_part_of_day (categorical)
    - t_rush_hour (0/1)  -- rough proxy (7–9, 16–19)
    """
    out = df.copy()
    s = out[dt_col]
    out[f"{prefix}dayofweek"] = s.dt.dayofweek
    out[f"{prefix}weekend"] = out[f"{prefix}dayofweek"].isin([5, 6]).astype(int)
    # prefer Hour if already present; else compute ad hoc
    hour = out["Hour"] if "Hour" in out.columns else s.dt.hour
    out[f"{prefix}part_of_day"] = hour.apply(_part_of_day)
    out[f"{prefix}rush_hour"] = hour.isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    return out


def derive_age_from_year_of_birth(
    df: pd.DataFrame,
    yob_col: str = "Year_of_birth",
    reference_year_col: str = "Accident_Year",
    target_col: str = "Age",
    clip_range: tuple[int, int] = (0, 110),
) -> pd.DataFrame:
    """
    Derive age = Accident_Year - Year_of_birth (as used in your EDA).
    """
    out = df.copy()
    # numeric coercions
    yob = pd.to_numeric(out[yob_col], errors="coerce")
    ref_year = pd.to_numeric(out[reference_year_col], errors="coerce")
    age = ref_year - yob
    out[target_col] = age.clip(lower=clip_range[0], upper=clip_range[1])
    return out

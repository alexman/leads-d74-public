"""
LeADS D7.4 – Exploratory Plots (Phase 1)
MIT License

Matplotlib-only EDA helpers tailored for the BAAC-like accident dataset.

Design principles
-----------------
- Single chart per function (no subplots) to keep Phase 1 simple.
- Sensible defaults; functions return the created Axes.
- Works with the public preprocessing utilities (e.g., parsed 'dt', 'Hour', etc.).
- Avoids heavyweight deps (no seaborn, no geopandas). Basic lat/long scatter only.

Important columns in typical inputs
-----------------------------------
- 'Date_and_hour' (string) -> use preprocessing.parse_datetime_column to create 'dt'
- 'Latitude', 'Longitude' (string/object) -> coerced internally for plotting
- category-like columns: 'Weather_condition', 'Collision', 'Surface', etc.

Phase 2
-------
Will include richer styling, interactive widgets, map baselayers, and
report-generation pipelines aligned with the complete D7.4 framework.
"""

from __future__ import annotations
from typing import Optional, Iterable, Tuple, List
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _validate_series_exists(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe.")


def plot_category_counts(
    df: pd.DataFrame,
    column: str,
    top_n: Optional[int] = 20,
    normalize: bool = False,
    title: Optional[str] = None,
    rotation: int = 45,
) -> plt.Axes:
    """
    Bar chart of category frequencies for a given column.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
        Categorical column name.
    top_n : int or None, default=20
        Show only the top N categories by count. If None, show all.
    normalize : bool, default=False
        If True, plot percentages instead of counts.
    title : str, optional
    rotation : int, default=45
        Rotation for x tick labels.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _validate_series_exists(df, column)
    s = df[column].astype("string").fillna("NA")
    counts = s.value_counts(normalize=normalize, dropna=False)
    if top_n is not None:
        counts = counts.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    counts.iloc[::-1].plot(kind="barh", ax=ax)  # horizontal for readability
    ax.set_xlabel("Percentage" if normalize else "Count")
    ax.set_ylabel(column)
    ax.set_title(title or f"{column} – {'Percentage' if normalize else 'Count'} (Top {top_n})")
    plt.tight_layout()
    return ax


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Histogram for a numeric column.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
        Numeric column to histogram; coerced to numeric with errors='coerce'.
    bins : int, default=30
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    _validate_series_exists(df, column)
    s = pd.to_numeric(df[column], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(s, bins=bins)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(title or f"Histogram of {column}")
    plt.tight_layout()
    return ax


def plot_correlation_heatmap(
    df: pd.DataFrame,
    numeric_columns: Optional[Iterable[str]] = None,
    title: str = "Correlation Heatmap",
) -> plt.Axes:
    """
    Heatmap of Pearson correlations over selected numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_columns : iterable of str, optional
        If None, auto-detect numeric columns.
    title : str, default="Correlation Heatmap"

    Returns
    -------
    matplotlib.axes.Axes
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_columns) == 0:
        raise ValueError("No numeric columns available for correlation.")

    corr = df[numeric_columns].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(numeric_columns)))
    ax.set_yticks(range(len(numeric_columns)))
    ax.set_xticklabels(numeric_columns, rotation=45, ha="right")
    ax.set_yticklabels(numeric_columns)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return ax


def plot_nulls_bar(
    df: pd.DataFrame,
    title: str = "Nulls per Column (%)",
    sort_desc: bool = True,
) -> plt.Axes:
    """
    Bar chart of null percentage per column.

    Parameters
    ----------
    df : pd.DataFrame
    title : str
    sort_desc : bool, default=True
        Sort by descending null percentage.

    Returns
    -------
    matplotlib.axes.Axes
    """
    null_pct = df.isna().mean().mul(100).round(2)
    if sort_desc:
        null_pct = null_pct.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    null_pct.plot(kind="barh", ax=ax)
    ax.set_xlabel("Null percentage (%)")
    ax.set_ylabel("Columns")
    ax.set_title(title)
    plt.tight_layout()
    return ax


def plot_time_series_counts(
    df: pd.DataFrame,
    dt_col: str = "dt",
    freq: str = "M",
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Line plot of counts over time at a specified frequency.

    Parameters
    ----------
    df : pd.DataFrame
    dt_col : str, default='dt'
        Parsed datetime column (timezone-aware is fine).
    freq : {'H','D','W','M','Q','Y'}, default='M'
        Resampling frequency.
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    _validate_series_exists(df, dt_col)
    s = pd.to_datetime(df[dt_col], errors="coerce")
    ts = (
        pd.Series(1, index=s)
        .sort_index()
        .resample(freq)
        .sum()
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts.index, ts.values)
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    ax.set_title(title or f"Events per {freq}")
    plt.tight_layout()
    return ax


def plot_geo_scatter(
    df: pd.DataFrame,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    title: str = "Accidents – Latitude/Longitude (raw scatter)",
    alpha: float = 0.6,
    size: int = 20,
) -> plt.Axes:
    """
    Simple latitude/longitude scatter (no basemap).

    Parameters
    ----------
    df : pd.DataFrame
    lat_col : str
    lon_col : str
    title : str
    alpha : float, default=0.6
    size : int, default=20

    Returns
    -------
    matplotlib.axes.Axes

    Notes
    -----
    - Coerces lat/long with errors='coerce'; drops invalid rows.
    - For real maps, Phase 2 will include basemaps and projections.
    """
    _validate_series_exists(df, lat_col)
    _validate_series_exists(df, lon_col)

    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    mask = lat.notna() & lon.notna()
    lat = lat[mask]
    lon = lon[mask]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(lon.values, lat.values, s=size, alpha=alpha)
    ax.set_xlabel(lon_col)
    ax.set_ylabel(lat_col)
    ax.set_title(title)
    # try to set aspect equal if ranges are comparable
    if lat.size > 1 and lon.size > 1:
        lat_range = lat.max() - lat.min()
        lon_range = lon.max() - lon.min()
        if lat_range > 0 and lon_range > 0:
            ax.set_aspect(lon_range / lat_range)
    plt.tight_layout()
    return ax


def plot_multivalue_token_counts(
    df: pd.DataFrame,
    column: str,
    sep: str = ",",
    top_n: Optional[int] = 20,
    normalize: bool = False,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Bar chart of token frequencies for a *multi-value* text column.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
        Column containing delimited tokens (e.g., 'A,B,C').
    sep : str, default=','
    top_n : int or None, default=20
        Show top N tokens; None means all tokens.
    normalize : bool, default=False
        If True, plot percentages instead of counts.
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    _validate_series_exists(df, column)

    # Split and flatten
    def _split(x) -> List[str]:
        if pd.isna(x):
            return []
        return [t.strip() for t in str(x).split(sep) if t is not None and t.strip() != ""]

    tokens = df[column].apply(_split)
    all_tokens = [t for L in tokens for t in L]

    if len(all_tokens) == 0:
        raise ValueError(f"No tokens found in column '{column}' using sep='{sep}'.")

    s = pd.Series(all_tokens)
    counts = s.value_counts(normalize=normalize)
    if top_n is not None:
        counts = counts.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    counts.iloc[::-1].plot(kind="barh", ax=ax)
    ax.set_xlabel("Percentage" if normalize else "Count")
    ax.set_ylabel(f"{column} tokens")
    ax.set_title(title or f"{column} – token {'percentage' if normalize else 'count'} (Top {top_n})")
    plt.tight_layout()
    return ax

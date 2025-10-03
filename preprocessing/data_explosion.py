"""
LeADS D7.4 – Data Explosion Utilities (Phase 1)
MIT License

Public version of the "data explosion" idea for BAAC-like inputs.

Key capabilities
----------------
1) detect_multivalue_columns: flag likely multi-value text fields
2) explode_aligned_columns: jointly explode *aligned* multi-value columns
   (mirrors the EDA notebook approach using an internal zip-like strategy)
3) one_hot_multivalue_columns: simplified one-hotting for multi-value fields

Notes
-----
- Default separator is ',' to match the shared EDA and sample dataset.
- Full, publication-grade encoders (e.g., token normalization, weighting,
  rare-token grouping, and reconciliation across tables) come in Phase 2.
"""

from __future__ import annotations
from typing import Iterable, List, Dict, Optional, Set
import pandas as pd

def detect_multivalue_columns(
    df: pd.DataFrame,
    candidate_columns: Optional[Iterable[str]] = None,
    sep: str = ",",
    min_share: float = 0.01,
) -> List[str]:
    """Heuristically detect columns likely to contain multi-value strings."""
    if candidate_columns is None:
        candidate_columns = [c for c in df.columns if df[c].dtype == "object"]

    flagged: List[str] = []
    for c in candidate_columns:
        s = df[c].dropna().astype(str)
        share = (s.str.contains(sep, regex=False)).mean() if len(s) else 0.0
        if share >= min_share:
            flagged.append(c)
    return flagged


def _split_to_list(x: object, sep: str) -> List[str]:
    """Split a cell into a list, keeping empty->[]; cast to str safely."""
    if pd.isna(x):
        return []
    parts = str(x).split(sep)
    # Strip and drop pure-empty tokens
    return [p.strip() for p in parts if p is not None and p.strip() != ""]


def _align_lists(row_lists: List[List[str]]) -> List[List[str]]:
    """
    Pad lists in a row to the same length (right-pad with None) so they can be zipped.
    This makes the explode operation robust to small per-column length mismatches.
    """
    max_len = max((len(L) for L in row_lists), default=0)
    aligned = []
    for L in row_lists:
        padded = L + [None] * (max_len - len(L))
        aligned.append(padded)
    return aligned


def explode_aligned_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    sep: str = ",",
    keep_empty_rows: bool = False,
    strict_equal_lengths: bool = False,
) -> pd.DataFrame:
    """
    Jointly explode a *set of aligned multi-value columns*.

    Example
    -------
    If the row has:
      Security_measures: "Seat Belt,Helmet"
      User_of_security_measures: "Yes,Yes"
      Sex: "Man,Woman"
    We split each to lists, align lengths, create tuples per index, then explode.

    Parameters
    ----------
    df : pd.DataFrame
    columns : iterable of str
        Columns to be jointly exploded.
    sep : str, default=','
        Multi-value separator.
    keep_empty_rows : bool
        If False, rows where *all* selected columns are empty will be dropped.
    strict_equal_lengths : bool
        If True, raise ValueError if per-row list lengths differ; if False,
        pad with None on shorter lists (public-safe default).

    Returns
    -------
    pd.DataFrame
        Exploded frame with original columns preserved and selected columns
        replaced by their exploded single values.
    """
    out = df.copy()

    # Build combined tuples
    split_cols = {c: out[c].apply(lambda x: _split_to_list(x, sep)) for c in columns}

    def _combine_row(idx) -> List[tuple]:
        lists = [split_cols[c].iat[idx] for c in columns]
        if strict_equal_lengths:
            lengths = {len(L) for L in lists}
            if len(lengths) > 1:
                raise ValueError(f"Row {idx} has unequal token lengths in {columns}: {lengths}")
            aligned = lists
        else:
            aligned = _align_lists(lists)
        # zip_longest-like (we padded already)
        tuples = list(zip(*aligned))
        # Drop all-None tuples
        tuples = [t for t in tuples if any(v is not None and str(v).strip() != "" for v in t)]
        return tuples

    combined = [ _combine_row(i) for i in range(len(out)) ]
    out = out.assign(_combined=combined).explode("_combined", ignore_index=True)

    if not keep_empty_rows:
        out = out[out["_combined"].notna()]

    # Unpack tuples back to the original columns
    if len(columns) == 1:
        # When only one column is passed, _combined is a scalar, not a tuple
        col = next(iter(columns))
        out[col] = out["_combined"]
    else:
        expanded = pd.DataFrame(out["_combined"].tolist(), columns=list(columns), index=out.index)
        for c in columns:
            out[c] = expanded[c]

    out = out.drop(columns=["_combined"]).reset_index(drop=True)
    return out


def one_hot_multivalue_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    sep: str = ",",
    min_count: int = 5,
    prefix_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Create binary indicator columns for tokens from multi-value fields.
    Public-friendly: only materialize tokens with frequency >= min_count.
    """
    out = df.copy()
    prefix_map = prefix_map or {}
    for col in columns:
        tokens = out[col].apply(lambda x: _split_to_list(x, sep))
        # Frequency table
        counts: Dict[str, int] = {}
        for L in tokens:
            for t in L:
                counts[t] = counts.get(t, 0) + 1
        keep: Set[str] = {t for t, c in counts.items() if c >= min_count}
        pref = prefix_map.get(col, col)
        for t in sorted(keep):
            col_name = f"{pref}__{t}".replace(" ", "_")
            out[col_name] = tokens.apply(lambda L, tok=t: int(tok in L))
    return out

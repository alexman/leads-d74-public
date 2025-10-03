"""
LeADS D7.4 – Public Visualization Package (Phase 1)
MIT License

Lightweight EDA plotting helpers for BAAC-like datasets, using matplotlib only.
Each function creates a single figure (no subplots) and returns the matplotlib
Axes so callers can further customize or save the plot.

Notes:
- Designed to pair with the public preprocessing utilities from Phase 1.
- Phase 2 (post-publication) will add plot themes, faceting, geospatial
  basemaps, interactive widgets, and standardized reporting templates.
"""

from .exploratory_plots import (
    plot_category_counts,
    plot_histogram,
    plot_correlation_heatmap,
    plot_nulls_bar,
    plot_time_series_counts,
    plot_geo_scatter,
    plot_multivalue_token_counts,
)

__all__ = [
    "plot_category_counts",
    "plot_histogram",
    "plot_correlation_heatmap",
    "plot_nulls_bar",
    "plot_time_series_counts",
    "plot_geo_scatter",
    "plot_multivalue_token_counts",
]

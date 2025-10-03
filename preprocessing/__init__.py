"""
LeADS D7.4 – Public Preprocessing Package (Phase 1)
MIT License

Lightweight utilities used in the public synthesis of D7.4:
- data_explosion: aligned multi-value "data explosion" (public version)
- temporal_features: timezone-aware parsing & temporal features
- data_validation: minimal schema checks and ID normalization

NOTE: Advanced feature engineering, conflict resolution, learned encoders,
and production pipelines (82% DL severity prediction, clustering, etc.)
arrive in Phase 2 (EUPL/GPL) after publication.
"""

from .data_explosion import (
    detect_multivalue_columns,
    explode_aligned_columns,
    one_hot_multivalue_columns,
)

from .temporal_features import (
    parse_datetime_column,
    add_temporal_features,
    derive_age_from_year_of_birth,
    add_accident_time_parts,
)

from .data_validation import (
    REQUIRED_COLUMNS,
    validate_required_columns,
    coerce_numeric_columns,
    basic_quality_report,
    normalize_id_column,
)

__all__ = [
    # data_explosion
    "detect_multivalue_columns",
    "explode_aligned_columns",
    "one_hot_multivalue_columns",
    # temporal_features
    "parse_datetime_column",
    "add_temporal_features",
    "derive_age_from_year_of_birth",
    "add_accident_time_parts",
    # data_validation
    "REQUIRED_COLUMNS",
    "validate_required_columns",
    "coerce_numeric_columns",
    "basic_quality_report",
    "normalize_id_column",
]

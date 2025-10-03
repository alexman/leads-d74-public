# Implementation Guide (Phase 1)

##This guide shows how to use the public utilities with BAAC-like accident data.

## 1) Load and basic normalization
```python
import pandas as pd
from preprocessing import normalize_id_column, parse_datetime_column, add_accident_time_parts

df = pd.read_csv("data/accidents-corporels-de-la-circulation-millesime_eng_columns_selected_data_translated_sample.csv")
df = normalize_id_column(df, id_col="ID_accident")
df = parse_datetime_column(df, source_col="Date_and_hour", target_col="dt", utc=True)
df = add_accident_time_parts(df, dt_col="dt")

## 2) Detect & handle multi-value fields
from preprocessing import detect_multivalue_columns, explode_aligned_columns, one_hot_multivalue_columns

mv_cols = detect_multivalue_columns(df)  # default sep=","
# Example of aligned explode for two related columns:
df_exp = explode_aligned_columns(
    df,
    columns=["Security_measures", "User_of_security_measures"],
    sep=",",
    strict_equal_lengths=False  # pad shorter lists with None
)
##3) Quick EDA
from preprocessing import basic_quality_report
from visualization import plot_category_counts, plot_nulls_bar, plot_time_series_counts, plot_geo_scatter

qr = basic_quality_report(df)
plot_nulls_bar(df)
plot_category_counts(df, "Weather_condition", top_n=10)
plot_time_series_counts(df, dt_col="dt", freq="M")
plot_geo_scatter(df, lat_col="Latitude", lon_col="Longitude")

##4) Phase 2 (post-publication)
'''
Learned encoders, probabilistic token reconciliation

Temporal normalization & holiday calendars

Clustering & severity prediction pipelines (~82% internal accuracy)

Schema contracts & cross-table integrity checks

Reporting templates (EUPL/GPL) 


## docs/data_requirements.md

# Data Requirements (Sample-Friendly)

This repo expects a BAAC-like CSV with (subset of) the following columns:

- **ID_accident** (string/integer-like; may require normalization)
- **Date_and_hour** (ISO or parseable string; tz-aware supported)
- **Latitude**, **Longitude** (optional but useful for plots)
- **Security_measures**, **User_of_security_measures** (often multi-value, comma-separated)
- **Place**, **Sex**, **User_category**, **Intersection**, **Weather_condition**, **Collision**, **Surface**, **Circulation**
- **Width_of_the_roadway**, **Width_of_the_central_bar**, **Number_of_channels** (numeric coercion recommended)
- **Road_category**, **Plan**, **Situation**, **Year_of_birth**, **Pedestrian_action**, **Health_condition**, **Point_of_shock**, **Manuver**, **Vehicle_category**
- Sparse/optional: **Reserved_lane**, **Infrastructure**, **Profile**

### Types (typical)
- `Date_and_hour`: string → parsed to datetime with `utc=True`
- Multi-value text: comma-separated (`"A,B,C"`)
- Numeric widths/channels: coercible to float/int

### Notes
- Rows with missing geo or mismatched multi-value lengths are handled safely (padding or dropping empty tuples).
- If your headers differ slightly, adjust the function arguments (e.g., `lat_col`, `lon_col`).
'''
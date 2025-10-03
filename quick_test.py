# quick_test.py
import os
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import (
    parse_datetime_column, add_accident_time_parts, normalize_id_column
)
from visualization import (
    plot_category_counts, plot_nulls_bar, plot_time_series_counts,
    plot_geo_scatter, plot_multivalue_token_counts
)

CSV_PATH = "data/accidents-corporels-de-la-circulation-millesime_eng_columns_selected_data_translated_sample.csv"
PLOTS_DIR = "plots"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def savefig(ax, filename: str):
    ensure_dir(PLOTS_DIR)
    out_path = os.path.join(PLOTS_DIR, filename)
    # save the current figure linked to this Axes
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

def guess_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    # Load
    df = pd.read_csv(CSV_PATH)
    print("Loaded rows:", len(df))
    print("Columns:", list(df.columns))

    # Basic normalization & time parsing
    if "ID_accident" in df.columns:
        df = normalize_id_column(df, id_col="ID_accident")
    df = parse_datetime_column(df, source_col="Date_and_hour", target_col="dt", utc=True)
    df = add_accident_time_parts(df, dt_col="dt")

    # 1) Category counts
    if "Weather_condition" in df.columns:
        ax = plot_category_counts(df, "Weather_condition", top_n=10)
        savefig(ax, "weather_condition_top10.png")
    else:
        print("Weather_condition not found — skipping category counts.")

    # 2) Nulls bar
    ax = plot_nulls_bar(df)
    savefig(ax, "nulls_per_column.png")

    # 3) Time series counts
    ax = plot_time_series_counts(df, dt_col="dt", freq="M")
    savefig(ax, "events_per_month.png")

    # 4) Geo scatter (auto-detect columns)
    lat_candidates = ["Latitude", "latitude", "lat", "Lat", "LAT", "Y", "y"]
    lon_candidates = ["Longitude", "longitude", "lon", "Lon", "LON", "X", "x"]
    lat_col = guess_col(df, lat_candidates)
    lon_col = guess_col(df, lon_candidates)
    if lat_col and lon_col:
        print(f"Using geo columns: {lat_col}, {lon_col}")
        ax = plot_geo_scatter(df, lat_col=lat_col, lon_col=lon_col)
        savefig(ax, "geo_scatter.png")
    else:
        print("Geo columns not found — skipping geo scatter.")

    # 5) Multi-value token counts
    if "Security_measures" in df.columns:
        ax = plot_multivalue_token_counts(df, "Security_measures", sep=",", top_n=15)
        savefig(ax, "security_measures_tokens.png")
    else:
        print("Security_measures not found — skipping token bar chart.")

    # Show windows if your setup supports GUI backends
    try:
        plt.show()
    except Exception as e:
        print(f"plt.show() skipped (backend likely non-interactive): {e}")

if __name__ == "__main__":
    main()

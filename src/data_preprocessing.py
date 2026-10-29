"""Phase 1 + 3: dataset audit and preprocessing.

Outputs
- data/processed/water_hourly_clean.csv   regular hourly series + flags
- outputs/metrics/data_audit.csv           per-column audit table
- outputs/metrics/data_audit_summary.json  record counts, gaps, decisions
"""
import json

import numpy as np
import pandas as pd

import config as C

NUM_COLS = ["Consumption", "Temperature", "Rain", "Sun"]


def read_variant(path):
    df = pd.read_csv(path, dtype=str)
    df["Time"] = df["Time"].str.strip()
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c].str.strip(), errors="coerce")
    df["start_hour"] = df["Time"].str.slice(0, 2).astype(int)
    df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    return df


def nan_runs(mask, labels):
    """Return (start_label, length) for each consecutive run of True."""
    runs, i, m = [], 0, np.asarray(mask)
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            runs.append((str(labels[i]), int(j - i)))
            i = j
        else:
            i += 1
    return runs


def audit_columns(df, name):
    rows = []
    desc = {
        "Date": ("string (M/D/YYYY)", "Calendar date", "-"),
        "Time": ("string (HH:MM-HH:MM)", "Hourly interval", "-"),
        "Consumption": ("float", "Water volume delivered in the hour", "m^3 per hour"),
        "Temperature": ("float", "Local air temperature", "deg C"),
        "Rain": ("float", "Local precipitation", "arbitrary units"),
        "Sun": ("float", "Local sunshine intensity", "arbitrary units"),
    }
    for col, (dtype, d, unit) in desc.items():
        s = df[col]
        n_miss = int(s.isna().sum())
        rows.append({
            "variant": name, "feature": col, "data_type": dtype, "description": d,
            "unit": unit, "missing": n_miss, "missing_pct": round(100 * n_miss / len(df), 3),
            "unique": int(s.nunique()),
            "min": s.min() if col in NUM_COLS else "",
            "max": s.max() if col in NUM_COLS else "",
        })
    return rows


def main():
    raw = read_variant(C.RAW_FILE)
    nonan = read_variant(C.NONAN_FILE)
    fixed = read_variant(C.FIXED_FILE)

    summary = {"source": "Weyrauch et al. (2024), Zenodo, doi:10.5281/zenodo.11045013, CC-BY-4.0"}
    for name, df in [("raw", raw), ("noNaN", nonan), ("noNaN_fixedTimechange", fixed)]:
        per_day = df.groupby("Date").size()
        summary[name] = {
            "rows": len(df),
            "days": int(per_day.size),
            "hours_per_day_counts": {str(k): int(v) for k, v in per_day.value_counts().items()},
            "duplicate_rows": int(df.duplicated(subset=["Date", "Time"] + NUM_COLS).sum()),
            "duplicate_date_time": int(df.duplicated(["Date", "Time"]).sum()),
            "missing_by_column": {c: int(df[c].isna().sum()) for c in ["Date", "Time"] + NUM_COLS},
            "negative_consumption": int((df.Consumption < 0).sum()),
            "zero_consumption": int((df.Consumption == 0).sum()),
            "sorted_chronologically": bool(df["date"].is_monotonic_increasing),
        }
    summary["raw"]["consumption_nan_runs"] = nan_runs(
        raw.Consumption.isna(), (raw.Date + " " + raw.Time).values)
    summary["raw"]["irregular_time_labels"] = sorted(
        set(raw.Time) - {f"{h:02d}:00-{(h + 1) % 24:02d}:00" for h in range(24)})

    audit = pd.DataFrame(audit_columns(raw, "raw") + audit_columns(fixed, "noNaN_fixedTimechange"))
    audit.to_csv(C.METRIC_DIR / "data_audit.csv", index=False)

    # ---- Build the modelling series from the time-change-fixed variant -----
    # It is the only variant with exactly 24 rows per day, which seasonal
    # periods of 24 and 168 hours require.
    fixed["timestamp"] = fixed["date"] + pd.to_timedelta(fixed["start_hour"], unit="h")
    fixed = fixed.sort_values("timestamp").reset_index(drop=True)
    assert fixed["timestamp"].is_unique, "duplicate timestamps"
    full_idx = pd.date_range(fixed.timestamp.iloc[0], fixed.timestamp.iloc[-1], freq=C.FREQ)
    assert len(full_idx) == len(fixed), "hourly grid has holes"

    # Flag every hour whose value the publisher created or changed, by
    # comparing with the raw file on the (date, start hour) key.
    raw_key = raw.drop_duplicates(["date", "start_hour"], keep="first").set_index(["date", "start_hour"])
    key = pd.MultiIndex.from_arrays([fixed.date, fixed.start_hour])
    raw_vals = raw_key["Consumption"].reindex(key).to_numpy()
    raw_nan = np.isnan(raw_vals)
    absent = ~key.isin(raw_key.index)
    changed = ~raw_nan & (np.abs(raw_vals - fixed.Consumption.to_numpy()) > 1e-9)
    dst_dates = set(raw.groupby("Date").size().loc[lambda s: s != 24].index)
    fixed["imputed_missing"] = raw_nan & ~absent
    fixed["dst_adjusted"] = absent | (changed & fixed.Date.isin(dst_dates).to_numpy())
    fixed["other_modified"] = changed & ~fixed.Date.isin(dst_dates).to_numpy()
    fixed["publisher_modified"] = fixed[["imputed_missing", "dst_adjusted", "other_modified"]].any(axis=1)

    clean = (fixed.set_index("timestamp")
             .rename(columns={"Consumption": C.TARGET, "Temperature": "temperature",
                              "Rain": "rain", "Sun": "sun"})
             [[C.TARGET, "temperature", "rain", "sun", "imputed_missing",
               "dst_adjusted", "other_modified", "publisher_modified"]])
    clean.index.name = "timestamp"
    clean["hour"] = clean.index.hour
    clean["dayofweek"] = clean.index.dayofweek
    clean["weekend"] = (clean.dayofweek >= 5).astype(int)
    clean["month"] = clean.index.month

    # Long gap: show where the publisher filled with multi-year averages.
    long_gap = clean.index[clean.imputed_missing]
    summary["modelling_series"] = {
        "variant_used": "waterconsumption_noNaN_fixedTimechange.csv",
        "start": str(clean.index[0]), "end": str(clean.index[-1]),
        "rows_before_cleaning": int(len(fixed)), "rows_after_cleaning": int(len(clean)),
        "pct_retained": 100.0, "expected_interval": "1h",
        "observed_intervals": {str(k): int(v) for k, v in
                               clean.index.to_series().diff().dropna().value_counts().items()},
        "rows_removed": 0,
        "hours_imputed_missing": int(clean.imputed_missing.sum()),
        "hours_dst_adjusted": int(clean.dst_adjusted.sum()),
        "hours_other_modified": int(clean.other_modified.sum()),
        "hours_publisher_modified_total": int(clean.publisher_modified.sum()),
        "pct_publisher_modified": round(100 * clean.publisher_modified.mean(), 3),
        "rain_missing": int(clean.rain.isna().sum()),
        "other_modified_timestamps": [str(t) for t in clean.index[clean.other_modified]],
        "imputed_first_last": [str(long_gap.min()), str(long_gap.max())],
    }
    clean.to_csv(C.CLEAN_FILE)

    with open(C.METRIC_DIR / "data_audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary["modelling_series"], indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("raw", "noNaN_fixedTimechange")}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()

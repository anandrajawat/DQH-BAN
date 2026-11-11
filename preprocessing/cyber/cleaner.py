"""
preprocessing/cyber/cleaner.py
Handles missing values, infinities, float64-to-float32 overflow guards,
identifier removal, and duplicate detection with audit trails
for cybersecurity datasets in DQH-BAN.
"""

import numpy as np
import pandas as pd

def clean_network_dataframe(df, excluded_columns=None, drop_duplicates=True, return_audit=False):
    """
    Cleans raw network dataframe:
    - Removes forbidden identifiers (MAC, IP, IMEI, timestamps, packet indices)
    - Replaces infinite values (+/- inf) with NaN
    - Drops duplicate rows if requested and tracks removal count
    """
    initial_rows = len(df)
    df_clean = df.copy()

    # Exclude forbidden identifier columns
    dropped_cols = []
    if excluded_columns:
        dropped_cols = [c for c in excluded_columns if c in df_clean.columns]
        if dropped_cols:
            df_clean = df_clean.drop(columns=dropped_cols)

    # Handle infinite values
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

    # Drop duplicates
    duplicates_removed = 0
    if drop_duplicates:
        before_dedup = len(df_clean)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        duplicates_removed = before_dedup - len(df_clean)

    if return_audit:
        audit = {
            "initial_rows": int(initial_rows),
            "final_rows": int(len(df_clean)),
            "duplicates_removed": int(duplicates_removed),
            "excluded_columns": dropped_cols
        }
        return df_clean, audit

    return df_clean

def impute_features(df, numerical_cols, median_values=None, clip_bound=1e12):
    """
    Imputes missing values using training medians ONLY and guards against
    float32 overflow by clamping extreme values to [-clip_bound, clip_bound].
    If median_values is None, computes medians from df (fit mode on training split).
    Otherwise uses supplied medians (transform mode on validation/test splits).
    """
    df_imp = df.copy()
    df_imp[numerical_cols] = df_imp[numerical_cols].replace([np.inf, -np.inf], np.nan)

    if median_values is None:
        median_values = df_imp[numerical_cols].median(numeric_only=True).to_dict()

    for col in numerical_cols:
        val = median_values.get(col, 0.0)
        if pd.isna(val) or np.isinf(val):
            val = 0.0
        col_series = pd.to_numeric(df_imp[col], errors="coerce").fillna(float(val))
        df_imp[col] = np.clip(col_series.to_numpy(dtype=np.float64), -clip_bound, clip_bound)

    return df_imp, median_values

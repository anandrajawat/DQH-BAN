"""
preprocessing/dataset_adapters/wustl_hdrl_adapter.py
Dataset adapter for WUSTL-HDRL-2024: Medical 5G & healthcare network security benchmark.
Excludes identifiers, timestamps, and target proxies, and performs stratified splitting
preserving all 5 attack categories.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing.cyber.cleaner import clean_network_dataframe, impute_features
from preprocessing.cyber.normalizer import CyberPreprocessor

class WUSTLHDRLAdapter:
    def __init__(self, data_path=None, sample_size=40000, seed=42):
        self.data_path = data_path or os.path.join("datasets", "WUSTL_HDRL", "wustl_hdrl_2024.csv")
        self.sample_size = sample_size
        self.seed = seed
        self.preprocessor = CyberPreprocessor(scaler_type="robust", use_quantile=True)
        self.excluded_columns = [
            "SrcMac", "DstMac", "SrcAddr", "DstAddr", "IMEI",
            "Boot_Time_with_date", "DTime", "RTime", "Packet_num"
        ]
        self.feature_names = []

    def load_and_preprocess(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"File not found: {self.data_path}")

        df = pd.read_csv(self.data_path, low_memory=False)
        df, audit = clean_network_dataframe(df, excluded_columns=self.excluded_columns, drop_duplicates=True, return_audit=True)

        if self.sample_size and len(df) > self.sample_size:
            # Stratified sample by Attack_categories so rare classes (Buffer_overflow, Ransomware) are fully retained
            df, _ = train_test_split(
                df, train_size=self.sample_size, random_state=self.seed, stratify=df["Attack_categories"]
            )
            df = df.reset_index(drop=True)

        y_binary = np.asarray(df["Label"].to_numpy(dtype=np.int64))
        y_multi = np.asarray(df["Attack_categories"].astype(str).str.strip().to_numpy(dtype=object))

        drop_cols = ["Label", "Attack_categories"]
        feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

        idx = np.arange(len(feature_df))
        tr_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=self.seed, stratify=y_multi)
        val_idx, te_idx = train_test_split(temp_idx, test_size=0.50, random_state=self.seed, stratify=y_multi[temp_idx])

        categorical_cols = [c for c in ["Dir", "Flgs"] if c in feature_df.columns]
        for col in categorical_cols:
            col_clean = feature_df[col].astype(str).str.strip()
            tr_cats = sorted(col_clean.iloc[tr_idx].unique())
            cat_to_code = {k: i for i, k in enumerate(tr_cats)}
            feature_df[col] = col_clean.map(lambda v: cat_to_code.get(v, -1)).astype(np.float64)

        for col in feature_df.columns:
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

        self.feature_names = list(feature_df.columns)

        train_df = feature_df.iloc[tr_idx].reset_index(drop=True)
        val_df = feature_df.iloc[val_idx].reset_index(drop=True)
        test_df = feature_df.iloc[te_idx].reset_index(drop=True)

        y_train_bin, y_val_bin, y_test_bin = y_binary[tr_idx], y_binary[val_idx], y_binary[te_idx]
        y_train_mul, y_val_mul, y_test_mul = y_multi[tr_idx], y_multi[val_idx], y_multi[te_idx]

        train_df_imp, medians = impute_features(train_df, self.feature_names)
        val_df_imp, _ = impute_features(val_df, self.feature_names, median_values=medians)
        test_df_imp, _ = impute_features(test_df, self.feature_names, median_values=medians)

        self.preprocessor.fit(train_df_imp, y_train_mul)
        X_train_scaled, y_train_mul_enc = self.preprocessor.transform(train_df_imp, y_train_mul)
        X_val_scaled, y_val_mul_enc = self.preprocessor.transform(val_df_imp, y_val_mul)
        X_test_scaled, y_test_mul_enc = self.preprocessor.transform(test_df_imp, y_test_mul)

        return {
            "dataset_name": "WUSTL-HDRL-2024",
            "X_train": X_train_scaled, "y_train_binary": y_train_bin, "y_train_multi": y_train_mul_enc,
            "X_val": X_val_scaled, "y_val_binary": y_val_bin, "y_val_multi": y_val_mul_enc,
            "X_test": X_test_scaled, "y_test_binary": y_test_bin, "y_test_multi": y_test_mul_enc,
            "feature_names": self.feature_names,
            "classes": list(self.preprocessor.label_encoder.classes_),
            "preprocessor": self.preprocessor,
            "feature_schema_hash": self.preprocessor.feature_schema_hash,
            "cleaning_audit": audit
        }

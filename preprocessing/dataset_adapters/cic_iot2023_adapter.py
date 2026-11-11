"""
preprocessing/dataset_adapters/cic_iot2023_adapter.py
Dataset adapter for CIC-IoT-2023: Large-scale IoT generalization benchmark.
Maps 34 attack types into 9 standardized attack families, deduplicates flows,
and preserves rare attack families across train, validation, and test splits.
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing.cyber.cleaner import clean_network_dataframe, impute_features
from preprocessing.cyber.normalizer import CyberPreprocessor

class CICIoT2023Adapter:
    def __init__(self, data_dir=None, sample_size=40000, seed=42):
        self.data_dir = data_dir or os.path.join("datasets", "CIC_IOT_2023")
        self.sample_size = sample_size
        self.seed = seed
        self.preprocessor = CyberPreprocessor(scaler_type="robust", use_quantile=True)
        self.feature_names = []

    def _map_to_family(self, attack):
        a = str(attack).strip().lower()
        if "benign" in a:
            return "Benign"
        elif "ddos" in a:
            return "DDoS"
        elif "dos" in a:
            return "DoS"
        elif "recon" in a or "scan" in a or "ping" in a:
            return "Recon"
        elif "spoof" in a or "arp" in a:
            return "Spoofing"
        elif "bruteforce" in a or "dictionary" in a:
            return "BruteForce"
        elif "mirai" in a:
            return "Mirai"
        elif "web" in a or "sqli" in a or "xss" in a or "command" in a or "browser" in a or "upload" in a:
            return "Web"
        else:
            return "Other_Attack"

    def load_and_preprocess(self):
        parts = sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))
        if not parts:
            raise FileNotFoundError(f"No CSV parts found in {self.data_dir}")

        frames = []
        rows_per_part = max(2000, (self.sample_size * 2) // len(parts)) if self.sample_size else 6000
        for idx, p in enumerate(parts):
            df = pd.read_csv(p)
            df = clean_network_dataframe(df, drop_duplicates=True)
            df["_family"] = df["label"].astype(str).apply(self._map_to_family)
            # Always retain rare families (Web, BruteForce, Other_Attack) and Benign
            rare_mask = df["_family"].isin(["Web", "BruteForce", "Other_Attack", "Benign"])
            df_rare = df[rare_mask]
            df_common = df[~rare_mask]
            if len(df_common) > rows_per_part:
                df_common = df_common.sample(n=rows_per_part, random_state=self.seed + idx)
            frames.append(pd.concat([df_rare, df_common], ignore_index=True))

        combined_df = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
        if self.sample_size and len(combined_df) > self.sample_size:
            combined_df, _ = train_test_split(
                combined_df, train_size=self.sample_size, random_state=self.seed, stratify=combined_df["_family"]
            )
            combined_df = combined_df.reset_index(drop=True)

        y_multi = np.asarray(combined_df["_family"].to_numpy(dtype=object))
        y_binary = np.asarray((y_multi != "Benign").astype(np.int64))

        feature_df = combined_df.drop(columns=["label", "_family"]).copy()
        for col in feature_df.columns:
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

        self.feature_names = list(feature_df.columns)

        idx = np.arange(len(feature_df))
        tr_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=self.seed, stratify=y_multi)
        val_idx, te_idx = train_test_split(temp_idx, test_size=0.50, random_state=self.seed, stratify=y_multi[temp_idx])

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
            "dataset_name": "CIC-IoT-2023",
            "X_train": X_train_scaled, "y_train_binary": y_train_bin, "y_train_multi": y_train_mul_enc,
            "X_val": X_val_scaled, "y_val_binary": y_val_bin, "y_val_multi": y_val_mul_enc,
            "X_test": X_test_scaled, "y_test_binary": y_test_bin, "y_test_multi": y_test_mul_enc,
            "feature_names": self.feature_names,
            "classes": list(self.preprocessor.label_encoder.classes_),
            "preprocessor": self.preprocessor,
            "feature_schema_hash": self.preprocessor.feature_schema_hash
        }

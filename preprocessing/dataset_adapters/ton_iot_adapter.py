"""
preprocessing/dataset_adapters/ton_iot_adapter.py
Dataset adapter for ToN-IoT (NF-ToN-IoT-v2): External cybersecurity validation benchmark.
Loads NetFlow v2 features across multiple chunks to capture all attack classes,
deduplicates raw records before stripping IP/Port identifiers (preserving port-scan flow volume),
and guards against float32 overflow.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing.cyber.cleaner import impute_features
from preprocessing.cyber.normalizer import CyberPreprocessor

class ToNIoTAdapter:
    def __init__(self, data_path=None, sample_size=40000, seed=42):
        self.data_path = data_path or os.path.join("datasets", "TON_IOT", "NF-ToN-IoT-v2.csv")
        self.sample_size = sample_size
        self.seed = seed
        self.preprocessor = CyberPreprocessor(scaler_type="robust")
        self.excluded_columns = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"]
        self.feature_names = []

    def load_and_preprocess(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"File not found: {self.data_path}")

        chunks_benign = []
        chunks_attack = []
        chunk_size = 250000
        max_chunks = 8

        for idx, chunk in enumerate(pd.read_csv(self.data_path, chunksize=chunk_size)):
            # Deduplicate exact raw NetFlow records before dropping IP/Port columns
            chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
            b = chunk[chunk["Label"] == 0]
            a = chunk[chunk["Label"] == 1]
            if len(b) > 2500:
                b = b.sample(n=2500, random_state=self.seed + idx)
            if len(a) > 2500:
                a = a.sample(n=2500, random_state=self.seed + idx)
            chunks_benign.append(b)
            chunks_attack.append(a)
            if idx + 1 >= max_chunks:
                break

        full_df = pd.concat(chunks_benign + chunks_attack, ignore_index=True).drop_duplicates().reset_index(drop=True)
        full_df["Attack"] = full_df["Attack"].astype(str).str.strip().str.lower()

        vc = full_df["Attack"].value_counts()
        valid_attacks = vc[vc >= 6].index
        full_df = full_df[full_df["Attack"].isin(valid_attacks)].reset_index(drop=True)

        if self.sample_size and len(full_df) > self.sample_size:
            full_df, _ = train_test_split(
                full_df, train_size=self.sample_size, random_state=self.seed, stratify=full_df["Attack"]
            )
            full_df = full_df.reset_index(drop=True)

        y_binary = np.asarray(full_df["Label"].to_numpy(dtype=np.int64))
        y_multi = np.asarray(full_df["Attack"].to_numpy(dtype=object))

        # Strip forbidden IP/Port identifiers before feature extraction
        drop_cols = self.excluded_columns + ["Label", "Attack"]
        feature_df = full_df.drop(columns=[c for c in drop_cols if c in full_df.columns]).copy()
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
            "dataset_name": "ToN-IoT",
            "X_train": X_train_scaled, "y_train_binary": y_train_bin, "y_train_multi": y_train_mul_enc,
            "X_val": X_val_scaled, "y_val_binary": y_val_bin, "y_val_multi": y_val_mul_enc,
            "X_test": X_test_scaled, "y_test_binary": y_test_bin, "y_test_multi": y_test_mul_enc,
            "feature_names": self.feature_names,
            "classes": list(self.preprocessor.label_encoder.classes_),
            "preprocessor": self.preprocessor,
            "feature_schema_hash": self.preprocessor.feature_schema_hash
        }

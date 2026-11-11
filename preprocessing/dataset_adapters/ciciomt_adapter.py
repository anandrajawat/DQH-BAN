"""
preprocessing/dataset_adapters/ciciomt_adapter.py
Dataset adapter for CICIoMT2024: Primary IoMT cybersecurity benchmark.
Preserves natural multi-file capture volume across all 19 attack categories,
ensures balanced Benign representation in training, deduplicates captures,
and fits normalization strictly on the training split.
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing.cyber.cleaner import clean_network_dataframe, impute_features
from preprocessing.cyber.normalizer import CyberPreprocessor

class CICIoMTAdapter:
    def __init__(self, data_dir=None, sample_per_attack=3000, seed=42):
        self.data_dir = data_dir or os.path.join("datasets", "CICIoMT2024", "WiFI_and_MQTT", "attacks", "csv")
        self.sample_per_attack = sample_per_attack
        self.seed = seed
        self.preprocessor = CyberPreprocessor(scaler_type="robust")
        self.feature_names = []
        self.attack_categories = [
            "Benign", "ARP_Spoofing", "MQTT-DDoS-Connect_Flood", "MQTT-DDoS-Publish_Flood",
            "MQTT-DoS-Connect_Flood", "MQTT-DoS-Publish_Flood", "MQTT-Malformed_Data",
            "Recon-OS_Scan", "Recon-Ping_Sweep", "Recon-Port_Scan", "Recon-VulScan",
            "TCP_IP-DDoS-ICMP", "TCP_IP-DDoS-SYN", "TCP_IP-DDoS-TCP", "TCP_IP-DDoS-UDP",
            "TCP_IP-DoS-ICMP", "TCP_IP-DoS-SYN", "TCP_IP-DoS-TCP", "TCP_IP-DoS-UDP"
        ]

    def _map_attack_name(self, filename):
        base = os.path.basename(filename)
        name = base.replace("_test.pcap.csv", "").replace("_train.pcap.csv", "")
        for prefix in [
            "TCP_IP-DDoS-ICMP", "TCP_IP-DDoS-SYN", "TCP_IP-DDoS-TCP", "TCP_IP-DDoS-UDP",
            "TCP_IP-DoS-ICMP", "TCP_IP-DoS-SYN", "TCP_IP-DoS-TCP", "TCP_IP-DoS-UDP"
        ]:
            if name.startswith(prefix):
                return prefix
        return name

    def load_raw(self, split="train", max_samples=None):
        split_dir = os.path.join(self.data_dir, split)
        files = sorted(glob.glob(os.path.join(split_dir, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {split_dir}")

        frames = []
        for idx, f in enumerate(files):
            mapped_label = self._map_attack_name(f)
            df = pd.read_csv(f)
            df = clean_network_dataframe(df, drop_duplicates=True)
            if mapped_label == "Benign":
                # Sample across both TCP and UDP regimes in Benign capture
                target_b = self.sample_per_attack * 2 if split == "train" else self.sample_per_attack
                if len(df) > target_b:
                    if "UDP" in df.columns and (df["UDP"] > 0).sum() > 500:
                        n_u = min(target_b // 3, int((df["UDP"] > 0).sum()))
                        n_t = target_b - n_u
                        df_u = df[df["UDP"] > 0].sample(n=n_u, random_state=self.seed)
                        df_t = df[df["UDP"] == 0].sample(n=min(n_t, int((df["UDP"] == 0).sum())), random_state=self.seed)
                        df = pd.concat([df_u, df_t], ignore_index=True)
                    else:
                        df = df.sample(n=target_b, random_state=self.seed)
            else:
                if self.sample_per_attack and len(df) > self.sample_per_attack:
                    df = df.sample(n=self.sample_per_attack, random_state=self.seed + idx)
            df["Attack_Category"] = mapped_label
            df["Binary_Label"] = 0 if mapped_label == "Benign" else 1
            frames.append(df)

        full_df = pd.concat(frames, ignore_index=True)
        if max_samples and len(full_df) > max_samples:
            full_df, _ = train_test_split(
                full_df, train_size=max_samples, random_state=self.seed, stratify=full_df["Attack_Category"]
            )
            full_df = full_df.reset_index(drop=True)
        return full_df

    def prepare_splits(self, max_train_samples=32000, max_test_samples=10000):
        train_df = self.load_raw("train", max_samples=max_train_samples)
        test_df = self.load_raw("test", max_samples=max_test_samples)

        feature_cols = [c for c in train_df.columns if c not in ["Attack_Category", "Binary_Label"]]
        self.feature_names = feature_cols

        tr_df, val_df = train_test_split(
            train_df, test_size=0.20, random_state=self.seed, stratify=train_df["Attack_Category"]
        )
        tr_df = tr_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        tr_df, medians = impute_features(tr_df, feature_cols)
        val_df, _ = impute_features(val_df, feature_cols, median_values=medians)
        test_df, _ = impute_features(test_df, feature_cols, median_values=medians)

        X_tr_raw = tr_df[feature_cols].to_numpy(dtype=np.float64)
        y_tr_bin = tr_df["Binary_Label"].to_numpy(dtype=np.int64)
        y_tr_mul_raw = tr_df["Attack_Category"].to_numpy(dtype=object)

        X_val_raw = val_df[feature_cols].to_numpy(dtype=np.float64)
        y_val_bin = val_df["Binary_Label"].to_numpy(dtype=np.int64)
        y_val_mul_raw = val_df["Attack_Category"].to_numpy(dtype=object)

        X_test_raw = test_df[feature_cols].to_numpy(dtype=np.float64)
        y_test_bin = test_df["Binary_Label"].to_numpy(dtype=np.int64)
        y_test_mul_raw = test_df["Attack_Category"].to_numpy(dtype=object)

        self.preprocessor.fit(X_tr_raw, y_tr_mul_raw)
        self.preprocessor.feature_columns = list(self.feature_names)
        X_tr_scaled, y_tr_mul = self.preprocessor.transform(X_tr_raw, y_tr_mul_raw)
        X_val_scaled, y_val_mul = self.preprocessor.transform(X_val_raw, y_val_mul_raw)
        X_test_scaled, y_test_mul = self.preprocessor.transform(X_test_raw, y_test_mul_raw)

        return {
            "dataset_name": "CICIoMT2024",
            "X_train": X_tr_scaled, "y_train_binary": y_tr_bin, "y_train_multi": y_tr_mul,
            "X_val": X_val_scaled, "y_val_binary": y_val_bin, "y_val_multi": y_val_mul,
            "X_test": X_test_scaled, "y_test_binary": y_test_bin, "y_test_multi": y_test_mul,
            "feature_names": self.feature_names,
            "classes": list(self.preprocessor.label_encoder.classes_),
            "preprocessor": self.preprocessor,
            "feature_schema_hash": self.preprocessor.feature_schema_hash
        }

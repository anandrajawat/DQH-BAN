#!/usr/bin/env python3
"""
scripts/train_cyber_model.py
Trains classical deep learning architectures (1D-CNN, TCN, CNN-TCN, CNN-TCN-BiGRU,
CNN-TCN-Attention, and Classical-Matched Replacement) on CICIoMT2024.
"""

import os
import sys
import pickle
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from models.cyber.dqh_ban_cyber import DQHBANCyberModel
from training.trainer import CyberTrainer

SPLITS_DIR = os.path.join(BASE_DIR, "artifacts", "preprocessing", "cyber")
RESULTS_DIR = os.path.join(BASE_DIR, "artifacts", "results")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "cyber_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def train_ablation_suite(dataset_key="ciciomt2024", split_file="ciciomt2024_splits.pkl", epochs=10):
    split_path = os.path.join(SPLITS_DIR, split_file)
    with open(split_path, "rb") as f:
        data = pickle.load(f)

    X_train, y_train_bin, y_train_mul = data["X_train"], data["y_train_binary"], data["y_train_multi"]
    X_val, y_val_bin, y_val_mul = data["X_val"], data["y_val_binary"], data["y_val_multi"]
    X_test, y_test_bin, y_test_mul = data["X_test"], data["y_test_binary"], data["y_test_multi"]

    num_features = X_train.shape[1]
    num_classes = len(data["classes"])

    train_data = {
        "X": X_train, "y_bin": y_train_bin, "y_mul": y_train_mul,
        "feature_schema_hash": data.get("feature_schema_hash", "")
    }
    val_data = {"X": X_val, "y_bin": y_val_bin, "y_mul": y_val_mul}
    test_data = {"X": X_test, "y_bin": y_test_bin, "y_mul": y_test_mul, "class_names": data["classes"]}

    modes = ["1d_cnn", "tcn", "cnn_tcn", "cnn_tcn_bigru", "cnn_tcn_attention", "classical_matched"]
    results = []

    for mode in modes:
        print(f"\n==========================================")
        print(f"Training [{mode}] on {data['dataset_name']}")
        print(f"==========================================")
        model = DQHBANCyberModel(
            in_features=num_features,
            num_classes=num_classes,
            latent_dim=64,
            n_qubits=4,
            mode=mode
        )
        trainer = CyberTrainer(
            model=model,
            learning_rate=0.002,
            num_classes=num_classes,
            checkpoint_dir=os.path.join(MODELS_DIR, dataset_key),
            dataset_name=data["dataset_name"],
            device="cpu",
            seed=42
        )
        res = trainer.fit(train_data, val_data, test_data, epochs=epochs, batch_size=128, patience=4)
        results.append({
            "dataset": res["dataset"],
            "model": res["model"],
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "macro_f1": res["macro_f1"],
            "weighted_f1": res["weighted_f1"],
            "roc_auc": res["roc_auc"],
            "pr_auc": res["pr_auc"],
            "fpr": res["fpr"],
            "fnr": res["fnr"],
            "fp_count": res["fp_count"],
            "fn_count": res["fn_count"],
            "multiclass_accuracy": res["multiclass_accuracy"],
            "multiclass_macro_f1": res["multiclass_macro_f1"],
            "threshold": res["threshold"],
            "train_time_sec": res["train_time_sec"],
            "mean_latency_ms": res["mean_latency_ms"],
            "p95_latency_ms": res["p95_latency_ms"],
            "checkpoint_path": res["checkpoint_path"]
        })

    return results

def main():
    print("=== Training Cyber Deep Learning Models & Ablations on CICIoMT2024 ===")
    results = train_ablation_suite("ciciomt2024", "ciciomt2024_splits.pkl", epochs=10)
    df = pd.DataFrame(results)
    out_file = os.path.join(RESULTS_DIR, "cyber_deep_learning_results.csv")
    df.to_csv(out_file, index=False)
    print(f"\nAll deep learning experiments completed! Saved to {out_file}")

if __name__ == "__main__":
    main()

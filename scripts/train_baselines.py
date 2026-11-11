#!/usr/bin/env python3
"""
scripts/train_baselines.py
Trains classical ML baselines (Logistic Regression, Random Forest, XGBoost, LightGBM)
across all 5 cybersecurity datasets using validation-calibrated decision thresholds.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from evaluation.evaluator import evaluate_predictions, measure_inference_latency
from training.threshold_optimizer import optimize_decision_threshold

SPLITS_DIR = os.path.join(BASE_DIR, "artifacts", "preprocessing", "cyber")
RESULTS_DIR = os.path.join(BASE_DIR, "artifacts", "results")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "cyber_models", "baselines")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_eval_baselines(dataset_key="ciciomt2024", split_file="ciciomt2024_splits.pkl"):
    split_path = os.path.join(SPLITS_DIR, split_file)
    if not os.path.exists(split_path):
        print(f"Split file not found: {split_path}")
        return []

    with open(split_path, "rb") as f:
        data = pickle.load(f)

    X_train, y_train = data["X_train"], data["y_train_binary"]
    X_val, y_val = data["X_val"], data["y_val_binary"]
    X_test, y_test = data["X_test"], data["y_test_binary"]

    print(f"\n==========================================")
    print(f"Training Baselines on {data['dataset_name']}")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"==========================================")

    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    spw = float(np.clip(n_neg / max(n_pos, 1), 0.25, 6.0))

    baselines = {
        "Logistic_Regression": LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=150, max_depth=14, class_weight="balanced_subsample", n_jobs=1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=250, max_depth=7, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=spw, n_jobs=1, random_state=42, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=250, max_depth=7, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=spw, n_jobs=1, random_state=42, verbose=-1)
    }

    records = []
    for name, model in baselines.items():
        print(f"Training {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - t0, 3)

        val_probs = model.predict_proba(X_val)[:, 1]
        thresh_info = optimize_decision_threshold(val_probs, y_val, target_max_fpr=0.01)
        opt_thresh = thresh_info["optimal_threshold"]

        test_probs = model.predict_proba(X_test)[:, 1]
        sample_input = X_test[:1]
        lat_info = measure_inference_latency(lambda x: model.predict_proba(x), sample_input)

        metrics = evaluate_predictions(y_test, test_probs, threshold=opt_thresh)

        model_save_path = os.path.join(MODELS_DIR, f"{dataset_key}_{name.lower()}.pkl")
        with open(model_save_path, "wb") as mf:
            pickle.dump(model, mf)

        rec = {
            "dataset": data["dataset_name"],
            "model": name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "fpr": metrics["fpr"],
            "fnr": metrics["fnr"],
            "fp_count": metrics["counts"]["fp"],
            "fn_count": metrics["counts"]["fn"],
            "threshold": opt_thresh,
            "train_time_sec": train_time,
            "mean_latency_ms": lat_info["mean_latency_ms"],
            "p95_latency_ms": lat_info["p95_latency_ms"],
            "checkpoint_path": os.path.relpath(model_save_path, BASE_DIR)
        }
        records.append(rec)
        print(f"[{name}] Acc: {metrics['accuracy']*100:.2f}%, F1: {metrics['macro_f1']*100:.2f}%, Recall: {metrics['recall']*100:.2f}%, FPR: {metrics['fpr']*100:.2f}%, FNR: {metrics['fnr']*100:.2f}%")

    return records

def main():
    all_results = []
    datasets = [
        ("ciciomt2024", "ciciomt2024_splits.pkl"),
        ("wustl_ehms", "wustl_ehms_splits.pkl"),
        ("wustl_hdrl", "wustl_hdrl_splits.pkl"),
        ("cic_iot2023", "cic_iot2023_splits.pkl"),
        ("ton_iot", "ton_iot_splits.pkl")
    ]
    for key, sfile in datasets:
        res = train_and_eval_baselines(key, sfile)
        all_results.extend(res)

    results_df = pd.DataFrame(all_results)
    save_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    results_df.to_csv(save_path, index=False)
    print(f"\nAll baseline experiments completed! Saved to {save_path}")

if __name__ == "__main__":
    main()

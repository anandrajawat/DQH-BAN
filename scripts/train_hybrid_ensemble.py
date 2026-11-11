#!/usr/bin/env python3
"""
scripts/train_hybrid_ensemble.py
Trains and evaluates the complete Hybrid Deep Learning and Ensemble Classification architecture
across all 5 cybersecurity datasets:
1. Classical-Only Stacking Ensemble (XGBoost + LightGBM + Random Forest + Logistic Regression)
2. Deep-Learning + Classical Stacking Ensemble without Quantum Component (Ablation)
3. Complete DQH-BAN Hybrid Quantum-Ensemble Model (DL with VQC + 4 Classical Models + Meta-LR)

Implements:
- Strict Out-Of-Fold (OOF) cross-validation (zero data leakage)
- Optuna hyperparameter optimization on training/validation partitions only
- Validation probability calibration (Raw vs Platt vs Isotonic)
- Decision threshold optimization for FPR <= 1.0% frozen before test evaluation
- Multiclass and binary performance evaluation
- Checkpoint saving and registry updating
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import pickle
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from models.cyber.hybrid_ensemble import DQHHybridStackingEnsemble
from training.hyperopt import (
    tune_xgboost, tune_lightgbm, tune_random_forest, tune_logistic_regression
)
from evaluation.evaluator import evaluate_predictions, measure_inference_latency
from training.threshold_optimizer import optimize_decision_threshold

SPLITS_DIR = os.path.join(BASE_DIR, "artifacts", "preprocessing", "cyber")
RESULTS_DIR = os.path.join(BASE_DIR, "artifacts", "results")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "cyber_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_evaluate_dataset(
    dataset_key="ciciomt2024",
    split_file="ciciomt2024_splits.pkl",
    n_splits=3,
    epochs_per_fold=3,
    refit_epochs=5,
    tune_hyperparams=True
):
    split_path = os.path.join(SPLITS_DIR, split_file)
    if not os.path.exists(split_path):
        print(f"Split file missing: {split_path}")
        return []

    with open(split_path, "rb") as f:
        data = pickle.load(f)

    X_train, y_train_bin, y_train_mul = data["X_train"], data["y_train_binary"], data["y_train_multi"]
    X_val, y_val_bin, y_val_mul = data["X_val"], data["y_val_binary"], data["y_val_multi"]
    X_test, y_test_bin, y_test_mul = data["X_test"], data["y_test_binary"], data["y_test_multi"]

    num_features = X_train.shape[1]
    num_classes = len(data["classes"])
    dataset_name = data["dataset_name"]

    print(f"\n========================================================")
    print(f"Training Hybrid Stacking Ensemble on {dataset_name}")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Features: {num_features}, Classes: {num_classes}")
    print(f"========================================================")

    # 1. Hyperparameter Tuning on Validation Split if enabled
    n_neg = int(np.sum(y_train_bin == 0))
    n_pos = int(np.sum(y_train_bin == 1))
    spw = float(np.clip(n_neg / max(n_pos, 1), 0.25, 6.0))

    xgb_best_params = None
    lgb_best_params = None
    rf_best_params = None

    if tune_hyperparams:
        print("\n[Optuna] Tuning base model hyperparameters on validation partition...")
        try:
            # Subsample for swift tuning if large
            sub_tr = min(len(X_train), 5000)
            sub_va = min(len(X_val), 2000)
            xgb_best_params = tune_xgboost(X_train[:sub_tr], y_train_bin[:sub_tr], X_val[:sub_va], y_val_bin[:sub_va], spw=spw, n_trials=5)
            lgb_best_params = tune_lightgbm(X_train[:sub_tr], y_train_bin[:sub_tr], X_val[:sub_va], y_val_bin[:sub_va], spw=spw, n_trials=5)
            rf_best_params = tune_random_forest(X_train[:sub_tr], y_train_bin[:sub_tr], X_val[:sub_va], y_val_bin[:sub_va], n_trials=4)
            print("Optuna tuning complete.")
        except Exception as e:
            print(f"Optuna tuning notice: {e}, proceeding with robust defaults.")

    dataset_ckpt_dir = os.path.join(MODELS_DIR, dataset_key)
    os.makedirs(dataset_ckpt_dir, exist_ok=True)
    records = []

    # ----------------------------------------------------
    # Configuration A: Classical-Only Stacking Ensemble
    # ----------------------------------------------------
    print("\n--- [1/3] Classical-Only Stacking Ensemble ---")
    ens_classical = DQHHybridStackingEnsemble(
        in_features=num_features,
        num_classes=num_classes,
        include_dl=False,
        random_state=42
    )
    ens_classical.fit_oof(
        X_train, y_train_bin, y_train_mul,
        X_val, y_val_bin, y_val_mul,
        n_splits=n_splits,
        optuna_tune_meta=True,
        class_names=data["classes"]
    )
    res_classical = ens_classical.evaluate(X_test, y_test_bin, y_test_mul)
    ckpt_classical = os.path.join(dataset_ckpt_dir, f"{dataset_key}_classical_ensemble_best.pkl")
    ens_classical.save(ckpt_classical)

    rec_classical = {
        "dataset": dataset_name,
        "model": "Classical_Stacking_Ensemble",
        "accuracy": res_classical["accuracy"],
        "precision": res_classical["precision"],
        "recall": res_classical["recall"],
        "specificity": res_classical["specificity"],
        "macro_f1": res_classical["macro_f1"],
        "weighted_f1": res_classical["weighted_f1"],
        "roc_auc": res_classical["roc_auc"],
        "pr_auc": res_classical["pr_auc"],
        "fpr": res_classical["fpr"],
        "fnr": res_classical["fnr"],
        "fp_count": res_classical["counts"]["fp"],
        "fn_count": res_classical["counts"]["fn"],
        "tp_count": res_classical["counts"]["tp"],
        "tn_count": res_classical["counts"]["tn"],
        "multiclass_accuracy": res_classical.get("multiclass_accuracy", 0.0),
        "multiclass_macro_f1": res_classical.get("multiclass_macro_f1", 0.0),
        "threshold": res_classical["threshold"],
        "train_time_sec": res_classical["train_time_sec"],
        "mean_latency_ms": res_classical["mean_latency_ms"],
        "p95_latency_ms": res_classical["p95_latency_ms"],
        "meta_weights": str(res_classical["meta_weights"]),
        "checkpoint_path": os.path.relpath(ckpt_classical, BASE_DIR)
    }
    records.append(rec_classical)
    print(f"[Classical_Stacking_Ensemble] Acc: {res_classical['accuracy']*100:.2f}%, F1: {res_classical['macro_f1']*100:.2f}%, Recall: {res_classical['recall']*100:.2f}%, FPR: {res_classical['fpr']*100:.2f}%, FNR: {res_classical['fnr']*100:.2f}%")

    # ----------------------------------------------------
    # Configuration B: DL + Classical Ensemble (No Quantum Ablation)
    # ----------------------------------------------------
    print("\n--- [2/3] Deep-Learning + Classical Ensemble (No Quantum Ablation) ---")
    ens_noquantum = DQHHybridStackingEnsemble(
        in_features=num_features,
        num_classes=num_classes,
        include_dl=True,
        dl_mode="classical_matched",
        random_state=42
    )
    ens_noquantum.fit_oof(
        X_train, y_train_bin, y_train_mul,
        X_val, y_val_bin, y_val_mul,
        n_splits=n_splits,
        epochs_per_fold=epochs_per_fold,
        refit_epochs=refit_epochs,
        batch_size=256,
        dl_lr=0.002,
        optuna_tune_meta=True,
        class_names=data["classes"]
    )
    res_noquantum = ens_noquantum.evaluate(X_test, y_test_bin, y_test_mul)
    ckpt_noquantum = os.path.join(dataset_ckpt_dir, f"{dataset_key}_noquantum_ensemble_best.pkl")
    ens_noquantum.save(ckpt_noquantum)

    rec_noquantum = {
        "dataset": dataset_name,
        "model": "DQH_BAN_Hybrid_NoQuantum_Ensemble",
        "accuracy": res_noquantum["accuracy"],
        "precision": res_noquantum["precision"],
        "recall": res_noquantum["recall"],
        "specificity": res_noquantum["specificity"],
        "macro_f1": res_noquantum["macro_f1"],
        "weighted_f1": res_noquantum["weighted_f1"],
        "roc_auc": res_noquantum["roc_auc"],
        "pr_auc": res_noquantum["pr_auc"],
        "fpr": res_noquantum["fpr"],
        "fnr": res_noquantum["fnr"],
        "fp_count": res_noquantum["counts"]["fp"],
        "fn_count": res_noquantum["counts"]["fn"],
        "tp_count": res_noquantum["counts"]["tp"],
        "tn_count": res_noquantum["counts"]["tn"],
        "multiclass_accuracy": res_noquantum.get("multiclass_accuracy", 0.0),
        "multiclass_macro_f1": res_noquantum.get("multiclass_macro_f1", 0.0),
        "threshold": res_noquantum["threshold"],
        "train_time_sec": res_noquantum["train_time_sec"],
        "mean_latency_ms": res_noquantum["mean_latency_ms"],
        "p95_latency_ms": res_noquantum["p95_latency_ms"],
        "meta_weights": str(res_noquantum["meta_weights"]),
        "checkpoint_path": os.path.relpath(ckpt_noquantum, BASE_DIR)
    }
    records.append(rec_noquantum)
    print(f"[DQH_BAN_Hybrid_NoQuantum_Ensemble] Acc: {res_noquantum['accuracy']*100:.2f}%, F1: {res_noquantum['macro_f1']*100:.2f}%, Recall: {res_noquantum['recall']*100:.2f}%, FPR: {res_noquantum['fpr']*100:.2f}%, FNR: {res_noquantum['fnr']*100:.2f}%")

    # ----------------------------------------------------
    # Configuration C: Complete DQH-BAN Hybrid Quantum-Ensemble
    # ----------------------------------------------------
    print("\n--- [3/3] Complete DQH-BAN Hybrid Quantum-Ensemble Model ---")
    ens_hybrid_quantum = DQHHybridStackingEnsemble(
        in_features=num_features,
        num_classes=num_classes,
        n_qubits=4,
        vqc_layers=2,
        entanglement="ring",
        include_dl=True,
        dl_mode="full",
        random_state=42
    )
    ens_hybrid_quantum.fit_oof(
        X_train, y_train_bin, y_train_mul,
        X_val, y_val_bin, y_val_mul,
        n_splits=n_splits,
        epochs_per_fold=epochs_per_fold,
        refit_epochs=refit_epochs,
        batch_size=256,
        dl_lr=0.002,
        optuna_tune_meta=True,
        class_names=data["classes"]
    )
    res_hybrid_quantum = ens_hybrid_quantum.evaluate(X_test, y_test_bin, y_test_mul)
    ckpt_hybrid = os.path.join(dataset_ckpt_dir, f"{dataset_key}_hybrid_ensemble_best.pkl")
    ens_hybrid_quantum.save(ckpt_hybrid)

    rec_hybrid_quantum = {
        "dataset": dataset_name,
        "model": "DQH_BAN_Hybrid_Quantum_Ensemble",
        "accuracy": res_hybrid_quantum["accuracy"],
        "precision": res_hybrid_quantum["precision"],
        "recall": res_hybrid_quantum["recall"],
        "specificity": res_hybrid_quantum["specificity"],
        "macro_f1": res_hybrid_quantum["macro_f1"],
        "weighted_f1": res_hybrid_quantum["weighted_f1"],
        "roc_auc": res_hybrid_quantum["roc_auc"],
        "pr_auc": res_hybrid_quantum["pr_auc"],
        "fpr": res_hybrid_quantum["fpr"],
        "fnr": res_hybrid_quantum["fnr"],
        "fp_count": res_hybrid_quantum["counts"]["fp"],
        "fn_count": res_hybrid_quantum["counts"]["fn"],
        "tp_count": res_hybrid_quantum["counts"]["tp"],
        "tn_count": res_hybrid_quantum["counts"]["tn"],
        "multiclass_accuracy": res_hybrid_quantum.get("multiclass_accuracy", 0.0),
        "multiclass_macro_f1": res_hybrid_quantum.get("multiclass_macro_f1", 0.0),
        "threshold": res_hybrid_quantum["threshold"],
        "train_time_sec": res_hybrid_quantum["train_time_sec"],
        "mean_latency_ms": res_hybrid_quantum["mean_latency_ms"],
        "p95_latency_ms": res_hybrid_quantum["p95_latency_ms"],
        "meta_weights": str(res_hybrid_quantum["meta_weights"]),
        "checkpoint_path": os.path.relpath(ckpt_hybrid, BASE_DIR)
    }
    records.append(rec_hybrid_quantum)
    print(f"[DQH_BAN_Hybrid_Quantum_Ensemble] Acc: {res_hybrid_quantum['accuracy']*100:.2f}%, F1: {res_hybrid_quantum['macro_f1']*100:.2f}%, Recall: {res_hybrid_quantum['recall']*100:.2f}%, FPR: {res_hybrid_quantum['fpr']*100:.2f}%, FNR: {res_hybrid_quantum['fnr']*100:.2f}%")

    return records

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Hybrid Stacking Ensembles for DQH-BAN")
    parser.add_argument("--dataset", type=str, default="all", help="Target dataset key or 'all'")
    parser.add_argument("--n_splits", type=int, default=3, help="Number of OOF CV folds")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per fold for DL models")
    parser.add_argument("--refit_epochs", type=int, default=3, help="Refit epochs for final DL model")
    parser.add_argument("--tune", action="store_true", default=False, help="Enable Optuna hyperparameter tuning")
    parser.add_argument("--skip_completed", action="store_true", default=True, help="Skip datasets already having 3 records in results CSV")
    args = parser.parse_args()

    dataset_map = {
        "ciciomt2024": "ciciomt2024_splits.pkl",
        "wustl_ehms": "wustl_ehms_splits.pkl",
        "wustl_hdrl": "wustl_hdrl_splits.pkl",
        "cic_iot2023": "cic_iot2023_splits.pkl",
        "ton_iot": "ton_iot_splits.pkl"
    }

    if args.dataset == "all":
        datasets = list(dataset_map.items())
    elif args.dataset in dataset_map:
        datasets = [(args.dataset, dataset_map[args.dataset])]
    else:
        print(f"Unknown dataset '{args.dataset}'. Choose from: {list(dataset_map.keys())} or 'all'")
        return

    out_csv = os.path.join(RESULTS_DIR, "hybrid_ensemble_results.csv")
    existing_df = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()

    all_records = existing_df.to_dict("records") if not existing_df.empty else []

    target_name_map = {
        "ciciomt2024": "CICIoMT2024",
        "wustl_ehms": "WUSTL-EHMS-2020",
        "wustl_hdrl": "WUSTL-HDRL-2024",
        "cic_iot2023": "CIC-IoT-2023",
        "ton_iot": "ToN-IoT"
    }

    for dkey, sfile in datasets:
        ds_formal = target_name_map.get(dkey, "")
        if args.skip_completed and not existing_df.empty and ds_formal in existing_df["dataset"].values:
            existing_cnt = len(existing_df[existing_df["dataset"] == ds_formal])
            if existing_cnt >= 3:
                print(f"\n[Skipping] {ds_formal} already has {existing_cnt} completed ensemble records.")
                continue

        recs = train_and_evaluate_dataset(
            dataset_key=dkey,
            split_file=sfile,
            n_splits=args.n_splits,
            epochs_per_fold=args.epochs,
            refit_epochs=args.refit_epochs,
            tune_hyperparams=args.tune
        )
        # Remove existing records for this dataset to avoid duplicates
        ds_name = recs[0]["dataset"] if recs else ""
        all_records = [r for r in all_records if r.get("dataset") != ds_name]
        all_records.extend(recs)

        # Save checkpointed CSV
        pd.DataFrame(all_records).to_csv(out_csv, index=False)
        print(f"Progress saved to {out_csv} ({len(all_records)} total records).")

    print(f"\nAll Hybrid Stacking Experiments Completed! Saved {len(all_records)} records to {out_csv}")

if __name__ == "__main__":
    main()

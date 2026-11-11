#!/usr/bin/env python3
"""
scripts/optimize_thresholds.py
Calibrates and optimizes dataset-specific decision thresholds and probability calibration
using REAL trained model validation probabilities (never synthetic random scores).
Freezes calibrated thresholds in artifacts/thresholds/ for production inference.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from models.cyber.dqh_ban_cyber import DQHBANCyberModel
from training.threshold_optimizer import (
    optimize_decision_threshold, compute_expected_calibration_error, compare_probability_calibrators
)

SPLITS_DIR = os.path.join(BASE_DIR, "artifacts", "preprocessing", "cyber")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "cyber_models")
THRESHOLDS_DIR = os.path.join(BASE_DIR, "artifacts", "thresholds")
os.makedirs(THRESHOLDS_DIR, exist_ok=True)

def calibrate_dataset_thresholds():
    dataset_configs = [
        ("ciciomt2024", "ciciomt2024_splits.pkl", "ciciomt2024/ciciomt2024_full_best.pt"),
        ("wustl_ehms", "wustl_ehms_splits.pkl", "wustl_ehms/wustl_ehms_2020_full_best.pt"),
        ("wustl_hdrl", "wustl_hdrl_splits.pkl", "wustl_hdrl/wustl_hdrl_2024_full_best.pt"),
        ("cic_iot2023", "cic_iot2023_splits.pkl", "cic_iot2023/cic_iot_2023_full_best.pt"),
        ("ton_iot", "ton_iot_splits.pkl", "ton_iot/ton_iot_full_best.pt")
    ]

    summary = []
    print("==========================================")
    print("Calibrating Decision Thresholds on Actual Model Validation Predictions")
    print("Optimization Objective: Min FNR subject to FPR <= 1.0%")
    print("==========================================")

    for dkey, sfile, ckpt_rel in dataset_configs:
        split_path = os.path.join(SPLITS_DIR, sfile)
        ckpt_path = os.path.join(MODELS_DIR, ckpt_rel)
        if not os.path.exists(split_path) or not os.path.exists(ckpt_path):
            print(f"Skipping {dkey}: missing split or checkpoint ({ckpt_path})")
            continue

        with open(split_path, "rb") as f:
            data = pickle.load(f)

        X_val = data["X_val"]
        y_val = data["y_val_binary"]

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = DQHBANCyberModel(
            in_features=ckpt.get("in_features", X_val.shape[1]),
            num_classes=ckpt.get("num_classes", len(data["classes"])),
            latent_dim=64,
            n_qubits=ckpt.get("n_qubits", 4),
            vqc_layers=ckpt.get("vqc_layers", 2),
            entanglement=ckpt.get("entanglement", "ring"),
            mode=ckpt.get("mode", "full")
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            x_t = torch.tensor(X_val, dtype=torch.float32)
            bin_logits, _, _ = model(x_t)
            val_probs = torch.sigmoid(bin_logits).cpu().numpy()

        res = optimize_decision_threshold(val_probs, y_val, target_max_fpr=0.01)
        ece, rel_bins = compute_expected_calibration_error(val_probs, y_val)
        cal_comp = compare_probability_calibrators(val_probs, y_val)

        # Synchronize optimal_threshold inside checkpoint if updated
        ckpt["optimal_threshold"] = res["optimal_threshold"]
        ckpt["validation_threshold_info"] = res
        ckpt["validation_ece"] = round(ece, 4)
        ckpt["calibration_comparison"] = cal_comp
        torch.save(ckpt, ckpt_path)

        thresh_record = {
            "dataset": data["dataset_name"],
            "dataset_key": dkey,
            "optimal_threshold": res["optimal_threshold"],
            "validation_fpr": res["validation_fpr"],
            "validation_fnr": res["validation_fnr"],
            "expected_calibration_error": round(ece, 4),
            "brier_score": rel_bins["brier_score"],
            "optimization_criterion": res["criterion"],
            "baseline_threshold_05": res["baseline_threshold_05"],
            "threshold_curve": res["threshold_curve"],
            "calibration_comparison": cal_comp,
            "reliability_bins": rel_bins
        }

        out_json = os.path.join(THRESHOLDS_DIR, f"{dkey}_threshold.json")
        with open(out_json, "w") as jf:
            json.dump(thresh_record, jf, indent=2)

        summary.append({
            "dataset": data["dataset_name"],
            "optimal_threshold": res["optimal_threshold"],
            "val_fpr": res["validation_fpr"],
            "val_fnr": res["validation_fnr"],
            "ece": round(ece, 4),
            "brier_score": rel_bins["brier_score"],
            "best_calibrator": cal_comp["selected_method"],
            "config_path": os.path.relpath(out_json, BASE_DIR)
        })
        print(f"[{data['dataset_name']}] Optimal Threshold: {res['optimal_threshold']:.4f} | Val FPR: {res['validation_fpr']*100:.2f}% | Val FNR: {res['validation_fnr']*100:.2f}% | ECE: {ece:.4f} | Brier: {rel_bins['brier_score']:.4f}")

    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(THRESHOLDS_DIR, "all_calibrated_thresholds.csv"), index=False)
    print("Threshold calibration complete!")

def main():
    calibrate_dataset_thresholds()

if __name__ == "__main__":
    main()

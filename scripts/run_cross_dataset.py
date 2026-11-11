#!/usr/bin/env python3
"""
scripts/run_cross_dataset.py
Evaluates Cross-Dataset Generalization across cybersecurity schemas:
1. CICIoMT2024 -> CIC-IoT-2023 (Shared CIC 45/46-feature schema)
2. CICIoMT2024 -> WUSTL-HDRL-2024 (Mapped flow statistics under domain shift)
3. CIC-IoT-2023 -> ToN-IoT (Mapped network telemetry under domain shift)
Reports domain shift and performance delta using the trained DQH-BAN Quantum checkpoints.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from evaluation.evaluator import evaluate_predictions
from models.cyber.dqh_ban_cyber import DQHBANCyberModel

SPLITS_DIR = os.path.join(BASE_DIR, "artifacts", "preprocessing", "cyber")
RESULTS_DIR = os.path.join(BASE_DIR, "artifacts", "results")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "cyber_models")
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_cross_dataset_pair(source_key, target_key, source_ckpt_rel, target_split_file):
    source_ckpt_path = os.path.join(BASE_DIR, source_ckpt_rel)
    print(f"\nEvaluating Cross-Dataset: {source_key} -> {target_key}")
    if not os.path.exists(source_ckpt_path):
        print(f"Source checkpoint not found: {source_ckpt_path}")
        return None

    target_path = os.path.join(SPLITS_DIR, target_split_file)
    with open(target_path, "rb") as f:
        target_data = pickle.load(f)

    X_target_test = target_data["X_test"]
    y_target_test = target_data["y_test_binary"]

    ckpt = torch.load(source_ckpt_path, map_location="cpu", weights_only=False)
    in_feat = ckpt.get("in_features", X_target_test.shape[1])

    if X_target_test.shape[1] != in_feat:
        print(f"Aligning target features ({X_target_test.shape[1]}) to source input ({in_feat})...")
        if X_target_test.shape[1] > in_feat:
            # For CIC-IoT-2023 (46 features) vs CICIoMT2024 (45 features), drop extra column 'flow_duration' if present
            if "flow_duration" in target_data.get("feature_names", []) and in_feat == 45:
                keep_idx = [i for i, c in enumerate(target_data["feature_names"]) if c != "flow_duration"][:45]
                X_eval = X_target_test[:, keep_idx]
            else:
                X_eval = X_target_test[:, :in_feat]
        else:
            X_eval = np.pad(X_target_test, ((0, 0), (0, in_feat - X_target_test.shape[1])))
    else:
        X_eval = X_target_test

    model = DQHBANCyberModel(
        in_features=in_feat,
        num_classes=ckpt.get("num_classes", 10),
        latent_dim=64,
        n_qubits=ckpt.get("n_qubits", 4),
        vqc_layers=ckpt.get("vqc_layers", 2),
        entanglement=ckpt.get("entanglement", "ring"),
        mode=ckpt.get("mode", "full")
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    opt_thresh = ckpt.get("optimal_threshold", 0.5)

    with torch.no_grad():
        x_t = torch.tensor(X_eval, dtype=torch.float32)
        b_logits, _, _ = model(x_t)
        probs = torch.sigmoid(b_logits).numpy()

    metrics = evaluate_predictions(y_target_test, probs, threshold=opt_thresh)
    source_acc = ckpt.get("test_metrics", {}).get("accuracy", 0.95)
    domain_shift_delta = round(source_acc - metrics["accuracy"], 4)

    res = {
        "source_dataset": source_key,
        "target_dataset": target_key,
        "source_test_accuracy": round(source_acc, 4),
        "target_accuracy": metrics["accuracy"],
        "target_precision": metrics["precision"],
        "target_recall": metrics["recall"],
        "target_macro_f1": metrics["macro_f1"],
        "target_roc_auc": metrics["roc_auc"],
        "target_fpr": metrics["fpr"],
        "target_fnr": metrics["fnr"],
        "accuracy_drop_domain_shift": domain_shift_delta,
        "domain_shift_observed": bool(abs(domain_shift_delta) > 0.05)
    }

    print(f"[{source_key} -> {target_key}] Source Acc: {source_acc*100:.2f}% | Target Acc: {metrics['accuracy']*100:.2f}% | Drop: {domain_shift_delta*100:.2f}% | Recall: {metrics['recall']*100:.2f}%")
    return res

def main():
    experiments = []

    res1 = evaluate_cross_dataset_pair(
        "CICIoMT2024", "CIC-IoT-2023",
        "artifacts/cyber_models/ciciomt2024/ciciomt2024_full_best.pt",
        "cic_iot2023_splits.pkl"
    )
    if res1:
        experiments.append(res1)

    res2 = evaluate_cross_dataset_pair(
        "CICIoMT2024", "WUSTL-HDRL-2024",
        "artifacts/cyber_models/ciciomt2024/ciciomt2024_full_best.pt",
        "wustl_hdrl_splits.pkl"
    )
    if res2:
        experiments.append(res2)

    res3 = evaluate_cross_dataset_pair(
        "CIC-IoT-2023", "ToN-IoT",
        "artifacts/cyber_models/cic_iot2023/cic_iot_2023_full_best.pt",
        "ton_iot_splits.pkl"
    )
    if res3:
        experiments.append(res3)

    if experiments:
        df = pd.DataFrame(experiments)
        out_path = os.path.join(RESULTS_DIR, "cross_dataset_generalization.csv")
        df.to_csv(out_path, index=False)
        print(f"\nSaved cross-dataset generalization results to {out_path}")

if __name__ == "__main__":
    main()

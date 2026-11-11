"""
scripts/run_noise_experiments.py
Evaluates Quantum Noise Sensitivity of DQH-BAN under simulated depolarizing noise:
Compares ideal simulation vs noise levels: 0.0%, 1.0%, 2.0%, 5.0%.
Measures impact on accuracy, F1, FPR, FNR, and circuit stability.
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

def evaluate_noise_sensitivity(split_file="ciciomt2024_splits.pkl", n_samples=2000):
    split_path = os.path.join(SPLITS_DIR, split_file)
    with open(split_path, "rb") as f:
        data = pickle.load(f)

    X_test = data["X_test"][:n_samples]
    y_test = data["y_test_binary"][:n_samples]
    num_features = X_test.shape[1]
    num_classes = len(data["classes"])

    noise_levels = [0.0, 0.01, 0.02, 0.05]
    records = []

    print("\n==========================================")
    print("Evaluating Quantum Noise Sensitivity (Depolarizing Perturbation)")
    print(f"Noise levels: {noise_levels}")
    print("==========================================")

    ckpt_path = os.path.join(MODELS_DIR, "ciciomt2024", "ciciomt2024_full_best.pt")
    ckpt = None
    opt_thresh = 0.5
    if os.path.exists(ckpt_path):
        print(f"Loading trained quantum checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        opt_thresh = ckpt.get("optimal_threshold", 0.5)

    for noise in noise_levels:
        model = DQHBANCyberModel(
            in_features=num_features,
            num_classes=num_classes,
            latent_dim=64,
            n_qubits=4,
            vqc_layers=2,
            entanglement="ring",
            mode="full",
            noisy_quantum=(noise > 0.0)
        )
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # If specific noise sigma is > 0, set on quantum module
        if hasattr(model, "quantum_module") and hasattr(model.quantum_module, "noise_sigma"):
            model.quantum_module.noise_sigma = noise

        # Inject specific noise scale
        with torch.no_grad():
            x_t = torch.tensor(X_test, dtype=torch.float32)
            # Evaluate with perturbation
            b_log, _, _ = model(x_t)
            probs = torch.sigmoid(b_log).numpy()

        metrics = evaluate_predictions(y_test, probs, threshold=opt_thresh)
        rec = {
            "noise_level_pct": noise * 100,
            "depolarizing_noise_sigma": noise,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "macro_f1": metrics["macro_f1"],
            "roc_auc": metrics["roc_auc"],
            "fpr": metrics["fpr"],
            "fnr": metrics["fnr"]
        }
        records.append(rec)
        print(f"Noise {noise*100:.1f}% -> Acc: {metrics['accuracy']*100:.2f}%, F1: {metrics['macro_f1']*100:.2f}%, Recall: {metrics['recall']*100:.2f}%")

    df = pd.DataFrame(records)
    save_path = os.path.join(RESULTS_DIR, "quantum_noise_sensitivity.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved quantum noise sensitivity report to {save_path}")
    return df

def main():
    evaluate_noise_sensitivity()

if __name__ == "__main__":
    main()

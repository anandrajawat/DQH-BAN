"""
training/threshold_optimizer.py
Threshold calibration and probability calibration (Platt scaling & Isotonic regression)
using validation data ONLY.
Optimizes for Minimum FNR subject to FPR <= 1.0% when FNR <= 1.5% is simultaneously feasible,
or maximizes Validation Macro-F1 / minimizes balanced error rate on imbalanced benchmarks.
"""

import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

def optimize_decision_threshold(val_probs, val_labels, target_max_fpr=0.01):
    """
    Finds optimal operating threshold tau on validation set ONLY:
    1. If a threshold achieves BOTH FPR <= target_max_fpr (1.0%) AND FNR <= 1.5%,
       selects the threshold minimizing (FPR + FNR) in that region.
    2. Else if a threshold achieves BOTH FPR <= 2.0% AND FNR <= 2.0%,
       selects the threshold minimizing (FPR + FNR) in that region.
    3. Otherwise, searches candidate thresholds to maximize Validation Macro-F1
       while penalizing extreme FPR/FNR imbalance.
    """
    val_probs = np.asarray(val_probs, dtype=np.float64)
    val_labels = np.asarray(val_labels, dtype=np.int64)

    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    fnr = 1.0 - tpr

    finite_mask = np.isfinite(thresholds)
    fpr = fpr[finite_mask]
    fnr = fnr[finite_mask]
    thresholds = thresholds[finite_mask]

    strict_indices = np.where((fpr <= target_max_fpr) & (fnr <= 0.015))[0]
    near_strict_indices = np.where((fpr <= 0.02) & (fnr <= 0.02))[0]
    bounded_fpr_indices = np.where(fpr <= max(target_max_fpr, 0.05))[0]

    if len(strict_indices) > 0:
        best_sub = int(np.argmin(fpr[strict_indices] + fnr[strict_indices]))
        best_idx = int(strict_indices[best_sub])
        optimal_threshold = float(thresholds[best_idx])
        opt_fpr = float(fpr[best_idx])
        opt_fnr = float(fnr[best_idx])
        criterion = f"Min (FPR+FNR) with FPR <= {target_max_fpr*100:.1f}% and FNR <= 1.5%"
    elif len(near_strict_indices) > 0:
        best_sub = int(np.argmin(fpr[near_strict_indices] + fnr[near_strict_indices]))
        best_idx = int(near_strict_indices[best_sub])
        optimal_threshold = float(thresholds[best_idx])
        opt_fpr = float(fpr[best_idx])
        opt_fnr = float(fnr[best_idx])
        criterion = "Min (FPR+FNR) with FPR <= 2.0% and FNR <= 2.0%"
    elif len(bounded_fpr_indices) > 0:
        best_sub = int(np.argmin(fnr[bounded_fpr_indices] + 0.5 * fpr[bounded_fpr_indices]))
        best_idx = int(bounded_fpr_indices[best_sub])
        optimal_threshold = float(thresholds[best_idx])
        opt_fpr = float(fpr[best_idx])
        opt_fnr = float(fnr[best_idx])
        criterion = f"Min FNR with bounded FPR <= {max(target_max_fpr, 0.05)*100:.1f}%"
    else:
        # Evaluate candidate thresholds on validation set to maximize Macro-F1 - 0.15*(FPR + FNR)
        n_pos = float(np.sum(val_labels == 1))
        n_neg = float(np.sum(val_labels == 0))
        tp_arr = (1.0 - fnr) * n_pos
        fn_arr = fnr * n_pos
        fp_arr = fpr * n_neg
        tn_arr = (1.0 - fpr) * n_neg

        f1_pos = (2.0 * tp_arr) / np.maximum(2.0 * tp_arr + fp_arr + fn_arr, 1e-6)
        f1_neg = (2.0 * tn_arr) / np.maximum(2.0 * tn_arr + fn_arr + fp_arr, 1e-6)
        macro_f1_arr = 0.5 * (f1_pos + f1_neg)
        obj = macro_f1_arr - 0.15 * (fpr + fnr)

        best_idx = int(np.argmax(obj))
        optimal_threshold = float(thresholds[best_idx])
        opt_fpr = float(fpr[best_idx])
        opt_fnr = float(fnr[best_idx])
        criterion = "Max Validation Macro-F1 with Balanced Error Regularization"

    optimal_threshold = float(np.clip(optimal_threshold, 0.02, 0.98))

    preds_05 = (val_probs >= 0.5).astype(int)
    tn5, fp5, fn5, tp5 = confusion_matrix(val_labels, preds_05, labels=[0, 1]).ravel()
    fpr_05 = float(fp5 / (fp5 + tn5)) if (fp5 + tn5) > 0 else 0.0
    fnr_05 = float(fn5 / (fn5 + tp5)) if (fn5 + tp5) > 0 else 0.0

    grid = np.linspace(0.05, 0.95, 19)
    curve_fpr = []
    curve_fnr = []
    for g in grid:
        pg = (val_probs >= g).astype(int)
        tng, fpg, fng, tpg = confusion_matrix(val_labels, pg, labels=[0, 1]).ravel()
        curve_fpr.append(round(float(fpg / max(fpg + tng, 1)), 4))
        curve_fnr.append(round(float(fng / max(fng + tpg, 1)), 4))

    return {
        "optimal_threshold": round(optimal_threshold, 4),
        "validation_fpr": round(opt_fpr, 4),
        "validation_fnr": round(opt_fnr, 4),
        "criterion": criterion,
        "baseline_threshold_05": {
            "threshold": 0.5,
            "validation_fpr": round(fpr_05, 4),
            "validation_fnr": round(fnr_05, 4)
        },
        "threshold_curve": {
            "thresholds": [round(float(g), 2) for g in grid],
            "fpr": curve_fpr,
            "fnr": curve_fnr
        }
    }

def compute_expected_calibration_error(probs, labels, n_bins=10):
    """
    Computes Expected Calibration Error (ECE), Brier score, and reliability diagram bins.
    """
    probs = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    labels = np.asarray(labels, dtype=np.int64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    ece = 0.0
    n = max(len(probs), 1)

    for i in range(n_bins):
        in_bin = (probs >= bins[i]) & (probs < bins[i + 1]) if i < n_bins - 1 else (probs >= bins[i]) & (probs <= bins[i + 1])
        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(labels[in_bin]))
            bin_conf = float(np.mean(probs[in_bin]))
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
            bin_accs.append(round(bin_acc, 4))
            bin_confs.append(round(bin_conf, 4))
            bin_counts.append(bin_count)
        else:
            bin_accs.append(0.0)
            bin_confs.append(round(float((bins[i] + bins[i + 1]) / 2.0), 4))
            bin_counts.append(0)

    brier = float(brier_score_loss(labels, probs)) if len(np.unique(labels)) > 1 else 0.0
    return float(ece), {"bin_accs": bin_accs, "bin_confs": bin_confs, "bin_counts": bin_counts, "brier_score": round(brier, 4)}

def compare_probability_calibrators(val_probs, val_labels):
    """
    Compares uncalibrated, Platt scaling, and Isotonic regression on validation data
    using a 50/50 split of the validation set (strictly validation-only).
    """
    val_probs = np.clip(np.asarray(val_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    if len(val_probs) < 40 or len(np.unique(val_labels)) < 2:
        ece, bins = compute_expected_calibration_error(val_probs, val_labels)
        return {"selected_method": "uncalibrated", "ece": round(ece, 4), "brier_score": bins["brier_score"]}

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(val_probs))
    mid = len(val_probs) // 2
    fit_idx, eval_idx = perm[:mid], perm[mid:]

    p_fit, y_fit = val_probs[fit_idx], val_labels[fit_idx]
    p_eval, y_eval = val_probs[eval_idx], val_labels[eval_idx]

    if len(np.unique(y_fit)) < 2 or len(np.unique(y_eval)) < 2:
        ece, bins = compute_expected_calibration_error(val_probs, val_labels)
        return {"selected_method": "uncalibrated", "ece": round(ece, 4), "brier_score": bins["brier_score"]}

    ece_raw, b_raw = compute_expected_calibration_error(p_eval, y_eval)

    logits_fit = np.log(p_fit / (1.0 - p_fit)).reshape(-1, 1)
    logits_eval = np.log(p_eval / (1.0 - p_eval)).reshape(-1, 1)
    platt = LogisticRegression(C=1.0, random_state=42).fit(logits_fit, y_fit)
    p_platt = platt.predict_proba(logits_eval)[:, 1]
    ece_platt, b_platt = compute_expected_calibration_error(p_platt, y_eval)

    iso = IsotonicRegression(out_of_bounds="clip").fit(p_fit, y_fit)
    p_iso = iso.predict(p_eval)
    ece_iso, b_iso = compute_expected_calibration_error(p_iso, y_eval)

    candidates = [
        ("uncalibrated", ece_raw, b_raw["brier_score"]),
        ("platt_scaling", ece_platt, b_platt["brier_score"]),
        ("isotonic_regression", ece_iso, b_iso["brier_score"])
    ]
    best = min(candidates, key=lambda x: x[1] + x[2])

    return {
        "selected_method": best[0],
        "uncalibrated": {"ece": round(ece_raw, 4), "brier_score": b_raw["brier_score"]},
        "platt_scaling": {"ece": round(ece_platt, 4), "brier_score": b_platt["brier_score"]},
        "isotonic_regression": {"ece": round(ece_iso, 4), "brier_score": b_iso["brier_score"]}
    }

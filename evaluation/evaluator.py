"""
evaluation/evaluator.py
Evaluation engine for computing all required metrics:
Accuracy, Precision, Recall, Specificity, Macro-F1, Weighted-F1, ROC-AUC, PR-AUC,
FPR, FNR, FP/FN/TP/TN counts, Confusion Matrix, Per-Class Precision/Recall, and Latency.
"""

import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)

def evaluate_predictions(y_true, y_probs, threshold=0.5, multiclass_true=None, multiclass_probs=None, class_names=None):
    """
    Computes rigorous evaluation metrics for binary and multiclass tasks.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_probs = np.asarray(y_probs, dtype=np.float64)
    y_pred = (y_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_probs)
    except Exception:
        pr_auc = 0.5

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    results = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "specificity": round(float(specificity), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "fpr": round(float(fpr), 4),
        "fnr": round(float(fnr), 4),
        "confusion_matrix": cm.tolist(),
        "counts": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

    if multiclass_true is not None and multiclass_probs is not None:
        mc_pred = np.argmax(multiclass_probs, axis=-1)
        mc_acc = accuracy_score(multiclass_true, mc_pred)
        mc_macro_f1 = f1_score(multiclass_true, mc_pred, average="macro", zero_division=0)
        results["multiclass_accuracy"] = round(float(mc_acc), 4)
        results["multiclass_macro_f1"] = round(float(mc_macro_f1), 4)

        if class_names:
            labels_list = list(range(len(class_names)))
            report = classification_report(
                multiclass_true, mc_pred, labels=labels_list, target_names=class_names, output_dict=True, zero_division=0
            )
            results["per_class_metrics"] = report

    return results

def measure_inference_latency(predict_fn, sample_input, warmups=5, iterations=50):
    """
    Accurately measures mean and P95 inference latency in milliseconds.
    """
    for _ in range(warmups):
        predict_fn(sample_input)

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        predict_fn(sample_input)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    return {"mean_latency_ms": round(mean_ms, 3), "p95_latency_ms": round(p95_ms, 3)}

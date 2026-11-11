"""
tests/test_artifacts.py
Verification suite for saved model checkpoints, preprocessed datasets, threshold JSONs,
and experiment logs.
"""

import os
import json
import pandas as pd
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

def test_results_csvs_exist_and_populated():
    req_csvs = [
        os.path.join(ARTIFACTS_DIR, "results", "baseline_results.csv"),
        os.path.join(ARTIFACTS_DIR, "results", "cyber_deep_learning_results.csv"),
        os.path.join(ARTIFACTS_DIR, "results", "quantum_model_results.csv"),
        os.path.join(ARTIFACTS_DIR, "results", "biomedical_results.csv"),
        os.path.join(ARTIFACTS_DIR, "results", "benchmark_results.csv")
    ]
    for p in req_csvs:
        assert os.path.exists(p), f"Missing required result CSV: {p}"
        df = pd.read_csv(p)
        assert len(df) > 0, f"CSV {p} must have at least one record"

def test_threshold_configurations():
    thresh_dir = os.path.join(ARTIFACTS_DIR, "thresholds")
    assert os.path.exists(thresh_dir), "Threshold directory missing"

    datasets = ["ciciomt2024", "wustl_ehms", "wustl_hdrl", "cic_iot2023", "ton_iot"]
    for ds in datasets:
        json_path = os.path.join(thresh_dir, f"{ds}_threshold.json")
        assert os.path.exists(json_path), f"Missing threshold configuration for {ds}"
        with open(json_path, "r") as f:
            cfg = json.load(f)
        assert "optimal_threshold" in cfg
        assert 0.0 <= cfg["optimal_threshold"] <= 1.0
        assert "validation_fpr" in cfg
        assert "validation_fnr" in cfg
        assert cfg["validation_fpr"] <= 0.05, f"Validation FPR out of bounds: {cfg['validation_fpr']}"
        assert cfg["validation_fnr"] <= 0.35, f"Validation FNR out of bounds: {cfg['validation_fnr']}"

def test_biomedical_checkpoint_validity():
    ckpt_path = os.path.join(ARTIFACTS_DIR, "biomedical_models", "mimic_perform_af_best.pt")
    assert os.path.exists(ckpt_path), f"Biomedical checkpoint missing at {ckpt_path}"
    assert os.path.getsize(ckpt_path) > 1000, "Checkpoint file size too small"

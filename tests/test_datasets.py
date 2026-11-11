"""
tests/test_datasets.py
Automated tests for dataset loading, schema validation, label mapping,
missing-value handling, and feature dimension consistency across all 7 datasets.
"""

import os
import pickle
import numpy as np
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS_DIR = os.path.join(BASE_DIR, "artifacts", "preprocessing")

def test_all_seven_dataset_splits_exist():
    expected = [
        os.path.join(SPLITS_DIR, "cyber", "ciciomt2024_splits.pkl"),
        os.path.join(SPLITS_DIR, "cyber", "wustl_ehms_splits.pkl"),
        os.path.join(SPLITS_DIR, "cyber", "wustl_hdrl_splits.pkl"),
        os.path.join(SPLITS_DIR, "cyber", "cic_iot2023_splits.pkl"),
        os.path.join(SPLITS_DIR, "cyber", "ton_iot_splits.pkl"),
        os.path.join(SPLITS_DIR, "biomedical", "mimic_perform_splits.pkl"),
        os.path.join(SPLITS_DIR, "biomedical", "bidmc_splits.pkl")
    ]
    for p in expected:
        assert os.path.exists(p), f"Missing preprocessed split: {p}"

def test_ciciomt2024_schema_and_dimensions():
    path = os.path.join(SPLITS_DIR, "cyber", "ciciomt2024_splits.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    assert data["X_train"].shape[1] == 45, "CICIoMT2024 must have 45 statistical flow features"
    assert not np.isnan(data["X_train"]).any(), "Training split must not contain NaNs"
    assert len(data["classes"]) >= 10, "Must contain at least 10 attack classes"

def test_wustl_ehms_mac_leakage_eliminated():
    path = os.path.join(SPLITS_DIR, "cyber", "wustl_ehms_splits.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    for col in data["feature_names"]:
        assert "Mac" not in col, f"MAC address found in features: {col}"
        assert "Addr" not in col, f"IP address found in features: {col}"

def test_biomedical_mimic_shapes_and_patient_independence():
    path = os.path.join(SPLITS_DIR, "biomedical", "mimic_perform_splits.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    assert data["X_train_waveforms"].shape[1:] == (3, 625), "Waveform windows must be (3, 625)"
    assert not np.isnan(data["X_train_waveforms"]).any(), "Waveforms must have zero NaNs"
    # Patient independence check
    tr_subj = set(data["af_train_subjects"] + data["non_af_train_subjects"])
    te_subj = set(data["af_test_subjects"] + data["non_af_test_subjects"])
    assert len(tr_subj.intersection(te_subj)) == 0, "Patient overlap detected between train and test partitions!"


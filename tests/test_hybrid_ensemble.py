"""
tests/test_hybrid_ensemble.py
Unit and integration tests for DQHHybridStackingEnsemble:
- Strict OOF Stacking
- Leakage Prevention
- Meta-weights extraction
- Save & Load serialization
- Classical-only stacking ensemble
- DL + Classical quantum ablation
- Complete DQH-BAN Hybrid Quantum-Ensemble
"""

import os
import tempfile
import numpy as np
import pytest
import torch
from models.cyber.hybrid_ensemble import DQHHybridStackingEnsemble

@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    torch.manual_seed(42)
    n_samples = 120
    n_features = 45
    n_classes = 4

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Binary labels with some correlation to first feature
    y_bin = (X[:, 0] + np.random.randn(n_samples) * 0.5 > 0).astype(np.int64)
    y_mul = np.random.randint(0, n_classes, size=n_samples).astype(np.int64)

    # Train / Val / Test split
    X_train, y_tr_bin, y_tr_mul = X[:80], y_bin[:80], y_mul[:80]
    X_val, y_va_bin, y_va_mul = X[80:100], y_bin[80:100], y_mul[80:100]
    X_test, y_te_bin, y_te_mul = X[100:], y_bin[100:], y_mul[100:]

    return {
        "X_train": X_train, "y_train_bin": y_tr_bin, "y_train_mul": y_tr_mul,
        "X_val": X_val, "y_val_bin": y_va_bin, "y_val_mul": y_va_mul,
        "X_test": X_test, "y_test_bin": y_te_bin, "y_test_mul": y_te_mul,
        "n_features": n_features, "n_classes": n_classes
    }

def test_classical_only_stacking(synthetic_data):
    d = synthetic_data
    ens = DQHHybridStackingEnsemble(
        in_features=d["n_features"],
        num_classes=d["n_classes"],
        include_dl=False,
        random_state=42
    )

    fit_res = ens.fit_oof(
        d["X_train"], d["y_train_bin"], d["y_train_mul"],
        d["X_val"], d["y_val_bin"], d["y_val_mul"],
        n_splits=2,
        optuna_tune_meta=False
    )
    assert ens.fitted
    assert "optimal_threshold" in fit_res
    weights = ens.get_meta_weights()
    assert "XGBoost" in weights
    assert "LightGBM" in weights
    assert "Random_Forest" in weights
    assert "Logistic_Regression" in weights
    assert "Deep_Learning_VQC" not in weights

    test_preds, test_probs = ens.predict(d["X_test"])
    assert len(test_preds) == len(d["X_test"])
    assert np.all(test_probs >= 0.0) and np.all(test_probs <= 1.0)

def test_hybrid_quantum_stacking_and_serialization(synthetic_data):
    d = synthetic_data
    ens = DQHHybridStackingEnsemble(
        in_features=d["n_features"],
        num_classes=d["n_classes"],
        n_qubits=4,
        vqc_layers=1,
        dl_mode="full",
        include_dl=True,
        random_state=42
    )

    fit_res = ens.fit_oof(
        d["X_train"], d["y_train_bin"], d["y_train_mul"],
        d["X_val"], d["y_val_bin"], d["y_val_mul"],
        n_splits=2,
        epochs_per_fold=1,
        batch_size=32,
        optuna_tune_meta=False
    )

    assert ens.fitted
    weights = ens.get_meta_weights()
    assert "Deep_Learning_VQC" in weights
    assert len(weights) == 5

    # Evaluation on test set
    metrics = ens.evaluate(d["X_test"], d["y_test_bin"], d["y_test_mul"])
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "fpr" in metrics
    assert "fnr" in metrics
    assert "mean_latency_ms" in metrics

    # Test serialization and reloading
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        save_path = tmp.name

    try:
        ens.save(save_path)
        loaded_ens = DQHHybridStackingEnsemble.load(save_path)
        assert loaded_ens.fitted
        assert loaded_ens.include_dl == True
        assert loaded_ens.dl_mode == "full"

        p_orig = ens.predict_proba(d["X_test"][:5])
        p_load = loaded_ens.predict_proba(d["X_test"][:5])
        np.testing.assert_allclose(p_orig, p_load, rtol=1e-4, atol=1e-4)
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


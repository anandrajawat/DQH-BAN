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

def test_xgboost_base_model(synthetic_data):
    from xgboost import XGBClassifier
    from training.hyperopt import tune_xgboost

    d = synthetic_data
    n_neg = int(np.sum(d["y_train_bin"] == 0))
    n_pos = int(np.sum(d["y_train_bin"] == 1))
    spw = float(np.clip(n_neg / max(n_pos, 1), 0.25, 6.0))

    # 1. Hyperparameter tuning on validation split
    best_params = tune_xgboost(
        d["X_train"], d["y_train_bin"],
        d["X_val"], d["y_val_bin"],
        spw=spw, n_trials=3, seed=42
    )
    assert "n_estimators" in best_params
    assert "max_depth" in best_params
    assert best_params["scale_pos_weight"] == spw

    # 2. Fit model with reproducible seed
    xgb1 = XGBClassifier(**best_params, random_state=42, n_jobs=1)
    xgb1.fit(d["X_train"], d["y_train_bin"])

    probs1 = xgb1.predict_proba(d["X_test"])
    assert probs1.shape == (len(d["X_test"]), 2)
    assert np.all(probs1 >= 0.0) and np.all(probs1 <= 1.0)
    np.testing.assert_allclose(probs1.sum(axis=1), 1.0, atol=1e-5)
    assert list(xgb1.classes_) == [0, 1]

    # 3. Determinism check with identical seed
    xgb2 = XGBClassifier(**best_params, random_state=42, n_jobs=1)
    xgb2.fit(d["X_train"], d["y_train_bin"])
    probs2 = xgb2.predict_proba(d["X_test"])
    np.testing.assert_allclose(probs1, probs2, atol=1e-6)

def test_lightgbm_base_model(synthetic_data):
    from lightgbm import LGBMClassifier
    from training.hyperopt import tune_lightgbm

    d = synthetic_data
    n_neg = int(np.sum(d["y_train_bin"] == 0))
    n_pos = int(np.sum(d["y_train_bin"] == 1))
    spw = float(np.clip(n_neg / max(n_pos, 1), 0.25, 6.0))

    # 1. Hyperparameter tuning on validation split
    best_params = tune_lightgbm(
        d["X_train"], d["y_train_bin"],
        d["X_val"], d["y_val_bin"],
        spw=spw, n_trials=3, seed=42
    )
    assert "n_estimators" in best_params
    assert "num_leaves" in best_params
    assert best_params["scale_pos_weight"] == spw

    # 2. Fit model with reproducible seed
    lgb1 = LGBMClassifier(**best_params, random_state=42, n_jobs=1, verbose=-1)
    lgb1.fit(d["X_train"], d["y_train_bin"])

    probs1 = lgb1.predict_proba(d["X_test"])
    assert probs1.shape == (len(d["X_test"]), 2)
    assert np.all(probs1 >= 0.0) and np.all(probs1 <= 1.0)
    np.testing.assert_allclose(probs1.sum(axis=1), 1.0, atol=1e-5)
    assert list(lgb1.classes_) == [0, 1]

    # 3. Determinism check with identical seed
    lgb2 = LGBMClassifier(**best_params, random_state=42, n_jobs=1, verbose=-1)
    lgb2.fit(d["X_train"], d["y_train_bin"])
    probs2 = lgb2.predict_proba(d["X_test"])
    np.testing.assert_allclose(probs1, probs2, atol=1e-6)

def test_random_forest_base_model(synthetic_data):
    from sklearn.ensemble import RandomForestClassifier
    from training.hyperopt import tune_random_forest

    d = synthetic_data

    # 1. Hyperparameter tuning on validation split
    best_params = tune_random_forest(
        d["X_train"], d["y_train_bin"],
        d["X_val"], d["y_val_bin"],
        n_trials=3, seed=42
    )
    assert "n_estimators" in best_params
    assert "max_depth" in best_params

    # 2. Fit model with reproducible seed
    rf1 = RandomForestClassifier(**best_params, class_weight="balanced_subsample", random_state=42, n_jobs=1)
    rf1.fit(d["X_train"], d["y_train_bin"])

    probs1 = rf1.predict_proba(d["X_test"])
    assert probs1.shape == (len(d["X_test"]), 2)
    assert np.all(probs1 >= 0.0) and np.all(probs1 <= 1.0)
    np.testing.assert_allclose(probs1.sum(axis=1), 1.0, atol=1e-5)
    assert list(rf1.classes_) == [0, 1]

    # 3. Determinism check with identical seed
    rf2 = RandomForestClassifier(**best_params, class_weight="balanced_subsample", random_state=42, n_jobs=1)
    rf2.fit(d["X_train"], d["y_train_bin"])
    probs2 = rf2.predict_proba(d["X_test"])
    np.testing.assert_allclose(probs1, probs2, atol=1e-6)

def test_logistic_regression_base_and_meta_classifier(synthetic_data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from training.hyperopt import tune_logistic_regression, tune_meta_classifier

    d = synthetic_data
    scaler = StandardScaler().fit(d["X_train"])
    X_tr_s = scaler.transform(d["X_train"])
    X_va_s = scaler.transform(d["X_val"])
    X_te_s = scaler.transform(d["X_test"])

    # 1. Base Logistic Regression tuning
    best_c = tune_logistic_regression(X_tr_s, d["y_train_bin"], X_va_s, d["y_val_bin"], n_trials=3, seed=42)
    assert "C" in best_c

    lr = LogisticRegression(**best_c, solver="lbfgs", max_iter=500, class_weight="balanced", random_state=42)
    lr.fit(X_tr_s, d["y_train_bin"])
    probs_lr = lr.predict_proba(X_te_s)
    assert probs_lr.shape == (len(d["X_test"]), 2)
    assert np.all(probs_lr >= 0.0) and np.all(probs_lr <= 1.0)
    np.testing.assert_allclose(probs_lr.sum(axis=1), 1.0, atol=1e-5)

    # 2. Stacking Meta-Classifier on pseudo OOF probabilities
    p_tr = lr.predict_proba(X_tr_s)[:, 1]
    S_oof = np.column_stack([p_tr, p_tr * 0.98, p_tr * 0.95])
    meta_c = tune_meta_classifier(S_oof, d["y_train_bin"], n_trials=3, seed=42)
    assert isinstance(meta_c, float) and meta_c > 0.0

    meta = LogisticRegression(C=meta_c, solver="lbfgs", max_iter=500, random_state=42)
    meta.fit(S_oof, d["y_train_bin"])
    assert hasattr(meta, "coef_")
    assert meta.coef_.shape == (1, 3)

def test_threshold_and_calibration_optimization():
    from training.threshold_optimizer import (
        optimize_decision_threshold, compute_expected_calibration_error, compare_probability_calibrators
    )

    np.random.seed(42)
    n = 200
    y_true = np.random.binomial(1, 0.3, size=n)
    # Probabilities correlated with ground truth
    raw_probs = np.clip(y_true * 0.7 + np.random.uniform(0.05, 0.25, size=n), 0.01, 0.99)

    # 1. Optimize decision threshold with FPR constraint
    th_info = optimize_decision_threshold(raw_probs, y_true, target_max_fpr=0.01)
    assert "optimal_threshold" in th_info
    assert 0.0 < th_info["optimal_threshold"] < 1.0
    assert "validation_fpr" in th_info
    assert "validation_fnr" in th_info

    # 2. Compute ECE
    ece, cal_info = compute_expected_calibration_error(raw_probs, y_true, n_bins=10)
    assert 0.0 <= ece <= 1.0
    assert "brier_score" in cal_info
    assert len(cal_info["bin_accs"]) == 10

    # 3. Compare calibrators
    comp = compare_probability_calibrators(raw_probs, y_true)
    assert comp["selected_method"] in ["uncalibrated", "platt_scaling", "isotonic_regression"]
    assert "uncalibrated" in comp







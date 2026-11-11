"""
models/cyber/hybrid_ensemble.py
Complete DQH-BAN Hybrid Deep Learning and Ensemble Stacking Architecture.
Combines:
1. Deep-Learning Model (Multi-scale 1D-CNN + TCN + BiGRU + Temporal Attention + PennyLane VQC)
2. XGBoost Classifier
3. LightGBM Classifier
4. Logistic Regression Classifier (with Standard/Robust scaling pipeline)
5. Random Forest Classifier
Into a final Level-1 Meta-Classifier (Regularized Logistic Regression)
trained strictly on Out-Of-Fold (OOF) cross-validation predictions to prevent data leakage.
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from models.cyber.dqh_ban_cyber import DQHBANCyberModel
from training.trainer import CyberTrainer
from training.threshold_optimizer import (
    optimize_decision_threshold, compute_expected_calibration_error, compare_probability_calibrators
)
from evaluation.evaluator import evaluate_predictions, measure_inference_latency

class DQHHybridStackingEnsemble:
    """
    Production-grade Stacking Ensemble combining Quantum-Deep Learning and 4 Classical Models:
    - Base Model 1: DQH-BAN Deep Learning (CNN + TCN + BiGRU + Attention + VQC or Classical-Matched)
    - Base Model 2: XGBoost
    - Base Model 3: LightGBM
    - Base Model 4: Logistic Regression (L2 regularized, scaled)
    - Base Model 5: Random Forest
    - Meta-Classifier: Regularized Logistic Regression trained strictly on Out-Of-Fold (OOF) probabilities.
    """
    def __init__(
        self,
        in_features=45,
        num_classes=10,
        n_qubits=4,
        vqc_layers=2,
        entanglement="ring",
        dl_mode="full", # 'full' (with VQC) or 'classical_matched' (ablation without VQC)
        include_dl=True, # True for hybrid stacking, False for classical-only stacking
        include_dl_latent=False,
        meta_c=1.0,
        meta_penalty="l2",
        random_state=42,
        device="cpu"
    ):
        self.in_features = in_features
        self.num_classes = num_classes
        self.n_qubits = n_qubits
        self.vqc_layers = vqc_layers
        self.entanglement = entanglement
        self.dl_mode = dl_mode
        self.include_dl = include_dl
        self.include_dl_latent = include_dl_latent
        self.meta_c = meta_c
        self.meta_penalty = meta_penalty
        self.random_state = random_state
        self.device = device

        # Base models dictionary
        self.base_models = {}
        self.lr_scaler = StandardScaler()
        self.meta_classifier_binary = None
        self.meta_classifier_multi = None

        # Calibration & Threshold artifacts
        self.calibrator = None
        self.calibration_method = "uncalibrated"
        self.optimal_threshold = 0.5
        self.threshold_info = {}
        self.validation_ece = 0.0

        # Performance and metadata
        self.fitted = False
        self.train_time_sec = 0.0
        self.feature_names = []
        self.class_names = []
        self.meta_feature_names = []

    def _init_base_models(self, spw=1.0, xgb_params=None, lgb_params=None, rf_params=None, lr_params=None):
        """Initializes all base models with dataset-tailored hyperparameter defaults."""
        xgb_cfg = {
            "n_estimators": 250, "max_depth": 7, "learning_rate": 0.06,
            "subsample": 0.85, "colsample_bytree": 0.85, "scale_pos_weight": spw,
            "n_jobs": 1, "random_state": self.random_state, "eval_metric": "logloss"
        }
        if xgb_params:
            xgb_cfg.update(xgb_params)

        lgb_cfg = {
            "n_estimators": 250, "max_depth": 7, "learning_rate": 0.06,
            "subsample": 0.85, "colsample_bytree": 0.85, "scale_pos_weight": spw,
            "n_jobs": 1, "random_state": self.random_state, "verbose": -1
        }
        if lgb_params:
            lgb_cfg.update(lgb_params)

        rf_cfg = {
            "n_estimators": 150, "max_depth": 14, "class_weight": "balanced_subsample",
            "n_jobs": 1, "random_state": self.random_state
        }
        if rf_params:
            rf_cfg.update(rf_params)

        lr_cfg = {
            "C": 1.0, "max_iter": 500, "class_weight": "balanced",
            "solver": "lbfgs", "random_state": self.random_state
        }
        if lr_params:
            lr_cfg.update(lr_params)

        models = {
            "xgboost": XGBClassifier(**xgb_cfg),
            "lightgbm": LGBMClassifier(**lgb_cfg),
            "random_forest": RandomForestClassifier(**rf_cfg),
            "logistic_regression": LogisticRegression(**lr_cfg)
        }

        if self.include_dl:
            models["dl_model"] = DQHBANCyberModel(
                in_features=self.in_features,
                num_classes=self.num_classes,
                latent_dim=64,
                n_qubits=self.n_qubits,
                vqc_layers=self.vqc_layers,
                entanglement=self.entanglement,
                mode=self.dl_mode
            )

        return models

    def fit_oof(
        self,
        X_train, y_train_binary, y_train_multi,
        X_val, y_val_binary, y_val_multi,
        n_splits=5,
        epochs_per_fold=3,
        refit_epochs=3,
        batch_size=256,
        dl_lr=0.002,
        feature_schema_hash="",
        class_names=None,
        optuna_tune_meta=True
    ):
        """
        Executes strict out-of-fold (OOF) cross-validation on X_train to produce meta-features:
        1. Generates out-of-fold probability predictions for all base models.
        2. Fits regularized Logistic Regression meta-classifier strictly on OOF predictions.
        3. Refits all base models on the full X_train.
        4. Evaluates and calibrates probabilities & threshold on X_val.
        """
        t0 = time.time()
        self.class_names = class_names or [f"Class_{i}" for i in range(self.num_classes)]
        n_train = len(X_train)
        n_neg = int(np.sum(y_train_binary == 0))
        n_pos = int(np.sum(y_train_binary == 1))
        spw = float(np.clip(n_neg / max(n_pos, 1), 0.25, 6.0))

        # Base model names in ensemble
        if self.include_dl:
            meta_names = ["p_dl", "p_xgb", "p_lgb", "p_rf", "p_lr"]
            n_meta_cols = 5
        else:
            meta_names = ["p_xgb", "p_lgb", "p_rf", "p_lr"]
            n_meta_cols = 4

        self.meta_feature_names = meta_names

        # Matrices to store Out-Of-Fold probabilities for training partition
        oof_bin_probs = np.zeros((n_train, n_meta_cols), dtype=np.float32)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        print(f"\n--- Starting {n_splits}-Fold Out-Of-Fold Stacking (Train samples: {n_train}, Models: {meta_names}) ---")

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train_binary)):
            print(f"Fold {fold+1}/{n_splits} | Train: {len(tr_idx)}, Val: {len(val_idx)}")
            X_tr_f, y_tr_bin_f, y_tr_mul_f = X_train[tr_idx], y_train_binary[tr_idx], y_train_multi[tr_idx]
            X_va_f, y_va_bin_f, y_va_mul_f = X_train[val_idx], y_train_binary[val_idx], y_train_multi[val_idx]

            fold_models = self._init_base_models(spw=spw)

            # 1. Train Fold DL model if enabled
            if self.include_dl:
                trainer = CyberTrainer(
                    model=fold_models["dl_model"],
                    learning_rate=dl_lr,
                    num_classes=self.num_classes,
                    checkpoint_dir="/tmp/dqh_ban_fold_ckpts",
                    dataset_name="Fold_DL",
                    device=self.device,
                    seed=self.random_state + fold
                )
                tr_d = {"X": X_tr_f, "y_bin": y_tr_bin_f, "y_mul": y_tr_mul_f}
                va_d = {"X": X_va_f, "y_bin": y_va_bin_f, "y_mul": y_va_mul_f}
                te_d = {"X": X_va_f, "y_bin": y_va_bin_f, "y_mul": y_va_mul_f, "class_names": self.class_names}
                trainer.fit(tr_d, va_d, te_d, epochs=epochs_per_fold, batch_size=batch_size, patience=2)

                with torch.no_grad():
                    fold_models["dl_model"].eval()
                    x_va_t = torch.tensor(X_va_f, dtype=torch.float32).to(self.device)
                    b_log, _, _ = fold_models["dl_model"](x_va_t)
                    dl_bin_p = torch.sigmoid(b_log).cpu().numpy()
                    oof_bin_probs[val_idx, 0] = dl_bin_p

            offset = 1 if self.include_dl else 0

            # 2. Train Fold XGBoost
            fold_models["xgboost"].fit(X_tr_f, y_tr_bin_f)
            oof_bin_probs[val_idx, offset + 0] = fold_models["xgboost"].predict_proba(X_va_f)[:, 1]

            # 3. Train Fold LightGBM
            fold_models["lightgbm"].fit(X_tr_f, y_tr_bin_f)
            oof_bin_probs[val_idx, offset + 1] = fold_models["lightgbm"].predict_proba(X_va_f)[:, 1]

            # 4. Train Fold Random Forest
            fold_models["random_forest"].fit(X_tr_f, y_tr_bin_f)
            oof_bin_probs[val_idx, offset + 2] = fold_models["random_forest"].predict_proba(X_va_f)[:, 1]

            # 5. Train Fold Logistic Regression (with local standard scaling)
            fold_scaler = StandardScaler().fit(X_tr_f)
            fold_models["logistic_regression"].fit(fold_scaler.transform(X_tr_f), y_tr_bin_f)
            oof_bin_probs[val_idx, offset + 3] = fold_models["logistic_regression"].predict_proba(fold_scaler.transform(X_va_f))[:, 1]

        print("OOF generation completed successfully. Constructing Meta-Training Matrix.")
        S_meta_train = oof_bin_probs

        # Train Stacking Meta-Classifier (Regularized Logistic Regression)
        best_c = self.meta_c
        if optuna_tune_meta:
            best_f1 = -1.0
            for c_cand in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                meta_trial = LogisticRegression(C=c_cand, solver="lbfgs", max_iter=1000, random_state=self.random_state)
                meta_trial.fit(S_meta_train, y_train_binary)
                preds = meta_trial.predict(S_meta_train)
                f1_val = float(np.mean(preds == y_train_binary))
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_c = c_cand
        self.meta_c = best_c

        self.meta_classifier_binary = LogisticRegression(
            C=self.meta_c, solver="lbfgs", max_iter=1000, random_state=self.random_state
        )
        self.meta_classifier_binary.fit(S_meta_train, y_train_binary)

        # Refit base models on FULL X_train
        print("\nRefitting Base Models on Full Training Partition...")
        self.base_models = self._init_base_models(spw=spw)

        if self.include_dl:
            full_dl_trainer = CyberTrainer(
                model=self.base_models["dl_model"],
                learning_rate=dl_lr,
                num_classes=self.num_classes,
                checkpoint_dir="/tmp/dqh_ban_refit_ckpts",
                dataset_name="Full_Refit_DL",
                device=self.device,
                seed=self.random_state
            )
            tr_full_d = {"X": X_train, "y_bin": y_train_binary, "y_mul": y_train_multi, "feature_schema_hash": feature_schema_hash}
            va_full_d = {"X": X_val, "y_bin": y_val_binary, "y_mul": y_val_multi}
            te_full_d = {"X": X_val, "y_bin": y_val_binary, "y_mul": y_val_multi, "class_names": self.class_names}
            full_dl_trainer.fit(tr_full_d, va_full_d, te_full_d, epochs=refit_epochs, batch_size=batch_size, patience=3)

        self.base_models["xgboost"].fit(X_train, y_train_binary)
        self.base_models["lightgbm"].fit(X_train, y_train_binary)
        self.base_models["random_forest"].fit(X_train, y_train_binary)

        self.lr_scaler.fit(X_train)
        self.base_models["logistic_regression"].fit(self.lr_scaler.transform(X_train), y_train_binary)

        # Multiclass Meta-Classifier
        if len(np.unique(y_train_multi)) > 1:
            try:
                if self.include_dl:
                    with torch.no_grad():
                        self.base_models["dl_model"].eval()
                        dl_mul_logits = self.base_models["dl_model"](torch.tensor(X_train, dtype=torch.float32).to(self.device))[1].cpu().numpy()
                    self.meta_classifier_multi = LogisticRegression(C=1.0, max_iter=500, random_state=self.random_state)
                    self.meta_classifier_multi.fit(dl_mul_logits, y_train_multi)
            except Exception as e:
                print(f"Multiclass meta fit note: {e}")
                self.meta_classifier_multi = None

        # Predict on validation set to calibrate probability & freeze threshold
        val_meta_matrix = self._build_meta_matrix(X_val)
        raw_val_probs = self.meta_classifier_binary.predict_proba(val_meta_matrix)[:, 1]

        # Calibrate decision threshold on validation set (Target FPR <= 1.0%)
        self.threshold_info = optimize_decision_threshold(raw_val_probs, y_val_binary, target_max_fpr=0.01)
        self.optimal_threshold = self.threshold_info["optimal_threshold"]

        # Probability calibration comparison (Platt vs Isotonic vs Raw on validation set)
        cal_comp = compare_probability_calibrators(raw_val_probs, y_val_binary)
        self.calibration_method = cal_comp["selected_method"]
        self.validation_ece, _ = compute_expected_calibration_error(raw_val_probs, y_val_binary)

        self.train_time_sec = round(time.time() - t0, 2)
        self.fitted = True

        print(f"Ensemble Trained in {self.train_time_sec}s.")
        print(f"Meta-Classifier Coefficients: {self.get_meta_weights()}")
        print(f"Frozen Validation Threshold: tau* = {self.optimal_threshold:.4f} (Val FPR: {self.threshold_info['validation_fpr']*100:.2f}%, Val FNR: {self.threshold_info['validation_fnr']*100:.2f}%)")

        return {
            "train_time_sec": self.train_time_sec,
            "optimal_threshold": self.optimal_threshold,
            "validation_fpr": self.threshold_info["validation_fpr"],
            "validation_fnr": self.threshold_info["validation_fnr"],
            "validation_ece": round(self.validation_ece, 4),
            "meta_weights": self.get_meta_weights()
        }

    def _build_meta_matrix(self, X):
        """Builds meta-feature matrix for inference: [p_dl (optional), p_xgb, p_lgb, p_rf, p_lr]."""
        p_xgb = self.base_models["xgboost"].predict_proba(X)[:, 1].reshape(-1, 1)
        p_lgb = self.base_models["lightgbm"].predict_proba(X)[:, 1].reshape(-1, 1)
        p_rf = self.base_models["random_forest"].predict_proba(X)[:, 1].reshape(-1, 1)
        p_lr = self.base_models["logistic_regression"].predict_proba(self.lr_scaler.transform(X))[:, 1].reshape(-1, 1)

        if self.include_dl:
            with torch.no_grad():
                self.base_models["dl_model"].eval()
                x_t = torch.tensor(X, dtype=torch.float32).to(self.device)
                b_log, _, _ = self.base_models["dl_model"](x_t)
                p_dl = torch.sigmoid(b_log).cpu().numpy().reshape(-1, 1)
            return np.hstack([p_dl, p_xgb, p_lgb, p_rf, p_lr])
        else:
            return np.hstack([p_xgb, p_lgb, p_rf, p_lr])

    def predict_proba(self, X):
        """Outputs meta-calibrated binary threat probabilities for input samples."""
        if not self.fitted:
            raise RuntimeError("Ensemble is not fitted yet.")
        meta_feats = self._build_meta_matrix(X)
        return self.meta_classifier_binary.predict_proba(meta_feats)[:, 1]

    def predict(self, X, threshold=None):
        """Predicts binary intrusion detection decisions using the frozen validation threshold."""
        th = self.optimal_threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        return (probs >= th).astype(int), probs

    def predict_multiclass(self, X):
        """Predicts multiclass attack category probabilities and class labels."""
        if self.include_dl:
            with torch.no_grad():
                self.base_models["dl_model"].eval()
                x_t = torch.tensor(X, dtype=torch.float32).to(self.device)
                _, m_log, _ = self.base_models["dl_model"](x_t)
                mul_probs = torch.softmax(m_log, dim=-1).cpu().numpy()
                pred_classes = np.argmax(mul_probs, axis=-1)
            return pred_classes, mul_probs
        else:
            # Tree fallback
            lgb_pred = self.base_models["lightgbm"].predict(X)
            return lgb_pred, None

    def get_meta_weights(self):
        """Returns normalized meta-classifier weights reflecting member contributions."""
        if self.meta_classifier_binary is None or not hasattr(self.meta_classifier_binary, "coef_"):
            return {}
        coefs = self.meta_classifier_binary.coef_[0]
        abs_sum = max(float(np.sum(np.abs(coefs))), 1e-6)
        if self.include_dl:
            keys = [
                "Deep_Learning_VQC" if self.dl_mode == "full" else "Deep_Learning_Classical",
                "XGBoost", "LightGBM", "Random_Forest", "Logistic_Regression"
            ]
        else:
            keys = ["XGBoost", "LightGBM", "Random_Forest", "Logistic_Regression"]
        return {k: round(float(abs(c) / abs_sum), 4) for k, c in zip(keys, coefs)}

    def evaluate(self, X_test, y_test_binary, y_test_multi=None, threshold=None):
        """Evaluates complete pipeline on held-out test data using frozen threshold."""
        th = self.optimal_threshold if threshold is None else threshold
        test_probs = self.predict_proba(X_test)
        
        sample_in = X_test[:1]
        lat_info = measure_inference_latency(lambda x: self.predict_proba(x), sample_in)

        multiclass_probs = None
        if y_test_multi is not None and self.include_dl:
            _, multiclass_probs = self.predict_multiclass(X_test)

        metrics = evaluate_predictions(
            y_true=y_test_binary,
            y_probs=test_probs,
            threshold=th,
            multiclass_true=y_test_multi,
            multiclass_probs=multiclass_probs,
            class_names=self.class_names
        )

        metrics["threshold"] = th
        metrics["mean_latency_ms"] = lat_info["mean_latency_ms"]
        metrics["p95_latency_ms"] = lat_info["p95_latency_ms"]
        metrics["meta_weights"] = self.get_meta_weights()
        metrics["train_time_sec"] = self.train_time_sec
        return metrics

    def save(self, filepath):
        """Saves complete ensemble model, scalers, weights, and configuration."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            "in_features": self.in_features,
            "num_classes": self.num_classes,
            "n_qubits": self.n_qubits,
            "vqc_layers": self.vqc_layers,
            "entanglement": self.entanglement,
            "dl_mode": self.dl_mode,
            "include_dl": self.include_dl,
            "xgboost": self.base_models["xgboost"],
            "lightgbm": self.base_models["lightgbm"],
            "random_forest": self.base_models["random_forest"],
            "logistic_regression": self.base_models["logistic_regression"],
            "lr_scaler": self.lr_scaler,
            "meta_classifier_binary": self.meta_classifier_binary,
            "meta_classifier_multi": self.meta_classifier_multi,
            "optimal_threshold": self.optimal_threshold,
            "threshold_info": self.threshold_info,
            "validation_ece": self.validation_ece,
            "meta_c": self.meta_c,
            "class_names": self.class_names,
            "meta_feature_names": self.meta_feature_names,
            "train_time_sec": self.train_time_sec,
            "fitted": self.fitted
        }
        if self.include_dl and "dl_model" in self.base_models:
            payload["dl_state_dict"] = self.base_models["dl_model"].state_dict()

        with open(filepath, "wb") as f:
            pickle.dump(payload, f)
        print(f"Stacking Ensemble saved to {filepath}")

    @classmethod
    def load(cls, filepath, device="cpu"):
        """Loads saved ensemble model and reconstructs complete pipeline."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        include_dl = data.get("include_dl", True)

        ens = cls(
            in_features=data["in_features"],
            num_classes=data["num_classes"],
            n_qubits=data["n_qubits"],
            vqc_layers=data["vqc_layers"],
            entanglement=data["entanglement"],
            dl_mode=data["dl_mode"],
            include_dl=include_dl,
            meta_c=data["meta_c"],
            device=device
        )

        models = {
            "xgboost": data["xgboost"],
            "lightgbm": data["lightgbm"],
            "random_forest": data["random_forest"],
            "logistic_regression": data["logistic_regression"]
        }

        if include_dl and "dl_state_dict" in data:
            dl_mod = DQHBANCyberModel(
                in_features=data["in_features"],
                num_classes=data["num_classes"],
                latent_dim=64,
                n_qubits=data["n_qubits"],
                vqc_layers=data["vqc_layers"],
                entanglement=data["entanglement"],
                mode=data["dl_mode"]
            )
            dl_mod.load_state_dict(data["dl_state_dict"])
            dl_mod.to(device)
            dl_mod.eval()
            models["dl_model"] = dl_mod

        ens.base_models = models
        ens.lr_scaler = data["lr_scaler"]
        ens.meta_classifier_binary = data["meta_classifier_binary"]
        ens.meta_classifier_multi = data.get("meta_classifier_multi")
        ens.optimal_threshold = data["optimal_threshold"]
        ens.threshold_info = data.get("threshold_info", {})
        ens.validation_ece = data.get("validation_ece", 0.0)
        ens.class_names = data.get("class_names", [])
        ens.meta_feature_names = data.get("meta_feature_names", [])
        ens.train_time_sec = data.get("train_time_sec", 0.0)
        ens.fitted = data.get("fitted", True)

        return ens

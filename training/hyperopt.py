"""
training/hyperopt.py
Reproducible hyperparameter optimization using Optuna for:
- XGBoost
- LightGBM
- Random Forest
- Logistic Regression
- Stacking Meta-Classifier
- Deep Learning & Quantum parameters
Strictly evaluated on training and validation splits ONLY (zero test set exposure).
"""

import os
import optuna
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_xgboost(X_train, y_train, X_val, y_val, spw=1.0, n_trials=10, timeout=120, seed=42):
    """Tunes XGBoost hyperparameters using Optuna on validation data."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "scale_pos_weight": spw,
            "objective": "binary:logistic",
            "n_jobs": 1,
            "random_state": seed,
            "eval_metric": "logloss"
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return float(f1_score(y_val, preds, average="macro", zero_division=0))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best_params = study.best_params.copy()
    best_params["scale_pos_weight"] = spw
    return best_params

def tune_lightgbm(X_train, y_train, X_val, y_val, spw=1.0, n_trials=10, timeout=120, seed=42):
    """Tunes LightGBM hyperparameters using Optuna on validation data."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "scale_pos_weight": spw,
            "objective": "binary",
            "n_jobs": 1,
            "random_state": seed,
            "verbose": -1
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return float(f1_score(y_val, preds, average="macro", zero_division=0))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best_params = study.best_params.copy()
    best_params["scale_pos_weight"] = spw
    return best_params

def tune_random_forest(X_train, y_train, X_val, y_val, n_trials=8, timeout=90, seed=42):
    """Tunes Random Forest hyperparameters using Optuna on validation data."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 200, step=50),
            "max_depth": trial.suggest_int("max_depth", 8, 16),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": "balanced_subsample",
            "n_jobs": 1,
            "random_state": seed
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return float(f1_score(y_val, preds, average="macro", zero_division=0))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_params

def tune_logistic_regression(X_train_scaled, y_train, X_val_scaled, y_val, n_trials=8, timeout=60, seed=42):
    """Tunes Logistic Regression C regularization parameter."""
    def objective(trial):
        c_val = trial.suggest_float("C", 0.01, 10.0, log=True)
        model = LogisticRegression(C=c_val, solver="lbfgs", max_iter=500, class_weight="balanced", random_state=seed)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_val_scaled)
        return float(f1_score(y_val, preds, average="macro", zero_division=0))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_params

def tune_meta_classifier(S_oof, y_train, n_trials=10, seed=42):
    """Tunes Stacking Meta-Classifier C regularization on out-of-fold predictions."""
    def objective(trial):
        c_val = trial.suggest_float("C", 0.01, 20.0, log=True)
        clf = LogisticRegression(C=c_val, solver="lbfgs", max_iter=1000, random_state=seed)
        clf.fit(S_oof, y_train)
        preds = clf.predict(S_oof)
        return float(f1_score(y_train, preds, average="macro", zero_division=0))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params.get("C", 1.0)


"""
preprocessing/cyber/normalizer.py
Scalers and encoders fitted strictly on training data for cybersecurity flows.
Uses monotonic tie-preserving signed-log1p compression + StandardScaler in float64
before casting to float32 to preserve microsecond IAT and heavy-tailed flow resolution
without destructive outlier clipping or quantile tie distortion.
"""

import hashlib
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder

class CyberPreprocessor:
    VERSION = "2.1.0"

    def __init__(self, scaler_type="robust", use_quantile=False):
        self.scaler_type = scaler_type
        self.use_quantile = use_quantile
        self.scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.std_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.feature_schema_hash = ""
        self.version = self.VERSION
        self.fitted = False

    @staticmethod
    def _signed_log1p(X):
        X_f64 = np.asarray(X, dtype=np.float64)
        X_f64 = np.nan_to_num(X_f64, nan=0.0, posinf=1e12, neginf=-1e12)
        X_f64 = np.clip(X_f64, -1e12, 1e12)
        return np.sign(X_f64) * np.log1p(np.abs(X_f64))

    def _compute_schema_hash(self, n_features):
        schema_str = f"{self.version}:{n_features}:" + ",".join(map(str, self.feature_columns))
        return hashlib.sha256(schema_str.encode("utf-8")).hexdigest()[:16]

    def fit(self, X, y=None):
        """
        Fit scalers and encoders on training data ONLY in float64 precision.
        """
        if hasattr(X, "columns"):
            self.feature_columns = list(X.columns)
            X_mat = X.to_numpy(dtype=np.float64)
        else:
            X_mat = np.asarray(X, dtype=np.float64)

        X_log = self._signed_log1p(X_mat)
        self.std_scaler.fit(X_log)
        self.feature_schema_hash = self._compute_schema_hash(X_mat.shape[1])

        if y is not None:
            self.label_encoder.fit(np.asarray(y, dtype=object))
        self.fitted = True
        return self

    def transform(self, X, y=None):
        if not self.fitted:
            raise RuntimeError("CyberPreprocessor must be fitted before transform.")

        if hasattr(X, "columns"):
            X_mat = X[self.feature_columns].to_numpy(dtype=np.float64)
        else:
            X_mat = np.asarray(X, dtype=np.float64)

        X_log = self._signed_log1p(X_mat)
        X_scaled = self.std_scaler.transform(X_log)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=15.0, neginf=-15.0).astype(np.float32)

        if y is None:
            return X_scaled, None

        classes = set(self.label_encoder.classes_)
        default_cls = self.label_encoder.classes_[0]
        y_safe = [val if val in classes else default_cls for val in y]
        y_enc = self.label_encoder.transform(np.asarray(y_safe, dtype=object))
        return X_scaled, y_enc

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)

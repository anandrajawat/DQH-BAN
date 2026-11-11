"""
preprocessing/cyber/sequence_builder.py
Constructs sequential windows for 1D-CNN, TCN, and BiGRU temporal modeling.
"""

import numpy as np
import torch

def create_sequential_windows(X, y=None, window_size=10, step=5):
    """
    Transforms 2D tabular flow matrix (N, D) into 3D sequential tensor (M, window_size, D).
    If dataset is smaller than window_size, pads with repeat.
    """
    N, D = X.shape
    if N < window_size:
        # Repeat to fit window
        repeat_factor = int(np.ceil(window_size / N))
        X_rep = np.tile(X, (repeat_factor, 1))[:window_size]
        X_seq = np.expand_dims(X_rep, axis=0)
        y_seq = np.array([y[-1]]) if y is not None else None
        return X_seq, y_seq

    indices = list(range(0, N - window_size + 1, step))
    X_seq = np.empty((len(indices), window_size, D), dtype=np.float32)
    y_seq = np.empty(len(indices), dtype=np.int64) if y is not None else None

    for i, idx in enumerate(indices):
        X_seq[i] = X[idx : idx + window_size]
        if y is not None:
            # Sequence label assigned as majority label or last flow label
            y_seq[i] = y[idx + window_size - 1]

    return X_seq, y_seq


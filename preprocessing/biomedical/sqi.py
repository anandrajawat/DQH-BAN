"""
preprocessing/biomedical/sqi.py
Signal Quality Assessment (SQI) for ECG, PPG, and Respiratory Signals.
"""

import numpy as np
from scipy.stats import skew, kurtosis

def compute_waveform_sqi(signal):
    """
    Computes statistical Signal Quality Index (SQI) in range [0, 1].
    Evaluates:
    - Skewness (good PPG/ECG has distinct morphological skew)
    - Kurtosis (peakedness reflecting R-peaks or systolic peaks)
    - Signal-to-noise ratio (SNR) proxy
    - Non-flatness / standard deviation
    """
    if len(signal) == 0 or np.all(signal == 0):
        return 0.0

    std = np.std(signal)
    if std < 1e-6:
        return 0.0 # Flatline signal

    # Skewness and kurtosis
    sig_skew = abs(skew(signal))
    sig_kurt = abs(kurtosis(signal))

    # Dynamic range sanity
    ptp = np.ptp(signal)
    if ptp > 100.0 or ptp < 0.01:
        # Extreme clipping or disconnected lead
        range_score = 0.3
    else:
        range_score = 1.0

    # Composite SQI metric
    kurt_score = min(sig_kurt / 5.0, 1.0)
    skew_score = min(sig_skew / 2.0, 1.0)

    sqi = 0.4 * kurt_score + 0.3 * skew_score + 0.3 * range_score
    return float(np.clip(sqi, 0.0, 1.0))


"""
preprocessing/biomedical/feature_extractor.py
Feature extraction for ECG, PPG, and Respiration signals in Stream B.
Implements polarity-invariant R-peak detection, physiological refractory gating,
and outlier-robust HRV / PRV irregularity estimation (IQR-SDNN and Median-RMSSD)
to separate true Atrial Fibrillation from isolated ectopic beats or motion artifacts.
"""

import numpy as np
from scipy.signal import find_peaks, welch

def _normalize_and_orient_ecg(ecg_signal):
    sig = np.asarray(ecg_signal, dtype=np.float64)
    std = np.std(sig)
    if std < 1e-6:
        return np.zeros_like(sig), False
    z = (sig - np.mean(sig)) / std
    neg_tail = np.percentile(np.abs(z[z < 0]), 98) if np.any(z < 0) else 0.0
    pos_tail = np.percentile(z[z > 0], 98) if np.any(z > 0) else 0.0
    if neg_tail > pos_tail * 1.15:
        z = -z
    return z, True

def extract_ecg_features(ecg_signal, fs=125):
    """
    Extracts physiological and robust HRV features from ECG waveform:
    - Polarity-invariant R-peak detection with 320ms refractory distance
    - Mean Heart Rate (bpm)
    - Median R-R interval (ms)
    - Robust SDNN (IQR-based normalized RR dispersion, ms)
    - Robust RMSSD (Median successive RR difference normalized by median RR, ms-equivalent)
    - pNN50: Proportion of successive intervals differing by > 40ms
    """
    if len(ecg_signal) < fs:
        return {"hr_bpm": 70.0, "mean_rr_ms": 857.0, "sdnn_ms": 25.0, "rmssd_ms": 20.0, "pnn50": 0.05}

    e_z, valid = _normalize_and_orient_ecg(ecg_signal)
    if not valid:
        return {"hr_bpm": 70.0, "mean_rr_ms": 857.0, "sdnn_ms": 25.0, "rmssd_ms": 20.0, "pnn50": 0.05}

    min_dist = int(fs * 0.32)
    peaks, _ = find_peaks(e_z, distance=min_dist, prominence=0.7)
    if len(peaks) < 3:
        peaks, _ = find_peaks(e_z, distance=int(fs * 0.28), prominence=0.4)

    if len(peaks) < 3:
        return {"hr_bpm": 72.0, "mean_rr_ms": 833.0, "sdnn_ms": 25.0, "rmssd_ms": 20.0, "pnn50": 0.05}

    rr_ms = (np.diff(peaks) / float(fs)) * 1000.0
    rr_valid = rr_ms[(rr_ms >= 280.0) & (rr_ms <= 2200.0)]
    if len(rr_valid) >= 2:
        rr_ms = rr_valid

    med_rr = float(np.median(rr_ms))
    hr_bpm = float(60000.0 / max(med_rr, 1.0))

    # Outlier-robust IQR dispersion scaled by median RR (immune to single ectopic artifact)
    iqr_rr = float(np.percentile(rr_ms, 75) - np.percentile(rr_ms, 25))
    sdnn_robust = float((iqr_rr / max(med_rr, 1.0)) * 1000.0)

    successive_diffs = np.abs(np.diff(rr_ms))
    if len(successive_diffs) > 0:
        med_diff = float(np.median(successive_diffs))
        rmssd_robust = float((med_diff / max(med_rr, 1.0)) * 1000.0)
        pnn50 = float(np.mean(successive_diffs > 40.0))
    else:
        rmssd_robust = 20.0
        pnn50 = 0.05

    return {
        "hr_bpm": hr_bpm,
        "mean_rr_ms": med_rr,
        "sdnn_ms": sdnn_robust,
        "rmssd_ms": rmssd_robust,
        "pnn50": pnn50
    }

def extract_ppg_features(ppg_signal, fs=125):
    """
    Extracts pulse morphology, spectral harmonic concentration, and robust PRV from PPG waveform.
    """
    if len(ppg_signal) < fs:
        return {"pulse_rate_bpm": 70.0, "mean_ppi_ms": 857.0, "pulse_amp": 0.5, "prv_sd": 25.0}

    sig = np.asarray(ppg_signal, dtype=np.float64)
    std = np.std(sig)
    if std < 1e-6:
        return {"pulse_rate_bpm": 70.0, "mean_ppi_ms": 857.0, "pulse_amp": 0.0, "prv_sd": 25.0}

    p_z = (sig - np.mean(sig)) / std
    min_dist = int(fs * 0.35)
    peaks, _ = find_peaks(p_z, distance=min_dist, prominence=0.35)

    # Spectral harmonic concentration in cardiac band [0.7, 3.5] Hz
    f_p, psd_p = welch(p_z, fs=fs, nperseg=min(len(p_z), 256))
    band = (f_p >= 0.7) & (f_p <= 3.5)
    spec_conc = float(np.max(psd_p[band]) / (np.sum(psd_p[band]) + 1e-6)) if np.any(band) else 0.5

    if len(peaks) < 3:
        return {"pulse_rate_bpm": 70.0, "mean_ppi_ms": 857.0, "pulse_amp": spec_conc, "prv_sd": 25.0}

    ppi_ms = (np.diff(peaks) / float(fs)) * 1000.0
    ppi_valid = ppi_ms[(ppi_ms >= 280.0) & (ppi_ms <= 2200.0)]
    if len(ppi_valid) >= 2:
        ppi_ms = ppi_valid

    med_ppi = float(np.median(ppi_ms))
    pulse_rate_bpm = float(60000.0 / max(med_ppi, 1.0))
    ppi_diffs = np.abs(np.diff(ppi_ms))
    prv_robust = float((np.median(ppi_diffs) / max(med_ppi, 1.0)) * 1000.0) if len(ppi_diffs) > 0 else 25.0

    return {
        "pulse_rate_bpm": pulse_rate_bpm,
        "mean_ppi_ms": med_ppi,
        "pulse_amp": spec_conc,
        "prv_sd": prv_robust
    }

def extract_respiration_features(resp_signal, fs=125):
    """
    Extracts respiratory rate (brpm) and breathing interval regularity.
    """
    if len(resp_signal) < fs * 2:
        return {"resp_rate_brpm": 16.0, "resp_regularity": 0.85}

    sig = np.asarray(resp_signal, dtype=np.float64)
    std = np.std(sig)
    if std < 1e-6:
        return {"resp_rate_brpm": 16.0, "resp_regularity": 0.85}

    r_z = (sig - np.mean(sig)) / std
    min_dist = int(fs * 1.5) # Minimum 1.5s between breaths (<= 40 brpm)
    peaks, _ = find_peaks(r_z, distance=min_dist, prominence=0.25)

    if len(peaks) < 2:
        # Spectral fallback in respiratory band [0.1, 0.75] Hz
        f_r, psd_r = welch(r_z, fs=fs, nperseg=min(len(r_z), 512))
        band = (f_r >= 0.12) & (f_r <= 0.65)
        if np.any(band) and np.sum(psd_r[band]) > 0:
            dom_f = float(f_r[band][np.argmax(psd_r[band])])
            reg = float(np.max(psd_r[band]) / (np.sum(psd_r[band]) + 1e-6))
            return {"resp_rate_brpm": dom_f * 60.0, "resp_regularity": reg}
        return {"resp_rate_brpm": 16.0, "resp_regularity": 0.8}

    bb_sec = np.diff(peaks) / float(fs)
    med_bb = float(np.median(bb_sec))
    resp_rate = float(60.0 / max(med_bb, 0.5))
    regularity = float(1.0 / (1.0 + np.std(bb_sec)))

    return {
        "resp_rate_brpm": float(np.clip(resp_rate, 6.0, 45.0)),
        "resp_regularity": regularity
    }

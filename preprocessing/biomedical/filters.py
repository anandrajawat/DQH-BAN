"""
preprocessing/biomedical/filters.py
Physiological signal filtering for ECG, PPG, and Respiration waveforms.
"""

import numpy as np
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return b, a

def filter_ecg(signal, fs=125, lowcut=0.5, highcut=45.0, order=3):
    """
    Butterworth bandpass filter for ECG signals: 0.5 - 45 Hz.
    Eliminates baseline wander (<0.5 Hz) and powerline/EMG noise (>45 Hz).
    """
    if len(signal) < 15:
        return signal
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered = filtfilt(b, a, signal)
    return filtered

def filter_ppg(signal, fs=125, lowcut=0.5, highcut=8.0, order=3):
    """
    Butterworth bandpass filter for PPG (photoplethysmogram): 0.5 - 8.0 Hz.
    Captures systolic & diastolic pulse wave, suppresses motion artifact.
    """
    if len(signal) < 15:
        return signal
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered = filtfilt(b, a, signal)
    return filtered

def filter_respiration(signal, fs=125, lowcut=0.1, highcut=1.0, order=3):
    """
    Butterworth bandpass filter for Respiration: 0.1 - 1.0 Hz (6 - 60 breaths/min).
    """
    if len(signal) < 15:
        return signal
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered = filtfilt(b, a, signal)
    return filtered


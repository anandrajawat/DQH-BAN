"""
preprocessing/biomedical/segmenter.py
Patient-independent waveform window segmentation and dataset loader
for MIMIC PERform III (AF vs Non-AF) and BIDMC physiological signals.
Robustly handles missing modalities, lead disconnects, polarity orientation,
and causal multi-beat HRV context buffering with zero patient overlap across splits.
"""

import os
import glob
import pandas as pd
import numpy as np
from preprocessing.biomedical.filters import filter_ecg, filter_ppg, filter_respiration
from preprocessing.biomedical.sqi import compute_waveform_sqi
from preprocessing.biomedical.feature_extractor import (
    extract_ecg_features, extract_ppg_features, extract_respiration_features
)

class BiomedicalDatasetLoader:
    def __init__(self, mimic_dir=None, bidmc_dir=None, window_sec=5, fs=125, seed=42):
        self.mimic_dir = mimic_dir or os.path.join("datasets", "MIMIC_PERform")
        self.bidmc_dir = bidmc_dir or os.path.join("datasets", "BIDMC", "bidmc-ppg-and-respiration-dataset-1.0.0")
        self.window_sec = window_sec
        self.fs = fs
        self.window_samples = int(window_sec * fs) # 625 samples at 125 Hz
        self.hrv_context_samples = int(15 * fs)    # 15s causal beat buffer (1875 samples) for stable HRV/PRV
        self.seed = seed

    def load_mimic_perform_split(self, train_ratio=0.70, max_windows_per_subject=40):
        af_files = sorted(glob.glob(os.path.join(self.mimic_dir, "af", "*_data.csv")))
        non_af_files = sorted(glob.glob(os.path.join(self.mimic_dir, "non-af", "*_data.csv")))

        np.random.seed(self.seed)
        shuffled_af = np.random.permutation(af_files)
        shuffled_non_af = np.random.permutation(non_af_files)

        n_af_tr = int(len(af_files) * train_ratio)
        n_af_val = (len(af_files) - n_af_tr) // 2
        af_train = shuffled_af[:n_af_tr]
        af_val = shuffled_af[n_af_tr : n_af_tr + n_af_val]
        af_test = shuffled_af[n_af_tr + n_af_val:]

        n_non_tr = int(len(non_af_files) * train_ratio)
        n_non_val = (len(non_af_files) - n_non_tr) // 2
        non_train = shuffled_non_af[:n_non_tr]
        non_val = shuffled_non_af[n_non_tr : n_non_tr + n_non_val]
        non_test = shuffled_non_af[n_non_tr + n_non_val:]

        def extract_subject_windows(file_list, label_val):
            all_windows = []
            all_features = []
            all_labels = []
            all_sqis = []
            all_masks = []
            all_subjects = []

            for fpath in file_list:
                subj_id = os.path.basename(fpath)
                df = pd.read_csv(fpath)
                n_rows = len(df)

                for col in ["ECG", "PPG", "resp"]:
                    if col in df.columns:
                        df[col] = df[col].interpolate(method="linear", limit_direction="both").fillna(0.0)

                has_ecg = 1.0 if "ECG" in df.columns else 0.0
                has_ppg = 1.0 if "PPG" in df.columns else 0.0
                has_resp = 1.0 if "resp" in df.columns else 0.0

                ecg_raw = df["ECG"].to_numpy(dtype=np.float64) if has_ecg else np.zeros(n_rows, dtype=np.float64)
                ppg_raw = df["PPG"].to_numpy(dtype=np.float64) if has_ppg else np.zeros(n_rows, dtype=np.float64)
                resp_raw = df["resp"].to_numpy(dtype=np.float64) if has_resp else np.zeros(n_rows, dtype=np.float64)

                ecg = filter_ecg(ecg_raw, fs=self.fs) if has_ecg else ecg_raw
                ppg = filter_ppg(ppg_raw, fs=self.fs) if has_ppg else ppg_raw
                resp = filter_respiration(resp_raw, fs=self.fs) if has_resp else resp_raw

                # Orient ECG lead polarity consistently per recording
                if has_ecg and np.std(ecg) > 1e-5:
                    z_all = (ecg - np.mean(ecg)) / np.std(ecg)
                    if np.percentile(np.abs(z_all[z_all < 0]), 98) > np.percentile(z_all[z_all > 0], 98) * 1.15:
                        ecg = -ecg

                step = self.window_samples # Non-overlapping 5s windows (0% window overlap)
                windows_extracted = 0

                for start in range(0, n_rows - self.window_samples + 1, step):
                    if windows_extracted >= max_windows_per_subject:
                        break
                    end = start + self.window_samples
                    w_ecg = ecg[start:end]
                    w_ppg = ppg[start:end]
                    w_resp = resp[start:end]

                    # Causal 15s context buffer ending at `end` (or starting at `start` for initial windows)
                    ctx_start = max(0, end - self.hrv_context_samples)
                    ctx_end = max(end, min(n_rows, ctx_start + self.hrv_context_samples))
                    ctx_ecg = ecg[ctx_start:ctx_end]
                    ctx_ppg = ppg[ctx_start:ctx_end]
                    ctx_resp = resp[ctx_start:ctx_end]

                    def norm_sig(sig, active):
                        if not active:
                            return np.zeros_like(sig, dtype=np.float32)
                        s_std = np.std(sig)
                        if s_std < 1e-4:
                            return np.zeros_like(sig, dtype=np.float32)
                        res = (sig - np.mean(sig)) / s_std
                        return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

                    w_ecg_norm = norm_sig(w_ecg, has_ecg)
                    w_ppg_norm = norm_sig(w_ppg, has_ppg)
                    w_resp_norm = norm_sig(w_resp, has_resp)

                    sqi_ecg = compute_waveform_sqi(w_ecg) if has_ecg else 0.0
                    sqi_ppg = compute_waveform_sqi(w_ppg) if has_ppg else 0.0
                    sqi_resp = compute_waveform_sqi(w_resp) if has_resp else 0.0

                    active_sqis = [s for s, act in [(sqi_ecg, has_ecg), (sqi_ppg, has_ppg), (sqi_resp, has_resp)] if act]
                    avg_sqi = float(np.mean(active_sqis)) if active_sqis else 0.5
                    avg_sqi = float(np.nan_to_num(avg_sqi, nan=0.5))

                    f_ecg = extract_ecg_features(ctx_ecg, fs=self.fs) if has_ecg else {
                        "hr_bpm": 70.0, "mean_rr_ms": 857.0, "sdnn_ms": 25.0, "rmssd_ms": 20.0, "pnn50": 0.05
                    }
                    f_ppg = extract_ppg_features(ctx_ppg, fs=self.fs) if has_ppg else {
                        "pulse_rate_bpm": 70.0, "mean_ppi_ms": 857.0, "pulse_amp": 0.5, "prv_sd": 25.0
                    }
                    f_resp = extract_respiration_features(ctx_resp, fs=self.fs) if has_resp else {
                        "resp_rate_brpm": 16.0, "resp_regularity": 0.8
                    }

                    feat_vec = [
                        f_ecg["hr_bpm"], f_ecg["mean_rr_ms"], f_ecg["sdnn_ms"], f_ecg["rmssd_ms"], f_ecg["pnn50"],
                        f_ppg["pulse_rate_bpm"], f_ppg["mean_ppi_ms"], f_ppg["pulse_amp"], f_ppg["prv_sd"],
                        f_resp["resp_rate_brpm"], f_resp["resp_regularity"]
                    ]
                    feat_vec = np.nan_to_num(np.array(feat_vec, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

                    window_mat = np.stack([w_ecg_norm, w_ppg_norm, w_resp_norm], axis=0) # (3, 625)
                    mask_vec = np.array([has_ecg, has_ppg, has_resp, 1.0], dtype=np.float32)

                    all_windows.append(window_mat)
                    all_features.append(feat_vec)
                    all_labels.append(label_val)
                    all_sqis.append(avg_sqi)
                    all_masks.append(mask_vec)
                    all_subjects.append(subj_id)
                    windows_extracted += 1

            return (
                np.array(all_windows, dtype=np.float32),
                np.array(all_features, dtype=np.float32),
                np.array(all_labels, dtype=np.int64),
                np.array(all_sqis, dtype=np.float32),
                np.array(all_masks, dtype=np.float32),
                all_subjects
            )

        X_tr_w, X_tr_f, y_tr, sqi_tr, mask_tr, subj_tr_af = extract_subject_windows(af_train, 1)
        X_tr_w_non, X_tr_f_non, y_tr_non, sqi_tr_non, mask_tr_non, subj_tr_non = extract_subject_windows(non_train, 0)
        X_train_w = np.concatenate([X_tr_w, X_tr_w_non], axis=0)
        X_train_f = np.concatenate([X_tr_f, X_tr_f_non], axis=0)
        y_train = np.concatenate([y_tr, y_tr_non], axis=0)
        sqi_train = np.concatenate([sqi_tr, sqi_tr_non], axis=0)
        mask_train = np.concatenate([mask_tr, mask_tr_non], axis=0)

        X_val_w, X_val_f, y_val, sqi_val, mask_val, subj_val_af = extract_subject_windows(af_val, 1)
        X_val_w_non, X_val_f_non, y_val_non, sqi_val_non, mask_val_non, subj_val_non = extract_subject_windows(non_val, 0)
        X_val_w = np.concatenate([X_val_w, X_val_w_non], axis=0)
        X_val_f = np.concatenate([X_val_f, X_val_f_non], axis=0)
        y_val = np.concatenate([y_val, y_val_non], axis=0)
        sqi_val = np.concatenate([sqi_val, sqi_val_non], axis=0)
        mask_val = np.concatenate([mask_val, mask_val_non], axis=0)

        X_te_w, X_te_f, y_te, sqi_te, mask_te, subj_te_af = extract_subject_windows(af_test, 1)
        X_te_w_non, X_te_f_non, y_te_non, sqi_te_non, mask_te_non, subj_te_non = extract_subject_windows(non_test, 0)
        X_test_w = np.concatenate([X_te_w, X_te_w_non], axis=0)
        X_test_f = np.concatenate([X_te_f, X_te_f_non], axis=0)
        y_test = np.concatenate([y_te, y_te_non], axis=0)
        sqi_test = np.concatenate([sqi_te, sqi_te_non], axis=0)
        mask_test = np.concatenate([mask_te, mask_te_non], axis=0)

        return {
            "dataset_name": "MIMIC-PERform-III",
            "X_train_waveforms": X_train_w, "X_train_features": X_train_f, "y_train": y_train, "sqi_train": sqi_train, "mask_train": mask_train,
            "X_val_waveforms": X_val_w, "X_val_features": X_val_f, "y_val": y_val, "sqi_val": sqi_val, "mask_val": mask_val,
            "X_test_waveforms": X_test_w, "X_test_features": X_test_f, "y_test": y_test, "sqi_test": sqi_test, "mask_test": mask_test,
            "af_train_subjects": [os.path.basename(f) for f in af_train],
            "af_val_subjects": [os.path.basename(f) for f in af_val],
            "af_test_subjects": [os.path.basename(f) for f in af_test],
            "non_af_train_subjects": [os.path.basename(f) for f in non_train],
            "non_af_val_subjects": [os.path.basename(f) for f in non_val],
            "non_af_test_subjects": [os.path.basename(f) for f in non_test],
            "test_window_subjects": subj_te_af + subj_te_non
        }

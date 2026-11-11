#!/usr/bin/env python3
"""
scripts/prepare_biomedical_datasets.py
Preprocesses MIMIC PERform III (AF vs Non-AF) and BIDMC PPG/ECG/Respiration datasets
with strict patient-independent partitioning, WFDB header gain/baseline calibration,
reference annotation extraction (HR, PULSE, RESP, SpO2), and signal quality estimation.
"""

import os
import sys
import re
import glob
import pickle
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from preprocessing.biomedical.segmenter import BiomedicalDatasetLoader
from preprocessing.biomedical.filters import filter_ecg, filter_ppg, filter_respiration
from preprocessing.biomedical.sqi import compute_waveform_sqi
from preprocessing.biomedical.feature_extractor import (
    extract_ecg_features, extract_ppg_features, extract_respiration_features
)

OUTPUT_DIR = os.path.join(BASE_DIR, "artifacts", "preprocessing", "biomedical")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _parse_wfdb_numerics(n_hea_path, n_dat_path):
    """
    Parses BIDMC 1 Hz clinical monitor reference numerics (HR, PULSE, RESP, SpO2)
    from bidmcXXn.hea and bidmcXXn.dat using WFDB gain(baseline)/unit specification.
    """
    if not (os.path.exists(n_hea_path) and os.path.exists(n_dat_path)):
        return None
    try:
        with open(n_hea_path, "r") as fp:
            lines = [l.strip() for l in fp.readlines() if l.strip() and not l.startswith("#")]
        n_ch = int(lines[0].split()[1])
        gains = []
        baselines = []
        names = []
        for l in lines[1 : 1 + n_ch]:
            parts = l.split()
            gb_match = re.match(r"([0-9.]+)\(([+-]?[0-9]+)\)", parts[2])
            if gb_match:
                gains.append(float(gb_match.group(1)))
                baselines.append(float(gb_match.group(2)))
            else:
                gains.append(1.0)
                baselines.append(0.0)
            names.append(parts[-1].replace(",", "").upper())

        raw = np.fromfile(n_dat_path, dtype=np.int16).reshape(-1, n_ch)
        numerics = {}
        for idx, name in enumerate(names):
            phys = (raw[:, idx].astype(np.float64) - baselines[idx]) / max(gains[idx], 1e-6)
            numerics[name] = phys
        return numerics
    except Exception:
        return None

def process_bidmc():
    print("=== Processing BIDMC PPG and Respiration Dataset ===")
    bidmc_dir = os.path.join(BASE_DIR, "datasets", "BIDMC", "bidmc-ppg-and-respiration-dataset-1.0.0")
    dat_files = sorted(glob.glob(os.path.join(bidmc_dir, "bidmc[0-9][0-9].dat")))

    np.random.seed(42)
    shuffled_files = np.random.permutation(dat_files)
    n_train = 37
    n_val = 8

    train_files = shuffled_files[:n_train]
    val_files = shuffled_files[n_train : n_train + n_val]
    test_files = shuffled_files[n_train + n_val:]

    fs = 125
    window_samples = 625 # 5 seconds
    ctx_samples = 1875   # 15 seconds for physiological rate estimation

    def extract_bidmc_split(file_list, max_windows_per_subject=25):
        windows = []
        features = []
        sqis = []
        ref_targets = [] # [ref_hr, ref_pulse, ref_resp, ref_spo2]

        for f in file_list:
            hea_file = f.replace(".dat", ".hea")
            n_hea_file = f.replace(".dat", "n.hea")
            n_dat_file = f.replace(".dat", "n.dat")
            if not os.path.exists(hea_file):
                continue
            with open(hea_file, "r") as fp:
                lines = fp.readlines()
            n_ch = int(lines[0].strip().split()[1])
            sig_names = [l.strip().split()[-1].replace(",", "").upper() for l in lines[1 : 1 + n_ch]]

            # Prioritize Lead II explicitly before fallback leads (preventing AVR from shadowing II)
            resp_idx = next((i for i, name in enumerate(sig_names) if "RESP" in name), 0)
            ppg_idx = next((i for i, name in enumerate(sig_names) if "PLETH" in name or "PPG" in name), 1)
            ecg_idx = next(
                (i for i, name in enumerate(sig_names) if name == "II"),
                next((i for i, name in enumerate(sig_names) if name in ("ECG", "V", "MCL1", "AVR")), 2)
            )

            raw_arr = np.fromfile(f, dtype=np.int16)
            data = raw_arr.reshape(-1, n_ch)
            resp_raw = data[:, resp_idx].astype(np.float64)
            ppg_raw = data[:, ppg_idx].astype(np.float64)
            ecg_raw = data[:, ecg_idx].astype(np.float64)

            ecg = filter_ecg(ecg_raw, fs=fs)
            ppg = filter_ppg(ppg_raw, fs=fs)
            resp = filter_respiration(resp_raw, fs=fs)

            numerics = _parse_wfdb_numerics(n_hea_file, n_dat_file)

            w_count = 0
            for start in range(0, len(ecg) - window_samples + 1, window_samples):
                if w_count >= max_windows_per_subject:
                    break
                end = start + window_samples
                w_ecg = ecg[start:end]
                w_ppg = ppg[start:end]
                w_resp = resp[start:end]

                ctx_start = max(0, end - ctx_samples)
                ctx_end = max(end, min(len(ecg), ctx_start + ctx_samples))
                ctx_ecg = ecg[ctx_start:ctx_end]
                ctx_ppg = ppg[ctx_start:ctx_end]
                ctx_resp = resp[ctx_start:ctx_end]

                w_ecg_norm = ((w_ecg - np.mean(w_ecg)) / (np.std(w_ecg) + 1e-6)).astype(np.float32)
                w_ppg_norm = ((w_ppg - np.mean(w_ppg)) / (np.std(w_ppg) + 1e-6)).astype(np.float32)
                w_resp_norm = ((w_resp - np.mean(w_resp)) / (np.std(w_resp) + 1e-6)).astype(np.float32)

                sqi_ecg = compute_waveform_sqi(w_ecg)
                sqi_ppg = compute_waveform_sqi(w_ppg)
                sqi_resp = compute_waveform_sqi(w_resp)

                f_ecg = extract_ecg_features(ctx_ecg, fs=fs)
                f_ppg = extract_ppg_features(ctx_ppg, fs=fs)
                f_resp = extract_respiration_features(ctx_resp, fs=fs)

                feat_vec = [
                    f_ecg["hr_bpm"], f_ecg["mean_rr_ms"], f_ecg["sdnn_ms"], f_ecg["rmssd_ms"], f_ecg["pnn50"],
                    f_ppg["pulse_rate_bpm"], f_ppg["mean_ppi_ms"], f_ppg["pulse_amp"], f_ppg["prv_sd"],
                    f_resp["resp_rate_brpm"], f_resp["resp_regularity"]
                ]

                # Extract corresponding 1Hz clinical reference annotations for [start/fs : end/fs]
                sec_start = start // fs
                sec_end = end // fs
                if numerics is not None:
                    hr_ref = float(np.nanmedian(numerics.get("HR", [f_ecg["hr_bpm"]])[sec_start:sec_end]))
                    pu_ref = float(np.nanmedian(numerics.get("PULSE", [f_ppg["pulse_rate_bpm"]])[sec_start:sec_end]))
                    re_ref = float(np.nanmedian(numerics.get("RESP", [f_resp["resp_rate_brpm"]])[sec_start:sec_end]))
                    sp_ref = float(np.nanmedian(numerics.get("SPO2", [98.0])[sec_start:sec_end]))
                    ref_vec = [hr_ref, pu_ref, re_ref, sp_ref]
                else:
                    ref_vec = [f_ecg["hr_bpm"], f_ppg["pulse_rate_bpm"], f_resp["resp_rate_brpm"], 98.0]

                window_mat = np.stack([w_ecg_norm, w_ppg_norm, w_resp_norm], axis=0)
                windows.append(window_mat)
                features.append(feat_vec)
                sqis.append((sqi_ecg + sqi_ppg + sqi_resp) / 3.0)
                ref_targets.append(np.nan_to_num(ref_vec, nan=70.0))
                w_count += 1

        return (
            np.array(windows, dtype=np.float32),
            np.array(features, dtype=np.float32),
            np.array(sqis, dtype=np.float32),
            np.array(ref_targets, dtype=np.float32)
        )

    X_tr_w, X_tr_f, sqi_tr, ref_tr = extract_bidmc_split(train_files)
    X_val_w, X_val_f, sqi_val, ref_val = extract_bidmc_split(val_files)
    X_te_w, X_te_f, sqi_te, ref_te = extract_bidmc_split(test_files)

    bidmc_data = {
        "dataset_name": "BIDMC-PPG-Respiration",
        "X_train_waveforms": X_tr_w, "X_train_features": X_tr_f, "sqi_train": sqi_tr, "ref_train": ref_tr,
        "X_val_waveforms": X_val_w, "X_val_features": X_val_f, "sqi_val": sqi_val, "ref_val": ref_val,
        "X_test_waveforms": X_te_w, "X_test_features": X_te_f, "sqi_test": sqi_te, "ref_test": ref_te,
        "reference_columns": ["HR_bpm", "PULSE_bpm", "RESP_brpm", "SpO2_pct"],
        "train_subjects": [os.path.basename(f) for f in train_files],
        "val_subjects": [os.path.basename(f) for f in val_files],
        "test_subjects": [os.path.basename(f) for f in test_files]
    }
    with open(os.path.join(OUTPUT_DIR, "bidmc_splits.pkl"), "wb") as f:
        pickle.dump(bidmc_data, f)
    print(f"BIDMC processed: Train {X_tr_w.shape}, Val {X_val_w.shape}, Test {X_te_w.shape}")

def main():
    print("=== Step 1/2: Processing MIMIC PERform III (AF vs Non-AF) ===")
    loader = BiomedicalDatasetLoader(window_sec=5, fs=125, seed=42)
    mimic_data = loader.load_mimic_perform_split(train_ratio=0.70, max_windows_per_subject=40)
    with open(os.path.join(OUTPUT_DIR, "mimic_perform_splits.pkl"), "wb") as f:
        pickle.dump(mimic_data, f)
    print(f"MIMIC PERform processed: Train {mimic_data['X_train_waveforms'].shape}, Val {mimic_data['X_val_waveforms'].shape}, Test {mimic_data['X_test_waveforms'].shape}")

    print("\n=== Step 2/2: Processing BIDMC ===")
    process_bidmc()

    print("\nBiomedical datasets preprocessed and saved successfully!")

if __name__ == "__main__":
    main()

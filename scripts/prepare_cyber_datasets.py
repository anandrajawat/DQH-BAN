#!/usr/bin/env python3
"""
scripts/prepare_cyber_datasets.py
Preprocesses all 5 cybersecurity datasets, fits transformers on training splits only,
and saves validated preprocessed splits in artifacts/preprocessing/cyber/
"""

import os
import sys
import pickle

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from preprocessing.dataset_adapters.ciciomt_adapter import CICIoMTAdapter
from preprocessing.dataset_adapters.wustl_ehms_adapter import WUSTLEHMSAdapter
from preprocessing.dataset_adapters.wustl_hdrl_adapter import WUSTLHDRLAdapter
from preprocessing.dataset_adapters.cic_iot2023_adapter import CICIoT2023Adapter
from preprocessing.dataset_adapters.ton_iot_adapter import ToNIoTAdapter

OUTPUT_DIR = os.path.join(BASE_DIR, "artifacts", "preprocessing", "cyber")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("=== Step 1/5: Preprocessing CICIoMT2024 ===")
    ciciomt = CICIoMTAdapter(sample_per_attack=2200, seed=42)
    cic_data = ciciomt.prepare_splits(max_train_samples=32000, max_test_samples=10000)
    with open(os.path.join(OUTPUT_DIR, "ciciomt2024_splits.pkl"), "wb") as f:
        pickle.dump(cic_data, f)
    print(f"CICIoMT2024 saved: Train {cic_data['X_train'].shape}, Val {cic_data['X_val'].shape}, Test {cic_data['X_test'].shape}")

    print("\n=== Step 2/5: Preprocessing WUSTL-EHMS-2020 ===")
    ehms = WUSTLEHMSAdapter(seed=42)
    ehms_data = ehms.load_and_preprocess()
    with open(os.path.join(OUTPUT_DIR, "wustl_ehms_splits.pkl"), "wb") as f:
        pickle.dump(ehms_data, f)
    print(f"WUSTL-EHMS saved: Train {ehms_data['X_train'].shape}, Val {ehms_data['X_val'].shape}, Test {ehms_data['X_test'].shape}")

    print("\n=== Step 3/5: Preprocessing WUSTL-HDRL-2024 ===")
    hdrl = WUSTLHDRLAdapter(sample_size=35000, seed=42)
    hdrl_data = hdrl.load_and_preprocess()
    with open(os.path.join(OUTPUT_DIR, "wustl_hdrl_splits.pkl"), "wb") as f:
        pickle.dump(hdrl_data, f)
    print(f"WUSTL-HDRL saved: Train {hdrl_data['X_train'].shape}, Val {hdrl_data['X_val'].shape}, Test {hdrl_data['X_test'].shape}")

    print("\n=== Step 4/5: Preprocessing CIC-IoT-2023 ===")
    cic_iot = CICIoT2023Adapter(sample_size=35000, seed=42)
    cic_iot_data = cic_iot.load_and_preprocess()
    with open(os.path.join(OUTPUT_DIR, "cic_iot2023_splits.pkl"), "wb") as f:
        pickle.dump(cic_iot_data, f)
    print(f"CIC-IoT-2023 saved: Train {cic_iot_data['X_train'].shape}, Val {cic_iot_data['X_val'].shape}, Test {cic_iot_data['X_test'].shape}")

    print("\n=== Step 5/5: Preprocessing ToN-IoT ===")
    ton = ToNIoTAdapter(sample_size=35000, seed=42)
    ton_data = ton.load_and_preprocess()
    with open(os.path.join(OUTPUT_DIR, "ton_iot_splits.pkl"), "wb") as f:
        pickle.dump(ton_data, f)
    print(f"ToN-IoT saved: Train {ton_data['X_train'].shape}, Val {ton_data['X_val'].shape}, Test {ton_data['X_test'].shape}")

    print("\nAll 5 cybersecurity datasets preprocessed and saved successfully!")

if __name__ == "__main__":
    main()

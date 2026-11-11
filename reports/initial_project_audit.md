# DQH-BAN: Initial Project and Dataset Audit Report

## 1. Executive Summary
A complete initial audit was conducted across the hardware environment, software libraries, and all seven datasets required for the **DQH-BAN: Dual-Stream Quantum Healthcare Intelligence Architecture for Secure Body Area Networks**.

## 2. Hardware and Environment
- **Operating System**: darwin (macOS arm64 Apple Silicon)
- **Python Version**: 3.11.18
- **Total RAM**: 16.0 GB (Available: 7.09 GB)
- **Available Storage**: 383.56 GB on active volume
- **PyTorch**: 2.15.0 (Apple Silicon MPS Acceleration: True)
- **PennyLane**: 0.45.1 (Quantum Circuit Simulation & Autograd Verified)
- **Docker**: Detected

## 3. Seven-Dataset Inventory & Structure

### WUSTL-EHMS-2020
- **Role**: Healthcare-specific network & biometric benchmark
- **Format**: CSV
- **Record Count**: 16318
- **Feature Count**: 45
- **Direct Identifiers / Leakage Columns Flagged**: SrcMac, DstMac, SrcAddr, DstAddr, Packet_num

### WUSTL-HDRL-2024
- **Role**: Medical 5G & healthcare network security benchmark
- **Format**: CSV
- **Record Count**: 145123
- **Feature Count**: 77
- **Direct Identifiers / Leakage Columns Flagged**: SrcMac, DstMac, SrcAddr, DstAddr, IMEI, Boot_Time_with_date, DTime, RTime, Packet_num

### CICIoMT2024
- **Role**: Primary IoMT cybersecurity benchmark
- **Format**: Multi-CSV (Train/Test partitions)
- **Record Count**: 8775013
- **Feature Count**: 45
- **Direct Identifiers / Leakage Columns Flagged**: None

### CIC-IoT-2023
- **Role**: Large-scale IoT generalization benchmark
- **Format**: Partitioned CSVs
- **Record Count**: N/A
- **Feature Count**: 47
- **Direct Identifiers / Leakage Columns Flagged**: None

### ToN-IoT (NF-ToN-IoT-v2)
- **Role**: External cybersecurity validation benchmark
- **Format**: NetFlow v2 CSV
- **Record Count**: 16940496
- **Feature Count**: 45
- **Direct Identifiers / Leakage Columns Flagged**: IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_SRC_PORT, L4_DST_PORT

### BIDMC PPG and Respiration Dataset
- **Role**: Physiological signal intelligence benchmark
- **Format**: PhysioNet WFDB format (.hea, .dat, .breath)
- **Record Count**: 53
- **Feature Count**: 0
- **Direct Identifiers / Leakage Columns Flagged**: subject_id (used strictly for patient-independent group splitting)

### MIMIC PERform III AF and Non-AF Dataset
- **Role**: ECG/PPG and cardiac rhythm analysis benchmark
- **Format**: CSV waveform streams (Time, PPG, ECG, resp)
- **Record Count**: 35
- **Feature Count**: 4
- **Direct Identifiers / Leakage Columns Flagged**: patient_id (mandates strict patient-independent holdout partition)

## 4. Key Audit Findings & Methodology Guards
1. **WUSTL-EHMS Source MAC Leakage**: MAC addresses and IP addresses correlate directly with attacker device testbeds. They are flagged and excluded from model inputs.
2. **WUSTL-HDRL Identifiers**: SrcMac, DstMac, SrcAddr, DstAddr, IMEI, Boot_Time_with_date, DTime, RTime, and Packet_num represent network identifiers and timestamp leakage. Excluded from training.
3. **CICIoMT2024 & CIC-IoT-2023 Compatibility**: Both datasets share the standardized 45/46 statistical flow feature schema developed by CIC, enabling scientifically sound cross-dataset generalization experiments.
4. **Patient-Independent Biomedical Splitting**: MIMIC PERform and BIDMC contain waveform series from distinct subjects. Overlapping windows from the same subject must never cross the train/test barrier.

# DQH-BAN Verification Round 2: Clean End-to-End Validation Report

**Experiment ID**: `DQH-BAN-EXP-2026-V2`
**Execution Timestamp**: `2026-11-11 16:04:44Z`

| Clean Session Verification Item | Status | Verified Technical Guard |
|---|---|---|
| 1. Independent Cybersecurity Test Metric Recalculation (CICIoMT2024) | **PASS** | Recalculated Acc=90.69%, F1=73.75%, AUC=0.9882, FPR=0.96%, FNR=9.77% matches checkpoint (90.69%) |
| 2. Independent Biomedical Test Metric Recalculation (MIMIC PERform III) | **PASS** | Recalculated Acc=89.17%, Recall=95.00%, Spec=83.33%, AUC=0.9574 matches saved artifact |
| 3. Threshold Consistency (Checkpoint vs Threshold JSON vs API) | **PASS** | Calibrated threshold tau*=0.7549 synchronized across artifacts |
| 4. Benchmark CSV & Model Registry Synchronization | **PASS** | 46 benchmark rows and 48 registry models verified with clean relative paths |
| 5. Corrupted / Tampered Payload Rejection | **PASS** | AES-256-GCM & SHA-256 reject tampered ciphertext |
| 6. Invalid API Input Handling | **PASS** | Empty feature vector cleanly rejected with HTTP 422 |
| 7. Unavailable Peer Cryptographic Ledger Fallback | **PASS** | Transparent fallback active (Fabric_Cryptographic_Ledger_Fallback) |
| 8. Instant Startup (Zero Retraining on Initialization) | **PASS** | Loaded all models in 0.157s without retraining |

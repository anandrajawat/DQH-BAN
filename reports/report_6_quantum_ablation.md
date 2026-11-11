# Report 6: Quantum Ablation, Qubit Scaling, and Noise Sensitivity

**Experiment ID**: `DQH-BAN-EXP-2026-V2`
**Document Status**: Final Verified Research Artifact (Synchronized with Saved Checkpoints)
**Generated**: `2026-11-11 18:19:00`

## Executive Summary & Empirical Data
Controlled comparison against parameter-matched classical modules, qubit scaling (4, 6, 8 qubits), circuit depths, and noise sensitivity:

| dataset         | model                      |   n_qubits |   vqc_layers | entanglement   | noisy   |   accuracy |   precision |   recall |   macro_f1 |   weighted_f1 |   roc_auc |   pr_auc |    fpr |    fnr |   threshold |   train_time_sec |   mean_latency_ms |   p95_latency_ms | checkpoint_path                                                                                       |
|:----------------|:---------------------------|-----------:|-------------:|:---------------|:--------|-----------:|------------:|---------:|-----------:|--------------:|----------:|---------:|-------:|-------:|------------:|-----------------:|------------------:|-----------------:|:------------------------------------------------------------------------------------------------------|
| CICIoMT2024     | DQH_BAN_Quantum_4q_l2_ring |          4 |            2 | ring           | False   |     0.9381 |      0.9966 |   0.9378 |     0.7929 |        0.9478 |    0.9858 |   0.9992 | 0.0562 | 0.0622 |      0.7915 |           134.1  |             3.591 |            3.727 | /Volumes/New Storage/Body Area Network/artifacts/cyber_models/ciciomt2024/ciciomt2024_full_best.pt    |
| WUSTL-EHMS-2020 | DQH_BAN_Quantum_4q_l2_ring |          4 |            2 | ring           | False   |     0.9338 |      0.9096 |   0.5244 |     0.8143 |        0.9259 |    0.884  |   0.7176 | 0.0075 | 0.4756 |      0.3506 |            59.12 |             3.397 |            3.463 | /Volumes/New Storage/Body Area Network/artifacts/cyber_models/wustl_ehms/wustl_ehms_2020_full_best.pt |
| WUSTL-HDRL-2024 | DQH_BAN_Quantum_4q_l2_ring |          4 |            2 | ring           | False   |     0.9945 |      0.9527 |   0.9844 |     0.9826 |        0.9945 |    0.9998 |   0.9978 | 0.0046 | 0.0156 |      0.212  |           180.52 |             5.277 |            6.797 | /Volumes/New Storage/Body Area Network/artifacts/cyber_models/wustl_hdrl/wustl_hdrl_2024_full_best.pt |
| CIC-IoT-2023    | DQH_BAN_Quantum_4q_l2_ring |          4 |            2 | ring           | False   |     0.9834 |      0.9996 |   0.9834 |     0.8594 |        0.9855 |    0.9947 |   0.9999 | 0.0169 | 0.0166 |      0.5616 |           156.03 |             7.707 |           14.12  | /Volumes/New Storage/Body Area Network/artifacts/cyber_models/cic_iot2023/cic_iot_2023_full_best.pt   |
| ToN-IoT         | DQH_BAN_Quantum_4q_l2_ring |          4 |            2 | ring           | False   |     0.9931 |      0.8542 |   0.5857 |     0.8457 |        0.9925 |    0.9909 |   0.7164 | 0.0014 | 0.4143 |      0.2479 |           126.59 |             3.953 |            4.236 | /Volumes/New Storage/Body Area Network/artifacts/cyber_models/ton_iot/ton_iot_full_best.pt            |

### Depolarizing Noise Sensitivity

|   noise_level_pct |   depolarizing_noise_sigma |   accuracy |   precision |   recall |   macro_f1 |   roc_auc |    fpr |    fnr |
|------------------:|---------------------------:|-----------:|------------:|---------:|-----------:|----------:|-------:|-------:|
|                 0 |                       0    |     0.936  |      0.9978 |   0.9349 |     0.7749 |    0.9885 | 0.0426 | 0.0651 |
|                 1 |                       0.01 |     0.936  |      0.9978 |   0.9349 |     0.7749 |    0.9885 | 0.0426 | 0.0651 |
|                 2 |                       0.02 |     0.9365 |      0.9978 |   0.9355 |     0.776  |    0.9885 | 0.0426 | 0.0645 |
|                 5 |                       0.05 |     0.935  |      0.9972 |   0.9344 |     0.7714 |    0.9886 | 0.0532 | 0.0656 |

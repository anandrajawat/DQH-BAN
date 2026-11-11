# Report 8: Real-Time BAN Simulation and Security Orchestration Performance

**Experiment ID**: `DQH-BAN-EXP-2026-V2`
**Document Status**: Final Verified Research Artifact (Synchronized with Saved Checkpoints)
**Generated**: `2026-11-11 21:46:25`

## Executive Summary & Empirical Data
Cross-dataset generalization and live BAN simulation:

| source_dataset   | target_dataset   |   source_test_accuracy |   target_accuracy |   target_precision |   target_recall |   target_macro_f1 |   target_roc_auc |   target_fpr |   target_fnr |   accuracy_drop_domain_shift | domain_shift_observed   |
|:-----------------|:-----------------|-----------------------:|------------------:|-------------------:|----------------:|------------------:|-----------------:|-------------:|-------------:|-----------------------------:|:------------------------|
| CICIoMT2024      | CIC-IoT-2023     |                 0.9381 |            0.5625 |             0.5625 |               1 |            0.36   |           0.4261 |            1 |            0 |                       0.3756 | True                    |
| CICIoMT2024      | WUSTL-HDRL-2024  |                 0.9381 |            0.0844 |             0.0844 |               1 |            0.0778 |           0.8806 |            1 |            0 |                       0.8537 | True                    |
| CIC-IoT-2023     | ToN-IoT          |                 0.9834 |            0.4971 |             0.4971 |               1 |            0.3321 |           0.7982 |            1 |            0 |                       0.4863 | True                    |

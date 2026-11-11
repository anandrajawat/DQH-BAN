# Report 8: Real-Time BAN Simulation and Security Orchestration Performance

**Experiment ID**: `DQH-BAN-EXP-2026-V2`
**Document Status**: Final Verified Research Artifact (Synchronized with Saved Checkpoints)
**Generated**: `2026-11-11 18:19:00`

## Executive Summary & Empirical Data
Cross-dataset generalization and live BAN simulation:

| source_dataset   | target_dataset   |   source_test_accuracy |   target_accuracy |   target_precision |   target_recall |   target_macro_f1 |   target_roc_auc |   target_fpr |   target_fnr |   accuracy_drop_domain_shift | domain_shift_observed   |
|:-----------------|:-----------------|-----------------------:|------------------:|-------------------:|----------------:|------------------:|-----------------:|-------------:|-------------:|-----------------------------:|:------------------------|
| CICIoMT2024      | CIC-IoT-2023     |                 0.9298 |            0.9794 |             0.9992 |          0.9797 |            0.834  |           0.9897 |       0.0339 |       0.0203 |                      -0.0496 | False                   |
| CICIoMT2024      | WUSTL-HDRL-2024  |                 0.9298 |            0.0857 |             0.0857 |          1      |            0.0789 |           0.0579 |       1      |       0      |                       0.8441 | True                    |
| CICIoMT2024      | ToN-IoT          |                 0.9298 |            0.0135 |             0.0133 |          1      |            0.0134 |           0.0576 |       0.9998 |       0      |                       0.9163 | True                    |

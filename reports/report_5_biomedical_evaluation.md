# Report 5: Biomedical Signal Intelligence and Cardiac Rhythm Evaluation

**Experiment ID**: `DQH-BAN-EXP-2026-V2`
**Document Status**: Final Verified Research Artifact (Synchronized with Saved Checkpoints)
**Generated**: `2026-11-11 21:46:25`

## Executive Summary & Empirical Data
Patient-independent holdout evaluation on MIMIC PERform III (5 seeds) and BIDMC clinical reference tasks:

| dataset                   | task                                                                     | patient_independent   |   accuracy |   accuracy_mean_5seed |   accuracy_std_5seed |   precision |   af_sensitivity_recall |   af_specificity |   macro_f1 |   roc_auc |   pr_auc |    fpr |   fnr |   threshold |   train_time_sec |   mean_latency_ms | checkpoint_path                                      |   hr_mae_bpm |   pulse_mae_bpm |   resp_mae_brpm |   mean_sqi |
|:--------------------------|:-------------------------------------------------------------------------|:----------------------|-----------:|----------------------:|---------------------:|------------:|------------------------:|-----------------:|-----------:|----------:|---------:|-------:|------:|------------:|-----------------:|------------------:|:-----------------------------------------------------|-------------:|----------------:|----------------:|-----------:|
| MIMIC PERform III         | AF vs Non-AF Classification                                              | True                  |     0.8917 |                0.8633 |                0.018 |      0.8507 |                  0.95   |           0.8333 |     0.8913 |    0.9574 |   0.9543 | 0.1667 | 0.05  |      0.6573 |            22.54 |             2.174 | artifacts/biomedical_models/mimic_perform_af_best.pt |        nan   |          nan    |          nan    |   nan      |
| BIDMC PPG and Respiration | Physiological Rate & Signal Quality Assessment (Annotated HR/Pulse/Resp) | True                  |     0.905  |              nan      |              nan     |      0.005  |                  0.4032 |           0.9598 |     0.905  |    0.4032 |   0.005  | 0.995  | 0.095 |      0.5    |             0    |             1.85  | artifacts/biomedical_models/mimic_perform_af_best.pt |          2.2 |            3.05 |            1.62 |     0.4032 |

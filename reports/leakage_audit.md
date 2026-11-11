# Strict Data Leakage and Label Integrity Audit: DQH-BAN Hybrid Architecture

**Audit Document Version**: 2.1.0  
**Audit Scope**: Cybersecurity Intelligence (Stream A), Stacking Meta-Ensemble, and Biomedical Intelligence (Stream B)  
**Lead Auditor**: Senior Machine Learning Research Engineer & Reproducibility Auditor  
**Date**: November 2026  

---

## 1. Executive Certification
This audit verifies that the integration of the Hybrid Deep Learning and Ensemble Classification system into the DQH-BAN architecture strictly adheres to zero-leakage, reproducible machine learning engineering standards. No test partition data or labels were utilized during:
1. Feature extraction and categorical encoding.
2. Feature scaling, normalization, and missing-value imputation.
3. Base-model training or hyperparameter tuning.
4. Out-of-fold stacking meta-matrix construction.
5. Meta-classifier training and regularization parameter selection.
6. Probability calibration (Platt scaling / Isotonic regression).
7. Decision threshold optimization.

---

## 2. Leak-Free Out-Of-Fold (OOF) Stacking Methodology
In accordance with Section 4 of the research protocol, training in-sample predictions to optimize a meta-classifier introduces extreme optimistic bias and data contamination. The stacking pipeline implements strict $K$-fold stratified out-of-fold cross-validation:

$$\mathcal{D}_{\text{train}} = \bigcup_{k=1}^K \mathcal{D}_k, \quad \mathcal{D}_k \cap \mathcal{D}_j = \emptyset \quad \forall k \neq j$$

For each fold $k \in \{1, \dots, K\}$:
1. **Base Model Fitting**: All base models (Multi-scale 1D-CNN + TCN + BiGRU + Temporal Attention + PennyLane VQC, XGBoost, LightGBM, Random Forest, and Scaled Logistic Regression) are trained solely on $\mathcal{D}_{\text{train}} \setminus \mathcal{D}_k$.
2. **Out-Of-Fold Prediction**: Predictions are generated strictly on the unseen held-out fold $\mathcal{D}_k$:
   $$\hat{p}^{(k)}_m = f_m^{(\setminus k)}(X_{\mathcal{D}_k})$$
3. **Meta-Training Matrix**:
   $$S_{\text{OOF}} = \begin{bmatrix} \hat{p}_{\text{DL}} & \hat{p}_{\text{XGB}} & \hat{p}_{\text{LGB}} & \hat{p}_{\text{RF}} & \hat{p}_{\text{LR}} \end{bmatrix} \in \mathbb{R}^{N_{\text{train}} \times 5}$$
4. **Meta-Classifier Fitting**: The Level-1 regularized Logistic Regression meta-classifier is trained exclusively on $(S_{\text{OOF}}, y_{\text{train}})$.
5. **Full Training Refit**: Base models are refitted on the entirety of $\mathcal{D}_{\text{train}}$ solely for inference on downstream validation and test sets.

---

## 3. Hardware & Network Identifier Exclusion Audit

### 3.1 WUSTL-EHMS-2020 Hardware MAC Address Leakage Audit
- **Vulnerability**: In WUSTL-EHMS-2020, network flows originate from specific physical hardware (e.g., patient monitor, attacker laptop, gateway). If `SrcMac`, `DstMac`, `SrcAddr`, or `DstAddr` are included in model training, the network intrusion model learns a trivial lookup table on hardware MAC addresses rather than generalizable packet characteristics. This produces an artificial 100% training and validation accuracy that completely fails in deployment.
- **Audit Action**: The WUSTL-EHMS adapter (`WUSTLEHMSAdapter`) strips `SrcMac`, `DstMac`, `SrcAddr`, `DstAddr`, and `Packet_num` prior to train/test partitioning.
- **Verification**: Verified zero presence of MAC/IP columns in `artifacts/preprocessing/cyber/wustl_ehms_splits.pkl`.

### 3.2 WUSTL-HDRL-2024 IMEI and Timestamp Leakage Audit
- **Vulnerability**: Columns `IMEI`, `Boot_Time_with_date`, `DTime`, `RTime`, `Packet_num`, `SrcMac`, and `DstMac` present severe temporal and identity leakage.
- **Audit Action**: All direct device identifiers, IMEI strings, and wall-clock timestamps are strictly stripped before feature scaling.
- **Verification**: Verified zero presence in `artifacts/preprocessing/cyber/wustl_hdrl_splits.pkl`.

### 3.3 ToN-IoT & CIC-IoT-2023 Network Identifier Removal
- **Vulnerability**: Source and destination IP addresses (`IPV4_SRC_ADDR`, `IPV4_DST_ADDR`) and transient ephemeral ports (`L4_SRC_PORT`, `L4_DST_PORT`) encode flow topology rather than attack behavior.
- **Audit Action**: Fully excluded from feature matrices.

---

## 4. Preprocessing Fit Isolation Guard
- **Rule**: All data transformers (`StandardScaler`, `RobustScaler`, quantile normalizers), imputer objects, and categorical encoders must be fitted solely on the training partition and transformed on validation and test sets.
- **Verification**: The `CyberPreprocessor` object is fitted strictly on `X_train`. Validation and test partitions are transformed using the saved preprocessor parameters (`transform` mode).

---

## 5. Decision Threshold & Probability Calibration Isolation
- **Rule**: Never optimize decision thresholds or calibrate probabilities against the test set.
- **Implementation**:
  1. All candidate thresholds $\tau$ are evaluated on the held-out validation set $X_{\text{val}}$ to minimize false negatives subject to $\text{FPR} \le 1.0\%$.
  2. The selected threshold $\tau^*$ is permanently frozen into the model checkpoint JSON / PKL.
  3. The frozen threshold is evaluated on the untouched held-out test set $X_{\text{test}}$.

---

## 6. Biomedical Waveform Patient-Independent Split Guard
- **Vulnerability**: In continuous physiological signals (ECG, PPG, respiration), randomly partitioning sliding windows from the same subject into train and test sets leads to massive patient-identity memorization (data contamination).
- **Audit Action**: Subject-wise (GroupKFold) partitioning is strictly enforced on MIMIC PERform III and BIDMC datasets. Zero subject overlap exists between training, validation, and test splits.

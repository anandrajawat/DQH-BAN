# Strict Data Leakage and Label Integrity Audit

## 1. WUSTL-EHMS-2020 MAC-Address Label Proxy Audit
In WUSTL-EHMS-2020, network flows originate from specific physical hardware (e.g., patient monitor, attacker laptop, gateway). If `SrcMac`, `DstMac`, `SrcAddr`, or `DstAddr` are included in model training, the network intrusion model learns a trivial lookup table on hardware MAC addresses rather than generalizable packet characteristics. This produces an artificial 100% training and validation accuracy that completely fails in deployment. **Audit Decision**: All MAC and IP address columns are stripped from feature vectors before training.

## 2. WUSTL-HDRL-2024 Timestamp and Identifier Leakage Audit
Columns `IMEI`, `Boot_Time_with_date`, `DTime`, `RTime`, `Packet_num`, `SrcMac`, and `DstMac` present severe temporal and identity leakage. Network traffic load and server metrics (`scputimes_*`, `svmem_*`) must only be used if they reflect transmission load without encoding ground truth attack schedules. All direct device identifiers and wall-clock timestamps are strictly excluded.

## 3. Biomedical Waveform Patient-Independent Split Guard
In BIDMC and MIMIC PERform datasets, segmenting continuous physiological signals into overlapping sliding windows risks severe data contamination if windows from the same subject are randomly partitioned into train and test sets. **Audit Decision**: Group-based partitioning by Subject ID is strictly enforced. No patient present in the training set will ever appear in the validation or test set.

## 4. Preprocessing Fit Isolation Guard
All scalers (`StandardScaler`, `RobustScaler`, `MinMaxScaler`), imputer objects, and categorical encoders must be fitted solely on the training partition and transformed on test sets.

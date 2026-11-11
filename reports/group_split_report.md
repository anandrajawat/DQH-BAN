# Group-Based and Stratified Partitioning Protocol

## Cybersecurity Partitioning Protocol
- **CICIoMT2024**: Official `train` and `test` attack-stratified partitions provided by the dataset authors are respected and evaluated.
- **WUSTL-EHMS-2020**: Stratified 70/15/15 Train/Validation/Test split with random seed 42 after excluding identifier columns.
- **WUSTL-HDRL-2024**: Stratified 70/15/15 Train/Validation/Test split after excluding identifiers.
- **CIC-IoT-2023**: Stratified partition across all attack families.
- **ToN-IoT**: Stratified partition across all 9 attack categories.

## Biomedical Partitioning Protocol
- **MIMIC PERform III**: GroupKFold / Group-split by Patient ID. 19 AF subjects and 16 Non-AF subjects split into 70% train subjects (13 AF, 11 non-AF) and 30% test subjects (6 AF, 5 non-AF). Zero subject overlap.
- **BIDMC**: Group-split across 53 subjects. 37 train subjects, 8 validation subjects, 8 test subjects.

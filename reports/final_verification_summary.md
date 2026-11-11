# DQH-BAN: Final Two-Round Independent Verification Summary

**Experiment ID**: `DQH-BAN-EXP-2026-V2`

## Certification of Scientific & Technical Integrity
- **Verification Round 1 (16 Technical Checks)**: **PASSED**
- **Verification Round 2 (8 Clean End-to-End Checks)**: **PASSED**

### Verified Highlights
1. **All Seven Datasets Audited & Leak-Proofed**: Zero MAC/IP leakage in WUSTL-EHMS/HDRL/ToN-IoT; 0% patient overlap in MIMIC PERform III and BIDMC.
2. **Exact Metric Reproducibility**: Fresh-process inference on held-out test sets reproduces saved checkpoint and CSV metrics to 4 decimal places.
3. **Full Artifact Synchronization**: Checkpoints, preprocessors, threshold JSONs, `benchmark_results.csv`, `model_registry.json`, FastAPI endpoints, Streamlit dashboard, and all 27 figures use a single source of truth.

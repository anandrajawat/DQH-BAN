# Resource Estimation & Memory Management Strategy

| Component | Strategy | Memory Footprint | Disk Footprint |
|---|---|---|---|
| CICIoMT2024 | Per-file streaming / chunked parquet | ~1.5 GB | ~800 MB |
| WUSTL-EHMS-2020 | In-memory pandas | ~15 MB | ~4 MB |
| WUSTL-HDRL-2024 | In-memory pandas / chunked | ~150 MB | ~81 MB |
| CIC-IoT-2023 | 10 representative parts partition | ~800 MB | ~700 MB |
| ToN-IoT (NetFlow v2) | Chunked streaming & subsampling | ~1.2 GB | ~2.6 GB |
| BIDMC Waveforms | Per-subject window extraction | ~250 MB | ~200 MB |
| MIMIC PERform | Per-subject patient-independent CSV | ~300 MB | ~100 MB |
| Quantum Simulation | 4-8 Qubits, 2-3 layers, small batch size | ~200 MB | Negligible |

**Total Peak Memory Requirement during Training**: < 4.5 GB RAM (Well within 16 GB system memory).

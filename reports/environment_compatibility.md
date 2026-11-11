# Environment Compatibility Assessment

- **Host Platform**: Apple Silicon arm64 (macOS)
- **Memory Budget**: 16 GB Physical RAM. Memory-efficient chunking, Parquet caching, and single-dataset offline training are strictly mandated to maintain low memory pressure.
- **Deep Learning Acceleration**: PyTorch MPS is verified. CPU fallback is provided for non-MPS-supported quantum simulation operations.
- **Quantum Simulation**: PennyLane with `default.qubit` simulator is verified with PyTorch torch autograd backpropagation.
- **Fabric / Docker**: Docker Engine v29+ is operational with Docker Compose support.

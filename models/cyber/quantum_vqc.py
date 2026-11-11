"""
models/cyber/quantum_vqc.py
Variational Quantum Circuit (VQC) implementation using PennyLane for DQH-BAN.
Supports batched tensor execution, data re-uploading, 4/6/8 qubits, multiple depths,
ring/linear entanglement, and ideal/noisy simulation.
"""

import torch
import torch.nn as nn
import pennylane as qml

class VariationalQuantumCircuit(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, entanglement="ring", simulator="default.qubit", noisy=False):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement
        self.noisy = noisy
        self.noise_sigma = 0.02

        # Trainable quantum weights: (n_layers, n_qubits, 3) for RX, RY, RZ rotations
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        # Trainable data re-uploading input scaling per layer
        self.input_scale = nn.Parameter(torch.ones(n_layers, n_qubits))

        dev = qml.device(simulator, wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def _circuit(inputs, weights, input_scale):
            for l in range(self.n_layers):
                # Data re-uploading angle embedding
                scaled_in = inputs * input_scale[l]
                qml.AngleEmbedding(scaled_in, wires=range(self.n_qubits), rotation="Y")

                for i in range(self.n_qubits):
                    qml.RX(weights[l, i, 0], wires=i)
                    qml.RY(weights[l, i, 1], wires=i)
                    qml.RZ(weights[l, i, 2], wires=i)

                # Entanglement topology
                if self.entanglement == "ring":
                    for i in range(self.n_qubits):
                        qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                else: # linear
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qnode = _circuit

    def forward(self, x):
        """
        x: (Batch, n_qubits) in range [-pi, pi]
        Returns:
            out: (Batch, n_qubits) expectation values in [-1, 1]
        """
        x_in = x.to(self.weights.device, dtype=self.weights.dtype)

        res = self.qnode(x_in, self.weights, self.input_scale)
        if isinstance(res, list):
            out_tensor = torch.stack(res, dim=-1).to(dtype=torch.float32)
        else:
            out_tensor = res.to(dtype=torch.float32)

        if self.noisy:
            noise = torch.randn_like(out_tensor) * self.noise_sigma
            out_tensor = torch.clamp(out_tensor + noise, -1.0, 1.0)

        return out_tensor

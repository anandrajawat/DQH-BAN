"""
models/cyber/fusion.py
Residual Quantum-Classical Fusion block for combining classical latent representations
with quantum expectation measurements via gated residual modulation.
"""

import torch
import torch.nn as nn

class ResidualQuantumClassicalFusion(nn.Module):
    def __init__(self, classical_dim=64, quantum_dim=4, fused_dim=64, dropout=0.15):
        super().__init__()
        self.classical_dim = classical_dim
        self.quantum_dim = quantum_dim
        self.fused_dim = fused_dim

        # Project quantum expectations into classical latent space
        self.quantum_proj = nn.Sequential(
            nn.Linear(quantum_dim, classical_dim),
            nn.LayerNorm(classical_dim),
            nn.GELU(),
            nn.Linear(classical_dim, classical_dim)
        )

        # Gated residual fusion
        self.gate = nn.Sequential(
            nn.Linear(classical_dim * 2, classical_dim),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(classical_dim)
        self.dense = nn.Sequential(
            nn.Linear(classical_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.shortcut = nn.Linear(classical_dim, fused_dim) if classical_dim != fused_dim else nn.Identity()

    def forward(self, classical_feat, quantum_meas):
        """
        classical_feat: (Batch, classical_dim)
        quantum_meas:   (Batch, quantum_dim)
        """
        q_proj = self.quantum_proj(quantum_meas) # (B, classical_dim)
        gate_weights = self.gate(torch.cat([classical_feat, q_proj], dim=-1))
        fused = self.norm(classical_feat + gate_weights * q_proj)
        out = self.shortcut(classical_feat) + self.dense(fused)
        return out

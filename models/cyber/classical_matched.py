"""
models/cyber/classical_matched.py
Parameter-matched classical replacement for the Variational Quantum Circuit (VQC)
for fair, controlled quantum ablation studies.
"""

import torch
import torch.nn as nn

class ClassicalMatchedBlock(nn.Module):
    def __init__(self, in_features=4, out_features=4, target_params=24):
        """
        Builds a compact classical feed-forward network with parameter count
        matched to the corresponding VQC module.
        """
        super().__init__()
        # in_features * h + h + h * out_features + out_features = params
        # (in + out + 1)*h + out = params
        denom = in_features + out_features + 1
        hidden = max(2, (target_params - out_features) // denom)

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_features),
            nn.Tanh() # Matches quantum Pauli-Z expectation range [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


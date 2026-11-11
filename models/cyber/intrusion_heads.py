"""
models/cyber/intrusion_heads.py
Dual intrusion detection output heads:
- Head 1: Binary classification (Benign vs Malicious)
- Head 2: Multiclass classification (Actual dataset-specific attack categories)
"""

import torch
import torch.nn as nn

class DualIntrusionHeads(nn.Module):
    def __init__(self, in_features=64, num_multiclass=10):
        super().__init__()
        self.num_multiclass = num_multiclass

        # Binary intrusion head (Outputs 1 logit for BCEWithLogits or 2 logits)
        self.binary_head = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1) # Single logit for threat probability sigmoid
        )

        # Multiclass attack classifier head
        self.multiclass_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_multiclass)
        )

    def forward(self, x):
        binary_logit = self.binary_head(x).squeeze(-1) # (Batch,)
        multi_logits = self.multiclass_head(x)          # (Batch, num_multiclass)
        return binary_logit, multi_logits


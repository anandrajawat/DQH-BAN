"""
models/cyber/attention.py
Temporal Attention mechanism with scaled dot-product projection,
temperature normalization, residual sequence highway, and attention dropout.
Produces interpretable attention weights summing to 1 across the temporal dimension.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, in_features, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.scale = 1.0 / math.sqrt(hidden_dim)

        self.query_layer = nn.Linear(in_features, hidden_dim)
        self.score_layer = nn.Linear(hidden_dim, 1, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_features)

    def forward(self, x):
        """
        x: (Batch, Seq_Len, in_features)
        Returns:
            context: (Batch, in_features)
            weights: (Batch, Seq_Len)
        """
        if x.dim() == 2:
            return x, torch.ones(x.size(0), 1, device=x.device)

        # 1. Project and scale
        proj = torch.tanh(self.query_layer(x)) # (B, T, hidden_dim)
        scores = self.score_layer(proj).squeeze(-1) * self.scale # (B, T)

        # 2. Stable softmax over time steps
        weights = F.softmax(scores, dim=-1) # (B, T)
        weights_dropped = self.attn_dropout(weights)

        # 3. Weighted aggregation across sequence
        context = torch.bmm(weights_dropped.unsqueeze(1), x).squeeze(1) # (B, in_features)

        # 4. Residual skip from mean representation + LayerNorm
        global_mean = x.mean(dim=1)
        out = self.norm(context + global_mean)

        return out, weights

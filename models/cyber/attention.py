"""
models/cyber/attention.py
Temporal Attention mechanism with interpretable attention weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        self.query_layer = nn.Linear(in_features, hidden_dim)
        self.score_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        """
        x: (Batch, Seq_Len, Features)
        Returns:
            context: (Batch, Features)
            weights: (Batch, Seq_Len)
        """
        if x.dim() == 2:
            return x, torch.ones(x.size(0), 1, device=x.device)

        proj = torch.tanh(self.query_layer(x)) # (B, T, hidden_dim)
        scores = self.score_layer(proj).squeeze(-1) # (B, T)
        weights = F.softmax(scores, dim=-1) # (B, T)
        
        # Weighted sum over temporal dimension
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1) # (B, Features)
        return context, weights


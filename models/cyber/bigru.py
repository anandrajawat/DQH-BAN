"""
models/cyber/bigru.py
Lightweight Bidirectional GRU layer for sequential state tracking.
"""

import torch
import torch.nn as nn

class BiGRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.out_dim = hidden_size * 2

    def forward(self, x):
        """
        x: (Batch, Channels, Seq_Len)
        Returns:
            out: (Batch, Seq_Len, hidden_size * 2)
        """
        # Permute to (Batch, Seq_Len, Channels) for GRU
        x_perm = x.permute(0, 2, 1)
        out, _ = self.gru(x_perm)
        return out


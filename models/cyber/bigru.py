"""
models/cyber/bigru.py
Recurrent sequential feature encoder supporting bidirectional and unidirectional modes.
Enhanced with LayerNorm, residual highway projection, and gradient stabilization.
"""

import torch
import torch.nn as nn

class BiGRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, bidirectional=True, dropout=0.15):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.out_dim = hidden_size * 2 if bidirectional else hidden_size

        # Residual highway projection
        self.residual_proj = nn.Linear(input_size, self.out_dim) if input_size != self.out_dim else nn.Identity()
        self.norm = nn.LayerNorm(self.out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (Batch, Channels, Seq_Len) or (Batch, Seq_Len, Channels)
        Returns:
            out: (Batch, Seq_Len, out_dim)
        """
        if x.dim() != 3:
            raise ValueError(f"BiGRUEncoder expects 3D tensor, got shape {x.shape}")

        # Ensure (Batch, Seq_Len, input_size)
        if x.size(1) == self.input_size and x.size(2) != self.input_size:
            x_seq = x.permute(0, 2, 1) # (B, L, C)
        elif x.size(2) == self.input_size:
            x_seq = x
        else:
            # Default convention: permute (B, C, L) -> (B, L, C)
            x_seq = x.permute(0, 2, 1)

        gru_out, _ = self.gru(x_seq)
        res = self.residual_proj(x_seq)
        out = self.norm(gru_out + res)
        out = self.dropout(out)
        return out

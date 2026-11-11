"""
models/cyber/cnn_encoder.py
Multi-Scale 1D-CNN Encoder for capturing localized network flow patterns.
"""

import torch
import torch.nn as nn

class MultiScale1DCNN(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_sizes=[3, 5, 7], dropout=0.2):
        super().__init__()
        self.branches = nn.ModuleList()
        branch_channels = out_channels // len(kernel_sizes)
        
        for k in kernel_sizes:
            padding = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, branch_channels, kernel_size=k, padding=padding),
                    nn.BatchNorm1d(branch_channels),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )
            
        total_branch_out = branch_channels * len(kernel_sizes)
        self.proj = nn.Conv1d(total_branch_out, out_channels, kernel_size=1) if total_branch_out != out_channels else nn.Identity()
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        """
        x: (Batch, In_Channels, Seq_Len) or (Batch, In_Channels)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1) # (B, C, 1)

        res = self.residual(x)
        outs = [branch(x) for branch in self.branches]
        concat = torch.cat(outs, dim=1)
        out = self.proj(concat)
        out = self.act(self.norm(out + res))
        return out


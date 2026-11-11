"""
models/cyber/cnn_encoder.py
Multi-Scale 1D-CNN Encoder for capturing localized network flow patterns.
Enhanced with multi-scale receptive fields, exact channel budgeting,
channel-wise Squeeze-and-Excitation recalibration, and residual skip connection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelSEBlock1d(nn.Module):
    """Squeeze-and-Excitation block for 1D convolutional feature maps."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.fc2 = nn.Linear(reduced, channels, bias=False)
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        b, c, _ = x.size()
        y = x.mean(dim=-1) # (B, C)
        y = self.act(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).unsqueeze(-1) # (B, C, 1)
        return x * y

class MultiScale1DCNN(nn.Module):
    def __init__(self, in_channels, out_channels=32, kernel_sizes=[3, 5, 7], dropout=0.15):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        branch_channels = out_channels // len(kernel_sizes)
        self.branches = nn.ModuleList()
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
        self.se = ChannelSEBlock1d(out_channels)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        """
        x: (Batch, In_Channels, Seq_Len) or (Batch, In_Channels)
        Returns: (Batch, out_channels, Seq_Len)
        """
        if x.dim() == 2:
            if x.size(1) == self.in_channels:
                x = x.unsqueeze(-1) # (B, C, 1)
            else:
                x = x.unsqueeze(1)  # (B, 1, L)

        res = self.residual(x)
        branch_outs = [branch(x) for branch in self.branches]
        concat = torch.cat(branch_outs, dim=1)
        out = self.proj(concat)
        if hasattr(self, "se") and self.se is not None:
            out = self.se(out)
        out = self.act(self.norm(out + res))
        return out

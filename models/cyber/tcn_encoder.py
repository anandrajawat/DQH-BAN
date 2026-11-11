"""
models/cyber/tcn_encoder.py
Temporal Convolutional Network (TCN) with dilated causal convolutions,
spatial channel dropout, and residual blocks for capturing sequential network behavior.
Strictly causal: outputs at step t never depend on future steps t' > t.
"""

import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    """Slices off right-hand padding to enforce strict causal convolution."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.15):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout1d(dropout) if hasattr(nn, 'Dropout1d') else nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout1d(dropout) if hasattr(nn, 'Dropout1d') else nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.norm1, self.act1, self.drop1,
            self.conv2, self.chomp2, self.norm2, self.act2, self.drop2
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.final_act = nn.GELU()

    def forward(self, x):
        res = self.downsample(x)
        out = self.net(x)
        return self.final_act(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels=[32, 64], kernel_size=3, dropout=0.15):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_ch, out_ch, kernel_size,
                    stride=1, dilation=dilation_size, padding=padding, dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (Batch, In_Channels, Seq_Len)
        Returns: (Batch, Out_Channels, Seq_Len)
        """
        return self.network(x)

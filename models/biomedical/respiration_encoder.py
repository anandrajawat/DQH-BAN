"""
models/biomedical/respiration_encoder.py
Respiration Waveform Encoder: 1D-CNN + GRU temporal representation learner.
"""

import torch
import torch.nn as nn

class RespirationEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=16):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 8, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(8),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        self.gru = nn.GRU(input_size=32, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        cnn_feat = self.cnn(x)
        gru_in = cnn_feat.permute(0, 2, 1)
        gru_out, _ = self.gru(gru_in)
        pooled = gru_out.mean(dim=1)
        return self.out_proj(pooled)


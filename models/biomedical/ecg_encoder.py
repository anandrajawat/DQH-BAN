"""
models/biomedical/ecg_encoder.py
ECG Waveform Encoder: 1D-CNN + GRU temporal representation learner.
"""

import torch
import torch.nn as nn

class ECGEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.gru = nn.GRU(input_size=64, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        """
        x: (Batch, 1, 625) or (Batch, 625)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        cnn_feat = self.cnn(x) # (B, 64, L)
        gru_in = cnn_feat.permute(0, 2, 1) # (B, L, 64)
        gru_out, _ = self.gru(gru_in) # (B, L, 2 * hidden_dim)
        pooled = gru_out.mean(dim=1) # (B, 2 * hidden_dim)
        return self.out_proj(pooled) # (B, hidden_dim)


"""
models/biomedical/signal_fusion.py
Gated Physiological Fusion with Missing-Modality Masking, SQI Weighting,
and Direct Tabular HRV Residual Connection.
"""

import torch
import torch.nn as nn

class GatedPhysiologicalFusion(nn.Module):
    def __init__(self, ecg_dim=32, ppg_dim=32, resp_dim=16, tab_dim=11, out_dim=64, dropout=0.2):
        super().__init__()
        self.total_raw_dim = ecg_dim + ppg_dim + resp_dim + tab_dim

        # Projections to common latent space
        self.ecg_proj = nn.Sequential(nn.Linear(ecg_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        self.ppg_proj = nn.Sequential(nn.Linear(ppg_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        self.resp_proj = nn.Sequential(nn.Linear(resp_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

        # Modality importance gating
        self.gate_net = nn.Sequential(
            nn.Linear(out_dim * 4, 4),
            nn.Softmax(dim=-1)
        )

        self.fusion_norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.dense = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

    def forward(self, ecg_f, ppg_f, resp_f, tab_f, modality_mask=None, sqi=None):
        """
        ecg_f: (B, 32), ppg_f: (B, 32), resp_f: (B, 16), tab_f: (B, 11)
        modality_mask: (B, 4) boolean / float mask [ecg, ppg, resp, tab]
        sqi: (B,) optional signal quality weights
        """
        e_p = self.ecg_proj(ecg_f)
        p_p = self.ppg_proj(ppg_f)
        r_p = self.resp_proj(resp_f)
        t_p = self.tab_proj(tab_f)

        if modality_mask is not None:
            e_p = e_p * modality_mask[:, 0:1]
            p_p = p_p * modality_mask[:, 1:2]
            r_p = r_p * modality_mask[:, 2:3]
            t_p = t_p * modality_mask[:, 3:4]

        concat_all = torch.cat([e_p, p_p, r_p, t_p], dim=-1)
        gates = self.gate_net(concat_all) # (B, 4)

        if sqi is not None:
            sqi_weight = torch.clamp(sqi.unsqueeze(-1), 0.1, 1.0)
            e_p = e_p * sqi_weight
            p_p = p_p * sqi_weight

        fused = (
            gates[:, 0:1] * e_p +
            gates[:, 1:2] * p_p +
            gates[:, 2:3] * r_p +
            gates[:, 3:4] * t_p +
            t_p # Residual HRV/PRV physiological feature anchor
        )

        out = self.dense(self.drop(self.fusion_norm(fused)))
        return out

"""
models/biomedical/af_classifier.py
Stream B: Complete Physiological Signal Intelligence Model for AF vs Non-AF Classification.
"""

import torch
import torch.nn as nn
from models.biomedical.ecg_encoder import ECGEncoder
from models.biomedical.ppg_encoder import PPGEncoder
from models.biomedical.respiration_encoder import RespirationEncoder
from models.biomedical.signal_fusion import GatedPhysiologicalFusion

class AFClassifierModel(nn.Module):
    def __init__(self, tab_feature_dim=11, latent_dim=64):
        super().__init__()
        self.ecg_encoder = ECGEncoder(in_channels=1, hidden_dim=32)
        self.ppg_encoder = PPGEncoder(in_channels=1, hidden_dim=32)
        self.resp_encoder = RespirationEncoder(in_channels=1, hidden_dim=16)

        self.fusion = GatedPhysiologicalFusion(
            ecg_dim=32, ppg_dim=32, resp_dim=16, tab_dim=tab_feature_dim, out_dim=latent_dim
        )

        # AF/Non-AF Classifier Head (Binary: 0=Non-AF, 1=AF)
        self.classifier_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1) # Single logit for BCEWithLogits
        )

    def forward(self, waveforms, tabular_features, modality_mask=None, sqi=None):
        """
        waveforms: (Batch, 3, 625) where channel 0=ECG, 1=PPG, 2=Resp
        tabular_features: (Batch, 11)
        """
        ecg_w = waveforms[:, 0:1, :]
        ppg_w = waveforms[:, 1:2, :]
        resp_w = waveforms[:, 2:3, :]

        e_feat = self.ecg_encoder(ecg_w)
        p_feat = self.ppg_encoder(ppg_w)
        r_feat = self.resp_encoder(resp_w)

        fused = self.fusion(e_feat, p_feat, r_feat, tabular_features, modality_mask=modality_mask, sqi=sqi)
        logit = self.classifier_head(fused).squeeze(-1)
        return logit, fused


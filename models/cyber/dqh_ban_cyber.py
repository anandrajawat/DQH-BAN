"""
models/cyber/dqh_ban_cyber.py
Complete DQH-BAN Stream A Quantum-Assisted Cybersecurity Intelligence Architecture.
Retains Multi-Scale 1D-CNN, TCN, BiGRU, Temporal Attention, Quantum Angle Projection,
Variational Quantum Circuit (VQC), Residual Quantum-Classical Fusion, and Dual Intrusion Heads,
enhanced with Soft-Threshold Quantile Binning and Periodic Fourier Feature Gating
so individual flow feature identities and sharp decision boundaries are preserved.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cyber.cnn_encoder import MultiScale1DCNN
from models.cyber.tcn_encoder import TemporalConvNet
from models.cyber.bigru import BiGRUEncoder
from models.cyber.attention import TemporalAttention
from models.cyber.quantum_vqc import VariationalQuantumCircuit
from models.cyber.classical_matched import ClassicalMatchedBlock
from models.cyber.fusion import ResidualQuantumClassicalFusion
from models.cyber.intrusion_heads import DualIntrusionHeads

class PeriodicCoordinateFeatureEncoder(nn.Module):
    """
    Encodes exact per-feature coordinates using learnable soft-threshold quantile bins
    plus periodic Fourier harmonics and squeeze-and-excitation feature gating.
    """
    def __init__(self, in_features, latent_dim=64, n_Harmonics=6, n_bins=12, dropout=0.15):
        super().__init__()
        self.in_features = in_features
        self.freqs = nn.Parameter(torch.randn(in_features, n_Harmonics) * 1.5)
        self.phases = nn.Parameter(torch.zeros(in_features, n_Harmonics))
        init_bins = torch.linspace(-3.0, 3.0, n_bins).unsqueeze(0).repeat(in_features, 1)
        self.bins = nn.Parameter(init_bins)
        self.bin_temp = nn.Parameter(torch.ones(in_features, n_bins) * 4.0)

        self.feature_gate = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Sigmoid()
        )
        raw_dim = in_features * (2 * n_Harmonics + n_bins + 1)
        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )

    def forward(self, x_flat):
        gated = x_flat * self.feature_gate(x_flat)
        angles = gated.unsqueeze(-1) * self.freqs.unsqueeze(0) + self.phases.unsqueeze(0)
        soft_bins = torch.sigmoid((x_flat.unsqueeze(-1) - self.bins.unsqueeze(0)) * self.bin_temp.unsqueeze(0))
        feat = torch.cat([
            gated,
            torch.sin(angles).flatten(1),
            torch.cos(angles).flatten(1),
            soft_bins.flatten(1)
        ], dim=-1)
        return self.mlp(feat)

class DQHBANCyberModel(nn.Module):
    def __init__(
        self,
        in_features=45,
        num_classes=10,
        latent_dim=64,
        n_qubits=4,
        vqc_layers=2,
        entanglement="ring",
        mode="full", # 'full', 'classical_matched', 'cnn_tcn_bigru', 'cnn_tcn_attention', 'cnn_tcn_vqc', 'cnn_tcn', '1d_cnn', 'tcn'
        noisy_quantum=False
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.n_qubits = n_qubits
        self.vqc_layers = vqc_layers
        self.entanglement = entanglement
        self.mode = mode

        self.pos_emb = nn.Parameter(torch.randn(1, 32, in_features) * 0.02)

        self.coord_encoder = PeriodicCoordinateFeatureEncoder(
            in_features=in_features, latent_dim=latent_dim, n_Harmonics=6, n_bins=12, dropout=0.15
        )

        # 1. Multi-Scale 1D-CNN
        self.cnn = MultiScale1DCNN(in_channels=1, out_channels=32, kernel_sizes=[3, 5, 7], dropout=0.15)
        self.cnn_proj = nn.Sequential(
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )

        # 2. Temporal Convolutional Network
        self.tcn = TemporalConvNet(num_inputs=32, num_channels=[32, 64], kernel_size=3, dropout=0.15)

        # 3. BiGRU
        self.bigru = BiGRUEncoder(input_size=64, hidden_size=32, num_layers=1, dropout=0.1)

        # 4. Temporal Attention
        self.attention = TemporalAttention(in_features=64, hidden_dim=32)

        # 5. Latent Feature Projection
        self.latent_proj = nn.Sequential(
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )

        # 6. Quantum Angle Encoding Projection
        self.quantum_in_proj = nn.Sequential(
            nn.Linear(latent_dim, n_qubits),
            nn.Hardtanh(min_val=-math.pi, max_val=math.pi)
        )

        # 7. Quantum VQC or Classical Parameter-Matched Replacement
        if mode == "full" or mode == "cnn_tcn_vqc":
            self.quantum_module = VariationalQuantumCircuit(
                n_qubits=n_qubits, n_layers=vqc_layers, entanglement=entanglement, noisy=noisy_quantum
            )
        elif mode == "classical_matched":
            target_params = vqc_layers * n_qubits * 4
            self.quantum_module = ClassicalMatchedBlock(
                in_features=n_qubits, out_features=n_qubits, target_params=target_params
            )
        else:
            self.quantum_module = None

        # 8. Residual Quantum-Classical Fusion
        if self.quantum_module is not None:
            self.fusion = ResidualQuantumClassicalFusion(
                classical_dim=latent_dim, quantum_dim=n_qubits, fused_dim=latent_dim, dropout=0.15
            )
        else:
            self.fusion = nn.Identity()

        # 9. Dual Intrusion Output Heads
        self.heads = DualIntrusionHeads(in_features=latent_dim, num_multiclass=num_classes)

    def _align_flat_features(self, x):
        if x.dim() == 2:
            x_flat = x
        elif x.dim() == 3:
            x_flat = x[:, -1, :] if x.size(-1) == self.in_features else x.mean(dim=-1)
        else:
            x_flat = x.view(x.size(0), -1)

        if x_flat.size(-1) < self.in_features:
            x_flat = F.pad(x_flat, (0, self.in_features - x_flat.size(-1)))
        elif x_flat.size(-1) > self.in_features:
            x_flat = x_flat[:, :self.in_features]
        return x_flat

    def forward(self, x, return_latent=False):
        x_flat = self._align_flat_features(x)
        coord_latent = self.coord_encoder(x_flat)

        if x.dim() == 2:
            x_in = x_flat.unsqueeze(1)
        elif x.dim() == 3:
            x_in = x if x.size(1) == 1 else x_flat.unsqueeze(1)
        else:
            x_in = x_flat.unsqueeze(1)

        cnn_out = self.cnn(x_in)
        if cnn_out.size(-1) == self.in_features:
            cnn_out = cnn_out + self.pos_emb

        if self.mode == "1d_cnn":
            pooled = self.cnn_proj(cnn_out.mean(dim=-1))
            latent = self.latent_proj(pooled) + 0.65 * coord_latent
            bin_l, mul_l = self.heads(latent)
            if return_latent:
                return bin_l, mul_l, None, latent
            return bin_l, mul_l, None

        tcn_out = self.tcn(cnn_out)

        if self.mode == "tcn" or self.mode == "cnn_tcn":
            pooled = tcn_out.mean(dim=-1)
            latent = self.latent_proj(pooled) + 0.8 * coord_latent
            bin_l, mul_l = self.heads(latent)
            if return_latent:
                return bin_l, mul_l, None, latent
            return bin_l, mul_l, None

        if self.mode == "cnn_tcn_bigru":
            gru_out = self.bigru(tcn_out)
            context = gru_out.mean(dim=1)
            attn_weights = None
            classical_latent = self.latent_proj(context) + 0.9 * coord_latent
        elif self.mode != "cnn_tcn_vqc":
            gru_out = self.bigru(tcn_out)
            context, attn_weights = self.attention(gru_out)
            classical_latent = self.latent_proj(context) + coord_latent
        else:
            context = tcn_out.mean(dim=-1)
            attn_weights = None
            classical_latent = self.latent_proj(context) + coord_latent

        if self.quantum_module is not None:
            q_angles = self.quantum_in_proj(classical_latent)
            q_meas = self.quantum_module(q_angles)
            fused_feat = self.fusion(classical_latent, q_meas)
        else:
            fused_feat = classical_latent

        binary_logit, multi_logits = self.heads(fused_feat)
        if return_latent:
            return binary_logit, multi_logits, attn_weights, fused_feat
        return binary_logit, multi_logits, attn_weights

    def extract_latent(self, x):
        """Extracts the learned latent embedding vector (shape: batch_size x latent_dim)."""
        _, _, _, latent = self.forward(x, return_latent=True)
        return latent

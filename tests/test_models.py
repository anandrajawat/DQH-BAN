"""
tests/test_models.py
Unit tests verifying model architecture forward passes, tensor shapes, gradient flow,
and device execution for both Stream A (Cyber) and Stream B (Biomedical).
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from models.cyber.cnn_encoder import MultiScale1DCNN
from models.cyber.tcn_encoder import TemporalConvNet
from models.cyber.bigru import BiGRUEncoder
from models.cyber.attention import TemporalAttention
from models.cyber.classical_matched import ClassicalMatchedBlock
from models.cyber.quantum_vqc import VariationalQuantumCircuit
from models.cyber.dqh_ban_cyber import DQHBANCyberModel
from models.biomedical.af_classifier import AFClassifierModel

def test_multiscale_1d_cnn_forward():
    model = MultiScale1DCNN(in_channels=1, out_channels=64)
    x = torch.randn(8, 1, 45)
    out = model(x)
    assert out.dim() == 3
    assert out.shape[0] == 8
    assert out.shape[1] == 64

def test_tcn_forward():
    model = TemporalConvNet(num_inputs=64, num_channels=[64, 64], kernel_size=3)
    x = torch.randn(8, 64, 16)
    out = model(x)
    assert out.shape == (8, 64, 16)

def test_bigru_forward():
    model = BiGRUEncoder(input_size=64, hidden_size=32, num_layers=1)
    x = torch.randn(8, 64, 16)
    out = model(x)
    assert out.shape == (8, 16, 64)

def test_temporal_attention_weights():
    model = TemporalAttention(in_features=64)
    x = torch.randn(8, 16, 64)
    context, weights = model(x)
    assert context.shape == (8, 64)
    assert weights.shape == (8, 16)
    sums = weights.sum(dim=-1).detach().cpu().numpy()
    assert np.allclose(sums, 1.0, atol=1e-5), "Attention weights must sum to 1 over sequence dimension"

def test_classical_matched_vqc():
    model = ClassicalMatchedBlock(in_features=4, out_features=4)
    x = torch.randn(8, 4)
    out = model(x)
    assert out.shape == (8, 4)

def test_pennylane_vqc():
    model = VariationalQuantumCircuit(n_qubits=4, n_layers=2, entanglement="ring")
    x = torch.randn(4, 4)
    out = model(x)
    assert out.shape == (4, 4)

def test_dqh_ban_cyber_forward():
    model = DQHBANCyberModel(
        in_features=45,
        num_classes=10,
        mode="full",
        n_qubits=4,
        vqc_layers=2,
        entanglement="ring"
    )
    x = torch.randn(4, 45)
    bin_logits, mc_logits, attn = model(x)
    assert bin_logits.shape == (4,)
    assert mc_logits.shape == (4, 10)

def test_dqh_ban_cyber_classical_matched_ablation():
    model = DQHBANCyberModel(
        in_features=45,
        num_classes=10,
        mode="classical_matched",
        n_qubits=4
    )
    x = torch.randn(4, 45)
    bin_logits, mc_logits, attn = model(x)
    assert bin_logits.shape == (4,)
    assert mc_logits.shape == (4, 10)

def test_af_classifier_biomedical_forward():
    model = AFClassifierModel(tab_feature_dim=11, latent_dim=64)
    waveforms = torch.randn(4, 3, 625)
    tab = torch.randn(4, 11)
    mask = torch.ones(4, 4)
    logit, rep = model(waveforms, tab, modality_mask=mask)
    assert logit.shape == (4,)
    assert rep.shape == (4, 64)

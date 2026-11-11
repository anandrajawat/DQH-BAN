"""
tests/test_blockchain.py
Automated tests for AES-256-GCM encryption/decryption, SHA-256 cryptographic commitments,
off-chain encrypted vault storage, and Hyperledger Fabric smart contract interactions.
"""

import os
import json
import pytest
import numpy as np

from blockchain.encryption import BiomedicalCryptoEngine
from blockchain.off_chain_vault import EncryptedOffChainVault
from blockchain.fabric_client import FabricClient

def test_aes_256_gcm_encryption_decryption():
    crypto = BiomedicalCryptoEngine()
    assert len(crypto.key) == 32, "AES key must be 32 bytes (256 bits)"
    
    plaintext = {"patient_id": "TEST_001", "ecg_lead_ii": [0.12, 0.45, -0.21], "hr": 74}
    enc_dict = crypto.encrypt_payload(json.dumps(plaintext), associated_data="TestPatient:001")
    
    assert "ciphertext_hex" in enc_dict
    assert "nonce_hex" in enc_dict
    assert "sha256_hash" in enc_dict
    
    decrypted_str, is_valid = crypto.decrypt_payload(enc_dict)
    assert is_valid is True
    decrypted = json.loads(decrypted_str)
    assert decrypted["patient_id"] == "TEST_001"
    assert decrypted["hr"] == 74

def test_sha256_hash_verification():
    data = "Sensitive medical telemetry sequence: 120/80 mmHg, SpO2 99%"
    h = BiomedicalCryptoEngine.compute_sha256(data)
    assert len(h) == 64, "SHA-256 hex digest must be 64 characters"
    assert BiomedicalCryptoEngine.compute_sha256(data) == h
    assert BiomedicalCryptoEngine.compute_sha256(data + "_tampered") != h

def test_off_chain_vault_storage_and_integrity(tmp_path):
    vault_dir = str(tmp_path / "vault")
    vault = EncryptedOffChainVault(vault_dir=vault_dir)
    
    telemetry = {"sensor": "ECG_PATCH_01", "samples": [1.0, 2.0, 3.0]}
    res = vault.store_biomedical_record("DEV_01", "SUBJ_01", telemetry)
    
    assert "record_id" in res
    assert "sha256_hash" in res
    
    retrieved = vault.retrieve_and_decrypt_record(res["record_id"])
    assert retrieved["integrity_verified"] is True
    assert retrieved["telemetry"]["sensor"] == "ECG_PATCH_01"

def test_fabric_client_lifecycle():
    ledger_path = "/tmp/test_fabric_ledger.json"
    if os.path.exists(ledger_path):
        os.remove(ledger_path)
    client = FabricClient(ledger_state_path=ledger_path)
    
    # 1. Device Registration
    reg_res = client.register_device("DEV_TEST_99", "ECG", "Org1MSP")
    assert reg_res["status"] == "SUCCESS"
    assert "block_hash" in reg_res
    
    # 2. Verify Device
    v_res = client.verify_device("DEV_TEST_99")
    assert v_res["authenticated"] is True
    
    # 3. Record Security Event
    event_res = client.record_security_event(
        event_id="EVT_001",
        device_id="DEV_TEST_99",
        event_type="SYN_FLOOD",
        threat_score=0.992,
        severity="CRITICAL",
        action_taken="QUARANTINE"
    )
    assert event_res["status"] == "SUCCESS"
    
    # 4. Telemetry Commitment & Verification
    h = BiomedicalCryptoEngine.compute_sha256("dummy_payload")
    comm_res = client.record_data_hash("REC_001", "DEV_TEST_99", h)
    assert comm_res["status"] == "SUCCESS"
    
    verify_res = client.verify_data_integrity("REC_001", h)
    assert verify_res["verified"] is True

"""
blockchain/off_chain_vault.py
Encrypted off-chain storage manager for Body Area Network biomedical records.
Stores AES-256-GCM ciphertexts off-chain and commits SHA-256 hashes on-chain.
"""

import os
import json
import time
from blockchain.encryption import BiomedicalCryptoEngine

class EncryptedOffChainVault:
    def __init__(self, vault_dir="artifacts/blockchain/off_chain_vault", crypto_engine=None):
        self.vault_dir = vault_dir
        os.makedirs(vault_dir, exist_ok=True)
        self.crypto_engine = crypto_engine or BiomedicalCryptoEngine()

    def store_biomedical_record(self, device_id, subject_id, telemetry_payload):
        """
        Encrypts patient record with AES-256-GCM, writes ciphertext to off-chain vault,
        and returns the record ID, timestamp, and SHA-256 cryptographic hash for blockchain anchoring.
        """
        payload_str = json.dumps(telemetry_payload, sort_keys=True)
        t_record = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        record_id = f"REC_{device_id}_{int(time.time()*1000)}"

        encrypted_data = self.crypto_engine.encrypt_payload(
            payload_str,
            associated_data=f"{device_id}:{subject_id}:{record_id}"
        )

        record_file = os.path.join(self.vault_dir, f"{record_id}.enc")
        vault_entry = {
            "record_id": record_id,
            "device_id": device_id,
            "subject_id": subject_id,
            "timestamp": t_record,
            "encrypted_payload": encrypted_data,
            "sha256_hash": encrypted_data["sha256_hash"]
        }

        with open(record_file, "w") as f:
            json.dump(vault_entry, f, indent=2)

        return {
            "record_id": record_id,
            "device_id": device_id,
            "subject_id": subject_id,
            "timestamp": t_record,
            "sha256_hash": encrypted_data["sha256_hash"],
            "vault_file_path": record_file,
            "file_size_bytes": os.path.getsize(record_file)
        }

    def retrieve_and_decrypt_record(self, record_id):
        """
        Retrieves record from off-chain vault, decrypts it, and verifies SHA-256 integrity.
        """
        record_file = os.path.join(self.vault_dir, f"{record_id}.enc")
        if not os.path.exists(record_file):
            raise FileNotFoundError(f"Record {record_id} not found in off-chain vault.")

        with open(record_file, "r") as f:
            vault_entry = json.load(f)

        decrypted_bytes, integrity_valid = self.crypto_engine.decrypt_payload(vault_entry["encrypted_payload"])
        telemetry = json.loads(decrypted_bytes.decode('utf-8'))

        return {
            "record_id": record_id,
            "device_id": vault_entry["device_id"],
            "subject_id": vault_entry["subject_id"],
            "timestamp": vault_entry["timestamp"],
            "sha256_hash": vault_entry["sha256_hash"],
            "integrity_verified": integrity_valid,
            "telemetry": telemetry
        }


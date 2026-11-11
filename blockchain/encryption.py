"""
blockchain/encryption.py
Cryptographic security module for Body Area Networks:
- AES-256-GCM authenticated encryption and decryption
- SHA-256 integrity commitments for off-chain biomedical telemetry
- ECDH key agreement over SECP256R1
"""

import os
import hashlib
import json
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

class BiomedicalCryptoEngine:
    def __init__(self, key=None):
        """
        Initializes AES-256-GCM engine with a 32-byte (256-bit) key.
        """
        self.key = key or AESGCM.generate_key(bit_length=256)
        self.aesgcm = AESGCM(self.key)

    @staticmethod
    def compute_sha256(data_bytes):
        """
        Generates SHA-256 cryptographic digest string (64-character hex).
        """
        if isinstance(data_bytes, str):
            data_bytes = data_bytes.encode('utf-8')
        return hashlib.sha256(data_bytes).hexdigest()

    def encrypt_payload(self, plaintext_bytes, associated_data=None):
        """
        Encrypts plaintext using AES-256-GCM.
        Returns:
            nonce (12 bytes), ciphertext_with_tag (bytes), sha256_hash (str)
        """
        if isinstance(plaintext_bytes, str):
            plaintext_bytes = plaintext_bytes.encode('utf-8')
            
        nonce = os.urandom(12) # 96-bit random IV
        aad = associated_data.encode('utf-8') if isinstance(associated_data, str) else associated_data
        
        ciphertext = self.aesgcm.encrypt(nonce, plaintext_bytes, aad)
        data_hash = self.compute_sha256(plaintext_bytes)
        
        return {
            "nonce_hex": nonce.hex(),
            "ciphertext_hex": ciphertext.hex(),
            "sha256_hash": data_hash,
            "aad": associated_data
        }

    def decrypt_payload(self, encrypted_record):
        """
        Decrypts AES-256-GCM payload and verifies integrity commitment.
        """
        nonce = bytes.fromhex(encrypted_record["nonce_hex"])
        ciphertext = bytes.fromhex(encrypted_record["ciphertext_hex"])
        aad = encrypted_record.get("aad")
        aad_bytes = aad.encode('utf-8') if isinstance(aad, str) else aad
        
        decrypted_bytes = self.aesgcm.decrypt(nonce, ciphertext, aad_bytes)
        recalculated_hash = self.compute_sha256(decrypted_bytes)
        
        integrity_valid = (recalculated_hash == encrypted_record["sha256_hash"])
        if not integrity_valid:
            raise ValueError("Integrity verification failed! SHA-256 hash mismatch.")
            
        return decrypted_bytes, integrity_valid

    @staticmethod
    def generate_ecdh_keypair():
        """
        Generates SECP256R1 ECDH private/public keypair.
        """
        private_key = ec.generate_private_key(ec.SECP256R1())
        return private_key, private_key.public_key()

    @staticmethod
    def derive_shared_secret(private_key, peer_public_key, salt=b"DQH-BAN-KDF"):
        """
        Derives a 256-bit symmetric key using ECDH and HKDF-SHA256.
        """
        shared_key = private_key.exchange(ec.ECDH(), peer_public_key)
        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=b"ban-telemetry-encryption"
        ).derive(shared_key)
        return derived_key


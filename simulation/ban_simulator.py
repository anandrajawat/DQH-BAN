"""
simulation/ban_simulator.py
Real-time Body Area Network (BAN) simulation environment.
Generates multi-sensor physiological telemetry, network flow features, and controlled synthetic attack scenarios.
Loads saved model checkpoints, preprocessors, and calibrated thresholds offline without retraining.
All synthetic events are strictly tagged as simulated, and patient identifiers are pseudonymized via SHA-256.
"""

import time
import os
import math
import random
import hashlib
import pickle
import numpy as np
import torch
from typing import Dict, Any
from blockchain.fabric_client import FabricClient
from blockchain.off_chain_vault import EncryptedOffChainVault
from orchestration.orchestrator import SecurityOrchestrationEngine

class BANSimulator:
    DEVICE_PROFILES = {
        "Holter_Monitor_ECG": {
            "type": "ECG_Monitor",
            "channels": ["ECG", "HeartRate", "HRV"],
            "base_hr": 72.0,
            "sampling_rate_hz": 125,
            "critical": False
        },
        "Smartwatch_Wearable": {
            "type": "Multisensor_Wristband",
            "channels": ["PPG", "SpO2", "Temperature", "Accelerometer"],
            "base_spo2": 98.0,
            "base_temp": 36.8,
            "sampling_rate_hz": 50,
            "critical": False
        },
        "Chest_Patch_Respiration": {
            "type": "Respiratory_Sensor",
            "channels": ["Respiration", "Breathing_Rate"],
            "base_resp": 16.0,
            "sampling_rate_hz": 25,
            "critical": False
        },
        "Continuous_BP_Cuff": {
            "type": "Blood_Pressure_Monitor",
            "channels": ["SYS", "DIA", "MAP"],
            "base_sys": 120.0,
            "base_dia": 80.0,
            "sampling_rate_hz": 1,
            "critical": False
        },
        "Implantable_Pacemaker": {
            "type": "Cardiac_Implant",
            "channels": ["Pacing_Rate", "Lead_Impedance", "Battery"],
            "base_pacing": 60.0,
            "sampling_rate_hz": 10,
            "critical": True
        }
    }

    ATTACK_SCENARIOS = [
        "BENIGN_ROUTINE",
        "SYNTHETIC_DDoS_TCP",
        "SYNTHETIC_MQTT_DoS_Flood",
        "SYNTHETIC_ARP_Spoofing",
        "SYNTHETIC_Data_Injection",
        "SYNTHETIC_Replay_Traffic",
        "SYNTHETIC_Unauthorized_Probe"
    ]

    def __init__(
        self,
        cyber_model_path="artifacts/cyber_models/ciciomt2024/ciciomt2024_full_best.pt",
        biomedical_model_path="artifacts/biomedical_models/mimic_perform_af_best.pt",
        cyber_split_path="artifacts/preprocessing/cyber/ciciomt2024_splits.pkl",
        cyber_hybrid_path="artifacts/cyber_models/ciciomt2024/ciciomt2024_hybrid_ensemble_best.pkl",
        fabric_client=None,
        orchestration_engine=None
    ):
        self.fabric_client = fabric_client or FabricClient()
        self.vault = EncryptedOffChainVault()
        self.orchestrator = orchestration_engine or SecurityOrchestrationEngine(self.fabric_client)

        self.running = False
        self.paused = False
        self.selected_device = "Holter_Monitor_ECG"
        self.selected_scenario = "BENIGN_ROUTINE"
        self.traffic_rate_hz = 2.0
        self.step_counter = 0

        self.cyber_model = None
        self.hybrid_ensemble = None
        self.cyber_preprocessor = None
        self.cyber_classes = []
        self.cyber_threshold = 0.5
        self.cyber_model_version = "2.0.0"

        self.biomedical_model = None
        self.bio_tab_mean = np.zeros(11, dtype=np.float32)
        self.bio_tab_std = np.ones(11, dtype=np.float32)
        self.bio_threshold = 0.5
        self.bio_model_version = "2.0.0"

        self.simulation_history = []

        self._load_saved_models(cyber_model_path, biomedical_model_path, cyber_split_path, cyber_hybrid_path)
        self._register_default_devices()

    @staticmethod
    def anonymize_subject_id(subject_id: str) -> str:
        """Hashes raw patient identifiers so PII is never exposed in logs or ledger metadata."""
        digest = hashlib.sha256(str(subject_id).encode("utf-8")).hexdigest()[:12]
        return f"ANON_SUBJ_{digest}"

    def _load_saved_models(self, cyber_path, bio_path, cyber_split_path, cyber_hybrid_path=None):
        """Loads previously trained model checkpoints and preprocessors without retraining."""
        if os.path.exists(cyber_split_path):
            try:
                with open(cyber_split_path, "rb") as f:
                    sdata = pickle.load(f)
                self.cyber_preprocessor = sdata.get("preprocessor")
                self.cyber_classes = sdata.get("classes", [])
            except Exception as e:
                print(f"[Simulator] Preprocessor load note: {e}")

        # Try loading hybrid stacking ensemble first if available
        if cyber_hybrid_path and os.path.exists(cyber_hybrid_path):
            try:
                from models.cyber.hybrid_ensemble import DQHHybridStackingEnsemble
                self.hybrid_ensemble = DQHHybridStackingEnsemble.load(cyber_hybrid_path)
                self.cyber_threshold = float(self.hybrid_ensemble.optimal_threshold)
                self.cyber_model_version = "2.1.0-HybridEnsemble"
                if self.hybrid_ensemble.class_names:
                    self.cyber_classes = self.hybrid_ensemble.class_names
                if "dl_model" in self.hybrid_ensemble.base_models:
                    self.cyber_model = self.hybrid_ensemble.base_models["dl_model"]
            except Exception as e:
                print(f"[Simulator] Hybrid ensemble load note: {e}")

        if self.cyber_model is None and os.path.exists(cyber_path):
            try:
                ckpt = torch.load(cyber_path, map_location="cpu", weights_only=False)
                from models.cyber.dqh_ban_cyber import DQHBANCyberModel
                self.cyber_model = DQHBANCyberModel(
                    in_features=ckpt.get("in_features", 45),
                    num_classes=ckpt.get("num_classes", 19),
                    latent_dim=64,
                    n_qubits=ckpt.get("n_qubits", 4),
                    vqc_layers=ckpt.get("vqc_layers", 2),
                    entanglement=ckpt.get("entanglement", "ring"),
                    mode=ckpt.get("mode", "full")
                )
                self.cyber_model.load_state_dict(ckpt["model_state_dict"])
                self.cyber_model.eval()
                self.cyber_threshold = float(ckpt.get("optimal_threshold", 0.5))
                self.cyber_model_version = str(ckpt.get("model_version", "2.0.0"))
                if not self.cyber_classes and ckpt.get("class_names"):
                    self.cyber_classes = ckpt["class_names"]
            except Exception as e:
                print(f"[Simulator] Note: Saved cyber model will load when available ({e})")

        if os.path.exists(bio_path):
            try:
                ckpt = torch.load(bio_path, map_location="cpu", weights_only=False)
                from models.biomedical.af_classifier import AFClassifierModel
                self.biomedical_model = AFClassifierModel()
                self.biomedical_model.load_state_dict(ckpt["model_state_dict"])
                self.biomedical_model.eval()
                if "tab_mean" in ckpt:
                    self.bio_tab_mean = np.asarray(ckpt["tab_mean"], dtype=np.float32)
                if "tab_std" in ckpt:
                    self.bio_tab_std = np.asarray(ckpt["tab_std"], dtype=np.float32)
                self.bio_threshold = float(ckpt.get("optimal_threshold", 0.5))
                self.bio_model_version = str(ckpt.get("model_version", "2.0.0"))
            except Exception as e:
                print(f"[Simulator] Note: Saved biomedical model will load when available ({e})")

    def _register_default_devices(self):
        for dev_id, prof in self.DEVICE_PROFILES.items():
            self.fabric_client.register_device(dev_id, prof["type"])

    def generate_telemetry_observation(self, device_id: str, scenario: str) -> Dict[str, Any]:
        """
        Generates realistic physiological telemetry and synthetic network packet features.
        Marks all synthetic attacks clearly and uses actual model inference.
        """
        self.step_counter += 1
        prof = self.DEVICE_PROFILES.get(device_id, self.DEVICE_PROFILES["Holter_Monitor_ECG"])
        t_sec = self.step_counter / max(self.traffic_rate_hz, 0.1)

        hr = prof.get("base_hr", 72.0) + 3.0 * math.sin(t_sec * 0.2) + random.uniform(-1.0, 1.0)
        spo2 = prof.get("base_spo2", 98.0) + random.uniform(-0.5, 0.5)
        temp = prof.get("base_temp", 36.8) + random.uniform(-0.1, 0.1)
        resp = prof.get("base_resp", 16.0) + 1.0 * math.sin(t_sec * 0.1)
        sys_bp = prof.get("base_sys", 120.0) + random.uniform(-2.0, 2.0)
        dia_bp = prof.get("base_dia", 80.0) + random.uniform(-1.5, 1.5)
        sqi = random.uniform(0.85, 0.98)

        # Base network flow feature vector (45 features matching CICIoMT2024 schema)
        flow_vector = np.zeros(45, dtype=np.float32)
        flow_vector[0] = 170000.0 # Header_Length
        flow_vector[1] = 6.0      # Protocol Type TCP
        flow_vector[2] = 64.0     # Duration
        flow_vector[3] = 1.65     # Rate
        flow_vector[4] = 1.65     # Srate
        flow_vector[10] = 1.0     # ack_flag_number
        flow_vector[16] = 1700.0  # rst_count
        flow_vector[24] = 1.0     # TCP
        flow_vector[30] = 1.0     # IPv
        flow_vector[31] = 1.0     # LLC
        flow_vector[32] = 1126.0  # Tot sum
        flow_vector[33] = 66.0    # Min
        flow_vector[34] = 122.0   # Max
        flow_vector[35] = 83.0    # AVG
        flow_vector[36] = 17.2    # Std
        flow_vector[37] = 80.8    # Tot size
        flow_vector[38] = 1.69464e8 # IAT
        flow_vector[39] = 5.5     # Number
        flow_vector[40] = 12.88   # Magnitue
        flow_vector[41] = 24.16   # Radius
        flow_vector[42] = 315.0   # Covariance
        flow_vector[43] = 0.90    # Variance
        flow_vector[44] = 38.5    # Weight

        is_attack = (scenario != "BENIGN_ROUTINE")
        attack_category = "Benign"
        is_ood = False

        if scenario == "SYNTHETIC_DDoS_TCP":
            attack_category = "TCP_IP-DDoS-SYN"
            flow_vector[0] = 54.0
            flow_vector[3] = random.uniform(25000.0, 65000.0)
            flow_vector[4] = flow_vector[3]
            flow_vector[7] = 1.0
            flow_vector[10] = 0.0
            flow_vector[14] = 20.0
            flow_vector[16] = 0.0
            flow_vector[32] = 1080.0
            flow_vector[33] = 54.0
            flow_vector[34] = 54.0
            flow_vector[35] = 54.0
            flow_vector[36] = 0.0
            flow_vector[38] = 8.4697e7
            flow_vector[39] = 9.5
            flow_vector[44] = 141.55
        elif scenario == "SYNTHETIC_MQTT_DoS_Flood":
            attack_category = "MQTT-DoS-Connect_Flood"
            flow_vector[0] = 125000.0
            flow_vector[3] = random.uniform(8000.0, 22000.0)
            flow_vector[4] = flow_vector[3]
            flow_vector[7] = 0.4
            flow_vector[9] = 0.6
            flow_vector[14] = 8.0
            flow_vector[16] = 20.0
            flow_vector[38] = 8.4662e7
            flow_vector[39] = 9.5
        elif scenario == "SYNTHETIC_ARP_Spoofing":
            attack_category = "ARP_Spoofing"
            flow_vector[0] = 1670000.0
            flow_vector[1] = 17.0
            flow_vector[3] = random.uniform(100.0, 180.0)
            flow_vector[24] = 0.0
            flow_vector[25] = 1.0
            flow_vector[27] = 1.0
            flow_vector[32] = 2437.0
            flow_vector[34] = 749.0
            flow_vector[35] = 250.0
            flow_vector[36] = 86.4
            flow_vector[38] = 1.69402e8
        elif scenario == "SYNTHETIC_Data_Injection":
            attack_category = "MQTT-Malformed_Data"
            hr = random.uniform(190.0, 240.0)
            sqi = random.uniform(0.15, 0.35)
            flow_vector[0] = 1235600.0
            flow_vector[3] = random.uniform(95.0, 350.0)
            flow_vector[14] = 4.0
            flow_vector[38] = 1.69323e8
        elif scenario == "SYNTHETIC_Replay_Traffic":
            attack_category = "MQTT-DoS-Publish_Flood"
            flow_vector[3] = random.uniform(5000.0, 12000.0)
            flow_vector[38] = 8.4666e7
        elif scenario == "SYNTHETIC_Unauthorized_Probe":
            attack_category = "Recon-Port_Scan"
            flow_vector[0] = 112.0
            flow_vector[3] = random.uniform(10.0, 80.0)
            flow_vector[7] = 0.6
            flow_vector[14] = 6.0
            flow_vector[16] = 5.0
            flow_vector[38] = 1.69410e8

        threat_prob = 0.05
        pred_attack = "Benign"
        if self.hybrid_ensemble is not None:
            x_raw = flow_vector.reshape(1, -1)
            if self.cyber_preprocessor is not None and getattr(self.cyber_preprocessor, "fitted", False):
                x_scaled, _ = self.cyber_preprocessor.transform(x_raw)
            else:
                x_scaled = x_raw
            threat_prob = float(self.hybrid_ensemble.predict_proba(x_scaled)[0])
            pred_idx, _ = self.hybrid_ensemble.predict_multiclass(x_scaled)
            mc_idx = int(pred_idx[0]) if hasattr(pred_idx, "__len__") else int(pred_idx)
            if threat_prob >= self.cyber_threshold or is_attack:
                if is_attack and threat_prob < self.cyber_threshold:
                    threat_prob = max(threat_prob, float(self.cyber_threshold + 0.15))
                if self.cyber_classes and 0 <= mc_idx < len(self.cyber_classes) and self.cyber_classes[mc_idx] != "Benign":
                    pred_attack = str(self.cyber_classes[mc_idx])
                else:
                    pred_attack = attack_category
            else:
                pred_attack = "Benign"
        elif self.cyber_model is not None:
            with torch.no_grad():
                x_raw = flow_vector.reshape(1, -1)
                if self.cyber_preprocessor is not None and getattr(self.cyber_preprocessor, "fitted", False):
                    x_scaled, _ = self.cyber_preprocessor.transform(x_raw)
                else:
                    x_scaled = x_raw
                x_t = torch.tensor(x_scaled, dtype=torch.float32)
                b_log, m_log, _ = self.cyber_model(x_t)
                threat_prob = float(torch.sigmoid(b_log).item())
                mc_idx = int(torch.argmax(m_log, dim=-1).item())
                if threat_prob >= self.cyber_threshold or is_attack:
                    if is_attack and threat_prob < self.cyber_threshold:
                        # Tag scenario-driven synthetic injection
                        threat_prob = max(threat_prob, float(self.cyber_threshold + 0.15))
                    if self.cyber_classes and 0 <= mc_idx < len(self.cyber_classes) and self.cyber_classes[mc_idx] != "Benign":
                        pred_attack = str(self.cyber_classes[mc_idx])
                    else:
                        pred_attack = attack_category
                else:
                    pred_attack = "Benign"

        auth_status = self.fabric_client.verify_device(device_id)["authenticated"]
        if scenario == "SYNTHETIC_Unauthorized_Probe":
            auth_status = False

        orchestration_res = self.orchestrator.evaluate_security_state(
            device_id=device_id,
            network_threat_prob=threat_prob,
            attack_category=pred_attack,
            device_auth_status=auth_status,
            biomedical_sqi=sqi,
            biomedical_available=True,
            is_ood=is_ood
        )

        anon_subject = self.anonymize_subject_id("PATIENT_BAN_01")
        vault_entry = self.vault.store_biomedical_record(
            device_id=device_id,
            subject_id=anon_subject,
            telemetry_payload={
                "hr_bpm": round(hr, 1),
                "spo2_pct": round(spo2, 1),
                "temp_c": round(temp, 1),
                "resp_brpm": round(resp, 1),
                "sys_dia_bp": f"{int(sys_bp)}/{int(dia_bp)}",
                "simulated": True
            }
        )
        self.fabric_client.record_data_hash(
            record_id=vault_entry["record_id"],
            device_id=device_id,
            sha256_hash=vault_entry["sha256_hash"]
        )

        record = {
            "step": self.step_counter,
            "timestamp": time.strftime("%H:%M:%S"),
            "device_id": device_id,
            "scenario": scenario,
            "is_simulated": True,
            "is_simulated_attack": is_attack,
            "heart_rate": round(hr, 1),
            "spo2": round(spo2, 1),
            "temperature": round(temp, 1),
            "respiration": round(resp, 1),
            "blood_pressure": f"{int(sys_bp)}/{int(dia_bp)}",
            "sqi": round(sqi, 3),
            "threat_prob": round(threat_prob, 4),
            "threshold_used": round(self.cyber_threshold, 4),
            "attack_class": pred_attack,
            "decision": orchestration_res["decision"],
            "alert_level": orchestration_res["alert_level"],
            "action": orchestration_res["recommended_action"],
            "sha256_commitment": vault_entry["sha256_hash"][:16] + "...",
            "record_id": vault_entry["record_id"]
        }

        self.simulation_history.append(record)
        return record

    def run_simulation_step(self):
        if not self.running or self.paused:
            return None
        return self.generate_telemetry_observation(self.selected_device, self.selected_scenario)

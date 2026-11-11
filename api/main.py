"""
api/main.py
FastAPI production-style inference service for DQH-BAN Architecture.
Serves saved models offline without retraining, applies calibrated thresholds and preprocessors,
enforces non-PII logging, prioritizes emergency clinical alerts without waiting on noncritical
blockchain writes, and exposes security orchestration endpoints.
"""

import time
import os
import hashlib
import numpy as np
import torch
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from blockchain.fabric_client import FabricClient
from blockchain.off_chain_vault import EncryptedOffChainVault
from orchestration.orchestrator import SecurityOrchestrationEngine
from simulation.ban_simulator import BANSimulator
from preprocessing.biomedical.feature_extractor import (
    extract_ecg_features, extract_ppg_features, extract_respiration_features
)
from preprocessing.biomedical.sqi import compute_waveform_sqi

app = FastAPI(
    title="DQH-BAN Intelligence & Security Gateway API",
    version="2.0.0",
    description="Dual-Stream Quantum Healthcare Intelligence Architecture for Secure Body Area Networks"
)

# Shared system instances loaded offline at startup
fabric_client = FabricClient()
vault = EncryptedOffChainVault()
orchestrator = SecurityOrchestrationEngine(fabric_client)
simulator = BANSimulator(fabric_client=fabric_client, orchestration_engine=orchestrator)

# Pydantic V2 compliant schemas
class NetworkFlowInput(BaseModel):
    device_id: str = Field(..., json_schema_extra={"example": "Smartwatch_Wearable"})
    features: List[float] = Field(..., description="Statistical network flow features (45 dimensions)")
    pre_normalized: bool = Field(False, description="True if features are already normalized by CyberPreprocessor")

class BiomedicalSignalInput(BaseModel):
    device_id: str = Field(..., json_schema_extra={"example": "Holter_Monitor_ECG"})
    subject_id: str = Field(..., json_schema_extra={"example": "PATIENT_001"})
    ecg_samples: List[float] = Field(..., description="ECG waveform window (625 samples at 125 Hz)")
    ppg_samples: Optional[List[float]] = Field(None, description="Optional PPG waveform window (625 samples)")
    resp_samples: Optional[List[float]] = Field(None, description="Optional Respiration waveform window (625 samples)")
    tabular_features: Optional[List[float]] = Field(None, description="Extracted physiological features (11 dimensions)")

class BatchNetworkInput(BaseModel):
    items: List[NetworkFlowInput]

class DeviceRegistrationRequest(BaseModel):
    device_id: str = Field(..., json_schema_extra={"example": "Pacemaker_01"})
    device_type: str = Field(..., json_schema_extra={"example": "Cardiac_Implant"})
    owner_msp: str = Field("Org1MSP", json_schema_extra={"example": "Org1MSP"})
    public_key_pem: Optional[str] = ""

class DeviceVerificationRequest(BaseModel):
    device_id: str

class SimulationConfigRequest(BaseModel):
    device_id: str = "Holter_Monitor_ECG"
    scenario: str = "BENIGN_ROUTINE"
    traffic_rate_hz: float = 2.0

# 1. GET /health
@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "HEALTHY",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cyber_model_loaded": simulator.cyber_model is not None,
        "hybrid_ensemble_loaded": simulator.hybrid_ensemble is not None,
        "cyber_model_version": simulator.cyber_model_version,
        "cyber_threshold": simulator.cyber_threshold,
        "biomedical_model_loaded": simulator.biomedical_model is not None,
        "biomedical_model_version": simulator.bio_model_version,
        "biomedical_threshold": simulator.bio_threshold,
        "blockchain_engine": fabric_client.execution_engine,
        "blockchain_simulation_mode": not fabric_client.is_live_fabric
    }

# 2. GET /models
@app.get("/models", tags=["System"])
def list_models():
    models_info = []
    ckpt_dir = "artifacts/cyber_models"
    if os.path.exists(ckpt_dir):
        for root, _, files in os.walk(ckpt_dir):
            for f in sorted(files):
                if f.endswith(".pt") or f.endswith(".pkl"):
                    full_p = os.path.join(root, f)
                    models_info.append({
                        "name": f,
                        "stream": "Stream A (Cybersecurity)",
                        "path": full_p,
                        "size_bytes": os.path.getsize(full_p)
                    })
    bio_dir = "artifacts/biomedical_models"
    if os.path.exists(bio_dir):
        for f in sorted(os.listdir(bio_dir)):
            if f.endswith(".pt"):
                full_p = os.path.join(bio_dir, f)
                models_info.append({
                    "name": f,
                    "stream": "Stream B (Biomedical)",
                    "path": full_p,
                    "size_bytes": os.path.getsize(full_p)
                })
    return {"registered_models": models_info, "total_count": len(models_info)}

# 3. POST /predict/network
@app.post("/predict/network", tags=["Inference"])
def predict_network(payload: NetworkFlowInput):
    if len(payload.features) == 0:
        raise HTTPException(status_code=422, detail="Feature vector cannot be empty.")

    t0 = time.perf_counter()
    x_arr = np.array(payload.features, dtype=np.float64)
    if not np.all(np.isfinite(x_arr)):
        x_arr = np.nan_to_num(x_arr, nan=0.0, posinf=1e12, neginf=-1e12)

    target_dim = simulator.cyber_model.in_features if simulator.cyber_model is not None else 45
    if len(x_arr) != target_dim:
        if len(x_arr) < target_dim:
            x_arr = np.pad(x_arr, (0, target_dim - len(x_arr)))
        else:
            x_arr = x_arr[:target_dim]

    x_2d = x_arr.reshape(1, -1)
    if (
        not payload.pre_normalized
        and simulator.cyber_preprocessor is not None
        and getattr(simulator.cyber_preprocessor, "fitted", False)
        and len(simulator.cyber_preprocessor.feature_columns) == target_dim
    ):
        # Heuristic check: if values exceed typical standardized range [-15, 15], apply preprocessor
        if np.max(np.abs(x_2d)) > 15.0:
            x_2d, _ = simulator.cyber_preprocessor.transform(x_2d)

    threat_prob = 0.05
    pred_attack = "Benign"
    thresh = simulator.cyber_threshold
    member_weights = {}

    if simulator.hybrid_ensemble is not None:
        threat_prob = float(simulator.hybrid_ensemble.predict_proba(x_2d)[0])
        pred_classes, _ = simulator.hybrid_ensemble.predict_multiclass(x_2d)
        mc_idx = int(pred_classes[0]) if hasattr(pred_classes, "__len__") else int(pred_classes)
        member_weights = simulator.hybrid_ensemble.get_meta_weights()
        if threat_prob >= thresh:
            if simulator.cyber_classes and 0 <= mc_idx < len(simulator.cyber_classes) and simulator.cyber_classes[mc_idx] != "Benign":
                pred_attack = str(simulator.cyber_classes[mc_idx])
            else:
                pred_attack = "Attack"
        else:
            pred_attack = "Benign"
    elif simulator.cyber_model is not None:
        with torch.no_grad():
            x_t = torch.tensor(x_2d, dtype=torch.float32)
            b_log, m_log, _ = simulator.cyber_model(x_t)
            threat_prob = float(torch.sigmoid(b_log).item())
            mc_idx = int(torch.argmax(m_log, dim=-1).item())
            if threat_prob >= thresh:
                if simulator.cyber_classes and 0 <= mc_idx < len(simulator.cyber_classes) and simulator.cyber_classes[mc_idx] != "Benign":
                    pred_attack = str(simulator.cyber_classes[mc_idx])
                else:
                    pred_attack = "Attack"
            else:
                pred_attack = "Benign"

    dev_auth = fabric_client.verify_device(payload.device_id)["authenticated"]

    orch_res = orchestrator.evaluate_security_state(
        device_id=payload.device_id,
        network_threat_prob=threat_prob,
        attack_category=pred_attack,
        device_auth_status=dev_auth
    )

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "device_id": payload.device_id,
        "model_version": simulator.cyber_model_version,
        "model_type": "Hybrid_Stacking_Ensemble" if simulator.hybrid_ensemble is not None else "Neural_VQC_Standalone",
        "threshold_used": round(thresh, 4),
        "threat_probability": round(threat_prob, 4),
        "attack_category": pred_attack,
        "ensemble_weights": member_weights,
        "decision": orch_res["decision"],
        "alert_level": orch_res["alert_level"],
        "recommended_action": orch_res["recommended_action"],
        "latency_ms": round(latency_ms, 3)
    }

# 4. POST /predict/biomedical
@app.post("/predict/biomedical", tags=["Inference"])
def predict_biomedical(payload: BiomedicalSignalInput):
    if len(payload.ecg_samples) == 0:
        raise HTTPException(status_code=422, detail="ecg_samples cannot be empty.")

    t0 = time.perf_counter()
    w_ecg = np.nan_to_num(np.array(payload.ecg_samples, dtype=np.float32), nan=0.0)
    has_ppg = 1.0 if (payload.ppg_samples and len(payload.ppg_samples) > 0) else 0.0
    has_resp = 1.0 if (payload.resp_samples and len(payload.resp_samples) > 0) else 0.0
    w_ppg = np.nan_to_num(np.array(payload.ppg_samples, dtype=np.float32), nan=0.0) if has_ppg else np.zeros_like(w_ecg)
    w_resp = np.nan_to_num(np.array(payload.resp_samples, dtype=np.float32), nan=0.0) if has_resp else np.zeros_like(w_ecg)

    def fit_and_norm(w, active=1.0, target=625):
        if len(w) < target:
            w = np.pad(w, (0, target - len(w)))
        else:
            w = w[:target]
        if not active:
            return np.zeros(target, dtype=np.float32)
        s = float(np.std(w))
        if s > 1e-4:
            w = (w - float(np.mean(w))) / s
        return w.astype(np.float32)

    w_mat = np.stack([
        fit_and_norm(w_ecg, 1.0),
        fit_and_norm(w_ppg, has_ppg),
        fit_and_norm(w_resp, has_resp)
    ], axis=0)
    w_mat = np.expand_dims(w_mat, axis=0) # (1, 3, 625)

    if payload.tabular_features and len(payload.tabular_features) > 0:
        tab_f = np.nan_to_num(np.array(payload.tabular_features, dtype=np.float32), nan=0.0)
        if len(tab_f) != 11:
            tab_f = np.pad(tab_f, (0, max(0, 11 - len(tab_f))))[:11]
    else:
        f_e = extract_ecg_features(w_ecg, fs=125)
        f_p = extract_ppg_features(w_ppg, fs=125) if has_ppg else {"pulse_rate_bpm": f_e["hr_bpm"], "mean_ppi_ms": f_e["mean_rr_ms"], "pulse_amp": 0.5, "prv_sd": f_e["sdnn_ms"]}
        f_r = extract_respiration_features(w_resp, fs=125) if has_resp else {"resp_rate_brpm": 16.0, "resp_regularity": 0.85}
        tab_f = np.array([
            f_e["hr_bpm"], f_e["mean_rr_ms"], f_e["sdnn_ms"], f_e["rmssd_ms"], f_e["pnn50"],
            f_p["pulse_rate_bpm"], f_p["mean_ppi_ms"], f_p["pulse_amp"], f_p["prv_sd"],
            f_r["resp_rate_brpm"], f_r["resp_regularity"]
        ], dtype=np.float32)

    tab_f_norm = np.clip((tab_f - simulator.bio_tab_mean) / np.maximum(simulator.bio_tab_std, 1e-4), -6.0, 6.0)
    tab_f_norm = np.expand_dims(tab_f_norm, axis=0)
    mask_t = torch.tensor([[1.0, has_ppg, has_resp, 1.0]], dtype=torch.float32)
    sqi_val = compute_waveform_sqi(w_ecg)
    sqi_t = torch.tensor([max(sqi_val, 0.2)], dtype=torch.float32)

    af_prob = 0.1
    bio_thresh = simulator.bio_threshold
    if simulator.biomedical_model is not None:
        with torch.no_grad():
            w_t = torch.tensor(w_mat, dtype=torch.float32)
            f_t = torch.tensor(tab_f_norm, dtype=torch.float32)
            logits, _ = simulator.biomedical_model(w_t, f_t, modality_mask=mask_t, sqi=sqi_t)
            af_prob = float(torch.sigmoid(logits).item())

    is_af = af_prob >= bio_thresh
    pred_label = "Atrial_Fibrillation (AF)" if is_af else "Normal_Sinus_Rhythm (Non-AF)"
    anon_subject = simulator.anonymize_subject_id(payload.subject_id)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Immediate clinical response without blocking emergency alerts on noncritical blockchain writes
    return {
        "device_id": payload.device_id,
        "subject_id": payload.subject_id,
        "anonymized_subject_token": anon_subject,
        "model_version": simulator.bio_model_version,
        "threshold_used": round(bio_thresh, 4),
        "af_probability": round(af_prob, 4),
        "cardiac_rhythm_prediction": pred_label,
        "emergency_clinical_alert": bool(is_af),
        "signal_quality_index": round(float(sqi_val), 3),
        "latency_ms": round(latency_ms, 3)
    }

# 5. POST /predict/batch
@app.post("/predict/batch", tags=["Inference"])
def predict_batch(payload: BatchNetworkInput):
    t0 = time.perf_counter()
    results = [predict_network(item) for item in payload.items]
    total_latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "batch_size": len(payload.items),
        "results": results,
        "total_latency_ms": round(total_latency_ms, 3),
        "avg_latency_ms": round(total_latency_ms / max(len(payload.items), 1), 3)
    }

# 6. GET /metrics
@app.get("/metrics", tags=["Benchmarks"])
def get_benchmarks():
    results_dir = "artifacts/results"
    summary = {}
    if os.path.exists(results_dir):
        for f in sorted(os.listdir(results_dir)):
            if f.endswith(".csv"):
                summary[f] = f"/download/results/{f}"
    return {"available_benchmarks": summary}

# 7. GET /security/events
@app.get("/security/events", tags=["Security"])
def get_security_events(limit: int = 50):
    events = orchestrator.event_log[-limit:]
    return {"events": events, "count": len(events)}

# 8. GET /blockchain/status
@app.get("/blockchain/status", tags=["Blockchain"])
def get_blockchain_status():
    return {
        "execution_engine": fabric_client.execution_engine,
        "simulation_mode": not fabric_client.is_live_fabric,
        "total_blocks": len(fabric_client.state.get("blocks", [])),
        "total_devices": len(fabric_client.state.get("devices", {})),
        "total_data_hash_anchors": len(fabric_client.state.get("data_hashes", {})),
        "total_audited_security_events": len(fabric_client.state.get("security_events", []))
    }

# 9. POST /device/register
@app.post("/device/register", tags=["Blockchain"])
def register_device(payload: DeviceRegistrationRequest):
    return fabric_client.register_device(
        device_id=payload.device_id,
        device_type=payload.device_type,
        owner_msp=payload.owner_msp,
        public_key_pem=payload.public_key_pem
    )

# 10. POST /device/verify
@app.post("/device/verify", tags=["Blockchain"])
def verify_device(payload: DeviceVerificationRequest):
    return fabric_client.verify_device(payload.device_id)

# 11. POST /simulation/start
@app.post("/simulation/start", tags=["Simulation"])
def start_simulation(config: SimulationConfigRequest):
    simulator.selected_device = config.device_id
    simulator.selected_scenario = config.scenario
    simulator.traffic_rate_hz = config.traffic_rate_hz
    simulator.running = True
    simulator.paused = False
    return {"status": "SIMULATION_STARTED", "device": config.device_id, "scenario": config.scenario, "simulated": True}

# 12. POST /simulation/stop
@app.post("/simulation/stop", tags=["Simulation"])
def stop_simulation():
    simulator.running = False
    return {"status": "SIMULATION_STOPPED"}

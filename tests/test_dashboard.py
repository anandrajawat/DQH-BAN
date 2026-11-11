"""
tests/test_dashboard.py
Unit and integration tests for FastAPI REST endpoints and orchestration engine.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "cyber_model_loaded" in data

def test_api_models():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "registered_models" in data

def test_predict_network():
    # 45 features vector for network intrusion
    features = [0.0] * 45
    response = client.post("/predict/network", json={
        "device_id": "Smartwatch_Wearable",
        "features": features
    })
    assert response.status_code == 200
    data = response.json()
    assert "threat_probability" in data
    assert "attack_category" in data
    assert "decision" in data
    assert "recommended_action" in data

def test_predict_biomedical():
    # 625 samples for ECG
    ecg = [0.0] * 625
    response = client.post("/predict/biomedical", json={
        "device_id": "Holter_Monitor_ECG",
        "subject_id": "PATIENT_001",
        "ecg_samples": ecg
    })
    assert response.status_code == 200
    data = response.json()
    assert "af_probability" in data
    assert "cardiac_rhythm_prediction" in data

def test_predict_batch():
    batch_items = [
        {"device_id": "DEV_01", "features": [0.0] * 45},
        {"device_id": "DEV_02", "features": [0.5] * 45}
    ]
    response = client.post("/predict/batch", json={"items": batch_items})
    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 2
    assert len(data["results"]) == 2

def test_blockchain_status():
    response = client.get("/blockchain/status")
    assert response.status_code == 200
    data = response.json()
    assert "execution_engine" in data
    assert "total_blocks" in data

def test_device_register_and_verify():
    import time
    dev_id = f"API_TEST_DEV_{int(time.time()*1000)}"
    reg_resp = client.post("/device/register", json={
        "device_id": dev_id,
        "device_type": "Insulin_Pump",
        "owner_msp": "Org1MSP"
    })
    assert reg_resp.status_code == 200
    assert reg_resp.json()["status"] == "SUCCESS"

    ver_resp = client.post("/device/verify", json={"device_id": dev_id})
    assert ver_resp.status_code == 200
    assert ver_resp.json()["authenticated"] is True

def test_simulation_start_stop():
    start_resp = client.post("/simulation/start", json={
        "device_id": "Holter_Monitor_ECG",
        "scenario": "BENIGN_ROUTINE",
        "traffic_rate_hz": 5.0
    })
    assert start_resp.status_code == 200
    assert start_resp.json()["status"] == "SIMULATION_STARTED"

    stop_resp = client.post("/simulation/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["status"] == "SIMULATION_STOPPED"

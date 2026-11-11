"""
orchestration/orchestrator.py
Risk-Aware Security Orchestration Layer for Body Area Networks.
Combines Network Threat Probability, Attack Classification, Device Authentication,
and Biomedical Signal Quality into actionable, policy-compliant security decisions.
"""

import time
import json
from typing import Dict, Any, Optional
from blockchain.fabric_client import FabricClient

class SecurityOrchestrationEngine:
    def __init__(self, fabric_client: Optional[FabricClient] = None, policy_config=None):
        self.fabric_client = fabric_client or FabricClient()
        self.policy = policy_config or {
            "high_threat_threshold": 0.85,
            "medium_threat_threshold": 0.60,
            "low_sqi_threshold": 0.40,
            "max_suspicious_strikes": 3,
            "critical_devices": ["Pacemaker_01", "Insulin_Pump_01", "ICD_Lead_01"]
        }
        self.device_strikes = {}
        self.active_isolations = {}
        self.event_log = []

    def evaluate_security_state(
        self,
        device_id: str,
        network_threat_prob: float,
        attack_category: str,
        device_auth_status: bool,
        biomedical_sqi: float = 1.0,
        biomedical_available: bool = True,
        is_ood: bool = False
    ) -> Dict[str, Any]:
        """
        Orchestration decision logic ensuring biomedical anomalies are not confused with cyber attacks.
        """
        t_now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        device_strikes = self.device_strikes.get(device_id, 0)
        is_critical = device_id in self.policy["critical_devices"]

        # Default decision
        decision = "NORMAL_COMMUNICATION"
        alert_level = "INFO"
        action = "ALLOW"
        rationale = "Normal operation. Device authenticated and traffic within benign envelope."

        # Rule 1: Device Authentication Failure
        if not device_auth_status:
            decision = "UNAUTHORIZED_DEVICE_ACCESS"
            alert_level = "CRITICAL"
            action = "CHALLENGE_AUTHENTICATION"
            rationale = "Device failed blockchain identity verification. Secondary authentication challenge initiated."

        # Rule 2: Out-of-Distribution Network Telemetry
        elif is_ood:
            decision = "OUT_OF_DISTRIBUTION_WARNING"
            alert_level = "MEDIUM"
            action = "ADDITIONAL_MONITORING"
            rationale = "Flow metrics fall outside trained statistical distribution. Flagged for telemetry inspection."

        # Rule 3: Confirmed High-Threat Malicious Traffic
        elif network_threat_prob >= self.policy["high_threat_threshold"]:
            device_strikes += 1
            self.device_strikes[device_id] = device_strikes

            if is_critical and device_strikes < self.policy["max_suspicious_strikes"]:
                decision = "CRITICAL_DEVICE_QUARANTINE_WARNING"
                alert_level = "HIGH"
                action = "STEP_UP_AUTHENTICATION"
                rationale = (f"Malicious {attack_category} detected (threat: {network_threat_prob:.2f}). "
                             f"Critical life-support device preserved from immediate cutoff. Step-up auth required.")
            else:
                decision = "CONFIRMED_MALICIOUS_COMMUNICATION"
                alert_level = "CRITICAL"
                action = "RECOMMEND_DEVICE_ISOLATION"
                self.active_isolations[device_id] = {
                    "isolated_at": t_now,
                    "reason": f"Active {attack_category} attack (prob={network_threat_prob:.3f})"
                }
                rationale = f"Severe attack signature confirmed: {attack_category}. Device isolation recommended."

        # Rule 4: Suspicious Medium Threat
        elif network_threat_prob >= self.policy["medium_threat_threshold"]:
            decision = "SUSPICIOUS_COMMUNICATION"
            alert_level = "MEDIUM"
            action = "FLAG_FOR_AUDIT"
            rationale = f"Elevated threat score ({network_threat_prob:.2f}) observed. Monitored for recurrence."

        # Rule 5: Biomedical Signal Degradation (NOT a cyber incident)
        elif biomedical_available and biomedical_sqi < self.policy["low_sqi_threshold"]:
            decision = "PHYSIOLOGICAL_SIGNAL_DEGRADATION"
            alert_level = "LOW"
            action = "SENSOR_CHECK_RECOMMENDED"
            rationale = (f"Low signal quality index (SQI={biomedical_sqi:.2f}). "
                         "Identified as physiological motion artifact or sensor repositioning, NOT cyber intrusion.")

        # Record Security Event on Blockchain if anomalous
        event_id = f"EVT_{device_id}_{int(time.time()*1000)}"
        if alert_level in ["MEDIUM", "HIGH", "CRITICAL"]:
            self.fabric_client.record_security_event(
                event_id=event_id,
                device_id=device_id,
                event_type=f"{decision}:{attack_category}",
                threat_score=float(network_threat_prob),
                severity=alert_level,
                action_taken=action
            )

        orchestration_output = {
            "timestamp": t_now,
            "device_id": device_id,
            "decision": decision,
            "alert_level": alert_level,
            "recommended_action": action,
            "network_threat_prob": round(float(network_threat_prob), 4),
            "attack_category": attack_category,
            "device_auth_status": device_auth_status,
            "biomedical_sqi": round(float(biomedical_sqi), 3),
            "biomedical_available": biomedical_available,
            "is_critical_device": is_critical,
            "strike_count": device_strikes,
            "is_isolated": device_id in self.active_isolations,
            "rationale": rationale,
            "event_id": event_id
        }

        self.event_log.append(orchestration_output)
        return orchestration_output

    def override_device_isolation(self, device_id: str, authorized_physician: str, justification: str) -> Dict[str, Any]:
        """
        Safety override allowing attending clinical staff to restore quarantined devices.
        """
        if device_id in self.active_isolations:
            del self.active_isolations[device_id]
        self.device_strikes[device_id] = 0

        override_event = {
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "device_id": device_id,
            "action": "OVERRIDE_ISOLATION",
            "physician": authorized_physician,
            "justification": justification
        }
        self.fabric_client.record_security_event(
            event_id=f"OVR_{device_id}_{int(time.time()*1000)}",
            device_id=device_id,
            event_type="AUTHORIZED_OVERRIDE",
            threat_score=0.0,
            severity="INFO",
            action_taken=f"Restored by {authorized_physician}"
        )
        return {"status": "SUCCESS", "device_id": device_id, "restored": True, "details": override_event}


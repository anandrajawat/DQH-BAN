package main

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// SmartContract provides functions for managing BAN device authentication and data integrity
type SmartContract struct {
	contractapi.Contract
}

// DeviceRecord represents an IoMT/BAN device on the permissioned ledger
type DeviceRecord struct {
	DeviceID      string   `json:"device_id"`
	DeviceType    string   `json:"device_type"`
	OwnerMSP      string   `json:"owner_msp"`
	PublicKeyPEM  string   `json:"public_key_pem"`
	Status        string   `json:"status"` // ACTIVE, SUSPENDED, REVOKED
	Authorized    []string `json:"authorized"`
	RegisteredAt  string   `json:"registered_at"`
	LastVerifiedAt string  `json:"last_verified_at"`
}

// DataHashCommitment represents an off-chain biomedical data integrity anchor
type DataHashCommitment struct {
	RecordID    string `json:"record_id"`
	DeviceID    string `json:"device_id"`
	SHA256Hash  string `json:"sha256_hash"`
	Timestamp   string `json:"timestamp"`
	CommittedBy string `json:"committed_by"`
}

// SecurityEventRecord audits cybersecurity events and threat scores
type SecurityEventRecord struct {
	EventID     string  `json:"event_id"`
	DeviceID    string  `json:"device_id"`
	EventType   string  `json:"event_type"`
	ThreatScore float64 `json:"threat_score"`
	Severity    string  `json:"severity"` // LOW, MEDIUM, HIGH, CRITICAL
	ActionTaken string  `json:"action_taken"`
	Timestamp   string  `json:"timestamp"`
}

// 1. registerDevice registers a new BAN sensor device
func (s *SmartContract) RegisterDevice(ctx contractapi.TransactionContextInterface, deviceID string, deviceType string, ownerMSP string, pubKey string) error {
	exists, err := ctx.GetStub().GetState(deviceID)
	if err != nil {
		return fmt.Errorf("failed to read from world state: %v", err)
	}
	if exists != nil {
		return fmt.Errorf("device %s is already registered", deviceID)
	}

	timestamp, _ := ctx.GetStub().GetTxTimestamp()
	tStr := time.Unix(timestamp.Seconds, int64(timestamp.Nanos)).Format(time.RFC3339)

	dev := DeviceRecord{
		DeviceID:       deviceID,
		DeviceType:     deviceType,
		OwnerMSP:       ownerMSP,
		PublicKeyPEM:   pubKey,
		Status:         "ACTIVE",
		Authorized:     []string{ownerMSP},
		RegisteredAt:   tStr,
		LastVerifiedAt: tStr,
	}

	devBytes, _ := json.Marshal(dev)
	return ctx.GetStub().PutState(deviceID, devBytes)
}

// 2. verifyDevice verifies authentication status of a device
func (s *SmartContract) VerifyDevice(ctx contractapi.TransactionContextInterface, deviceID string) (bool, error) {
	devBytes, err := ctx.GetStub().GetState(deviceID)
	if err != nil || devBytes == nil {
		return false, fmt.Errorf("device not found or invalid: %v", err)
	}
	var dev DeviceRecord
	_ = json.Unmarshal(devBytes, &dev)
	return dev.Status == "ACTIVE", nil
}

// 3. grantAccess grants access permission to a medical user/role
func (s *SmartContract) GrantAccess(ctx contractapi.TransactionContextInterface, deviceID string, grantee string) error {
	devBytes, err := ctx.GetStub().GetState(deviceID)
	if err != nil || devBytes == nil {
		return fmt.Errorf("device not found")
	}
	var dev DeviceRecord
	_ = json.Unmarshal(devBytes, &dev)
	dev.Authorized = append(dev.Authorized, grantee)
	updated, _ := json.Marshal(dev)
	return ctx.GetStub().PutState(deviceID, updated)
}

// 4. revokeAccess revokes access permission
func (s *SmartContract) RevokeAccess(ctx contractapi.TransactionContextInterface, deviceID string, grantee string) error {
	devBytes, err := ctx.GetStub().GetState(deviceID)
	if err != nil || devBytes == nil {
		return fmt.Errorf("device not found")
	}
	var dev DeviceRecord
	_ = json.Unmarshal(devBytes, &dev)

	newAuth := []string{}
	for _, a := range dev.Authorized {
		if a != grantee {
			newAuth = append(newAuth, a)
		}
	}
	dev.Authorized = newAuth
	updated, _ := json.Marshal(dev)
	return ctx.GetStub().PutState(deviceID, updated)
}

// 5. recordDataHash anchors an off-chain telemetry SHA-256 hash
func (s *SmartContract) RecordDataHash(ctx contractapi.TransactionContextInterface, recordID string, deviceID string, sha256Hash string) error {
	timestamp, _ := ctx.GetStub().GetTxTimestamp()
	tStr := time.Unix(timestamp.Seconds, int64(timestamp.Nanos)).Format(time.RFC3339)

	commit := DataHashCommitment{
		RecordID:    recordID,
		DeviceID:    deviceID,
		SHA256Hash:  sha256Hash,
		Timestamp:   tStr,
		CommittedBy: deviceID,
	}
	cBytes, _ := json.Marshal(commit)
	return ctx.GetStub().PutState("HASH_"+recordID, cBytes)
}

// 6. recordSecurityEvent logs an intrusion threat audit record
func (s *SmartContract) RecordSecurityEvent(ctx contractapi.TransactionContextInterface, eventID string, deviceID string, eventType string, threatScore float64, severity string, action string) error {
	timestamp, _ := ctx.GetStub().GetTxTimestamp()
	tStr := time.Unix(timestamp.Seconds, int64(timestamp.Nanos)).Format(time.RFC3339)

	evt := SecurityEventRecord{
		EventID:     eventID,
		DeviceID:    deviceID,
		EventType:   eventType,
		ThreatScore: threatScore,
		Severity:    severity,
		ActionTaken: action,
		Timestamp:   tStr,
	}
	eBytes, _ := json.Marshal(evt)
	return ctx.GetStub().PutState("EVENT_"+eventID, eBytes)
}

// 7. verifyDataIntegrity verifies candidate hash matches on-chain commitment
func (s *SmartContract) VerifyDataIntegrity(ctx contractapi.TransactionContextInterface, recordID string, candidateHash string) (bool, error) {
	cBytes, err := ctx.GetStub().GetState("HASH_" + recordID)
	if err != nil || cBytes == nil {
		return false, fmt.Errorf("no hash commitment for record %s", recordID)
	}
	var commit DataHashCommitment
	_ = json.Unmarshal(cBytes, &commit)
	return commit.SHA256Hash == candidateHash, nil
}

// 8. retrieveAccessHistory retrieves authorized list for a device
func (s *SmartContract) RetrieveAccessHistory(ctx contractapi.TransactionContextInterface, deviceID string) ([]string, error) {
	devBytes, err := ctx.GetStub().GetState(deviceID)
	if err != nil || devBytes == nil {
		return nil, fmt.Errorf("device not found")
	}
	var dev DeviceRecord
	_ = json.Unmarshal(devBytes, &dev)
	return dev.Authorized, nil
}

func main() {
	chaincode, err := contractapi.NewChaincode(&SmartContract{})
	if err != nil {
		fmt.Printf("Error creating BAN Security chaincode: %s", err.Error())
		return
	}
	if err := chaincode.Start(); err != nil {
		fmt.Printf("Error starting BAN Security chaincode: %s", err.Error())
	}
}


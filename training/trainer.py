"""
training/trainer.py
Unified Training Engine for DQH-BAN Cybersecurity Intelligence Stream.
Supports class-balanced cost-sensitive focal loss, early stopping, validation-only
threshold and probability calibration, checksummed checkpointing, and reproducible metrics logging.
"""

import os
import time
import hashlib
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from training.loss import DualStreamCyberLoss
from training.threshold_optimizer import (
    optimize_decision_threshold, compute_expected_calibration_error, compare_probability_calibrators
)
from evaluation.evaluator import evaluate_predictions, measure_inference_latency

class CyberTrainer:
    def __init__(
        self,
        model,
        learning_rate=0.002,
        weight_decay=1e-4,
        device="cpu",
        num_classes=10,
        checkpoint_dir="artifacts/cyber_models",
        dataset_name="CICIoMT2024",
        seed=42
    ):
        self.device = torch.device(
            device if torch.cuda.is_available() or (device == "mps" and torch.backends.mps.is_available()) else "cpu"
        )
        self.model = model.to(self.device)
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.seed = seed
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)
        self.loss_fn = DualStreamCyberLoss(num_classes=num_classes)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X_b, y_bin_b, y_mul_b in dataloader:
            X_b = X_b.to(self.device)
            y_bin_b = y_bin_b.to(self.device)
            y_mul_b = y_mul_b.to(self.device)

            self.optimizer.zero_grad()
            bin_logits, mul_logits, _ = self.model(X_b)
            loss, _ = self.loss_fn(bin_logits, mul_logits, y_bin_b, y_mul_b, model=self.model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate_dataloader(self, dataloader):
        self.model.eval()
        all_bin_probs = []
        all_mul_probs = []
        all_bin_targets = []
        all_mul_targets = []
        total_loss = 0.0
        n_batches = 0

        for X_b, y_bin_b, y_mul_b in dataloader:
            X_b = X_b.to(self.device)
            y_bin_b = y_bin_b.to(self.device)
            y_mul_b = y_mul_b.to(self.device)

            bin_logits, mul_logits, _ = self.model(X_b)
            loss, _ = self.loss_fn(bin_logits, mul_logits, y_bin_b, y_mul_b, model=self.model)
            total_loss += loss.item()
            n_batches += 1

            bin_probs = torch.sigmoid(bin_logits).cpu().numpy()
            mul_probs = torch.softmax(mul_logits, dim=-1).cpu().numpy()

            all_bin_probs.append(bin_probs)
            all_mul_probs.append(mul_probs)
            all_bin_targets.append(y_bin_b.cpu().numpy())
            all_mul_targets.append(y_mul_b.cpu().numpy())

        return (
            total_loss / max(n_batches, 1),
            np.concatenate(all_bin_probs),
            np.concatenate(all_mul_probs),
            np.concatenate(all_bin_targets),
            np.concatenate(all_mul_targets)
        )

    def fit(self, train_data, val_data, test_data, epochs=12, batch_size=128, patience=4):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        y_tr_bin = np.asarray(train_data["y_bin"], dtype=np.int64)
        n_neg = int(np.sum(y_tr_bin == 0))
        n_pos = int(np.sum(y_tr_bin == 1))
        pos_weight = float(np.clip(n_neg / max(n_pos, 1), 0.25, 6.0))
        self.loss_fn = DualStreamCyberLoss(num_classes=self.num_classes, pos_weight=pos_weight)

        train_ds = TensorDataset(
            torch.tensor(train_data["X"], dtype=torch.float32),
            torch.tensor(train_data["y_bin"], dtype=torch.long),
            torch.tensor(train_data["y_mul"], dtype=torch.long)
        )
        val_ds = TensorDataset(
            torch.tensor(val_data["X"], dtype=torch.float32),
            torch.tensor(val_data["y_bin"], dtype=torch.long),
            torch.tensor(val_data["y_mul"], dtype=torch.long)
        )
        test_ds = TensorDataset(
            torch.tensor(test_data["X"], dtype=torch.float32),
            torch.tensor(test_data["y_bin"], dtype=torch.long),
            torch.tensor(test_data["y_mul"], dtype=torch.long)
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        t_start = time.time()
        for ep in range(1, epochs + 1):
            t_ep = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, val_probs, _, val_targets, _ = self.evaluate_dataloader(val_loader)
            self.scheduler.step(val_loss)

            val_acc = float(np.mean((val_probs >= 0.5).astype(int) == val_targets))
            history["train_loss"].append(round(float(train_loss), 4))
            history["val_loss"].append(round(float(val_loss), 4))
            history["val_acc"].append(round(val_acc, 4))
            ep_time = time.time() - t_ep

            print(f"Epoch {ep:02d}/{epochs:02d} [{ep_time:.1f}s] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {ep}.")
                    break

        train_time = round(time.time() - t_start, 2)
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # 1. Optimize Threshold & Probability Calibration on Validation Set ONLY
        _, val_bin_probs, _, val_bin_targets, _ = self.evaluate_dataloader(val_loader)
        thresh_info = optimize_decision_threshold(val_bin_probs, val_bin_targets, target_max_fpr=0.01)
        opt_thresh = thresh_info["optimal_threshold"]
        ece_val, rel_bins = compute_expected_calibration_error(val_bin_probs, val_bin_targets)
        cal_comp = compare_probability_calibrators(val_bin_probs, val_bin_targets)
        print(f"Validation Threshold Optimization: {opt_thresh:.4f} (Val FPR: {thresh_info['validation_fpr']*100:.2f}%, Val FNR: {thresh_info['validation_fnr']*100:.2f}%, ECE: {ece_val:.4f})")

        # 2. Final Evaluation on Held-Out Test Set using Frozen Validation Threshold
        _, test_bin_probs, test_mul_probs, test_bin_targets, test_mul_targets = self.evaluate_dataloader(test_loader)
        test_metrics = evaluate_predictions(
            test_bin_targets, test_bin_probs, threshold=opt_thresh,
            multiclass_true=test_mul_targets, multiclass_probs=test_mul_probs,
            class_names=test_data.get("class_names")
        )

        # 3. Measure Test Inference Latency
        sample_in = torch.tensor(test_data["X"][:1], dtype=torch.float32).to(self.device)
        def infer_fn(x):
            with torch.no_grad():
                self.model(x)
        lat_info = measure_inference_latency(infer_fn, sample_in)

        # 4. Save Artifacts with Metadata & Checksum
        tag = f"{self.dataset_name.lower().replace('-', '_')}_{self.model.mode}"
        model_ckpt_path = os.path.join(self.checkpoint_dir, f"{tag}_best.pt")
        ckpt_payload = {
            "model_version": "2.0.0",
            "dataset": self.dataset_name,
            "seed": self.seed,
            "model_state_dict": self.model.state_dict(),
            "mode": self.model.mode,
            "in_features": self.model.in_features,
            "num_classes": self.model.num_classes,
            "n_qubits": self.model.n_qubits,
            "vqc_layers": getattr(self.model, "vqc_layers", 2),
            "entanglement": getattr(self.model, "entanglement", "ring"),
            "feature_schema_hash": train_data.get("feature_schema_hash", ""),
            "preprocessing_version": "2.0.0",
            "class_names": test_data.get("class_names", []),
            "optimal_threshold": opt_thresh,
            "validation_threshold_info": thresh_info,
            "validation_ece": round(ece_val, 4),
            "validation_reliability_bins": rel_bins,
            "calibration_comparison": cal_comp,
            "test_metrics": test_metrics,
            "history": history,
            "train_time_sec": train_time,
            "latency": lat_info
        }
        torch.save(ckpt_payload, model_ckpt_path)

        with open(model_ckpt_path, "rb") as f_chk:
            ckpt_checksum = hashlib.sha256(f_chk.read()).hexdigest()[:16]

        result_summary = {
            "dataset": self.dataset_name,
            "model": self.model.mode,
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
            "roc_auc": test_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"],
            "fpr": test_metrics["fpr"],
            "fnr": test_metrics["fnr"],
            "fp_count": test_metrics["counts"]["fp"],
            "fn_count": test_metrics["counts"]["fn"],
            "tp_count": test_metrics["counts"]["tp"],
            "tn_count": test_metrics["counts"]["tn"],
            "multiclass_accuracy": test_metrics.get("multiclass_accuracy", 0.0),
            "multiclass_macro_f1": test_metrics.get("multiclass_macro_f1", 0.0),
            "threshold": opt_thresh,
            "val_fpr": thresh_info["validation_fpr"],
            "val_fnr": thresh_info["validation_fnr"],
            "val_ece": round(ece_val, 4),
            "train_time_sec": train_time,
            "mean_latency_ms": lat_info["mean_latency_ms"],
            "p95_latency_ms": lat_info["p95_latency_ms"],
            "checkpoint_path": os.path.relpath(model_ckpt_path),
            "checkpoint_checksum": ckpt_checksum,
            "test_probs": test_bin_probs,
            "test_targets": test_bin_targets,
            "val_probs": val_bin_probs,
            "val_targets": val_bin_targets,
            "history": history
        }

        print(f"Final Test Result [{self.model.mode}] -> Acc: {test_metrics['accuracy']*100:.2f}%, F1: {test_metrics['macro_f1']*100:.2f}%, Recall: {test_metrics['recall']*100:.2f}%, FPR: {test_metrics['fpr']*100:.2f}%, FNR: {test_metrics['fnr']*100:.2f}%")
        return result_summary

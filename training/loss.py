"""
training/loss.py
Configurable Class-Balanced Cost-Sensitive Focal Loss for DQH-BAN Intrusion Detection:
L = lambda_1 * L_focal + lambda_2 * L_FN + lambda_3 * L_reg
Correctly balances positive/negative class weights based on training class distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CostSensitiveFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, pos_weight=1.0, lambda_focal=1.0, lambda_fn=0.25, lambda_reg=0.0001):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = float(pos_weight)
        self.lambda_focal = lambda_focal
        self.lambda_fn = lambda_fn
        self.lambda_reg = lambda_reg

    def forward(self, binary_logits, binary_targets, model=None):
        """
        binary_logits:  (Batch,) unnormalized logits
        binary_targets: (Batch,) 0 (benign) or 1 (attack)
        """
        targets = binary_targets.float()
        probs = torch.sigmoid(binary_logits)

        # Class-balanced BCE + Focal modulation
        pw = torch.tensor(self.pos_weight, dtype=binary_logits.dtype, device=binary_logits.device)
        bce = F.binary_cross_entropy_with_logits(binary_logits, targets, pos_weight=pw, reduction="none")
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * ((1.0 - p_t) ** self.gamma)
        loss_focal = (focal_weight * bce).mean()

        # Smooth False-Negative & False-Positive balanced margin surrogate
        fn_surrogate = targets * torch.square(1.0 - probs) + 0.5 * (1.0 - targets) * torch.square(probs)
        loss_fn = fn_surrogate.mean()

        loss_reg = 0.0
        if model is not None and self.lambda_reg > 0:
            for p in model.parameters():
                if p.requires_grad:
                    loss_reg = loss_reg + torch.sum(p ** 2)

        total_loss = (
            self.lambda_focal * loss_focal +
            self.lambda_fn * loss_fn +
            self.lambda_reg * loss_reg
        )

        return total_loss, {"loss_focal": float(loss_focal.item()), "loss_fn": float(loss_fn.item())}

class DualStreamCyberLoss(nn.Module):
    def __init__(self, num_classes=10, pos_weight=1.0, lambda_binary=1.0, lambda_multi=0.4, lambda_fn=0.25):
        super().__init__()
        self.binary_loss_fn = CostSensitiveFocalLoss(pos_weight=pos_weight, lambda_fn=lambda_fn)
        self.multi_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
        self.lambda_binary = lambda_binary
        self.lambda_multi = lambda_multi

    def forward(self, binary_logits, multi_logits, binary_targets, multi_targets, model=None):
        bin_loss, loss_dict = self.binary_loss_fn(binary_logits, binary_targets, model)
        mul_loss = self.multi_loss_fn(multi_logits, multi_targets)
        total_loss = self.lambda_binary * bin_loss + self.lambda_multi * mul_loss
        loss_dict["loss_multi"] = float(mul_loss.item())
        loss_dict["total_loss"] = float(total_loss.item())
        return total_loss, loss_dict

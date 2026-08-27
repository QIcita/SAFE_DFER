"""Standalone Aware Calibration Loss (ACL) for SAFE."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeCrossEntropy(nn.Module):
    def __init__(
        self,
        label_smoothing: float = 0.0,
        class_weight: Optional[Sequence[float] | torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if class_weight is not None and not isinstance(class_weight, torch.Tensor):
            class_weight = torch.tensor(class_weight, dtype=torch.float32)
        self.class_weight = class_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = (
            self.class_weight.to(logits.device)
            if self.class_weight is not None
            else None
        )
        return F.cross_entropy(
            logits,
            targets,
            weight=weight,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )


class ConfusionAwareCalibrationLoss(nn.Module):
    """CAL: confusing-class loss with margin and category-balance factors."""

    def __init__(
        self,
        num_classes: int,
        topk_confusing: int = 1,
        confusing_mode: str = "max",
        fuse_mode: str = "rms",
        alpha_margin: float = 1.0,
        alpha_balance: float = 1.0,
        balance_clamp_min: float = 0.1,
        balance_clamp_max: float = 10.0,
        detach_weights: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.topk_confusing = topk_confusing
        self.confusing_mode = confusing_mode
        self.fuse_mode = fuse_mode
        self.alpha_margin = alpha_margin
        self.alpha_balance = alpha_balance
        self.balance_clamp_min = balance_clamp_min
        self.balance_clamp_max = balance_clamp_max
        self.detach_weights = detach_weights
        self.eps = eps

    @staticmethod
    def _get_ground_truth_logit(
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return logits.gather(1, targets.unsqueeze(1)).squeeze(1)

    def _get_confusing_logit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.ones_like(logits, dtype=torch.bool, device=logits.device)
        mask.scatter_(1, targets.unsqueeze(1), False)
        non_target_logits = logits.masked_fill(~mask, float("-inf"))

        if self.topk_confusing == 1:
            return non_target_logits.max(dim=1)

        topk_logits, topk_idx = torch.topk(
            non_target_logits,
            k=self.topk_confusing,
            dim=1,
        )
        if self.confusing_mode == "max":
            return topk_logits[:, 0], topk_idx[:, 0]
        if self.confusing_mode == "logsumexp":
            return torch.logsumexp(topk_logits, dim=1), topk_idx[:, 0]
        raise ValueError(f"Unsupported confusing_mode: {self.confusing_mode}")

    @staticmethod
    def _margin_aware_factor(
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)
        p_y = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        return 1.0 - p_y

    def _category_balance_factor(
        self,
        targets: torch.Tensor,
        class_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if class_counts is None:
            class_counts = (
                torch.bincount(
                    targets,
                    minlength=self.num_classes,
                )
                .float()
                .to(targets.device)
            )
        else:
            class_counts = class_counts.float().to(targets.device)

        total_samples = class_counts.sum().clamp_min(1.0)
        factors = total_samples / (self.num_classes * class_counts.clamp_min(1.0))
        factors = torch.clamp(
            factors,
            min=self.balance_clamp_min,
            max=self.balance_clamp_max,
        )
        return factors[targets]

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_counts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        s_y = self._get_ground_truth_logit(logits, targets)
        s_mc, idx_mc = self._get_confusing_logit(logits, targets)

        # -log(exp(s_y) / (exp(s_y) + exp(s_mc))) = softplus(s_mc - s_y)
        loss_ca = F.softplus(s_mc - s_y)
        f_margin = self._margin_aware_factor(logits, targets)
        f_balance = self._category_balance_factor(targets, class_counts)

        if self.detach_weights:
            f_margin = f_margin.detach()
            f_balance = f_balance.detach()

        if self.fuse_mode == "rms":
            weighted = torch.sqrt(
                0.5
                * (
                    self.alpha_margin * (f_margin * loss_ca) ** 2
                    + self.alpha_balance * (f_balance * loss_ca) ** 2
                )
                + self.eps
            )
        elif self.fuse_mode == "weighted_sum":
            weighted = (
                self.alpha_margin * f_margin * loss_ca
                + self.alpha_balance * f_balance * loss_ca
            )
        else:
            raise ValueError(f"Unsupported fuse_mode: {self.fuse_mode}")

        loss_cal = weighted.mean()
        auxiliary = {
            "s_y": s_y.detach(),
            "s_mc": s_mc.detach(),
            "idx_mc": idx_mc.detach(),
            "loss_ca_mean": loss_ca.mean().detach(),
            "f_margin_mean": f_margin.mean().detach(),
            "f_balance_mean": f_balance.mean().detach(),
        }
        return loss_cal, auxiliary


class SupervisedContrastiveLoss(nn.Module):
    """FCL: supervised contrastive learning on video-level features."""

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.eps = eps

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.size(0)

        features = F.normalize(features, p=2, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        targets = targets.contiguous().view(-1, 1)
        positive_mask = torch.eq(targets, targets.T).float().to(device)
        logits_mask = torch.ones_like(positive_mask) - torch.eye(
            batch_size,
            device=device,
        )
        positive_mask = positive_mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_probability = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp_min(self.eps)
        )

        positive_count = positive_mask.sum(dim=1)
        valid_mask = positive_count > 0
        if valid_mask.sum() == 0:
            return features.sum() * 0.0

        mean_log_probability = (positive_mask * log_probability).sum(
            dim=1
        ) / positive_count.clamp_min(1.0)
        loss = -(self.temperature / self.base_temperature) * mean_log_probability
        return loss[valid_mask].mean()


class SAFE_ACL(nn.Module):
    """Complete SAFE objective: CE + lambda_cal * CAL + lambda_fcl * FCL."""

    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.0,
        lambda_cal: float = 0.05,
        lambda_fcl: float = 0.001,
        topk_confusing: int = 1,
        confusing_mode: str = "max",
        fuse_mode: str = "rms",
        alpha_margin: float = 1.0,
        alpha_balance: float = 1.0,
        balance_clamp_min: float = 0.1,
        balance_clamp_max: float = 10.0,
        contrast_temperature: float = 0.07,
        detach_weights: bool = True,
        class_weight: Optional[Sequence[float] | torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.ce_loss = SafeCrossEntropy(label_smoothing, class_weight)
        self.cal_loss = ConfusionAwareCalibrationLoss(
            num_classes=num_classes,
            topk_confusing=topk_confusing,
            confusing_mode=confusing_mode,
            fuse_mode=fuse_mode,
            alpha_margin=alpha_margin,
            alpha_balance=alpha_balance,
            balance_clamp_min=balance_clamp_min,
            balance_clamp_max=balance_clamp_max,
            detach_weights=detach_weights,
        )
        self.fcl_loss = SupervisedContrastiveLoss(temperature=contrast_temperature)
        self.lambda_cal = lambda_cal
        self.lambda_fcl = lambda_fcl

    def forward(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor,
        class_counts: Optional[torch.Tensor] = None,
        lambda_cal: Optional[float] = None,
        lambda_fcl: Optional[float] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        lam_cal = self.lambda_cal if lambda_cal is None else lambda_cal
        lam_fcl = self.lambda_fcl if lambda_fcl is None else lambda_fcl

        loss_ce = self.ce_loss(logits, targets)
        loss_cal, cal_aux = self.cal_loss(logits, targets, class_counts)
        loss_fcl = self.fcl_loss(features, targets)
        total_loss = loss_ce + lam_cal * loss_cal + lam_fcl * loss_fcl

        loss_dict = {
            "loss_total": total_loss.detach(),
            "loss_ce": loss_ce.detach(),
            "loss_cal": loss_cal.detach(),
            "loss_fcl": loss_fcl.detach(),
            "lambda_cal": torch.tensor(lam_cal, device=logits.device),
            "lambda_fcl": torch.tensor(lam_fcl, device=logits.device),
        }
        loss_dict.update(cal_aux)
        return total_loss, loss_dict


# Paper-facing module name.
ACL = SAFE_ACL

__all__ = [
    "SafeCrossEntropy",
    "ConfusionAwareCalibrationLoss",
    "SupervisedContrastiveLoss",
    "SAFE_ACL",
    "ACL",
]

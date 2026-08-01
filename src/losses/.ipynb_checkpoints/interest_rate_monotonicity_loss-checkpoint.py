"""Interest-rate sensitivity loss."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.losses.base_loss import BaseLoss


class InterestRateMonotonicityLoss(BaseLoss):
    """Match d(V/S)/dr to the classical solver sensitivity target."""

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        del model, context

        target_mean = batch["target_mean"].to(predictions.device)
        target_std = batch["target_std"].to(predictions.device)
        target_sensitivity = self.require_batch_tensor(
            batch,
            "interest_rate_sensitivity_target",
        ).to(predictions.device)

        reserve_ratio = predictions * target_std + target_mean
        reserve_ratio_sensitivity = self.first_derivative(
            predictions=reserve_ratio,
            batch=batch,
            name="scenario_interest_rate",
        )
        return F.huber_loss(
            reserve_ratio_sensitivity,
            target_sensitivity,
            reduction=self.reduction,
            delta=0.01,
        )

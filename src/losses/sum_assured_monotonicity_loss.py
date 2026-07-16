"""Sum-assured monotonicity loss.

Created: 2026-06-10  Revised: 2026-06-11
Purpose: Enforce dV/dS >= 0 via finite-difference perturbation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.data.dataset import FEATURE_INDEX, FEATURE_SCALES
from src.losses.base_loss import BaseLoss

_S_IDX = FEATURE_INDEX["sum_assured"]
# Perturbation: £5,000 change / scale
_DELTA = 5_000.0 / FEATURE_SCALES["sum_assured"]


class SumAssuredMonotonicityLoss(BaseLoss):
    """Enforce dV/dS >= 0 using finite-difference output comparison.

    Why finite difference:
        The sum_assured feature has the largest raw scale (£950k range) so
        its normalised gradient dz/d(S_norm) is small relative to its actual
        economic importance. Direct output comparison gives a clear signal.
    """

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        del context
        features = batch["features"]
        raw_sum_assured = batch["raw_features"][:, _S_IDX : _S_IDX + 1]
        target_mean = batch["target_mean"].to(predictions.device)
        target_std = batch["target_std"].to(predictions.device)

        features_high = features.clone()
        features_high[:, _S_IDX] = features_high[:, _S_IDX] + _DELTA

        pred_high = model(features_high)
        sum_assured_high = raw_sum_assured + 5_000.0

        reserve_base = (predictions * target_std + target_mean) * raw_sum_assured
        reserve_high = (pred_high * target_std + target_mean) * sum_assured_high

        # dV/dS < 0 is a violation — penalise relu(V(S) - V(S+Δ))
        violation = reserve_base - reserve_high
        return self.reduce(torch.relu(violation) * 10.0)

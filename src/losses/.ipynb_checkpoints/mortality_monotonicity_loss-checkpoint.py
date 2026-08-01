"""Mortality monotonicity loss.

Created: 2026-06-03  Revised: 2026-06-11
Purpose: Enforce dV/dμ >= 0 via finite-difference perturbation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.data.dataset import FEATURE_INDEX, FEATURE_SCALES
from src.losses.base_loss import BaseLoss

_MU_IDX = FEATURE_INDEX["mortality"]
# Perturbation: 0.001 absolute mortality intensity change / scale
_DELTA = 0.001 / FEATURE_SCALES["mortality"]


class MortalityMonotonicityLoss(BaseLoss):
    """Enforce dV/dμ >= 0 using finite-difference output comparison.

    Why finite difference:
        Autograd dz/d(mu_norm) is negligibly small in the standardised output
        space, giving near-zero weight updates. Finite difference gives a direct,
        magnitude-proportional signal: the more the model wrongly predicts a
        lower reserve at higher mortality, the larger the penalty.
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

        # Perturb mortality upward by Δμ
        features_high = features.clone()
        features_high[:, _MU_IDX] = features_high[:, _MU_IDX] + _DELTA

        pred_high = model(features_high)

        # dV/dμ < 0 is a violation — penalise relu(V(μ) - V(μ+Δ))
        violation = predictions - pred_high
        return self.reduce(torch.relu(violation) * 10.0)
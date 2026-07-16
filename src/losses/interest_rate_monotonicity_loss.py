"""Interest-rate monotonicity loss.

Created: 2026-06-03  Revised: 2026-07-02
Purpose: Match solver-implied local reserve sensitivity to interest rates using
a robust supervised derivative objective.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.losses.base_loss import BaseLoss


class InterestRateMonotonicityLoss(BaseLoss):
    """Match PINN interest sensitivity to the classical actuarial benchmark.

    Scientific Context:
        For standard term insurance under Thiele dynamics, a higher discount
        rate generally lowers the present value of future insurer liabilities.
        The desired local behaviour can be expressed as:

            d(V / S) / dr

        where ``V`` is reserve, ``S`` is sum assured, and ``r`` is the
        interest-rate assumption. Rather than only constraining the sign, this
        loss learns the actual local slope implied by the classical solver:

            g_thiele(t) ≈ ((V(r + Δr) / S) - (V(r - Δr) / S)) / (2Δr)

        The model predicts a standardized reserve ratio ``z`` such that:

            v = V / S = z * sigma + mu

        so autodiff on ``v`` gives a directly comparable normalized slope:

            g_pinn = d(v) / dr

        The loss is a Huber penalty:

            L_rate = huber(g_pinn - g_thiele)

        which is less brittle than plain MSE when a few policies have steeper
        sensitivities than the rest.

    Business Interpretation:
        This objective teaches the model not just that reserves should usually
        fall when rates rise, but by how much. That makes the learned surrogate
        more useful for market stress testing, ALM sensitivity analysis, and
        interactive reserve dashboards where slope realism matters.
    """

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
        target_sensitivity = self.require_batch_tensor(batch, "interest_rate_sensitivity_target").to(predictions.device)

        reserve_ratio = predictions * target_std + target_mean
        reserve_ratio_sensitivity = self.first_derivative(
            reserve_ratio,
            batch=batch,
            name="scenario_interest_rate",
        )
        return F.huber_loss(reserve_ratio_sensitivity, target_sensitivity, reduction=self.reduction, delta=0.01)

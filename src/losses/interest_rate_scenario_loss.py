"""Interest-rate scenario loss.

Created: 2026-07-02
Purpose: Match Thiele shocked reserve curves under fixed-policy interest-rate
perturbations.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.data.dataset import FEATURE_INDEX, FEATURE_SCALES
from src.losses.base_loss import BaseLoss

_R_IDX = FEATURE_INDEX["scenario_interest_rate"]


class InterestRateScenarioLoss(BaseLoss):
    """Match classical shocked reserve curves for fixed-policy rate scenarios.

    Scientific Context:
        The business-facing diagnostic that failed was not a local derivative at
        one time point. It was a scenario curve: hold policy attributes fixed,
        shock the interest-rate assumption, solve Thiele, and compare the
        resulting reserve path or peak reserve behavior.

        This loss therefore supervises the model on two local rate scenarios for
        each base record:

            v_down(t) = V(t; r - Δr) / S
            v_up(t)   = V(t; r + Δr) / S

        where ``V`` is reserve, ``r`` is the issue-rate or scenario-rate
        assumption, ``Δr`` is a small deterministic shock, and ``S`` is sum
        assured.

        The neural model predicts a standardized reserve ratio ``z`` and is
        converted back to reserve-ratio space via:

            v = z * sigma + mu

        The loss compares predicted shocked reserve ratios against classical
        shocked reserve ratios using a Huber penalty on both up and down
        scenarios. This is more aligned with the actual rate-sensitivity plot
        than penalizing only the sign of ``dV/dr``.

    Business Interpretation:
        This teaches the surrogate what the reserve curve should look like when
        treasury, ALM, or risk users move the rate assumption up or down while
        keeping the contract itself unchanged.
    """

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        """Compute the shocked-curve matching penalty.

        Args:
            model: Reserve model to evaluate under shocked interest rates.
            batch: Batch payload containing normalized inputs and classical
                shocked reserve-ratio targets.
            predictions: Base-model predictions for the unshocked inputs.
            context: Shared mutable context dictionary.

        Returns:
            torch.Tensor: Scalar scenario loss.
        """

        del predictions, context
        features = self.require_batch_tensor(batch, "features")
        target_mean = self.require_batch_tensor(batch, "target_mean").to(features.device)
        target_std = self.require_batch_tensor(batch, "target_std").to(features.device)
        shock_delta = self.require_batch_tensor(batch, "interest_rate_shock_delta").to(features.device)
        target_down = self.require_batch_tensor(batch, "interest_rate_shock_down_target").to(features.device)
        target_up = self.require_batch_tensor(batch, "interest_rate_shock_up_target").to(features.device)

        delta_normalized = shock_delta / FEATURE_SCALES["scenario_interest_rate"]

        features_down = features.clone()
        features_down[:, _R_IDX : _R_IDX + 1] = features_down[:, _R_IDX : _R_IDX + 1] - delta_normalized
        features_up = features.clone()
        features_up[:, _R_IDX : _R_IDX + 1] = features_up[:, _R_IDX : _R_IDX + 1] + delta_normalized

        pred_down = model(features_down)
        pred_up = model(features_up)

        reserve_ratio_down = pred_down * target_std + target_mean
        reserve_ratio_up = pred_up * target_std + target_mean

        loss_down = F.huber_loss(reserve_ratio_down, target_down, reduction=self.reduction, delta=0.01)
        loss_up = F.huber_loss(reserve_ratio_up, target_up, reduction=self.reduction, delta=0.01)
        return 0.5 * (loss_down + loss_up)

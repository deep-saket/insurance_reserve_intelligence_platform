"""Interest-rate peak reserve loss.

Created: 2026-07-02
Purpose: Match classical shocked peak reserves for fixed-policy
interest-rate scenarios.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.data.dataset import FEATURE_INDEX, FEATURE_SCALES
from src.losses.base_loss import BaseLoss

_T_IDX = FEATURE_INDEX["time"]
_R_IDX = FEATURE_INDEX["scenario_interest_rate"]
_MU_IDX = FEATURE_INDEX["mortality"]


class InterestRatePeakLoss(BaseLoss):
    """Match Thiele peak reserves under fixed-policy rate shocks.

    Scientific Context:
        The observed failure mode is a policy-level scenario diagnostic:
        for a fixed contract, the peak reserve produced by the neural surrogate
        across an interest-rate sweep does not match the classical Thiele peak
        reserve curve.

        This loss therefore supervises the model directly on shocked peak
        reserve points rather than only on local derivatives or random
        time-point rows. For each policy record, the dataset stores:

        - the classical peak time under ``r - Δr``
        - the classical peak mortality at that time
        - the classical shocked peak reserve ratio ``max_t V(t; r - Δr) / S``
        - the same quantities under ``r + Δr``

        The model is evaluated at those shocked peak coordinates and compared in
        reserve-ratio space:

            v = z * sigma + mu

        The loss is:

            L_peak = 0.5 * huber(v_down_pred, v_down_thiele)
                   + 0.5 * huber(v_up_pred,   v_up_thiele)

        This is a closer proxy for the business chart than pointwise shocked
        curves because it directly targets the peak-reserve quantities plotted
        in the rate-sensitivity analysis.

    Business Interpretation:
        This loss teaches the model the key output used by treasury, ALM, and
        reserve governance stakeholders: "for this policy, what is the maximum
        liability burden under lower or higher interest-rate scenarios?"
    """

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        """Compute the policy-level shocked peak reserve penalty.

        Args:
            model: Reserve model evaluated at shocked peak coordinates.
            batch: Batch payload containing base features and classical peak
                targets under shocked rate scenarios.
            predictions: Base predictions for the unshocked batch.
            context: Shared mutable context dictionary.

        Returns:
            torch.Tensor: Scalar peak-reserve loss.
        """

        del predictions, context
        features = self.require_batch_tensor(batch, "features")
        target_mean = self.require_batch_tensor(batch, "target_mean").to(features.device)
        target_std = self.require_batch_tensor(batch, "target_std").to(features.device)
        shock_delta = self.require_batch_tensor(batch, "interest_rate_shock_delta").to(features.device)

        down_peak_time = self.require_batch_tensor(batch, "interest_rate_shock_down_peak_time").to(features.device)
        up_peak_time = self.require_batch_tensor(batch, "interest_rate_shock_up_peak_time").to(features.device)
        down_peak_mortality = self.require_batch_tensor(batch, "interest_rate_shock_down_peak_mortality").to(features.device)
        up_peak_mortality = self.require_batch_tensor(batch, "interest_rate_shock_up_peak_mortality").to(features.device)
        down_peak_target = self.require_batch_tensor(batch, "interest_rate_shock_down_peak_target").to(features.device)
        up_peak_target = self.require_batch_tensor(batch, "interest_rate_shock_up_peak_target").to(features.device)

        delta_normalized = shock_delta / FEATURE_SCALES["scenario_interest_rate"]

        features_down_peak = features.clone()
        features_down_peak[:, _T_IDX : _T_IDX + 1] = down_peak_time / FEATURE_SCALES["time"]
        features_down_peak[:, _R_IDX : _R_IDX + 1] = features_down_peak[:, _R_IDX : _R_IDX + 1] - delta_normalized
        features_down_peak[:, _MU_IDX : _MU_IDX + 1] = down_peak_mortality / FEATURE_SCALES["mortality"]

        features_up_peak = features.clone()
        features_up_peak[:, _T_IDX : _T_IDX + 1] = up_peak_time / FEATURE_SCALES["time"]
        features_up_peak[:, _R_IDX : _R_IDX + 1] = features_up_peak[:, _R_IDX : _R_IDX + 1] + delta_normalized
        features_up_peak[:, _MU_IDX : _MU_IDX + 1] = up_peak_mortality / FEATURE_SCALES["mortality"]

        pred_down_peak = model(features_down_peak)
        pred_up_peak = model(features_up_peak)

        reserve_ratio_down_peak = pred_down_peak * target_std + target_mean
        reserve_ratio_up_peak = pred_up_peak * target_std + target_mean

        loss_down = F.huber_loss(reserve_ratio_down_peak, down_peak_target, reduction=self.reduction, delta=0.01)
        loss_up = F.huber_loss(reserve_ratio_up_peak, up_peak_target, reduction=self.reduction, delta=0.01)
        return 0.5 * (loss_down + loss_up)

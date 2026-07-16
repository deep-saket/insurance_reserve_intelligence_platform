"""PDE residual loss in v=V/S z-space."""
from __future__ import annotations
from typing import Any
import torch
from torch import nn
from src.losses.base_loss import BaseLoss

class PDEResidualLoss(BaseLoss):
    """Thiele ODE: dz/dt = r*(z+μ/σ) + P/(S·σ) - μ_mort·(1-z-μ/σ)"""
    def forward(self, model, batch, predictions, context):
        del model
        dz_dt   = self.first_derivative(predictions=predictions, batch=batch, name="time")
        r       = self.raw_feature(batch, "scenario_interest_rate")
        P       = self.raw_feature(batch, "premium")
        S       = self.raw_feature(batch, "sum_assured").clamp(min=1.0)
        mu_mort = self.raw_feature(batch, "mortality")
        t_mean  = batch["target_mean"].to(predictions.device)
        t_std   = batch["target_std"].to(predictions.device)
        z_offset = predictions + t_mean / t_std
        survival_gap = ((1.0 - t_mean) / t_std) - predictions
        residual = dz_dt - r * z_offset - P / (S * t_std) + mu_mort * survival_gap
        context["pde_residual"] = residual
        return self.reduce(residual.pow(2))

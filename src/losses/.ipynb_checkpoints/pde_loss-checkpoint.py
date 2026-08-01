"""PDE residual loss in v=V/S z-space."""

from __future__ import annotations

from src.losses.base_loss import BaseLoss


class PDEResidualLoss(BaseLoss):
    """Thiele residual for standardized reserve-ratio predictions."""

    def forward(self, model, batch, predictions, context):
        del model

        dz_dt = self.first_derivative(
            predictions=predictions,
            batch=batch,
            name="time",
        )
        r = self.raw_feature(batch, "scenario_interest_rate")
        premium_ratio = self.raw_feature(batch, "premium_ratio")
        mu = self.raw_feature(batch, "mortality")
        target_mean = batch["target_mean"].detach().to(predictions.device)
        target_std = batch["target_std"].detach().to(predictions.device).clamp_min(1e-8)

        v = predictions * target_std + target_mean
        dv_dt = r * v + premium_ratio - mu * (1.0 - v)
        residual = dz_dt - (dv_dt / target_std)
        context["pde_residual"] = residual
        return self.reduce(residual.pow(2))

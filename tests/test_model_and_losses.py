"""Model and loss tests.

Created: 2026-05-31
Purpose: Validate reserve model forward passes and config-driven loss behavior.
"""

from __future__ import annotations

import torch

from src.losses.total_loss import TotalLoss
from src.models.factory import ModelFactory
from src.utils.config import LossConfig, LossSettingsConfig, ModelConfig


def test_model_forward_and_total_loss() -> None:
    """Verify that the reserve model and total loss produce valid tensor outputs."""
    model = ModelFactory.create_pinn(ModelConfig())
    raw_features = torch.rand(8, 7)
    raw_features[:, 0] *= 30.0
    raw_features[:, 1] *= 45.0
    raw_features[:, 2] *= 0.1
    raw_features[:, 3] *= 0.1
    raw_features[:, 4] *= 2_000.0
    raw_features[:, 5] *= 950_000.0
    raw_features[:, 6] *= 0.05

    features = raw_features.clone()
    features[:, 0] /= 30.0
    features[:, 1] /= 45.0
    features[:, 2] /= 0.07
    features[:, 3] /= 0.07
    features[:, 4] /= 2_000.0
    features[:, 5] /= 950_000.0
    features[:, 6] /= 0.05
    features.requires_grad_(True)

    batch = {
        "features": features,
        "raw_features": raw_features,
        "target": torch.rand(8, 1),
        "sum_assured_scale": raw_features[:, 5:6].clamp(min=1.0),
        "target_mean": torch.zeros(8, 1),
        "target_std": torch.ones(8, 1),
        "term": torch.ones(8, 1) * 10.0,
        "terminal_mortality": raw_features[:, 6:7],
    }
    loss_fn = TotalLoss(LossConfig(), settings=LossSettingsConfig(reduction="mean", use_adaptive_weights=False))
    predictions = model(features)
    breakdown = loss_fn(model=model, batch=batch, predictions=predictions, context={})
    assert breakdown["total_loss"].item() >= 0.0
    assert "data_loss" in breakdown["components"]
    assert "pde_residual" in breakdown["context"]
    assert breakdown["context"]["pde_residual"].shape == (8, 1)

"""Boundary loss — V(T)=0 → z_T = -μ/σ."""
from __future__ import annotations
from typing import Any
import torch
from torch import nn
from src.data.dataset import FEATURE_INDEX, FEATURE_SCALES
from src.losses.base_loss import BaseLoss

class BoundaryLoss(BaseLoss):
    def forward(self, model, batch, predictions, context):
        features = self.require_batch_tensor(batch, "features")
        terms    = self.require_batch_tensor(batch, "term")
        terminal_mortality = self.require_batch_tensor(batch, "terminal_mortality")
        t_mean   = batch["target_mean"].to(features.device)
        t_std    = batch["target_std"].to(features.device)
        bf = features.clone()
        bf[:, FEATURE_INDEX["time"]:FEATURE_INDEX["time"]+1] = terms / FEATURE_SCALES["time"]
        bf[:, FEATURE_INDEX["mortality"]:FEATURE_INDEX["mortality"]+1] = (
            terminal_mortality / FEATURE_SCALES["mortality"]
        )
        z_boundary = model(bf)
        z_target   = -t_mean / t_std
        context["boundary_predictions"] = z_boundary
        return self.reduce((z_boundary - z_target).pow(2))

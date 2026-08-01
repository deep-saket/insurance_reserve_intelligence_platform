"""Base neural model interfaces.

Created: 2026-05-31
Purpose: Define abstract interfaces for reserve prediction models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseReserveModel(nn.Module, ABC):
    """Abstract reserve prediction model.

    Attributes:
        input_features: Ordered feature names expected by the model.
    """

    input_features: tuple[str, ...] = ("time", "age", "pricing_interest_rate", "scenario_interest_rate", "premium_ratio", "sum_assured", "mortality")

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict reserves for input features.

        Args:
            features: Input tensor containing policy and time features.

        Returns:
            torch.Tensor: Predicted reserve values.
        """




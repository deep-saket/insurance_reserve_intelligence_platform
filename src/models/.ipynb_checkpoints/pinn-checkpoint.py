"""PINN reserve model.

Created: 2026-05-31
Purpose: Define the physics-informed reserve model built on the MLP backbone.
"""

from __future__ import annotations

import torch

from src.models.base_model import BaseReserveModel
from src.models.mlp import MLP


class PINNReserveModel(BaseReserveModel):
    """Physics-informed reserve estimator implemented with an MLP backbone."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
        dropout: float,
        skip_connections: bool = False,
    ) -> None:
        """Initialize the reserve PINN.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Width of hidden layers.
            num_layers: Total number of linear layers including the output layer.
            activation: Activation function name.
            dropout: Dropout probability between hidden layers.
            skip_connections: Whether residual skip connections should be used in
                the hidden backbone.
        """
        super().__init__()
        self.backbone = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
            skip_connections=skip_connections,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict reserve values.

        Args:
            features: Input feature tensor.

        Returns:
            torch.Tensor: Predicted reserve tensor.
        """

        return self.backbone(features)

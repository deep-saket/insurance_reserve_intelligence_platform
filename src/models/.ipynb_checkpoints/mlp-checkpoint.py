"""Generic multilayer perceptron blocks.

Created: 2026-05-31
Purpose: Provide configurable feed-forward neural network building blocks.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def activation_factory(name: str) -> Callable[[], nn.Module]:
    """Map configuration strings to activation layers.

    Args:
        name: Activation function name from configuration.

    Returns:
        Callable[[], nn.Module]: Activation layer constructor.

    Raises:
        ValueError: If the activation name is unsupported.
    """

    registry: dict[str, Callable[[], nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    key = name.lower()
    if key not in registry:
        raise ValueError(f"Unsupported activation: {name}")
    return registry[key]


class MLP(nn.Module):
    """Configurable feed-forward network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
        dropout: float,
        skip_connections: bool = False,
    ) -> None:
        """Initialize the MLP backbone.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Width of hidden layers.
            num_layers: Total number of linear layers including the output layer.
            activation: Activation function name.
            dropout: Dropout probability between hidden layers.
            skip_connections: Whether residual skip connections should be used
                between hidden layers when shapes are compatible.

        Raises:
            ValueError: If fewer than two layers are requested.
        """
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        act = activation_factory(activation)
        self.skip_connections = skip_connections
        self.hidden_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        current_dim = input_dim
        self.projections = nn.ModuleList()   # 1x1 projections for skip when dims differ
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(current_dim, hidden_dim))
            self.activations.append(act())
            self.dropouts.append(nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())
            # If input and output dims differ, project the residual so skip always works
            if skip_connections and current_dim != hidden_dim:
                self.projections.append(nn.Linear(current_dim, hidden_dim, bias=False))
            else:
                self.projections.append(nn.Identity())
            current_dim = hidden_dim
        self.output_layer = nn.Linear(current_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            features: Input feature tensor.

        Returns:
            torch.Tensor: Network output tensor.
        """

        hidden = features
        for layer, activation, dropout, proj in zip(
            self.hidden_layers, self.activations, self.dropouts, self.projections
        ):
            residual = hidden
            hidden = dropout(activation(layer(hidden)))
            if self.skip_connections:
                hidden = hidden + proj(residual)   # proj is Identity when dims match
        return self.output_layer(hidden)
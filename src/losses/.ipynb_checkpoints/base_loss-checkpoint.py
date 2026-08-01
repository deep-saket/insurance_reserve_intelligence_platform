"""Base utilities for reserve loss functions.

Created: 2026-06-03
Purpose: Provide a common loss interface and derivative helpers for PINN and KINN objectives.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from src.data.dataset import FEATURE_INDEX, FEATURE_SCALES


class BaseLoss(nn.Module, ABC):
    """Abstract base class for all reserve-learning loss functions.

    Scientific Context:
        The reserve model mixes supervised terms, physics constraints, and
        actuarial knowledge constraints. Each concrete loss follows the same
        interface so the experiment engine can assemble objectives from YAML.

    Business Interpretation:
        Each loss is a governance rule. Some rules enforce fit to benchmark
        reserves, while others enforce economically sensible reserve behavior
        under mortality, age, interest-rate, and contractual constraints.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Initialize the base loss helper.

        Args:
            reduction: Scalar reduction to apply. Supported values are ``mean``
                and ``sum``.
        """
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction '{reduction}'. Available reductions: mean, sum")
        self.reduction = reduction

    @abstractmethod
    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        """Compute one scalar loss term."""

    def reduce(self, values: torch.Tensor) -> torch.Tensor:
        """Reduce a tensor of penalties to a scalar.

        Args:
            values: Element-wise penalty tensor.

        Returns:
            torch.Tensor: Reduced scalar tensor.
        """

        if self.reduction == "mean":
            return values.mean()
        return values.sum()

    @staticmethod
    def require_batch_tensor(batch: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        """Fetch a required tensor from a batch payload.

        Args:
            batch: Batch dictionary produced by the dataset and trainer.
            key: Required tensor name.

        Returns:
            torch.Tensor: Requested tensor.

        Raises:
            KeyError: If the batch does not contain the required tensor.
        """

        if key not in batch:
            raise KeyError(f"Batch is missing required tensor '{key}'. Available keys: {sorted(batch.keys())}")
        return batch[key]

    @staticmethod
    def raw_feature(batch: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        """Retrieve one feature in real actuarial units.

        Args:
            batch: Batch dictionary with either ``raw_features`` or normalized
                ``features``.
            name: Canonical feature name.

        Returns:
            torch.Tensor: Feature column in real-world units.
        """

        index = FEATURE_INDEX[name]
        if "raw_features" in batch:
            return batch["raw_features"][:, index : index + 1]
        return batch["features"][:, index : index + 1] * FEATURE_SCALES[name]

    @staticmethod
    def normalized_feature(batch: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        """Retrieve one model input feature in normalized units.

        Args:
            batch: Batch dictionary containing normalized model inputs.
            name: Canonical feature name.

        Returns:
            torch.Tensor: Feature column in normalized model-input units.
        """

        index = FEATURE_INDEX[name]
        return batch["features"][:, index : index + 1]

    @staticmethod
    def _gradient_matrix(predictions: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Compute first derivatives with respect to normalized features.

        Args:
            predictions: Reserve predictions.
            features: Normalized feature matrix with gradients enabled.

        Returns:
            torch.Tensor: Jacobian rows for each batch element.
        """

        return torch.autograd.grad(
            outputs=predictions,
            inputs=features,
            grad_outputs=torch.ones_like(predictions),
            create_graph=True,
            retain_graph=True,
        )[0]

    def first_derivative(self, predictions: torch.Tensor, batch: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        """Compute a first derivative in real actuarial units.

        Args:
            predictions: Reserve predictions.
            batch: Batch dictionary containing normalized features.
            name: Variable name such as ``time`` or ``mortality``.

        Returns:
            torch.Tensor: ``dV/d(variable)`` in real-world units.
        """

        features = self.require_batch_tensor(batch, "features")

        grads = self._gradient_matrix(
            predictions,
            features,
        )
        
        if name == "scenario_interest_rate":
            scale = float(batch["interest_std"][0])
        else:
            scale = FEATURE_SCALES[name]
        
        index = FEATURE_INDEX[name]
        
        return grads[:, index:index+1] / scale

    def second_derivative(self, predictions: torch.Tensor, batch: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        """Compute a second derivative in real actuarial units.

        Args:
            predictions: Reserve predictions.
            batch: Batch dictionary containing normalized features.
            name: Variable name such as ``time``.

        Returns:
            torch.Tensor: ``dÂ²V/d(variable)Â²`` in real-world units.
        """

        features = self.require_batch_tensor(batch, "features")
        first_grads = self._gradient_matrix(predictions, features)
        index = FEATURE_INDEX[name]
        second_grads = torch.autograd.grad(
            outputs=first_grads[:, index : index + 1],
            inputs=features,
            grad_outputs=torch.ones_like(first_grads[:, index : index + 1]),
            create_graph=True,
            retain_graph=True,
        )[0]
        if name == "scenario_interest_rate":
            scale = float(batch["interest_std"][0])
        else:
            scale = FEATURE_SCALES[name]
        return second_grads[:, index : index + 1] / (scale**2)


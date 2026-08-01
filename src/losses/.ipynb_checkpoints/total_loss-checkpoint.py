"""Config-driven total-loss assembly.

Created: 2026-06-03
Purpose: Build experiment-specific PINN, KINN, and hybrid objectives from YAML configuration.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.losses.registry import LOSS_REGISTRY
from src.utils.config import LossConfig, LossSettingsConfig


class TotalLoss(nn.Module):
    """Assemble the configured reserve-learning objective.

    Scientific Context:
        This module converts YAML into an executable objective by composing data,
        PDE, boundary, monotonicity, solvency, smoothness, portfolio, and
        regularization terms.

    Business Interpretation:
        This is the experiment switchboard that allows the same model to be run
        as a pure PINN, a knowledge-informed neural network, a hybrid model, or
        a supervised-only reserve model.
    """

    def __init__(self, config: LossConfig, settings: LossSettingsConfig | None = None) -> None:
        """Initialize the config-driven total loss.

        Args:
            config: Named loss configuration.
            settings: Shared loss settings.

        Raises:
            ValueError: If the configuration references unknown loss names or if
                an enabled loss has no weight.
            NotImplementedError: If adaptive weighting is requested.
        """

        super().__init__()
        self.config = config
        self.settings = settings or LossSettingsConfig()
        if self.settings.use_adaptive_weights:
            raise NotImplementedError(
                "Adaptive loss weighting is not implemented in this release. "
                "Set loss_settings.use_adaptive_weights=false."
            )
        self._validate_config()
        self.losses = nn.ModuleDict(
            {
                name: LOSS_REGISTRY[name](reduction=self.settings.reduction)
                for name, term in self.config.terms.items()
                if term.enabled
            }
        )
        self.active_loss_names = list(self.losses.keys())

    def _validate_config(self) -> None:
        """Validate configured loss names and weights."""

        configured_names = set(self.config.terms.keys())
        available_names = set(LOSS_REGISTRY.keys())
        unknown_names = sorted(configured_names - available_names)
        if unknown_names:
            raise ValueError(
                "Configured loss names are not registered: "
                f"{unknown_names}. Available loss names: {sorted(available_names)}"
            )

        missing_weight_names = sorted(
            name for name, term in self.config.terms.items() if term.enabled and term.weight is None
        )
        if missing_weight_names:
            raise ValueError(
                "Enabled losses must define a weight. Missing weights for: "
                f"{missing_weight_names}"
            )

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute the weighted total loss and its components.

        Args:
            model: Reserve model being optimized.
            batch: Batch dictionary containing model inputs and targets.
            predictions: Current reserve predictions for the batch.
            context: Optional shared dictionary for diagnostics.

        Returns:
            dict[str, Any]: Total loss tensor plus component breakdowns.
        """

        local_context = {} if context is None else context
        raw_components: dict[str, torch.Tensor] = {}
        weighted_components: dict[str, torch.Tensor] = {}
        total_loss = predictions.new_tensor(0.0)

        for name, loss_module in self.losses.items():
            raw_value = loss_module(model=model, batch=batch, predictions=predictions, context=local_context)
            weight = self.config.terms[name].weight
            if weight is None:
                raise ValueError(f"Enabled loss '{name}' does not define a weight.")
            weighted_value = raw_value * float(weight)
            raw_components[name] = raw_value
            weighted_components[name] = weighted_value
            total_loss = total_loss + weighted_value

        return {
            "total_loss": total_loss,
            "components": raw_components,
            "weighted_components": weighted_components,
            "context": local_context,
        }

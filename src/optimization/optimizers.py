"""Shared optimizer result contracts.

Created: 2026-07-02
Purpose: Provide structured outputs for scipy-based optimization wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Structured business optimization result.

    Business Interpretation:
        The optimizer returns both the decision recommendation and the business
        objective value, keeping numerical optimizer details separate from the
        actuarial decision layer.
    """

    objective_name: str
    optimal_values: dict[str, float]
    objective_value: float
    success: bool
    method: str
    message: str = ""
    diagnostics: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)

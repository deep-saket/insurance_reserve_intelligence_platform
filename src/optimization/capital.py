"""Capital allocation optimization workflows.

Created: 2026-07-02
Purpose: Allocate capital subject to reserve and solvency business constraints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.optimize import minimize_scalar

from src.actuarial.policy import Policy
from src.optimization.objectives import reserve, solvency_ratio
from src.optimization.optimizers import OptimizationResult

if TYPE_CHECKING:
    from src.optimization.predictor import ReservePredictor


def optimize_capital_allocation(
    policy: Policy,
    predictor: ReservePredictor,
    capital_bounds: tuple[float, float] | None = None,
    solvency_threshold: float = 1.5,
    time_point: float = 0.0,
    penalty_weight: float = 1_000_000.0,
) -> OptimizationResult:
    """Minimize allocated capital subject to solvency ratio discipline."""

    reserve_amount = max(reserve(policy=policy, predictor=predictor, time_point=time_point), 0.0)
    lower, upper = capital_bounds or (0.0, max(reserve_amount * solvency_threshold * 3.0, 1.0))
    history: list[dict[str, float]] = []

    def objective(capital_value: float) -> float:
        ratio = solvency_ratio(
            policy=policy,
            predictor=predictor,
            capital=float(capital_value),
            time_point=time_point,
        )
        solvency_shortfall = max(0.0, float(solvency_threshold) - ratio)
        reserve_shortfall = max(0.0, reserve_amount - float(capital_value))
        objective_value = float(capital_value) + penalty_weight * (
            solvency_shortfall**2 + reserve_shortfall**2
        )
        history.append(
            {
                "iteration": float(len(history)),
                "objective": objective_value,
                "premium": float(policy.premium),
                "reserve": reserve_amount,
                "capital": float(capital_value),
                "profit": 0.0,
                "solvency_ratio": ratio,
            }
        )
        return objective_value

    result = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
    capital_value = float(result.x)
    ratio = solvency_ratio(
        policy=policy,
        predictor=predictor,
        capital=capital_value,
        time_point=time_point,
    )
    return OptimizationResult(
        objective_name="capital_allocation",
        optimal_values={"capital": capital_value},
        objective_value=capital_value,
        success=bool(result.success and ratio >= solvency_threshold),
        method="scipy_minimize_scalar_bounded_penalty",
        message=str(result.message),
        diagnostics={
            "reserve": reserve_amount,
            "solvency_ratio": ratio,
            "solvency_threshold": float(solvency_threshold),
        },
        history=history,
    )

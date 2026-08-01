"""Portfolio optimization workflows.

Created: 2026-07-02
Purpose: Optimize portfolio-level decisions around profit, reserve, and solvency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from src.actuarial.policy import Policy
from src.optimization.objectives import portfolio_metrics, portfolio_objective
from src.optimization.optimizers import OptimizationResult

if TYPE_CHECKING:
    from src.optimization.predictor import ReservePredictor


def optimize_portfolio_premiums(
    policies: Sequence[Policy],
    predictor: ReservePredictor,
    premium_bounds: Sequence[tuple[float, float]] | None = None,
    time_point: float | Sequence[float] = 0.0,
    cost_of_capital: float = 0.06,
    required_capital_factor: float = 1.5,
    reserve_penalty_rate: float = 0.02,
    expense_rate: float | None = 0.05,
    fixed_expense: float | None = None,
    diversification_credit: float = 0.0,
    solvency_threshold: float | None = None,
) -> OptimizationResult:
    """Optimize premiums for a portfolio while tracking capital and solvency."""

    if not policies:
        return OptimizationResult(
            objective_name="portfolio_premium_profit",
            optimal_values={},
            objective_value=0.0,
            success=True,
            method="scipy_minimize_slsqp",
            message="No policies supplied.",
        )
    bounds = list(
        premium_bounds
        or [(max(float(policy.premium) * 0.5, 1e-6), float(policy.premium) * 2.0) for policy in policies]
    )
    x0 = np.asarray([float(policy.premium) for policy in policies], dtype=float)
    history: list[dict[str, float]] = []

    def with_premiums(values: Sequence[float]) -> list[Policy]:
        return [
            replace(policy, premium=float(premium))
            for policy, premium in zip(policies, values)
        ]

    def objective(values: Sequence[float]) -> float:
        objective_value = portfolio_objective(
            policies=with_premiums(values),
            predictor=predictor,
            time_point=time_point,
            cost_of_capital=cost_of_capital,
            required_capital_factor=required_capital_factor,
            reserve_penalty_rate=reserve_penalty_rate,
            expense_rate=expense_rate,
            fixed_expense=fixed_expense,
            diversification_credit=diversification_credit,
        )
        metrics = portfolio_metrics(
            policies=with_premiums(values),
            predictor=predictor,
            time_point=time_point,
            cost_of_capital=cost_of_capital,
            required_capital_factor=required_capital_factor,
            reserve_penalty_rate=reserve_penalty_rate,
            expense_rate=expense_rate,
            fixed_expense=fixed_expense,
            diversification_credit=diversification_credit,
        )
        history.append(
            {
                "iteration": float(len(history)),
                "objective": float(objective_value),
                "premium": float(np.mean(values)),
                "reserve": metrics.reserve,
                "capital": metrics.capital,
                "profit": metrics.profit,
                "solvency_ratio": metrics.solvency_ratio,
            }
        )
        return -objective_value

    constraints = []
    if solvency_threshold is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda values: portfolio_metrics(
                    policies=with_premiums(values),
                    predictor=predictor,
                    time_point=time_point,
                    cost_of_capital=cost_of_capital,
                    required_capital_factor=required_capital_factor,
                    reserve_penalty_rate=reserve_penalty_rate,
                    expense_rate=expense_rate,
                    fixed_expense=fixed_expense,
                    diversification_credit=diversification_credit,
                ).solvency_ratio
                - float(solvency_threshold),
            }
        )

    result = minimize(objective, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP")
    optimized_policies = with_premiums(result.x)
    metrics = portfolio_metrics(
        policies=optimized_policies,
        predictor=predictor,
        time_point=time_point,
        cost_of_capital=cost_of_capital,
        required_capital_factor=required_capital_factor,
        reserve_penalty_rate=reserve_penalty_rate,
        expense_rate=expense_rate,
        fixed_expense=fixed_expense,
        diversification_credit=diversification_credit,
    )
    return OptimizationResult(
        objective_name="portfolio_premium_profit",
        optimal_values={
            f"{policy.policy_id}.premium": float(premium)
            for policy, premium in zip(policies, result.x)
        },
        objective_value=float(-result.fun),
        success=bool(result.success),
        method="scipy_minimize_slsqp",
        message=str(result.message),
        diagnostics={
            "portfolio_profit": metrics.profit,
            "portfolio_reserve": metrics.reserve,
            "portfolio_capital": metrics.capital,
            "portfolio_solvency_ratio": metrics.solvency_ratio,
        },
        history=history,
    )

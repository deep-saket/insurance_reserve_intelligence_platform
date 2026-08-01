"""Product design optimization workflows.

Created: 2026-07-02
Purpose: Optimize premium, coverage, and term decisions around the reserve oracle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.optimize import minimize

from src.actuarial.policy import Policy
from src.optimization.constraints import reserve_non_negative, solvency_at_least
from src.optimization.objectives import (
    ProductDesign,
    apply_product_design,
    product_design_objective,
    profit_breakdown,
)
from src.optimization.optimizers import OptimizationResult

if TYPE_CHECKING:
    from src.optimization.predictor import ReservePredictor


def optimize_product_design(
    base_policy: Policy,
    predictor: ReservePredictor,
    premium_bounds: tuple[float, float],
    coverage_bounds: tuple[float, float],
    term_bounds: tuple[float, float],
    interest_rate_bounds: tuple[float, float] | None = None,
    capital: float | None = None,
    solvency_threshold: float = 1.5,
    time_point: float = 0.0,
    cost_of_capital: float = 0.06,
    required_capital_factor: float = 1.5,
    reserve_penalty_rate: float = 0.02,
    expense_rate: float | None = 0.05,
    fixed_expense: float | None = None,
    coverage_growth_weight: float = 0.0,
    term_growth_weight: float = 0.0,
) -> OptimizationResult:
    """Optimize product design variables for maximum product objective."""

    bounds = [
        premium_bounds,
        coverage_bounds,
        term_bounds,
        interest_rate_bounds or (float(base_policy.interest_rate), float(base_policy.interest_rate)),
    ]
    x0 = [
        float(base_policy.premium),
        float(base_policy.sum_assured),
        float(base_policy.term),
        float(base_policy.interest_rate),
    ]
    history: list[dict[str, float]] = []

    def design_from_values(values: list[float]) -> ProductDesign:
        return ProductDesign(
            premium=float(values[0]),
            sum_assured=float(values[1]),
            term=int(round(float(values[2]))),
            interest_rate=float(values[3]),
        )

    def objective(values: list[float]) -> float:
        objective_value = product_design_objective(
            base_policy=base_policy,
            design=design_from_values(values),
            predictor=predictor,
            time_point=time_point,
            cost_of_capital=cost_of_capital,
            required_capital_factor=required_capital_factor,
            reserve_penalty_rate=reserve_penalty_rate,
            expense_rate=expense_rate,
            fixed_expense=fixed_expense,
            coverage_growth_weight=coverage_growth_weight,
            term_growth_weight=term_growth_weight,
        )
        candidate = candidate_policy(values)
        breakdown = profit_breakdown(
            policy=candidate,
            predictor=predictor,
            time_point=time_point,
            cost_of_capital=cost_of_capital,
            required_capital_factor=required_capital_factor,
            reserve_penalty_rate=reserve_penalty_rate,
            expense_rate=expense_rate,
            fixed_expense=fixed_expense,
        )
        history.append(
            {
                "iteration": float(len(history)),
                "objective": float(objective_value),
                "premium": float(candidate.premium),
                "reserve": breakdown.reserve,
                "capital": max(breakdown.reserve, 0.0) * float(required_capital_factor),
                "profit": breakdown.profit,
                "sum_assured": float(candidate.sum_assured),
                "term": float(candidate.term),
                "interest_rate": float(candidate.interest_rate),
            }
        )
        return -objective_value

    def candidate_policy(values: list[float]) -> Policy:
        return apply_product_design(base_policy, design_from_values(values))

    constraints = [
        {"type": "ineq", "fun": lambda values: reserve_non_negative(candidate_policy(values), predictor, time_point)},
    ]
    if capital is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda values: solvency_at_least(
                    candidate_policy(values),
                    predictor,
                    capital=float(capital),
                    threshold=solvency_threshold,
                    time_point=time_point,
                ),
            }
        )

    result = minimize(objective, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP")
    optimal_design = design_from_values(result.x)
    optimal_policy = apply_product_design(base_policy, optimal_design)
    breakdown = profit_breakdown(
        policy=optimal_policy,
        predictor=predictor,
        time_point=time_point,
        cost_of_capital=cost_of_capital,
        required_capital_factor=required_capital_factor,
        reserve_penalty_rate=reserve_penalty_rate,
        expense_rate=expense_rate,
        fixed_expense=fixed_expense,
    )
    return OptimizationResult(
        objective_name="product_design",
        optimal_values={
            "premium": optimal_design.premium,
            "sum_assured": optimal_design.sum_assured,
            "term": float(optimal_design.term),
            "interest_rate": optimal_design.interest_rate,
        },
        objective_value=float(-result.fun),
        success=bool(result.success),
        method="scipy_minimize_slsqp",
        message=str(result.message),
        diagnostics={
            "profit": breakdown.profit,
            "reserve": breakdown.reserve,
            "capital_cost": breakdown.capital_cost,
            "expense_cost": breakdown.expense_cost,
            "reserve_cost": breakdown.reserve_cost,
        },
        history=history,
    )

"""Premium optimization workflows.

Created: 2026-07-02
Purpose: Optimize policy premium using business objectives and ReservePredictor.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from scipy.optimize import minimize_scalar

from src.actuarial.policy import Policy
from src.optimization.objectives import pricing_objective, profit_breakdown
from src.optimization.optimizers import OptimizationResult

if TYPE_CHECKING:
    from src.optimization.predictor import ReservePredictor


def optimize_premium(
    policy: Policy,
    predictor: ReservePredictor,
    bounds: tuple[float, float] | None = None,
    time_point: float = 0.0,
    cost_of_capital: float = 0.06,
    required_capital_factor: float = 1.5,
    reserve_penalty_rate: float = 0.02,
    expense_rate: float | None = 0.05,
    fixed_expense: float | None = None,
    market_premium: float | None = None,
    market_penalty_rate: float = 0.001,
) -> OptimizationResult:
    """Find the premium that maximizes pricing profit.

    SciPy minimizes, so this wrapper returns negative profit internally.

    The market premium penalty prevents the optimizer from always choosing the
    highest possible premium just because premium income increases.
    """

    original_premium = float(policy.premium)

    premium_bounds = bounds or (
        max(original_premium * 0.5, 1e-6),
        original_premium * 2.0,
    )

    reference_market_premium = (
        original_premium if market_premium is None else float(market_premium)
    )

    history: list[dict[str, float]] = []

    def objective(premium_value: float) -> float:
        candidate = replace(policy, premium=float(premium_value))

        profit_value = pricing_objective(
            policy=candidate,
            predictor=predictor,
            time_point=time_point,
            cost_of_capital=cost_of_capital,
            required_capital_factor=required_capital_factor,
            reserve_penalty_rate=reserve_penalty_rate,
            expense_rate=expense_rate,
            fixed_expense=fixed_expense,
            market_premium=reference_market_premium,
            market_penalty_rate=market_penalty_rate,
        )

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

        business_reserve = max(float(breakdown.reserve), 0.0)
        capital = business_reserve * float(required_capital_factor)

        history.append(
            {
                "iteration": float(len(history)),
                "objective": float(profit_value),
                "premium": float(premium_value),
                "reserve": business_reserve,
                "capital": capital,
                "profit": float(breakdown.profit),
            }
        )

        return -float(profit_value)

    result = minimize_scalar(
        objective,
        bounds=premium_bounds,
        method="bounded",
    )

    optimal_policy = replace(policy, premium=float(result.x))

    final_profit = pricing_objective(
        policy=optimal_policy,
        predictor=predictor,
        time_point=time_point,
        cost_of_capital=cost_of_capital,
        required_capital_factor=required_capital_factor,
        reserve_penalty_rate=reserve_penalty_rate,
        expense_rate=expense_rate,
        fixed_expense=fixed_expense,
        market_premium=reference_market_premium,
        market_penalty_rate=market_penalty_rate,
    )

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

    business_reserve = max(float(breakdown.reserve), 0.0)

    return OptimizationResult(
        objective_name="pricing_profit",
        optimal_values={"premium": float(result.x)},
        objective_value=float(final_profit),
        success=bool(result.success),
        method="scipy_minimize_scalar_bounded",
        message=str(result.message),
        diagnostics={
            "reserve": business_reserve,
            "capital": business_reserve * float(required_capital_factor),
            "capital_cost": float(breakdown.capital_cost),
            "expense_cost": float(breakdown.expense_cost),
            "reserve_cost": float(breakdown.reserve_cost),
            "market_premium": reference_market_premium,
            "market_penalty_rate": float(market_penalty_rate),
        },
        history=history,
    )
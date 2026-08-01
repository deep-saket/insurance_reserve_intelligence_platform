"""Business objective functions for reserve-driven optimization.

Created: 2026-07-02
Purpose: Keep actuarial business mathematics separate from scipy optimizers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from src.actuarial.policy import Policy

if TYPE_CHECKING:
    from src.optimization.predictor import ReservePredictor


@dataclass(frozen=True, slots=True)
class ProfitBreakdown:
    """Auditable profitability components for a policy decision.

    Business Interpretation:
        Insurance profit is not just premium less claims. It also reflects
        operating expenses, the drag from reserve requirements, and the cost of
        capital needed to support the liability.
    """

    premium_income: float
    expected_claim_cost: float
    reserve: float
    capital_cost: float
    expense_cost: float
    reserve_cost: float
    profit: float


@dataclass(frozen=True, slots=True)
class PortfolioMetrics:
    """Portfolio-level business quantities used by optimization objectives."""

    premium_income: float
    expected_claim_cost: float
    reserve: float
    capital: float
    capital_cost: float
    expense_cost: float
    reserve_cost: float
    profit: float
    solvency_ratio: float


@dataclass(frozen=True, slots=True)
class ProductDesign:
    """Decision variables used by product and pricing optimizers."""

    premium: float
    sum_assured: float
    term: int
    interest_rate: float


def model_reserve(policy: Policy, predictor: ReservePredictor, time_point: float = 0.0) -> float:
    """Raw reserve predicted by the neural model.

    This may be negative because the PINN predicts the learned actuarial reserve
    ratio directly.
    """
    return float(predictor.predict(policy=policy, time_point=time_point))


def reserve(policy: Policy, predictor: ReservePredictor, time_point: float = 0.0) -> float:
    """Business reserve used for optimization.

    Reserve = max(ModelReserve, 0)

    This prevents negative reserves from creating fake profit or fake capital relief.
    """
    return max(model_reserve(policy, predictor, time_point), 0.0)


def reserve_objective(policy: Policy, predictor: ReservePredictor, time_point: float = 0.0) -> float:
    """Return reserve objective value for minimization."""

    return reserve(policy=policy, predictor=predictor, time_point=time_point)


def premium_income(policy: Policy) -> float:
    """Return policy premium income for the objective horizon."""

    return float(policy.premium)


def apply_product_design(policy: Policy, design: ProductDesign) -> Policy:
    """Return a policy copy with product-design decision variables applied."""

    return replace(
        policy,
        premium=float(design.premium),
        sum_assured=float(design.sum_assured),
        term=int(round(design.term)),
        interest_rate=float(design.interest_rate),
    )


def expected_claim_cost(policy: Policy, time_point: float = 0.0) -> float:
    """Approximate discounted expected claim cost.

    TODO:
        Replace this with full actuarial EPV:

        ExpectedClaim = integral mu(t) S(t) Benefit exp(-rt) dt
    """
    mortality = float(policy.mortality_profile.intensity_at(time_point))
    death_benefit = float(policy.sum_assured)
    interest_rate = float(policy.interest_rate)

    remaining_term = max(float(policy.term) - float(time_point), 0.0)
    average_claim_time = remaining_term / 2.0

    discount_factor = (1.0 + interest_rate) ** (-average_claim_time)

    return mortality * death_benefit * discount_factor

def expense_cost(
    policy: Policy,
    time_point: float = 0.0,
    expense_rate: float | None = 0.05,
    fixed_expense: float | None = None,
) -> float:
    """Return simple policy expense cost.

    Business Interpretation:
        Expenses can be supplied explicitly as a premium percentage and fixed
        amount. If omitted, the policy's own expense curve is used as the fixed
        operating expense estimate.
    """

    variable_expense = float(policy.premium) * float(expense_rate or 0.0)
    fixed_component = 0.0 if fixed_expense is None else float(fixed_expense)
    return variable_expense + fixed_component


def reserve_cost(
    policy: Policy,
    predictor: ReservePredictor,
    time_point: float = 0.0,
    reserve_penalty_rate: float = 0.02,
) -> float:
    """Return direct reserve penalty used in profitability objectives."""

    reserve_amount = max(reserve(policy=policy, predictor=predictor, time_point=time_point), 0.0)
    return reserve_amount * float(reserve_penalty_rate)


def required_capital(
    policy: Policy,
    predictor: ReservePredictor,
    time_point: float = 0.0,
    required_capital_factor: float = 1.5,
) -> float:
    """Required Capital = Reserve x Required Capital Factor."""
    business_reserve = reserve(policy, predictor, time_point)
    return business_reserve * required_capital_factor


def capital_cost(
    policy: Policy,
    predictor: ReservePredictor,
    time_point: float = 0.0,
    cost_of_capital: float = 0.06,
    required_capital_factor: float = 1.5,
) -> float:
    """Return cost of required capital associated with the predicted reserve."""

    return required_capital(
        policy=policy,
        predictor=predictor,
        time_point=time_point,
        required_capital_factor=required_capital_factor,
    ) * float(cost_of_capital)


def solvency_ratio(
    policy: Policy,
    predictor: ReservePredictor,
    capital: float,
    time_point: float = 0.0,
) -> float:
    """Solvency Ratio = Capital / Reserve."""
    business_reserve = reserve(policy, predictor, time_point)

    if business_reserve <= 1e-8:
        return float("inf")

    return float(capital) / business_reserve


def profit_breakdown(
    policy: Policy,
    predictor: ReservePredictor,
    time_point: float = 0.0,
    cost_of_capital: float = 0.06,
    required_capital_factor: float = 1.5,
    reserve_penalty_rate: float = 0.02,
    expense_rate: float | None = 0.05,
    fixed_expense: float | None = None,
) -> ProfitBreakdown:
    """Return premium, claims, expenses, reserve cost, capital cost, and profit.

    Equation:
        Profit = Premium Income - Claims - Expenses - Capital Cost - Reserve Cost
    """

    income = premium_income(policy)
    claims = expected_claim_cost(policy, time_point=time_point)
    reserve_amount = reserve(policy=policy, predictor=predictor, time_point=time_point)
    capital = max(reserve_amount, 0.0) * float(required_capital_factor) * float(cost_of_capital)
    expenses = expense_cost(
        policy=policy,
        time_point=time_point,
        expense_rate=expense_rate,
        fixed_expense=fixed_expense,
    )
    reserve_drag = max(reserve_amount, 0.0) * float(reserve_penalty_rate)
    profit_value = income - claims - expenses - capital - reserve_drag
    return ProfitBreakdown(
        premium_income=income,
        expected_claim_cost=claims,
        reserve=reserve_amount,
        capital_cost=capital,
        expense_cost=expenses,
        reserve_cost=reserve_drag,
        profit=profit_value,
    )


def profit(
    policy: Policy,
    predictor: ReservePredictor,
    time_point: float = 0.0,
    cost_of_capital: float = 0.06,
    required_capital_factor: float = 1.5,
    reserve_penalty_rate: float = 0.02,
    expense_rate: float | None = 0.05,
    fixed_expense: float | None = None,
) -> float:
    """Return policy profit under current business assumptions.

    Equation:
        Profit = Premium Income - Claims - Expenses - Capital Cost - Reserve Cost
    """

    return profit_breakdown(
        policy=policy,
        predictor=predictor,
        time_point=time_point,
        cost_of_capital=cost_of_capital,
        required_capital_factor=required_capital_factor,
        reserve_penalty_rate=reserve_penalty_rate,
        expense_rate=expense_rate,
        fixed_expense=fixed_expense,
    ).profit


def profit_objective(
    policy: Policy,
    predictor: ReservePredictor,
    time_point: float = 0.0,
    cost_of_capital: float = 0.06,
    required_capital_factor: float = 1.5,
    reserve_penalty_rate: float = 0.02,
    expense_rate: float | None = 0.05,
    fixed_expense: float | None = None,
) -> float:
    """Return profit objective value for maximization."""

    return profit(
        policy=policy,
        predictor=predictor,
        time_point=time_point,
        cost_of_capital=cost_of_capital,
        required_capital_factor=required_capital_factor,
        reserve_penalty_rate=reserve_penalty_rate,
        expense_rate=expense_rate,
        fixed_expense=fixed_expense,
    )


def pricing_objective(
    policy: Policy,
    predictor: ReservePredictor,
    time_point: float = 0.0,
    cost_of_capital: float = 0.06,
    required_capital_factor: float = 1.5,
    reserve_penalty_rate: float = 0.02,
    expense_rate: float | None = 0.05,
    fixed_expense: float | None = None,
    market_premium: float | None = None,
    market_penalty_rate: float = 0.001,
) -> float:
    """Return pricing profit after optional market premium penalty.

    Profit = Premium - Claims - Expenses - Capital Cost - Reserve Cost

    The market penalty prevents the optimizer from always selecting the highest
    possible premium.
    """

    base_profit = profit(
        policy=policy,
        predictor=predictor,
        time_point=time_point,
        cost_of_capital=cost_of_capital,
        required_capital_factor=required_capital_factor,
        reserve_penalty_rate=reserve_penalty_rate,
        expense_rate=expense_rate,
        fixed_expense=fixed_expense,
    )

    if market_premium is None:
        return base_profit

    excess_premium = max(float(policy.premium) - float(market_premium), 0.0)
    market_penalty = float(market_penalty_rate) * excess_premium * excess_premium

    return base_profit - market_penalty
def product_objective(
    policy: Policy,
    predictor: ReservePredictor,
    time_point: float = 0.0,
    cost_of_capital: float = 0.06,
    required_capital_factor: float = 1.5,
    reserve_penalty_rate: float = 0.02,
    expense_rate: float | None = 0.05,
    fixed_expense: float | None = None,
    coverage_growth_weight: float = 0.0,
    term_growth_weight: float = 0.0,
) -> float:
    """Return product-design objective for premium, coverage, and term decisions.

    Business Interpretation:
        Product optimization can value profitable growth as well as raw profit.
        The optional growth weights let optimizers reward larger coverage and
        longer terms while still pricing reserve and capital drag.
    """

    base_profit = profit_objective(
        policy=policy,
        predictor=predictor,
        time_point=time_point,
        cost_of_capital=cost_of_capital,
        required_capital_factor=required_capital_factor,
        reserve_penalty_rate=reserve_penalty_rate,
        expense_rate=expense_rate,
        fixed_expense=fixed_expense,
    )
    coverage_value = float(coverage_growth_weight) * float(policy.sum_assured)
    term_value = float(term_growth_weight) * float(policy.term)
    return base_profit + coverage_value + term_value


def product_design_objective(
    base_policy: Policy,
    design: ProductDesign,
    predictor: ReservePredictor,
    time_point: float = 0.0,
    cost_of_capital: float = 0.06,
    required_capital_factor: float = 1.5,
    reserve_penalty_rate: float = 0.02,
    expense_rate: float | None = 0.05,
    fixed_expense: float | None = None,
    coverage_growth_weight: float = 0.0,
    term_growth_weight: float = 0.0,
) -> float:
    """Return product objective for explicit optimizer decision variables."""

    designed_policy = apply_product_design(base_policy, design)
    return product_objective(
        policy=designed_policy,
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


def capital_objective(
    policy: Policy,
    predictor: ReservePredictor,
    time_point: float = 0.0,
    mode: str = "min_reserve",
    required_capital_factor: float = 1.5,
) -> float:
    """Return capital objective value.

    ``mode='min_reserve'`` returns reserve for minimization.
    ``mode='max_roe'`` returns return on equity for maximization.

    Equation:
        ROE = Profit / Required Capital
    """

    reserve_amount = reserve(policy=policy, predictor=predictor, time_point=time_point)
    if mode == "min_reserve":
        return reserve_amount
    if mode == "max_roe":
        capital_amount = required_capital(
            policy=policy,
            predictor=predictor,
            time_point=time_point,
            required_capital_factor=required_capital_factor,
        )
        return profit(policy=policy, predictor=predictor, time_point=time_point) / max(
            capital_amount,
            1.0,
        )
    raise ValueError(f"Unsupported capital objective mode: {mode}")


def solvency_objective(
    policy: Policy,
    predictor: ReservePredictor,
    capital: float | None = None,
    time_point: float = 0.0,
    required_capital_factor: float = 1.5,
) -> float:
    """Return solvency ratio objective value for maximization."""

    return solvency_ratio(
        policy=policy,
        predictor=predictor,
        capital=capital,
        time_point=time_point,
        required_capital_factor=required_capital_factor,
    )


def portfolio_metrics(
    policies: Sequence[Policy],
    predictor: ReservePredictor,
    time_point: float | Sequence[float] = 0.0,
    cost_of_capital: float = 0.06,
    required_capital_factor: float = 1.5,
    reserve_penalty_rate: float = 0.02,
    expense_rate: float | None = 0.05,
    fixed_expense: float | None = None,
    diversification_credit: float = 0.0,
) -> PortfolioMetrics:
    """Return aggregate portfolio profit, reserve, capital, and solvency."""

    time_points = _resolve_time_points(policies=policies, time_point=time_point)
    breakdowns = [
        profit_breakdown(
            policy=policy,
            predictor=predictor,
            time_point=valuation_time,
            cost_of_capital=cost_of_capital,
            required_capital_factor=required_capital_factor,
            reserve_penalty_rate=reserve_penalty_rate,
            expense_rate=expense_rate,
            fixed_expense=fixed_expense,
        )
        for policy, valuation_time in zip(policies, time_points)
    ]
    total_reserve = sum(item.reserve for item in breakdowns)
    total_capital = max(total_reserve, 0.0) * float(required_capital_factor)
    diversified_capital = total_capital * max(0.0, 1.0 - float(diversification_credit))
    total_profit = sum(item.profit for item in breakdowns)
    ratio = float("inf") if total_reserve <= 0.0 else diversified_capital / total_reserve
    return PortfolioMetrics(
        premium_income=sum(item.premium_income for item in breakdowns),
        expected_claim_cost=sum(item.expected_claim_cost for item in breakdowns),
        reserve=total_reserve,
        capital=diversified_capital,
        capital_cost=sum(item.capital_cost for item in breakdowns),
        expense_cost=sum(item.expense_cost for item in breakdowns),
        reserve_cost=sum(item.reserve_cost for item in breakdowns),
        profit=total_profit,
        solvency_ratio=ratio,
    )


def portfolio_objective(
    policies: Sequence[Policy],
    predictor: ReservePredictor,
    time_point: float | Sequence[float] = 0.0,
    cost_of_capital: float = 0.06,
    required_capital_factor: float = 1.5,
    reserve_penalty_rate: float = 0.02,
    expense_rate: float | None = 0.05,
    fixed_expense: float | None = None,
    diversification_credit: float = 0.0,
) -> float:
    """Return portfolio profit objective value for maximization."""

    return portfolio_metrics(
        policies=policies,
        predictor=predictor,
        time_point=time_point,
        cost_of_capital=cost_of_capital,
        required_capital_factor=required_capital_factor,
        reserve_penalty_rate=reserve_penalty_rate,
        expense_rate=expense_rate,
        fixed_expense=fixed_expense,
        diversification_credit=diversification_credit,
    ).profit


def _clip_time(policy: Policy, time_point: float) -> float:
    """Clip valuation time into the valid policy duration."""

    return float(np.clip(float(time_point), 0.0, float(policy.term)))


def _resolve_time_points(
    policies: Sequence[Policy],
    time_point: float | Sequence[float],
) -> list[float]:
    """Return one valuation time per policy."""

    if isinstance(time_point, Sequence) and not isinstance(time_point, str):
        if len(time_point) != len(policies):
            raise ValueError("time_point sequence must match policies length.")
        return [float(value) for value in time_point]
    return [float(time_point)] * len(policies)

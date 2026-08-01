"""Reusable business constraints for reserve-driven optimization.

Created: 2026-07-02
Purpose: Express actuarial and commercial feasibility checks independently of optimizers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from src.actuarial.policy import Policy
from src.optimization.objectives import reserve, solvency_ratio

if TYPE_CHECKING:
    from src.optimization.predictor import ReservePredictor


ConstraintFunction = Callable[..., float]


def premium_positive(policy: Policy) -> float:
    """Return value >= 0 when premium is positive."""

    return float(policy.premium)


def premium_above(policy: Policy, minimum_premium: float) -> float:
    """Return value >= 0 when premium is at least a business minimum."""

    return float(policy.premium) - float(minimum_premium)


def premium_below(policy: Policy, maximum_premium: float) -> float:
    """Return value >= 0 when premium is at most a business maximum."""

    return float(maximum_premium) - float(policy.premium)


def reserve_non_negative(
    policy: Policy,
    predictor: ReservePredictor,
    time_point: float = 0.0,
) -> float:
    """Return value >= 0 when predicted reserve is non-negative."""

    return reserve(policy=policy, predictor=predictor, time_point=time_point)


def capital_covers_reserve(
    policy: Policy,
    predictor: ReservePredictor,
    capital: float,
    time_point: float = 0.0,
) -> float:
    """Return value >= 0 when allocated capital covers predicted reserve."""

    return float(capital) - reserve(policy=policy, predictor=predictor, time_point=time_point)


def solvency_at_least(
    policy: Policy,
    predictor: ReservePredictor,
    capital: float,
    threshold: float = 1.5,
    time_point: float = 0.0,
) -> float:
    """Return value >= 0 when solvency ratio is at least the threshold."""

    return solvency_ratio(
        policy=policy,
        predictor=predictor,
        capital=capital,
        time_point=time_point,
    ) - float(threshold)


def term_within(policy: Policy, minimum_term: float, maximum_term: float) -> tuple[float, float]:
    """Return lower and upper scipy-compatible term constraint values."""

    return (
        float(policy.term) - float(minimum_term),
        float(maximum_term) - float(policy.term),
    )


def coverage_within(
    policy: Policy,
    minimum_coverage: float,
    maximum_coverage: float,
) -> tuple[float, float]:
    """Return lower and upper scipy-compatible coverage constraint values."""

    return (
        float(policy.sum_assured) - float(minimum_coverage),
        float(maximum_coverage) - float(policy.sum_assured),
    )


def scipy_inequality(fun: ConstraintFunction) -> dict[str, object]:
    """Wrap a no-argument scalar constraint for scipy.optimize.

    SciPy inequality constraints are feasible when ``fun(x) >= 0``. Optimizer
    wrappers can close over their decision vector and call these business
    constraints without duplicating the equations.
    """

    return {"type": "ineq", "fun": lambda *args: float(fun(*args))}

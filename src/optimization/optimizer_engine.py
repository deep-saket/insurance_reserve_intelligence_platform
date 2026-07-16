"""Optimization workflows for reserve management.

Created: 2026-05-31
Purpose: Optimize reserve-related assumptions and pricing decisions using the trained model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from scipy.optimize import minimize, minimize_scalar

from src.actuarial.policy import Policy
from src.data.dataset import (
    FEATURE_INDEX,
    FEATURE_SCALES,
    build_policy_feature_array,
    normalize_raw_feature_array,
)
from src.utils.config import OptimizationConfig


@dataclass(slots=True)
class OptimizationResult:
    """Optimization summary.

    Attributes:
        variable_name: Decision variable being optimized.
        optimal_value: Best decision value found.
        objective_value: Objective value at the optimum.
        method: Optimization routine used.
        success: Whether the optimization run converged successfully.

    Business Interpretation:
        This object is the decision recommendation returned by the optimization
        engine for pricing or reserve-targeting workflows.
    """

    variable_name: str
    optimal_value: float
    objective_value: float
    method: str
    success: bool


class OptimizationEngine:
    """Optimize premiums, interest rates, and reserve-related objectives.

    Scientific Context:
        The engine mixes gradient-based search and SciPy-based numerical
        optimization on top of a differentiable reserve surrogate.

    Business Interpretation:
        It converts a reserve model into a decision-support tool for pricing,
        target setting, and solvency-aware profitability analysis.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        config: OptimizationConfig,
        target_mean: float,
        target_std: float,
    ) -> None:
        """Initialize the optimization engine.

        Args:
            model: Trained reserve model.
            device: Execution device for optimization runs.
            config: Optimization hyperparameters and constraints.
            target_mean: Mean used to standardize reserve-ratio targets.
            target_std: Standard deviation used to standardize reserve-ratio targets.
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)

    def _features_from_policy(self, policy: Policy) -> torch.Tensor:
        """Build an optimization feature tensor from a policy.

        Args:
            policy: Policy to convert.

        Returns:
            torch.Tensor: Single-row feature tensor.
        """
        raw = build_policy_feature_array(policy=policy, time_point=0.0)
        normalized = normalize_raw_feature_array(raw)
        return torch.tensor(normalized, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _predict(self, features: torch.Tensor) -> torch.Tensor:
        """Predict reserves for optimization inputs.

        Args:
            features: Feature tensor for reserve inference.

        Returns:
            torch.Tensor: Model reserve prediction tensor.
        """
        self.model.eval()
        z = self.model(features)
        sum_assured = features[:, FEATURE_INDEX["sum_assured"] : FEATURE_INDEX["sum_assured"] + 1]
        sum_assured = sum_assured * FEATURE_SCALES["sum_assured"]
        return (z * self.target_std + self.target_mean) * sum_assured

    def target_reserve_optimization(self, policy: Policy, target_reserve: float) -> OptimizationResult:
        """Find the interest rate that produces a target reserve.

        Args:
            policy: Policy whose reserve target is being calibrated.
            target_reserve: Desired reserve level.

        Returns:
            OptimizationResult: Optimization summary for the calibrated interest rate.

        Scientific Context:
            This is an inverse problem: solve for the assumption value that makes
            the reserve surface hit a prescribed target.

        Business Interpretation:
            It answers the question, "What rate environment would justify this
            liability level?"
        """

        def objective(interest_rate: float) -> float:
            features = self._features_from_policy(policy).clone()
            features[:, FEATURE_INDEX["scenario_interest_rate"]] = (
                interest_rate / FEATURE_SCALES["scenario_interest_rate"]
            )
            reserve = float(self._predict(features).detach().cpu().item())
            return (reserve - target_reserve) ** 2

        result = minimize_scalar(objective, bounds=(-0.02, 0.15), method="bounded")
        return OptimizationResult(
            variable_name="scenario_interest_rate",
            optimal_value=float(result.x),
            objective_value=float(result.fun),
            method="scipy_minimize_scalar",
            success=bool(result.success),
        )

    def premium_optimization(self, policy: Policy) -> OptimizationResult:
        """Use gradient ascent to maximize a simple profitability proxy.

        Args:
            policy: Policy whose premium is being optimized.

        Returns:
            OptimizationResult: Optimization summary for the premium decision.

        Business Interpretation:
            This provides a research-grade pricing lever for exploring premium
            adequacy while penalizing reserve intensity.
        """

        premium = torch.tensor([policy.premium], dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([premium], lr=self.config.learning_rate)

        for _ in range(self.config.steps):
            optimizer.zero_grad(set_to_none=True)
            features = self._features_from_policy(policy).clone()
            features[:, FEATURE_INDEX["premium"]] = premium / FEATURE_SCALES["premium"]
            reserve = self._predict(features)
            profitability = premium - 0.01 * reserve
            loss = -profitability.mean()
            loss.backward()
            optimizer.step()

        optimized_features = self._features_from_policy(policy).clone()
        optimized_features[:, FEATURE_INDEX["premium"]] = premium.detach() / FEATURE_SCALES["premium"]
        objective_value = float((premium - 0.01 * self._predict(optimized_features)).detach().cpu().item())
        return OptimizationResult(
            variable_name="premium",
            optimal_value=float(premium.detach().cpu().item()),
            objective_value=objective_value,
            method="gradient_based",
            success=True,
        )

    def constrained_premium_optimization(self, policy: Policy) -> OptimizationResult:
        """Maximize profitability subject to a reserve floor.

        Args:
            policy: Policy whose premium is being optimized.

        Returns:
            OptimizationResult: Optimization summary under the solvency constraint.

        Business Interpretation:
            This approximates pricing under capital discipline by discouraging
            solutions that breach a reserve-based solvency floor.
        """

        def objective(values: np.ndarray) -> float:
            premium_value = float(values[0])
            features = self._features_from_policy(policy).clone()
            features[:, FEATURE_INDEX["premium"]] = premium_value / FEATURE_SCALES["premium"]
            reserve = float(self._predict(features).detach().cpu().item())
            profitability = premium_value - 0.01 * reserve
            penalty = max(0.0, self.config.solvency_threshold - reserve) ** 2
            return -(profitability - 1000.0 * penalty)

        result = minimize(
            objective,
            x0=np.asarray([policy.premium], dtype=float),
            bounds=[(policy.premium * 0.5, policy.premium * 2.0)],
            method="L-BFGS-B",
        )
        return OptimizationResult(
            variable_name="premium",
            optimal_value=float(result.x[0]),
            objective_value=float(-result.fun),
            method="scipy_constrained",
            success=bool(result.success),
        )

    def bayesian_optimization_hook(
        self,
        policy: Policy,
        optimizer_fn: Callable[[Callable[[float], float]], tuple[float, float]],
    ) -> OptimizationResult:
        """Allow external Bayesian optimization libraries to plug in cleanly.

        Args:
            policy: Policy whose premium is being optimized.
            optimizer_fn: External optimizer callback returning the best point and objective.

        Returns:
            OptimizationResult: Optimization summary returned through the plugin hook.

        Business Interpretation:
            This keeps the architecture open for more advanced experimentation
            without rewriting the core engine.
        """

        def objective(premium_value: float) -> float:
            features = self._features_from_policy(policy).clone()
            features[:, FEATURE_INDEX["premium"]] = premium_value / FEATURE_SCALES["premium"]
            reserve = float(self._predict(features).detach().cpu().item())
            return -(premium_value - 0.01 * reserve)

        optimal_value, objective_value = optimizer_fn(objective)
        return OptimizationResult(
            variable_name="premium",
            optimal_value=float(optimal_value),
            objective_value=float(objective_value),
            method="bayesian_hook",
            success=True,
        )

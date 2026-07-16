"""Insurance liability digital twin workflows.

Created: 2026-05-31
Purpose: Simulate reserve behavior through time, scenarios, regimes, and portfolios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.actuarial.policy import Policy
from src.data.dataset import (
    FEATURE_INDEX,
    FEATURE_SCALES,
    build_policy_feature_array,
    normalize_raw_feature_array,
)
from src.data.simulator import PolicySimulator, ScenarioDefinition
from src.utils.config import DigitalTwinConfig


@dataclass(slots=True)
class RegimeDefinition:
    """Macroeconomic regime multipliers.

    Attributes:
        name: Regime name.
        interest_rate_shift: Additive shock to interest rates.
        mortality_multiplier: Multiplicative mortality shock.
        inflation_multiplier: Multiplicative inflation proxy applied to benefits.

    Business Interpretation:
        This object packages a macro narrative such as recession or mortality
        crisis into model-ready shocks.
    """

    name: str
    interest_rate_shift: float
    mortality_multiplier: float
    inflation_multiplier: float


class DigitalTwinEngine:
    """Scenario-aware digital twin for insurance liabilities.

    Scientific Context:
        The digital twin is a scenario-queryable surrogate over the liability
        state space, allowing repeated reserve evaluations under changing inputs.

    Business Interpretation:
        This is the interactive liability lab for forecasting, what-if analysis,
        and portfolio simulation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        config: DigitalTwinConfig,
        target_mean: float,
        target_std: float,
        simulator: PolicySimulator | None = None,
    ) -> None:
        """Initialize the digital twin engine.

        Args:
            model: Trained reserve model.
            device: Execution device for inference.
            config: Digital twin simulation settings.
            target_mean: Mean used to standardize reserve-ratio targets.
            target_std: Standard deviation used to standardize reserve-ratio targets.
            simulator: Optional policy simulator used for scenario cloning.
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.simulator = simulator

    def _feature_tensor(self, policy: Policy, time_point: float) -> torch.Tensor:
        """Build a single-step model input from a policy and time point.

        Args:
            policy: Policy to evaluate.
            time_point: Elapsed policy time.

        Returns:
            torch.Tensor: Single-row model input tensor.
        """
        raw = build_policy_feature_array(policy=policy, time_point=time_point)
        normalized = normalize_raw_feature_array(raw)
        return torch.tensor(normalized, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _predict_reserve(self, features: torch.Tensor) -> float:
        """Predict reserve in currency units from normalized features."""

        with torch.no_grad():
            z = float(self.model(features).item())
        sum_assured = float(features[0, FEATURE_INDEX["sum_assured"]].item()) * FEATURE_SCALES["sum_assured"]
        return float((z * self.target_std + self.target_mean) * sum_assured)

    def reserve_forecast(self, policy: Policy, steps: int | None = None) -> pd.DataFrame:
        """Forecast reserves over time for a policy.

        Args:
            policy: Policy to forecast.
            steps: Optional number of forecast steps.

        Returns:
            pd.DataFrame: Time-indexed reserve forecast.

        Business Interpretation:
            This produces the liability path that an actuary or CFO would inspect
            when asking how reserves evolve through the contract life.
        """

        self.model.eval()
        horizon = steps or self.config.forecast_horizon
        times = np.linspace(0.0, float(policy.term), horizon, dtype=float)
        reserves: list[float] = []
        with torch.no_grad():
            for time_point in times:
                reserves.append(self._predict_reserve(self._feature_tensor(policy, float(time_point))))
        return pd.DataFrame({"time": times, "reserve": reserves, "policy_id": policy.policy_id})

    def scenario_simulation(self, policies: list[Policy], scenario: ScenarioDefinition) -> pd.DataFrame:
        """Run a direct what-if scenario across a portfolio.

        Args:
            policies: Baseline policies to simulate.
            scenario: Scenario shock definition.

        Returns:
            pd.DataFrame: Baseline-versus-scenario reserve comparison table.

        Raises:
            ValueError: If no simulator is available for generating scenario clones.

        Business Interpretation:
            This is the portfolio what-if tool for changing assumptions and seeing
            the reserve impact immediately.
        """

        if self.simulator is None:
            raise ValueError("A PolicySimulator is required for scenario simulation.")
        scenario_policies = self.simulator.generate_scenario_policies(policies, scenario)
        baseline = self.portfolio_simulation(policies)
        stressed = self.portfolio_simulation(scenario_policies)
        return baseline.merge(stressed, on="policy_id", suffixes=("_base", "_scenario"))

    def regime_simulation(self, policies: list[Policy]) -> pd.DataFrame:
        """Simulate predefined macro regimes.

        Args:
            policies: Policies to simulate under each regime.

        Returns:
            pd.DataFrame: Reserve outcomes by policy and regime.

        Business Interpretation:
            This translates macro stories such as inflation or recession into
            liability outcomes that executives can compare.
        """

        regimes = [
            RegimeDefinition("base", 0.0, 1.0, 1.0),
            RegimeDefinition("soft_recession", -0.01, 1.05, 1.01),
            RegimeDefinition("inflationary", 0.02, 1.02, 1.05),
            RegimeDefinition("mortality_crisis", -0.005, 1.20, 1.02),
        ]
        rows: list[dict[str, float | str]] = []
        for regime in regimes:
            for policy in policies:
                features = self._feature_tensor(policy, 0.0)
                features[:, FEATURE_INDEX["scenario_interest_rate"]] += (
                    regime.interest_rate_shift / FEATURE_SCALES["scenario_interest_rate"]
                )
                features[:, FEATURE_INDEX["sum_assured"]] *= regime.inflation_multiplier
                features[:, FEATURE_INDEX["mortality"]] *= regime.mortality_multiplier
                reserve = self._predict_reserve(features)
                rows.append({"policy_id": policy.policy_id, "regime": regime.name, "reserve": reserve})
        return pd.DataFrame(rows)

    def portfolio_simulation(self, policies: list[Policy]) -> pd.DataFrame:
        """Predict portfolio-level reserves policy by policy.

        Args:
            policies: Portfolio to value.

        Returns:
            pd.DataFrame: Policy-level reserves plus portfolio aggregate.

        Business Interpretation:
            This is the digital twin's portfolio valuation view, useful for capital
            planning and segment comparisons.
        """

        rows: list[dict[str, float | str]] = []
        self.model.eval()
        with torch.no_grad():
            for policy in policies:
                reserve = self._predict_reserve(self._feature_tensor(policy, 0.0))
                rows.append({"policy_id": policy.policy_id, "reserve": reserve})
        frame = pd.DataFrame(rows)
        frame["portfolio_reserve"] = frame["reserve"].sum()
        return frame

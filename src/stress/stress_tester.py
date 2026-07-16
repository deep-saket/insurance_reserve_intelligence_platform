"""Stress testing workflows.

Created: 2026-05-31
Purpose: Apply insurance and macro shocks to reserve predictions and export scenario outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.actuarial.policy import Policy
from src.data.dataset import FEATURE_INDEX, FEATURE_SCALES, build_policy_feature_array, normalize_raw_feature_array
from src.utils.config import StressScenarioConfig
from src.visualization.stress_plots import plot_stress_comparison


@dataclass(slots=True)
class StressResult:
    """Stress test output summary.

    Attributes:
        policy_id: Policy identifier.
        scenario_name: Name of the applied stress scenario.
        before_reserve: Baseline reserve before the shock.
        after_reserve: Reserve after the shock.
        delta: Absolute reserve change.
        delta_pct: Relative reserve change.

    Business Interpretation:
        This is the per-policy shock impact record used to explain which contracts
        contribute most to adverse reserve movement.
    """

    policy_id: str
    scenario_name: str
    before_reserve: float
    after_reserve: float
    delta: float
    delta_pct: float


class StressTester:
    """Apply actuarial and macro shocks to reserve predictions.

    Scientific Context:
        The stress tester perturbs key state variables such as mortality,
        interest rate, premium persistence, and benefit size, then re-evaluates
        the reserve model.

    Business Interpretation:
        This is the scenario engine used for risk review, capital planning, and
        management what-if analysis.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        config: StressScenarioConfig,
        target_mean: float,
        target_std: float,
    ) -> None:
        """Initialize the stress tester.

        Args:
            model: Trained reserve model.
            device: Execution device for inference.
            config: Stress scenario configuration.
            target_mean: Mean used to standardize reserve-ratio targets.
            target_std: Standard deviation used to standardize reserve-ratio targets.
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)

    def _policy_features(self, policy: Policy, time_point: float = 0.0) -> torch.Tensor:
        """Build a baseline feature tensor for one policy.

        Args:
            policy: Policy to transform.
            time_point: Elapsed policy time for valuation.

        Returns:
            torch.Tensor: Single-row feature tensor.
        """
        raw = build_policy_feature_array(policy=policy, time_point=time_point)
        normalized = normalize_raw_feature_array(raw)
        return torch.tensor(normalized, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _predict(self, features: torch.Tensor) -> float:
        """Predict a scalar reserve value.

        Args:
            features: Feature tensor for model inference.

        Returns:
            float: Predicted reserve.
        """
        self.model.eval()
        with torch.no_grad():
            z = float(self.model(features).item())
        sum_assured = float(features[0, FEATURE_INDEX["sum_assured"]].item()) * FEATURE_SCALES["sum_assured"]
        return float((z * self.target_std + self.target_mean) * sum_assured)

    def _apply_shock(self, policy: Policy, scenario_name: str) -> torch.Tensor:
        """Apply a named shock to a policy feature vector.

        Args:
            policy: Policy to stress.
            scenario_name: Stress scenario identifier.

        Returns:
            torch.Tensor: Shocked feature tensor.

        Raises:
            ValueError: If the scenario name is unsupported.

        Business Interpretation:
            Each branch corresponds to a common actuarial or macro risk story such
            as higher mortality, lower rates, inflation pressure, or worse lapse behavior.
        """
        features = self._policy_features(policy)
        shocked = features.clone()
        if scenario_name == "mortality_shock":
            shocked[:, FEATURE_INDEX["mortality"]] *= 1.0 + self.config.mortality_shock
        elif scenario_name == "interest_rate_shock":
            shocked[:, FEATURE_INDEX["scenario_interest_rate"]] += (
                self.config.interest_rate_shock / FEATURE_SCALES["scenario_interest_rate"]
            )
        elif scenario_name == "inflation_shock":
            shocked[:, FEATURE_INDEX["sum_assured"]] *= 1.0 + self.config.inflation_shock
            shocked[:, FEATURE_INDEX["premium"]] *= 1.0 + 0.5 * self.config.inflation_shock
        elif scenario_name == "longevity_shock":
            shocked[:, FEATURE_INDEX["mortality"]] *= 1.0 + self.config.longevity_shock
        elif scenario_name == "lapse_shock":
            shocked[:, FEATURE_INDEX["premium"]] *= 1.0 - self.config.lapse_shock
        else:
            raise ValueError(f"Unsupported stress scenario: {scenario_name}")
        return shocked

    def run_scenario(self, policies: list[Policy], scenario_name: str) -> pd.DataFrame:
        """Run one stress scenario across a portfolio.

        Args:
            policies: Portfolio to stress.
            scenario_name: Stress scenario identifier.

        Returns:
            pd.DataFrame: Policy-level before-and-after reserve comparison.

        Business Interpretation:
            This is the scenario result table used to assess reserve vulnerability
            under a single adverse narrative.
        """

        results: list[StressResult] = []
        for policy in policies:
            baseline_features = self._policy_features(policy)
            shocked_features = self._apply_shock(policy, scenario_name)
            before = self._predict(baseline_features)
            after = self._predict(shocked_features)
            delta = after - before
            delta_pct = delta / before if abs(before) > 1e-12 else np.nan
            results.append(
                StressResult(
                    policy_id=policy.policy_id,
                    scenario_name=scenario_name,
                    before_reserve=before,
                    after_reserve=after,
                    delta=delta,
                    delta_pct=delta_pct,
                )
            )
        frame = pd.DataFrame([asdict(result) for result in results])
        return frame

    def run_all(self, policies: list[Policy], output_dir: str) -> dict[str, pd.DataFrame]:
        """Run all configured scenarios, write CSVs, and generate plots.

        Args:
            policies: Portfolio to stress.
            output_dir: Directory used for CSV and plot exports.

        Returns:
            dict[str, pd.DataFrame]: Scenario result tables keyed by scenario name.

        Business Interpretation:
            This is the batch stress workflow for generating a compact risk pack of
            reserve impacts across all standard scenarios.
        """

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        outputs: dict[str, pd.DataFrame] = {}
        for scenario_name in [
            "mortality_shock",
            "interest_rate_shock",
            "inflation_shock",
            "longevity_shock",
            "lapse_shock",
        ]:
            frame = self.run_scenario(policies, scenario_name)
            csv_path = Path(output_dir) / f"{scenario_name}.csv"
            frame.to_csv(csv_path, index=False)
            plot_stress_comparison(frame, str(Path(output_dir) / f"{scenario_name}.png"))
            outputs[scenario_name] = frame
        return outputs

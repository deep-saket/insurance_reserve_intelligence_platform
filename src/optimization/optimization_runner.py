"""End-to-end optimization runner.

Created: 2026-07-02
Purpose: Run pricing, capital, product, and portfolio optimization workflows.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.data.simulator import ScenarioDefinition
from src.optimization.capital import optimize_capital_allocation
from src.optimization.objectives import portfolio_metrics, profit_breakdown
from src.optimization.optimizers import OptimizationResult
from src.optimization.portfolio import optimize_portfolio_premiums
from src.optimization.predictor import ReservePredictor
from src.optimization.pricing import optimize_premium
from src.optimization.product_design import optimize_product_design
from src.pipeline import build_datasets, build_model, build_simulator
from src.utils.checkpoint import CheckpointManager
from src.utils.config import ConfigLoader, ExperimentConfig, ensure_directories
from src.utils.device import DeviceManager
from src.utils.seed import set_seed
from src.visualization.optimization_plots import plot_metric_sweep, plot_standard_convergence_set


OptimizationMode = Literal["all", "pricing", "capital", "product", "portfolio", "scenario"]


class OptimizationRunner:
    """Orchestrate optimization use cases from trained model to artifacts."""

    def __init__(
        self,
        config: ExperimentConfig,
        checkpoint_path: str | Path | None = None,
        output_dir: str | Path = "artifacts/optimization",
    ) -> None:
        self.config = config
        ensure_directories(self.config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device_manager = DeviceManager(
            preferred_device=self.config.trainer.device,
            prefer_mixed_precision=False,
        )
        self.train_dataset, self.policy, self.portfolio = self._build_training_context()
        self.model = build_model(self.config)
        self._load_checkpoint(checkpoint_path)
        self.predictor = ReservePredictor(
            model=self.model,
            training_statistics=self.train_dataset,
            device=self.device_manager.device,
        )

    @classmethod
    def from_config_path(
        cls,
        config_path: str | Path = "configs/config.yaml",
        checkpoint_path: str | Path | None = None,
        output_dir: str | Path = "artifacts/optimization",
    ) -> OptimizationRunner:
        config = ConfigLoader.load(config_path)
        set_seed(config.seed)
        return cls(config=config, checkpoint_path=checkpoint_path, output_dir=output_dir)
        
    def valuation_time(self, policy=None) -> float:
        policy = self.policy if policy is None else policy
        return min(max(float(policy.term) * 0.5, 1.0), float(policy.term))
        
    def run(self, mode: OptimizationMode = "all") -> dict[str, OptimizationResult]:
        results: dict[str, OptimizationResult] = {}
        if mode in ("all", "pricing"):
            results["pricing"] = self.run_pricing()
        if mode in ("all", "capital"):
            results["capital"] = self.run_capital()
        if mode in ("all", "product"):
            results["product_design"] = self.run_product_design()
        if mode in ("all", "portfolio"):
            results["portfolio"] = self.run_portfolio()
        if mode in ("all", "scenario"):
            self.run_scenarios()
        if results:
            self._write_summary(results)
        return results
    
    def run_pricing(self) -> OptimizationResult:
        result = optimize_premium(
            policy=self.policy,
            predictor=self.predictor,
            time_point=self.valuation_time(),
        )
        self._write_result("pricing", result)
        self._write_history("pricing", result)
        self._write_sweep_plots()
        return result
    
    
    def run_capital(self) -> OptimizationResult:
        result = optimize_capital_allocation(
            policy=self.policy,
            predictor=self.predictor,
            time_point=self.valuation_time(),
        )
        self._write_result("capital", result)
        self._write_history("capital", result)
        return result
    
    
    def run_product_design(self) -> OptimizationResult:
        valuation_time = self.valuation_time()
        baseline = self.baseline_breakdown()
    
        result = optimize_product_design(
            base_policy=self.policy,
            predictor=self.predictor,
            premium_bounds=(max(self.policy.premium * 0.5, 1e-6), self.policy.premium * 2.0),
            coverage_bounds=(max(self.policy.sum_assured * 0.5, 1.0), self.policy.sum_assured * 1.5),
            term_bounds=(max(1.0, self.policy.term * 0.5), self.policy.term * 1.5),
            interest_rate_bounds=(max(-0.01, self.policy.scenario_interest_rate - 0.02), self.policy.scenario_interest_rate + 0.02),
            capital=max(baseline.reserve, 0.0) * 1.5,
            time_point=valuation_time,
        )
    
        self._write_result("product_design", result)
        self._write_history("product_design", result)
        return result
    
    
    def run_portfolio(self) -> OptimizationResult:
        result = optimize_portfolio_premiums(
            policies=self.portfolio,
            predictor=self.predictor,
            diversification_credit=0.15,
            solvency_threshold=1.5,
            time_point=[self.valuation_time(policy) for policy in self.portfolio],
        )
        self._write_result("portfolio", result)
        self._write_history("portfolio", result)
        return result
    
    
    def baseline_breakdown(self):
        return profit_breakdown(
            policy=self.policy,
            predictor=self.predictor,
            time_point=self.valuation_time(),
        )

    def print_before_after(self, results: dict[str, OptimizationResult]) -> None:
        baseline = self.baseline_breakdown()
        print("-" * 56)
        print("Original Policy")
        print(f"Premium        {baseline.premium_income:,.2f}")
        print(f"Reserve        {baseline.reserve:,.2f}")
        print(f"Profit         {baseline.profit:,.2f}")
        print("-" * 56)
        for name, result in results.items():
            print(name.replace("_", " ").title())
            for key, value in result.optimal_values.items():
                print(f"{key:<14} {value:,.2f}")
            print(f"Objective      {result.objective_value:,.2f}")
            for key, value in result.diagnostics.items():
                print(f"{key:<14} {value:,.2f}")
            print("-" * 56)
    def run_scenarios(self) -> pd.DataFrame:
        simulator = build_simulator(self.config)
    
        scenarios = {
            "base": ScenarioDefinition(),
            "interest_plus_2pct": ScenarioDefinition(interest_rate_shift=0.02),
            "mortality_plus_20pct": ScenarioDefinition(mortality_multiplier=1.20),
            "inflation_plus_10pct": ScenarioDefinition(premium_multiplier=1.10),
        }
    
        rows: list[dict[str, float | str | bool]] = []
    
        for name, scenario in scenarios.items():
            policies = simulator.generate_scenario_policies(self.portfolio, scenario)
            valuation_times = [self.valuation_time(policy) for policy in policies]
    
            rows.append(
                self._result_row(
                    f"{name}_pricing",
                    optimize_premium(
                        policies[0],
                        self.predictor,
                        time_point=valuation_times[0],
                    ),
                )
            )
            rows.append(
                self._result_row(
                    f"{name}_capital",
                    optimize_capital_allocation(
                        policies[0],
                        self.predictor,
                        time_point=valuation_times[0],
                    ),
                )
            )
            rows.append(
                self._result_row(
                    f"{name}_portfolio",
                    optimize_portfolio_premiums(
                        policies,
                        self.predictor,
                        time_point=valuation_times,
                    ),
                )
            )
    
        frame = pd.DataFrame(rows)
        frame.to_csv(self.output_dir / "scenario.csv", index=False)
        return frame
    
    def _build_training_context(self):
        train_dataset, _, _, test_policies = build_datasets(self.config)
        portfolio = test_policies[: min(5, len(test_policies))]
        return train_dataset, portfolio[0], portfolio

    def _load_checkpoint(self, checkpoint_path: str | Path | None) -> None:
        path = Path(checkpoint_path) if checkpoint_path is not None else Path(self.config.paths.checkpoints_dir) / "best_model.pt"
        if not path.exists():
            return
        checkpoint = CheckpointManager(str(path.parent)).load(path, map_location=self.device_manager.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        self.model.load_state_dict(state_dict)

    def _write_result(self, name: str, result: OptimizationResult) -> None:
        pd.DataFrame([self._result_row(name, result)]).to_csv(self.output_dir / f"{name}.csv", index=False)

    def _write_history(self, name: str, result: OptimizationResult) -> None:
        if not result.history:
            return
        frame = pd.DataFrame(result.history)
        frame.to_csv(self.output_dir / f"{name}_convergence.csv", index=False)
        plot_standard_convergence_set(frame, self.output_dir, name)

    def _write_summary(self, results: dict[str, OptimizationResult]) -> None:
        baseline = self.baseline_breakdown()
        lines = [
            "Optimization Summary",
            "=" * 60,
            "Original Policy",
            f"Premium: {baseline.premium_income:,.2f}",
            f"Reserve: {baseline.reserve:,.2f}",
            f"Profit: {baseline.profit:,.2f}",
            "",
        ]
        for name, result in results.items():
            lines.extend(
                [
                    name.replace("_", " ").title(),
                    f"Success: {result.success}",
                    f"Objective: {result.objective_value:,.2f}",
                    f"Optimal Values: {result.optimal_values}",
                    f"Diagnostics: {result.diagnostics}",
                    "",
                ]
            )
        (self.output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    def _write_sweep_plots(self) -> None:
        sweep = self._metric_sweep()
        sweep.to_csv(self.output_dir / "business_sweeps.csv", index=False)
        for variable in ("premium", "sum_assured", "interest_rate"):
            subset = sweep[sweep["variable"] == variable]
            plot_metric_sweep(subset, "value", "profit", self.output_dir / f"profit_vs_{variable}.png", f"Profit vs {variable.replace('_', ' ').title()}")
            plot_metric_sweep(subset, "value", "reserve", self.output_dir / f"reserve_vs_{variable}.png", f"Reserve vs {variable.replace('_', ' ').title()}")
            

    def _metric_sweep(self) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        valuation_time = self.valuation_time()
    
        definitions = {
            "premium": np.linspace(max(self.policy.premium * 0.5, 1e-6), self.policy.premium * 1.8, 30),
            "sum_assured": np.linspace(max(self.policy.sum_assured * 0.5, 1.0), self.policy.sum_assured * 1.5, 30),
            "scenario_interest_rate": np.linspace(
                max(-0.01, self.policy.scenario_interest_rate - 0.03),
                self.policy.scenario_interest_rate + 0.03,
                30,
            ),
        }
    
        for variable, values in definitions.items():
            for value in values:
                if variable == "scenario_interest_rate":
                    candidate = replace(
                        self.policy,
                        scenario_interest_rate=float(value),
                        interest_rate=float(value),
                    )
                    plot_variable = "interest_rate"
                else:
                    candidate = replace(self.policy, **{variable: float(value)})
                    plot_variable = variable
    
                breakdown = profit_breakdown(
                    policy=candidate,
                    predictor=self.predictor,
                    time_point=valuation_time,
                )
    
                rows.append(
                    {
                        "variable": plot_variable,
                        "value": float(value),
                        "profit": breakdown.profit,
                        "reserve": breakdown.reserve,
                    }
                )
    
        return pd.DataFrame(rows)

    @staticmethod
    def _result_row(name: str, result: OptimizationResult) -> dict[str, float | str | bool]:
        row: dict[str, float | str | bool] = {
            "name": name,
            "objective_name": result.objective_name,
            "objective_value": result.objective_value,
            "success": result.success,
            "method": result.method,
            "message": result.message,
        }
        row.update(result.optimal_values)
        row.update(result.diagnostics)
        return row

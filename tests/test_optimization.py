"""Optimization tests.

Created: 2026-05-31
Purpose: Validate optimization-engine output contracts.
"""

from __future__ import annotations

from src.data.simulator import PolicySimulator
from src.optimization.optimizer_engine import OptimizationEngine
from src.pipeline import build_model
from src.utils.config import ConfigLoader
from src.utils.device import DeviceManager


def test_optimization_engine_returns_result_objects() -> None:
    """Verify that target reserve optimization returns a populated result object."""
    config = ConfigLoader.load("configs/config.yaml")
    simulator = PolicySimulator(
        age_range=(35, 35),
        term_range=(15, 15),
        interest_rate_range=(0.04, 0.04),
        sum_assured_range=(150000.0, 150000.0),
        seed=11,
    )
    policy = simulator.generate_random_policies(1)[0]
    engine = OptimizationEngine(
        model=build_model(config),
        device=DeviceManager(prefer_mixed_precision=False).device,
        config=config.optimization,
        target_mean=0.0,
        target_std=1.0,
    )
    result = engine.target_reserve_optimization(policy, target_reserve=1000.0)
    assert result.variable_name == "scenario_interest_rate"

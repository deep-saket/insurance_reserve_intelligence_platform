"""Digital twin tests.

Created: 2026-05-31
Purpose: Validate digital twin reserve forecasting behavior.
"""

from __future__ import annotations

from src.digital_twin.engine import DigitalTwinEngine
from src.pipeline import build_model
from src.utils.config import ConfigLoader
from src.utils.device import DeviceManager
from src.data.simulator import PolicySimulator


def test_digital_twin_forecast_has_rows() -> None:
    """Verify that reserve forecasting returns the requested number of rows."""
    config = ConfigLoader.load("configs/config.yaml")
    simulator = PolicySimulator(
        age_range=(40, 40),
        term_range=(10, 10),
        interest_rate_range=(0.03, 0.03),
        sum_assured_range=(100000.0, 100000.0),
        seed=3,
    )
    policy = simulator.generate_random_policies(1)[0]
    engine = DigitalTwinEngine(
        model=build_model(config),
        device=DeviceManager(prefer_mixed_precision=False).device,
        config=config.digital_twin,
        target_mean=0.0,
        target_std=1.0,
    )
    frame = engine.reserve_forecast(policy, steps=5)
    assert len(frame) == 5

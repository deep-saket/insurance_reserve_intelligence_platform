"""Run optimization workflows for the ActuaryTwin platform.

Created: 2026-05-31
Purpose: Execute reserve-driven pricing and target-search workflows.
"""

from __future__ import annotations

from pathlib import Path

from src.optimization.optimizer_engine import OptimizationEngine
from src.pipeline import build_dataloaders, build_model
from src.utils.checkpoint import CheckpointManager
from src.utils.config import ConfigLoader, ensure_directories
from src.utils.device import DeviceManager
from src.utils.seed import set_seed


def main() -> None:
    """Run the optimization entrypoint.

    Business Interpretation:
        This script exposes the model as a decision-support engine for pricing and
        reserve-target calibration experiments.
    """
    config = ConfigLoader.load(Path("configs/config.yaml"))
    ensure_directories(config)
    set_seed(config.seed)

    _, _, _, test_dataset, test_policies = build_dataloaders(config)
    device_manager = DeviceManager(preferred_device=config.trainer.device, prefer_mixed_precision=False)
    model = build_model(config)
    checkpoint_path = Path(config.paths.checkpoints_dir) / "best_model.pt"
    if checkpoint_path.exists():
        checkpoint = CheckpointManager(config.paths.checkpoints_dir).load(checkpoint_path, map_location=device_manager.device)
        model.load_state_dict(checkpoint["model_state_dict"])

    engine = OptimizationEngine(
        model=model,
        device=device_manager.device,
        config=config.optimization,
        target_mean=test_dataset.target_mean,
        target_std=test_dataset.target_std,
    )
    policy = test_policies[0]
    reserve_result = engine.target_reserve_optimization(policy, target_reserve=50_000.0)
    premium_result = engine.premium_optimization(policy)
    constrained_result = engine.constrained_premium_optimization(policy)
    print(reserve_result)
    print(premium_result)
    print(constrained_result)


if __name__ == "__main__":
    main()

"""Run stress testing for the ActuaryTwin platform.

Created: 2026-05-31
Purpose: Execute standard reserve stress scenarios and export reports.
"""

from __future__ import annotations

from pathlib import Path

from src.pipeline import build_dataloaders, build_model
from src.stress.stress_tester import StressTester
from src.utils.checkpoint import CheckpointManager
from src.utils.config import ConfigLoader, ensure_directories
from src.utils.device import DeviceManager
from src.utils.seed import set_seed


def main() -> None:
    """Run the stress-testing entrypoint.

    Business Interpretation:
        This script produces a reusable reserve stress pack for actuarial and risk
        review workflows.
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

    tester = StressTester(
        model=model,
        device=device_manager.device,
        config=config.stress,
        target_mean=test_dataset.target_mean,
        target_std=test_dataset.target_std,
    )
    results = tester.run_all(test_policies[:10], output_dir=config.paths.reports_dir)
    print(
        f"run={config.trainer.run_name} device={device_manager.summary()} "
        f"generated {len(results)} scenario reports in {config.paths.reports_dir}"
    )


if __name__ == "__main__":
    main()

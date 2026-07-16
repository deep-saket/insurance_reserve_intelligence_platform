"""Evaluate the trained ActuaryTwin model.

Created: 2026-05-31
Purpose: Evaluate reserve accuracy and export sensitivity analytics.
"""

from __future__ import annotations

from pathlib import Path

import torch
import pandas as pd

from src.actuarial.actuarial_solver import ThieleSolver
from src.data.dataset import build_policy_feature_array, normalize_raw_feature_array
from src.visualization.reserve_plots import plot_reserve_trajectory

from src.evaluators.evaluator import ReserveEvaluator
from src.pipeline import build_dataloaders, build_model
from src.utils.checkpoint import CheckpointManager
from src.utils.config import ConfigLoader, ensure_directories
from src.utils.device import DeviceManager
from src.utils.seed import set_seed


def main() -> None:
    """Run the evaluation entrypoint.

    Business Interpretation:
        This script validates the trained reserve engine before it is used in
        downstream analysis such as optimization or stress testing.
    """
    config = ConfigLoader.load(Path("configs/config.yaml"))
    ensure_directories(config)
    set_seed(config.seed)

    _, _, test_loader, test_dataset, test_policies = build_dataloaders(config)
    device_manager = DeviceManager(preferred_device=config.trainer.device, prefer_mixed_precision=False)
    model = build_model(config)

    checkpoint_path = Path(config.paths.checkpoints_dir) / "best_model.pt"
    if checkpoint_path.exists():
        checkpoint = CheckpointManager(config.paths.checkpoints_dir).load(checkpoint_path, map_location=device_manager.device)
        model.load_state_dict(checkpoint["model_state_dict"])

    evaluator = ReserveEvaluator(model=model, device=device_manager.device)
    result = evaluator.evaluate(test_loader)
    print(
        f"run={config.trainer.run_name} device={device_manager.summary()} "
        f"MSE={result.mse:.6f} MAE={result.mae:.6f} RMSE={result.rmse:.6f} R2={result.r2:.6f}"
    )

    sample_features = torch.stack([test_dataset[i]["features"] for i in range(100)])
    sample_raw_features = torch.stack([test_dataset[i]["raw_features"] for i in range(100)])
    output_csv = Path(config.paths.reports_dir) / "sensitivity_report.csv"
    evaluator.generate_sensitivity_report(
        sample_features,
        str(output_csv),
        raw_features=sample_raw_features,
        target_mean=test_dataset.target_mean,
        target_std=test_dataset.target_std,
    )
    print(f"Sensitivity report written to {output_csv}")

    # ==========================================
    # Reserve Trajectory Plot
    # ==========================================
    policy = test_policies[0]

    solver = ThieleSolver(
        method=config.solver.method,
        integration_step=config.solver.integration_step,
        rtol=config.solver.rtol,
        atol=config.solver.atol,
    )

    trajectory = solver.solve(
        policy=policy,
        num_steps=config.data.time_steps,
    )

    pinn_reserves: list[float] = []
    model.eval()
    for time_point in trajectory.times:
        features = torch.tensor(
            normalize_raw_feature_array(
                build_policy_feature_array(policy=policy, time_point=float(time_point))
            ),
            dtype=torch.float32,
            device=device_manager.device,
        ).unsqueeze(0)
        with torch.no_grad():
            prediction_z = model(features)
        reserve = (
            (prediction_z.item() * test_dataset.target_std + test_dataset.target_mean)
            * policy.sum_assured
        )
        pinn_reserves.append(float(reserve))

    trajectory_df = pd.DataFrame(
        {
            "time": trajectory.times,
            "classical_reserve": trajectory.reserves,
            "pinn_reserve": pinn_reserves,
        }
    )

    reserve_plot_path = (
        Path(config.paths.reports_dir)
        / "reserve_trajectory_comparison.png"
    )

    plot_reserve_trajectory(
        trajectory_df,
        str(reserve_plot_path),
    )

    print(f"Reserve plot written to {reserve_plot_path}")


if __name__ == "__main__":
    main()

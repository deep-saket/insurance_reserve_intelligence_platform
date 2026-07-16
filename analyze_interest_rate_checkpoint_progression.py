"""Interest-rate checkpoint progression analysis.

Created: 2026-07-02
Purpose: Evaluate how reserve-vs-interest-rate behavior evolves across saved
training checkpoints and export progression plots for debugging.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.actuarial.actuarial_solver import ThieleSolver
from src.data.dataset import build_policy_feature_array, normalize_raw_feature_array
from src.pipeline import build_dataloaders, build_model
from src.utils.checkpoint import CheckpointManager
from src.utils.config import ConfigLoader, ensure_directories
from src.utils.device import DeviceManager


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Track reserve-vs-rate curve quality across checkpoints.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config.")
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=None,
        help="Optional cap on the number of epoch checkpoints to analyze.",
    )
    return parser.parse_args()


def _checkpoint_epoch(path: Path) -> int:
    """Extract the epoch number from an ``epoch_XXX.pt`` checkpoint name."""

    return int(path.stem.split("_")[1])


def _predict_reserve(
    model: torch.nn.Module,
    device: torch.device,
    policy,
    time_point: float,
    target_mean: float,
    target_std: float,
) -> float:
    """Predict one reserve value in currency units.

    Args:
        model: Trained neural reserve model.
        device: Inference device.
        policy: Policy to value.
        time_point: Elapsed policy duration.
        target_mean: Training-set reserve-ratio mean.
        target_std: Training-set reserve-ratio standard deviation.

    Returns:
        float: Predicted reserve in monetary units.
    """

    features = torch.tensor(
        normalize_raw_feature_array(build_policy_feature_array(policy=policy, time_point=float(time_point))),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        z = float(model(features).item())
    return float((z * target_std + target_mean) * policy.sum_assured)


def _representative_policies(policies: list) -> list:
    """Pick a small deterministic evaluation subset.

    Args:
        policies: Candidate policies.

    Returns:
        list: Deduplicated representative policies.
    """

    candidates = [
        min(policies, key=lambda policy: policy.age),
        max(policies, key=lambda policy: policy.age),
        min(policies, key=lambda policy: policy.sum_assured),
        max(policies, key=lambda policy: policy.sum_assured),
    ]
    selected = []
    seen: set[str] = set()
    for policy in candidates:
        if policy.policy_id not in seen:
            selected.append(policy)
            seen.add(policy.policy_id)
    return selected


def _checkpoint_paths(checkpoint_dir: Path, max_checkpoints: int | None) -> list[Path]:
    """Return ordered epoch checkpoint paths to analyze.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        max_checkpoints: Optional limit on the number of checkpoints.

    Returns:
        list[Path]: Sorted checkpoint paths.
    """

    paths = sorted(checkpoint_dir.glob("epoch_*.pt"), key=_checkpoint_epoch)
    if max_checkpoints is not None and len(paths) > max_checkpoints:
        indices = np.linspace(0, len(paths) - 1, max_checkpoints, dtype=int)
        paths = [paths[index] for index in indices]
    return paths


def main() -> None:
    """Run checkpoint progression analysis for the configured experiment.

    Business Interpretation:
        This produces a training-filmstrip for interest-rate curve quality, so
        the team can see whether reserve-vs-rate behavior is improving steadily,
        stalling, or getting worse as the optimization proceeds.
    """

    args = parse_args()
    config = ConfigLoader.load(Path(args.config))
    ensure_directories(config)

    _, _, _, test_dataset, test_policies = build_dataloaders(config)
    selected_policies = _representative_policies(test_policies)
    rate_grid = np.linspace(config.data.interest_rate_min, config.data.interest_rate_max, 11)

    device_manager = DeviceManager(preferred_device=config.trainer.device, prefer_mixed_precision=False)
    solver = ThieleSolver(
        method=config.solver.method,
        integration_step=config.solver.integration_step,
        rtol=config.solver.rtol,
        atol=config.solver.atol,
    )
    checkpoint_dir = Path(config.paths.checkpoints_dir)
    checkpoint_paths = _checkpoint_paths(checkpoint_dir, args.max_checkpoints)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No epoch checkpoints found in {checkpoint_dir}")

    reports_dir = Path(config.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config).to(device_manager.device)
    manager = CheckpointManager(str(checkpoint_dir))

    curve_rows: list[dict[str, float | int | str]] = []
    metric_rows: list[dict[str, float | int]] = []

    for checkpoint_path in checkpoint_paths:
        checkpoint = manager.load(checkpoint_path, map_location=device_manager.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        epoch = int(checkpoint["epoch"])

        checkpoint_curve_rows: list[dict[str, float | int | str]] = []
        for policy in selected_policies:
            t_mid = 0.5 * float(policy.term)
            base_traj = solver.solve(policy=policy, num_steps=config.data.time_steps)
            base_peak_index = int(np.argmax(base_traj.reserves))
            t_peak_base = float(base_traj.times[base_peak_index])

            for rate in rate_grid:
                shocked_policy = replace(policy, scenario_interest_rate=float(rate))
                trajectory = solver.solve(shocked_policy, num_steps=config.data.time_steps)
                classical_mid = float(np.interp(t_mid, trajectory.times, trajectory.reserves))
                classical_basepeak = float(np.interp(t_peak_base, trajectory.times, trajectory.reserves))
                classical_peak = float(np.max(trajectory.reserves))

                pinn_trajectory = np.asarray(
                    [
                        _predict_reserve(
                            model=model,
                            device=device_manager.device,
                            policy=shocked_policy,
                            time_point=float(time_point),
                            target_mean=test_dataset.target_mean,
                            target_std=test_dataset.target_std,
                        )
                        for time_point in trajectory.times
                    ],
                    dtype=float,
                )
                pinn_mid = float(np.interp(t_mid, trajectory.times, pinn_trajectory))
                pinn_basepeak = float(np.interp(t_peak_base, trajectory.times, pinn_trajectory))
                pinn_peak = float(np.max(pinn_trajectory))

                checkpoint_curve_rows.append(
                    {
                        "epoch": epoch,
                        "policy_id": policy.policy_id,
                        "age": policy.age,
                        "term": policy.term,
                        "sum_assured": policy.sum_assured,
                        "scenario_interest_rate": float(rate),
                        "classical_mid_reserve": classical_mid,
                        "pinn_mid_reserve": pinn_mid,
                        "classical_basepeak_reserve": classical_basepeak,
                        "pinn_basepeak_reserve": pinn_basepeak,
                        "classical_peak_reserve": classical_peak,
                        "pinn_peak_reserve": pinn_peak,
                    }
                )

        curve_rows.extend(checkpoint_curve_rows)
        checkpoint_frame = pd.DataFrame(checkpoint_curve_rows)

        def mean_pct_error(classical_column: str, pinn_column: str) -> float:
            classical = checkpoint_frame[classical_column].to_numpy(dtype=float)
            predicted = checkpoint_frame[pinn_column].to_numpy(dtype=float)
            return float((np.abs(predicted - classical) / np.maximum(np.abs(classical), 1.0)).mean() * 100.0)

        metric_rows.append(
            {
                "epoch": epoch,
                "mid_mean_pct_error": mean_pct_error("classical_mid_reserve", "pinn_mid_reserve"),
                "basepeak_time_mean_pct_error": mean_pct_error(
                    "classical_basepeak_reserve",
                    "pinn_basepeak_reserve",
                ),
                "peak_mean_pct_error": mean_pct_error("classical_peak_reserve", "pinn_peak_reserve"),
                "validation_total_loss": float(checkpoint["validation_metrics"]["total_loss"]),
            }
        )

    curve_frame = pd.DataFrame(curve_rows).sort_values(["epoch", "policy_id", "scenario_interest_rate"])
    metrics_frame = pd.DataFrame(metric_rows).sort_values("epoch")
    curve_csv = reports_dir / "checkpoint_interest_curve_progression.csv"
    metrics_csv = reports_dir / "checkpoint_interest_curve_progression_metrics.csv"
    curve_frame.to_csv(curve_csv, index=False)
    metrics_frame.to_csv(metrics_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(metrics_frame["epoch"], metrics_frame["mid_mean_pct_error"], marker="o", label="Reserve at T/2")
    axes[0].plot(
        metrics_frame["epoch"],
        metrics_frame["basepeak_time_mean_pct_error"],
        marker="o",
        label="Reserve at baseline peak time",
    )
    axes[0].plot(metrics_frame["epoch"], metrics_frame["peak_mean_pct_error"], marker="o", label="Peak reserve")
    axes[0].set_xlabel("Checkpoint epoch")
    axes[0].set_ylabel("Mean percentage error vs Thiele")
    axes[0].set_title("Interest-rate curve error progression")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(metrics_frame["epoch"], metrics_frame["validation_total_loss"], marker="o", color="#b56576")
    axes[1].set_xlabel("Checkpoint epoch")
    axes[1].set_ylabel("Validation total loss")
    axes[1].set_title("Validation loss progression")
    axes[1].grid(alpha=0.25)

    fig.suptitle("Checkpoint progression for reserve-vs-rate behavior", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plot_path = reports_dir / "checkpoint_interest_curve_progression.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Curve rows written to {curve_csv}")
    print(f"Metric rows written to {metrics_csv}")
    print(f"Progression plot written to {plot_path}")
    print(metrics_frame.to_string(index=False))


if __name__ == "__main__":
    main()

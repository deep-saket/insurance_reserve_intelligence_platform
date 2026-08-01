"""Comprehensive PINN model validation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.actuarial.actuarial_solver import ThieleSolver
from src.data.dataset import FEATURE_INDEX, FEATURE_SCALES, build_policy_feature_array
from src.data.simulator import PolicySimulator
from src.pipeline import build_dataloaders, build_model
from src.utils.checkpoint import CheckpointManager
from src.utils.config import ConfigLoader
from src.utils.device import DeviceManager


def load_model(config, device):
    model = build_model(config).to(device)
    checkpoint_path = Path("artifacts") / config.trainer.run_name / "checkpoints" / "best_model.pt"
    checkpoint = CheckpointManager(checkpoint_path.parent).load(
        checkpoint_path,
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def make_features(dataset, raw_features, device, requires_grad=False):
    raw_tensor = torch.tensor(raw_features, dtype=torch.float32)
    features = dataset.normalize_features(raw_tensor).unsqueeze(0).to(device)
    if requires_grad:
        features.requires_grad_(True)
    return features


def predict_one(model, dataset, raw_features, sum_assured, device):
    features = make_features(dataset, raw_features, device)
    with torch.no_grad():
        z = model(features).item()
    return float(dataset.denormalize_target(z, sum_assured))


def predict_trajectory(model, policy, times, dataset, device):
    reserves = []

    for time_point in times:
        raw_features = build_policy_feature_array(
            policy=policy,
            time_point=float(time_point),
        )
        reserve = predict_one(
            model=model,
            dataset=dataset,
            raw_features=raw_features,
            sum_assured=policy.sum_assured,
            device=device,
        )
        reserves.append(reserve)

    return np.asarray(reserves)


def pde_residual(model, policy, times, dataset, device):
    residuals = []

    for time_point in times[1:-1]:
        mortality = policy.mortality_profile.intensity_at(float(time_point))
        raw_features = build_policy_feature_array(
            policy=policy,
            time_point=float(time_point),
            mortality=float(mortality),
        )
        features = make_features(
            dataset=dataset,
            raw_features=raw_features,
            device=device,
            requires_grad=True,
        )

        z = model(features)
        dz_dt_norm = torch.autograd.grad(
            outputs=z,
            inputs=features,
            grad_outputs=torch.ones_like(z),
            create_graph=False,
            retain_graph=False,
        )[0][0, FEATURE_INDEX["time"]]

        dV_dt = (
            dz_dt_norm.item()
            / FEATURE_SCALES["time"]
            * dataset.target_std
            * policy.sum_assured
        )

        reserve_ratio = z.item() * dataset.target_std + dataset.target_mean
        reserve = reserve_ratio * policy.sum_assured

        sum_assured = policy.sum_assured
        interest_rate = policy.scenario_interest_rate
        premium = policy.premium

        rhs = interest_rate * reserve + premium - mortality * (sum_assured - reserve)
        residuals.append((dV_dt - rhs) / max(abs(rhs), 1.0))

    return np.asarray(residuals)


@dataclass
class CheckResult:
    name: str
    passed: bool
    score: float
    summary: str
    details: list[str] = field(default_factory=list)


def check_accuracy(model, policies, solver, dataset, device, n_steps):
    mae_list = []
    rmse_list = []
    rel_err_list = []

    for policy in policies:
        trajectory = solver.solve(policy, num_steps=n_steps)
        classical = np.asarray(trajectory.reserves)
        pinn = predict_trajectory(model, policy, trajectory.times, dataset, device)

        classical_trimmed = classical[:-1]
        pinn_trimmed = pinn[:-1]

        mae = np.mean(np.abs(pinn_trimmed - classical_trimmed))
        rmse = np.sqrt(np.mean((pinn_trimmed - classical_trimmed) ** 2))
        peak = max(np.abs(classical_trimmed).max(), 1.0)
        rel = mae / peak * 100.0

        mae_list.append(mae)
        rmse_list.append(rmse)
        rel_err_list.append(rel)

    mean_mae = np.mean(mae_list)
    mean_rmse = np.mean(rmse_list)
    mean_rel = np.mean(rel_err_list)
    p90_rel = np.percentile(rel_err_list, 90)

    score = max(0.0, 100.0 - mean_rel * 5.0)
    passed = mean_rel < 10.0

    return CheckResult(
        name="Accuracy",
        passed=passed,
        score=round(score, 1),
        summary=f"Mean MAE: {mean_mae:,.0f} | Mean relative error: {mean_rel:.2f}% | P90 relative: {p90_rel:.2f}%",
        details=[
            f"Mean RMSE: {mean_rmse:,.0f}",
            f"Best policy rel error: {min(rel_err_list):.2f}%",
            f"Worst policy rel error: {max(rel_err_list):.2f}%",
            "PASS threshold: mean relative error < 10%",
        ],
    ), rel_err_list


def check_boundary(model, policies, dataset, device):
    errors = []

    for policy in policies:
        raw_features = build_policy_feature_array(
            policy=policy,
            time_point=float(policy.term),
        )
        reserve = predict_one(
            model=model,
            dataset=dataset,
            raw_features=raw_features,
            sum_assured=policy.sum_assured,
            device=device,
        )
        errors.append(abs(reserve))

    mean_error = np.mean(errors)
    max_error = np.max(errors)

    score = max(0.0, 100.0 - mean_error / 10.0)
    passed = mean_error < 500.0

    return CheckResult(
        name="Boundary condition V(T) = 0",
        passed=passed,
        score=round(score, 1),
        summary=f"Mean |V(T)|: {mean_error:,.1f} | Max |V(T)|: {max_error:,.1f}",
        details=[
            "Classical Thiele has V(T) = 0 by construction.",
            "PASS threshold: mean |V(T)| < 500",
        ],
    ), errors


def check_pde(model, policies, dataset, device, n_steps):
    all_residuals = []

    for policy in policies[:30]:
        times = np.linspace(0.0, float(policy.term), n_steps)
        residuals = pde_residual(model, policy, times, dataset, device)
        all_residuals.extend(residuals.tolist())

    all_residuals = np.asarray(all_residuals)
    mean_abs = np.mean(np.abs(all_residuals))
    max_abs = np.max(np.abs(all_residuals))

    score = max(0.0, 100.0 - mean_abs / 10.0)
    passed = mean_abs < 500.0

    return CheckResult(
        name="Physics (Thiele PDE residual)",
        passed=passed,
        score=round(score, 1),
        summary=f"Mean |relative PDE residual|: {mean_abs:,.3f} | Max: {max_abs:,.3f}",
        details=[
            "Residual = dV/dt - (r*V + P - mu*(S-V)).",
            "PASS threshold: mean relative residual < 500",
        ],
    ), all_residuals


def check_monotonicity(model, policies, dataset, device):
    results = {
        "dV/dr < 0": [],
        "dV/dmu > 0": [],
        "dV/dP > 0": [],
        "dV/dS > 0": [],
    }

    for policy in policies[:50]:
        time_point = float(policy.term) / 2.0
        mortality = policy.mortality_profile.intensity_at(time_point)

        base_raw = build_policy_feature_array(
            policy=policy,
            time_point=time_point,
            mortality=mortality,
        )
        base_reserve = predict_one(model, dataset, base_raw, policy.sum_assured, device)

        checks = [
            ("dV/dr < 0", "scenario_interest_rate", 0.01, False, policy.sum_assured),
            ("dV/dmu > 0", "mortality", 0.0005, True, policy.sum_assured),
            ("dV/dP > 0", "premium_ratio", 0.0003, True, policy.sum_assured),
            ("dV/dS > 0", "sum_assured", policy.sum_assured * 0.10, True, policy.sum_assured * 1.10),
        ]

        for label, feature_name, raw_delta, expected_positive, bumped_sum_assured in checks:
            bumped_raw = base_raw.copy()
            feature_index = FEATURE_INDEX[feature_name]
            bumped_raw[feature_index] += raw_delta

            bumped_reserve = predict_one(
                model=model,
                dataset=dataset,
                raw_features=bumped_raw,
                sum_assured=bumped_sum_assured,
                device=device,
            )

            delta_reserve = bumped_reserve - base_reserve
            results[label].append((delta_reserve > 0.0) == expected_positive)

    scores = {key: np.mean(values) * 100.0 for key, values in results.items()}
    overall = np.mean(list(scores.values()))
    passed = overall >= 70.0

    return CheckResult(
        name="Actuarial monotonicity",
        passed=passed,
        score=round(overall, 1),
        summary=f"Overall sign-correctness: {overall:.1f}%",
        details=[f"{key}: {value:.0f}% correct" for key, value in scores.items()],
    ), scores


def check_generalisation(model, solver, dataset, device, config, n_steps):
    ood_simulator = PolicySimulator(
        age_range=(70, 80),
        term_range=(30, 40),
        interest_rate_range=(0.08, 0.12),
        sum_assured_range=(1_000_000.0, 2_000_000.0),
        mortality_scale=config.data.mortality_scale,
        mortality_shape=config.data.mortality_shape,
        mortality_reference_age=config.data.mortality_reference_age,
        premium_loading=config.data.premium_loading,
        max_expiry_age=120,
        seed=999,
    )

    ood_policies = ood_simulator.generate_random_policies(100)
    rel_errors = []

    for policy in ood_policies:
        trajectory = solver.solve(policy, num_steps=n_steps)
        classical = np.asarray(trajectory.reserves[:-1])
        pinn = predict_trajectory(model, policy, trajectory.times, dataset, device)[:-1]

        peak = max(np.abs(classical).max(), 1.0)
        rel_errors.append(np.mean(np.abs(pinn - classical)) / peak * 100.0)

    mean_rel = np.mean(rel_errors)
    p90_rel = np.percentile(rel_errors, 90)

    score = max(0.0, 100.0 - mean_rel * 5.0)
    passed = mean_rel < 15.0

    return CheckResult(
        name="Generalisation (unseen policies)",
        passed=passed,
        score=round(score, 1),
        summary=f"OOD mean relative error: {mean_rel:.2f}% | P90: {p90_rel:.2f}%",
        details=["PASS threshold: OOD mean relative error < 15%"],
    ), rel_errors


def plot_validation(results, rel_errors, boundary_errors, pde_residuals, mono_scores, ood_errors, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    ax = axes[0]
    names = [result.name.split("(")[0].strip()[:30] for result in results]
    scores = [result.score for result in results]
    colors = ["#2ecc71" if result.passed else "#e74c3c" for result in results]
    bars = ax.barh(names, scores, color=colors, edgecolor="white")
    ax.set_xlim(0, 105)
    ax.axvline(70, color="#333", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Score (0-100)")
    ax.set_title("Validation scorecard", fontweight="bold")
    for bar, score in zip(bars, scores):
        ax.text(score + 1, bar.get_y() + bar.get_height() / 2, f"{score:.0f}", va="center")

    ax = axes[1]
    ax.hist(rel_errors, bins=20, color="#4C6EF5", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(rel_errors), color="#e74c3c", linewidth=2, label=f"Mean {np.mean(rel_errors):.1f}%")
    ax.axvline(10, color="#333", linewidth=1, linestyle="--", alpha=0.6, label="10% threshold")
    ax.set_xlabel("Relative error (%)")
    ax.set_ylabel("Policy count")
    ax.set_title("Accuracy: relative error distribution", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.hist(boundary_errors, bins=20, color="#7950F2", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(boundary_errors), color="#e74c3c", linewidth=2, label=f"Mean {np.mean(boundary_errors):,.0f}")
    ax.set_xlabel("|V(T)|")
    ax.set_ylabel("Policy count")
    ax.set_title("Boundary: |V(T)| should be near 0", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[3]
    clipped = np.clip(pde_residuals, -5, 5)
    ax.hist(clipped, bins=30, color="#2F9E44", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="#333", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Relative PDE residual")
    ax.set_ylabel("Count")
    ax.set_title("Physics: Thiele PDE residual", fontweight="bold")

    ax = axes[4]
    labels = list(mono_scores.keys())
    vals = list(mono_scores.values())
    colors = ["#2ecc71" if value >= 70 else "#e74c3c" for value in vals]
    ax.bar(labels, vals, color=colors, edgecolor="white")
    ax.axhline(70, color="#333", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% correct sign")
    ax.set_title("Monotonicity: sign correctness", fontweight="bold")
    ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    for i, value in enumerate(vals):
        ax.text(i, value + 1, f"{value:.0f}%", ha="center", fontsize=8)

    ax = axes[5]
    ax.hist(rel_errors, bins=15, alpha=0.7, color="#4C6EF5", label="In-distribution", edgecolor="white")
    ax.hist(ood_errors, bins=15, alpha=0.7, color="#F03E3E", label="Out-of-distribution", edgecolor="white")
    ax.axvline(np.mean(rel_errors), color="#4C6EF5", linewidth=2, linestyle="--")
    ax.axvline(np.mean(ood_errors), color="#F03E3E", linewidth=2, linestyle="--")
    ax.set_xlabel("Relative error (%)")
    ax.set_ylabel("Policy count")
    ax.set_title("Generalisation: in vs out of distribution", fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle("PINN Model Validation Report", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  plots -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-policies", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = ConfigLoader.load(Path("configs/config.yaml"))
    device = DeviceManager(
        preferred_device=config.trainer.device,
        prefer_mixed_precision=False,
    ).device

    print("Loading model...")
    model = load_model(config, device)

    print("Building test data...")
    _, _, _, test_dataset, test_policies = build_dataloaders(config)

    solver = ThieleSolver(
        method=config.solver.method,
        integration_step=config.solver.integration_step,
        rtol=config.solver.rtol,
        atol=config.solver.atol,
    )

    policies = test_policies[: args.n_policies]
    n_steps = config.data.time_steps

    print(f"\nRunning 5 validation checks on {len(policies)} policies...\n")

    results = []

    print("  [1/5] Accuracy...")
    r1, rel_errors = check_accuracy(model, policies, solver, test_dataset, device, n_steps)
    results.append(r1)

    print("  [2/5] Boundary condition...")
    r2, boundary_errors = check_boundary(model, policies, test_dataset, device)
    results.append(r2)

    print("  [3/5] PDE residual...")
    r3, pde_residuals = check_pde(model, policies, test_dataset, device, n_steps)
    results.append(r3)

    print("  [4/5] Monotonicity...")
    r4, mono_scores = check_monotonicity(model, policies, test_dataset, device)
    results.append(r4)

    print("  [5/5] Generalisation...")
    r5, ood_errors = check_generalisation(model, solver, test_dataset, device, config, n_steps)
    results.append(r5)

    lines = [
        "=" * 65,
        "  PINN MODEL VALIDATION REPORT",
        "=" * 65,
        "",
    ]

    overall_pass = sum(result.passed for result in results)
    lines.append(f"Overall: {overall_pass}/{len(results)} checks passed\n")

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"{status} [{result.score:5.1f}/100] {result.name}")
        lines.append(f"      {result.summary}")
        if args.verbose:
            for detail in result.details:
                lines.append(f"        {detail}")
        lines.append("")

    report = "\n".join(lines)
    print("\n" + report)

    out_dir = Path("artifacts") / config.trainer.run_name / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "validation_report.txt").write_text(report, encoding="utf-8")

    print("\nGenerating plots...")
    plot_validation(
        results,
        rel_errors,
        boundary_errors,
        pde_residuals,
        mono_scores,
        ood_errors,
        out_dir / "validation_plots.png",
    )

    print(f"\nFull report -> {out_dir / 'validation_report.txt'}")


if __name__ == "__main__":
    main()
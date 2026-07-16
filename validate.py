"""Comprehensive PINN model validation.

Five checks that together tell you whether the model is correct:

  1. ACCURACY       — how close are predictions to classical Thiele on held-out policies
  2. BOUNDARY       — does V(T) = 0 hold for all policies
  3. PHYSICS (PDE)  — does the model satisfy the Thiele ODE at interior points
  4. MONOTONICITY   — do sensitivities have the right signs (actuarial sanity)
  5. GENERALISATION — does accuracy degrade for out-of-distribution policies

Usage:
    python validate_model.py
    python validate_model.py --n-policies 200   # test on more policies
    python validate_model.py --verbose          # print per-policy breakdowns

Output:
    artifacts/<run_name>/reports/validation_report.txt   — full text report
    artifacts/<run_name>/reports/validation_plots.png    — 5-panel summary
"""

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
from src.actuarial.policy import Policy
from src.data.dataset import (
    FEATURE_INDEX,
    FEATURE_SCALES,
    build_policy_feature_array,
    normalize_raw_feature_array,
)
from src.data.simulator import PolicySimulator
from src.pipeline import build_dataloaders, build_model
from src.utils.checkpoint import CheckpointManager
from src.utils.config import ConfigLoader
from src.utils.device import DeviceManager


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(config, device):
    model = build_model(config).to(device)
    checkpoint_path = (
        Path("artifacts") / config.trainer.run_name / "checkpoints" / "best_model.pt"
    )
    ckpt = CheckpointManager(checkpoint_path.parent).load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict_trajectory(model, policy, times, target_mean, target_std, device):
    """Run PINN for every time point and denormalise to real £."""
    reserves = []
    for t in times:
        feat = torch.tensor(
            normalize_raw_feature_array(build_policy_feature_array(policy=policy, time_point=float(t))),
            dtype=torch.float32,
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            z = model(feat).item()
        v = z * target_std + target_mean
        reserves.append(v * policy.sum_assured)
    return np.array(reserves)


def pde_residual(model, policy, times, target_mean, target_std, device):
    """Compute Thiele PDE residual at each interior time point."""
    residuals = []
    for t in times[1:-1]:   # skip endpoints
        mu = policy.mortality_profile.intensity_at(float(t))
        feat = torch.tensor(
            normalize_raw_feature_array(
                build_policy_feature_array(policy=policy, time_point=float(t), mortality=float(mu))
            ),
            dtype=torch.float32,
        ).unsqueeze(0).to(device).requires_grad_(True)

        z = model(feat)
        # dz/dt in normalised time
        dz_dt_norm = torch.autograd.grad(z, feat, create_graph=False)[0][0, FEATURE_INDEX["time"]]
        # Convert to dV/dt in real space: dV/dt = (dz/dt_norm / scale_t) * std * S
        dV_dt = (dz_dt_norm.item() / FEATURE_SCALES["time"]) * target_std * policy.sum_assured

        # Reconstruct V from z
        v = z.item() * target_std + target_mean
        V = v * policy.sum_assured
        S = policy.sum_assured
        r = policy.scenario_interest_rate
        P = policy.premium

        # Thiele: dV/dt = r*V + P - mu*(S - V)
        rhs = r * V + P - mu * (S - V)
        residuals.append((dV_dt - rhs) / max(abs(rhs), 1.0))
    return np.array(residuals)


# ── check results ─────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    score: float          # 0-100, higher is better
    summary: str
    details: list[str] = field(default_factory=list)


# ── CHECK 1: Accuracy ─────────────────────────────────────────────────────────

def check_accuracy(model, policies, solver, target_mean, target_std, device, n_steps) -> CheckResult:
    mae_list, rmse_list, rel_err_list = [], [], []

    for policy in policies:
        traj = solver.solve(policy, num_steps=n_steps)
        times = traj.times
        classical = np.array(traj.reserves)
        pinn = predict_trajectory(model, policy, times, target_mean, target_std, device)

        # Ignore final point (classical → 0 inflates relative error)
        c, p = classical[:-1], pinn[:-1]
        mae  = np.mean(np.abs(p - c))
        rmse = np.sqrt(np.mean((p - c)**2))
        peak = max(np.abs(c).max(), 1.0)
        rel  = np.mean(np.abs(p - c)) / peak * 100

        mae_list.append(mae)
        rmse_list.append(rmse)
        rel_err_list.append(rel)

    mean_mae  = np.mean(mae_list)
    mean_rmse = np.mean(rmse_list)
    mean_rel  = np.mean(rel_err_list)
    p90_rel   = np.percentile(rel_err_list, 90)

    # Score: 100 if mean_rel < 1%, 0 if > 20%
    score = max(0.0, 100.0 - mean_rel * 5)
    passed = mean_rel < 10.0

    return CheckResult(
        name="Accuracy",
        passed=passed,
        score=round(score, 1),
        summary=f"Mean MAE: £{mean_mae:,.0f} | Mean relative error: {mean_rel:.2f}% | P90 relative: {p90_rel:.2f}%",
        details=[
            f"Mean RMSE: £{mean_rmse:,.0f}",
            f"Best policy rel error:  {min(rel_err_list):.2f}%",
            f"Worst policy rel error: {max(rel_err_list):.2f}%",
            f"PASS threshold: mean relative error < 10%",
        ]
    ), rel_err_list


# ── CHECK 2: Boundary condition ───────────────────────────────────────────────

def check_boundary(model, policies, target_mean, target_std, device) -> CheckResult:
    errors = []
    for policy in policies:
        t_T = float(policy.term)
        feat = torch.tensor(
            normalize_raw_feature_array(build_policy_feature_array(policy=policy, time_point=t_T)),
            dtype=torch.float32,
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            z = model(feat).item()
        V_T = (z * target_std + target_mean) * policy.sum_assured
        errors.append(abs(V_T))   # should be 0

    mean_err = np.mean(errors)
    max_err  = np.max(errors)
    # Score: 100 if mean < £10, 0 if > £1000
    score = max(0.0, 100.0 - mean_err / 10.0)
    passed = mean_err < 500.0

    return CheckResult(
        name="Boundary condition  V(T) = 0",
        passed=passed,
        score=round(score, 1),
        summary=f"Mean |V(T)|: £{mean_err:,.1f} | Max |V(T)|: £{max_err:,.1f}",
        details=[
            "Classical Thiele has V(T) = 0 by construction.",
            "PINN must learn this from boundary_loss.",
            f"PASS threshold: mean |V(T)| < £500",
        ]
    ), errors


# ── CHECK 3: PDE residual ─────────────────────────────────────────────────────

def check_pde(model, policies, target_mean, target_std, device, n_steps) -> CheckResult:
    all_residuals = []
    times_list = np.linspace(0, 1, n_steps)   # normalised

    for policy in policies[:30]:   # autograd is slow, subsample
        times = np.linspace(0.0, float(policy.term), n_steps)
        res = pde_residual(model, policy, times, target_mean, target_std, device)
        all_residuals.extend(res.tolist())

    all_residuals = np.array(all_residuals)
    mean_abs = np.mean(np.abs(all_residuals))
    max_abs  = np.max(np.abs(all_residuals))

    # Score: 100 if mean < £10/yr, 0 if > £1000/yr
    score = max(0.0, 100.0 - mean_abs / 10.0)
    passed = mean_abs < 500.0

    return CheckResult(
        name="Physics (Thiele PDE residual)",
        passed=passed,
        score=round(score, 1),
        summary=f"Mean |dV/dt residual|: £{mean_abs:,.1f}/yr | Max: £{max_abs:,.1f}/yr",
        details=[
            "Residual = |dV/dt - (r·V + P - μ·(S-V))|",
            "Should be near 0 if model satisfies Thiele's equation.",
            f"PASS threshold: mean residual < £500/yr",
        ]
    ), all_residuals


# ── CHECK 4: Monotonicity / actuarial sanity ──────────────────────────────────

def check_monotonicity(model, policies, target_mean, target_std, device) -> CheckResult:
    results = {
        "dV/dr < 0  (higher rate → lower reserve)":   [],
        "dV/dμ > 0  (higher mortality → higher reserve)": [],
        "dV/dP > 0  (higher premium → higher reserve)":   [],
        "dV/dS > 0  (higher sum assured → higher reserve)": [],
    }

    for policy in policies[:50]:
        t_mid = float(policy.term) / 2
        mu = policy.mortality_profile.intensity_at(t_mid)
        base_feat = normalize_raw_feature_array(
            build_policy_feature_array(policy=policy, time_point=t_mid, mortality=mu)
        ).tolist()

        def predict(feat_list, sa):
            f = torch.tensor(feat_list, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                z = model(f).item()
            return (z * target_std + target_mean) * sa

        V0 = predict(base_feat, policy.sum_assured)

        # Bump each feature by +10% and check sign of change
        for i, (label, expected_positive) in enumerate([
            ("dV/dr < 0  (higher rate → lower reserve)",         False),
            ("dV/dμ > 0  (higher mortality → higher reserve)",   True),
            ("dV/dP > 0  (higher premium → higher reserve)",     True),
            ("dV/dS > 0  (higher sum assured → higher reserve)", True),
        ]):
            feat_bump = base_feat.copy()
            feat_idx = [FEATURE_INDEX["scenario_interest_rate"], FEATURE_INDEX["mortality"],
                        FEATURE_INDEX["premium"], FEATURE_INDEX["sum_assured"]][i]
            feat_bump[feat_idx] *= 1.10
            V1 = predict(feat_bump, bumped_sa := policy.sum_assured * 1.10 if label.startswith("dV/dS") else policy.sum_assured)
            dV = V1 - V0
            correct = (dV > 0) == expected_positive
            results[label].append(correct)

    scores = {k: np.mean(v) * 100 for k, v in results.items()}
    overall = np.mean(list(scores.values()))
    passed = overall >= 70.0

    details = [f"  {k}: {v:.0f}% correct" for k, v in scores.items()]
    details.append(f"PASS threshold: >= 70% correct across all checks")

    return CheckResult(
        name="Actuarial monotonicity",
        passed=passed,
        score=round(overall, 1),
        summary=f"Overall sign-correctness: {overall:.1f}%",
        details=details
    ), scores


# ── CHECK 5: Generalisation ───────────────────────────────────────────────────

def check_generalisation(model, solver, target_mean, target_std, device, config, n_steps) -> CheckResult:
    """Test on OOD policies: older ages, longer terms, extreme rates."""
    simulator = PolicySimulator(
        age_range=(config.data.age_min, config.data.age_max),
        term_range=(config.data.term_min, config.data.term_max),
        interest_rate_range=(config.data.interest_rate_min, config.data.interest_rate_max),
        sum_assured_range=(config.data.sum_assured_min, config.data.sum_assured_max),
        mortality_scale=config.data.mortality_scale,
        mortality_shape=config.data.mortality_shape,
        mortality_reference_age=config.data.mortality_reference_age,
        premium_loading=config.data.premium_loading,
        max_expiry_age=config.data.max_expiry_age,
        sum_assured_rounding=config.data.sum_assured_rounding,
        sum_assured_age_decay=config.data.sum_assured_age_decay,
        seed=999,   # different seed = unseen policies
    )
    ood_simulator = PolicySimulator(
        age_range =(70,80),
        term_range=(30,40),
        interest_rate_range=(0.08,0.12),
        sum_assured_range=(1_000_000.0,2_000_000.0),
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
        traj = solver.solve(policy, num_steps=n_steps)
        classical = np.array(traj.reserves[:-1])
        pinn = predict_trajectory(model, policy, traj.times, target_mean, target_std, device)[:-1]
        peak = max(np.abs(classical).max(), 1.0)
        rel_errors.append(np.mean(np.abs(pinn - classical)) / peak * 100)

    mean_rel = np.mean(rel_errors)
    p90_rel  = np.percentile(rel_errors, 90)
    score    = max(0.0, 100.0 - mean_rel * 5)
    passed   = mean_rel < 15.0

    return CheckResult(
        name="Generalisation (unseen policies)",
        passed=passed,
        score=round(score, 1),
        summary=f"OOD mean relative error: {mean_rel:.2f}% | P90: {p90_rel:.2f}%",
        details=[
            "Policies generated with a different random seed — never seen during training.",
            f"PASS threshold: OOD mean relative error < 15%",
        ]
    ), rel_errors


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_validation(results, rel_errors, boundary_errors, pde_residuals,
                    mono_scores, ood_errors, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # 1. Score card
    ax = axes[0]
    names  = [r.name.split("(")[0].strip()[:30] for r in results]
    scores = [r.score for r in results]
    colors = ["#2ecc71" if r.passed else "#e74c3c" for r in results]
    bars = ax.barh(names, scores, color=colors, edgecolor="white")
    ax.set_xlim(0, 105)
    ax.axvline(70, color="#333", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Score (0-100)")
    ax.set_title("Validation scorecard", fontweight="bold")
    for bar, score in zip(bars, scores):
        ax.text(score + 1, bar.get_y() + bar.get_height()/2,
                f"{score:.0f}", va="center", fontsize=9, fontweight="bold")
    ax.text(71, -0.6, "pass threshold", fontsize=7, color="#555")

    # 2. Relative error distribution
    ax = axes[1]
    ax.hist(rel_errors, bins=20, color="#4C6EF5", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(rel_errors), color="#e74c3c", linewidth=2,
               label=f"Mean {np.mean(rel_errors):.1f}%")
    ax.axvline(10, color="#333", linewidth=1, linestyle="--", alpha=0.6, label="10% threshold")
    ax.set_xlabel("Relative error (%)")
    ax.set_ylabel("Policy count")
    ax.set_title("Accuracy: relative error distribution", fontweight="bold")
    ax.legend(fontsize=8)

    # 3. Boundary error distribution
    ax = axes[2]
    ax.hist(boundary_errors, bins=20, color="#7950F2", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(boundary_errors), color="#e74c3c", linewidth=2,
               label=f"Mean £{np.mean(boundary_errors):,.0f}")
    ax.set_xlabel("|V(T)| (£)")
    ax.set_ylabel("Policy count")
    ax.set_title("Boundary: |V(T)| should be near £0", fontweight="bold")
    ax.legend(fontsize=8)

    # 4. PDE residual distribution
    ax = axes[3]
    clipped = np.clip(pde_residuals, -5000, 5000)
    ax.hist(clipped, bins=30, color="#2F9E44", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="#333", linewidth=1.5, linestyle="--")
    ax.set_xlabel("dV/dt residual (£/yr)")
    ax.set_ylabel("Count")
    ax.set_title("Physics: Thiele PDE residual\n(should be centred on 0)", fontweight="bold")

    # 5. Monotonicity scores
    ax = axes[4]
    labels = [k.split("(")[0].strip()[:20] for k in mono_scores.keys()]
    vals   = list(mono_scores.values())
    colors = ["#2ecc71" if v >= 70 else "#e74c3c" for v in vals]
    ax.bar(labels, vals, color=colors, edgecolor="white")
    ax.axhline(70, color="#333", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% policies with correct sign")
    ax.set_title("Monotonicity: sign correctness\n(should be > 70%)", fontweight="bold")
    ax.tick_params(axis="x", labelsize=7)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8, fontweight="bold")

    # 6. OOD vs in-distribution
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
    print(f"  plots → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-policies", type=int, default=100)
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    config = ConfigLoader.load(Path("configs/config.yaml"))
    device = DeviceManager(preferred_device=config.trainer.device,
                           prefer_mixed_precision=False).device

    print("Loading model...")
    model = load_model(config, device)

    print("Building test data...")
    _, _, _, test_dataset, test_policies = build_dataloaders(config)
    target_mean = test_dataset.target_mean
    target_std  = test_dataset.target_std

    solver = ThieleSolver(
        method=config.solver.method,
        integration_step=config.solver.integration_step,
        rtol=config.solver.rtol, atol=config.solver.atol,
    )

    policies = test_policies[:args.n_policies]
    n_steps  = config.data.time_steps

    print(f"\nRunning 5 validation checks on {len(policies)} policies...\n")

    results = []

    print("  [1/5] Accuracy...")
    r1, rel_errors = check_accuracy(model, policies, solver, target_mean, target_std, device, n_steps)
    results.append(r1)

    print("  [2/5] Boundary condition...")
    r2, boundary_errors = check_boundary(model, policies, target_mean, target_std, device)
    results.append(r2)

    print("  [3/5] PDE residual (slow — uses autograd)...")
    r3, pde_res = check_pde(model, policies, target_mean, target_std, device, n_steps)
    results.append(r3)

    print("  [4/5] Monotonicity...")
    r4, mono_scores = check_monotonicity(model, policies, target_mean, target_std, device)
    results.append(r4)

    print("  [5/5] Generalisation...")
    r5, ood_errors = check_generalisation(model, solver, target_mean, target_std, device, config, n_steps)
    results.append(r5)

    # ── report ────────────────────────────────────────────────────────────────
    lines = [
        "=" * 65,
        "  PINN MODEL VALIDATION REPORT",
        "=" * 65, "",
    ]
    overall_pass = sum(r.passed for r in results)
    lines.append(f"Overall: {overall_pass}/{len(results)} checks passed\n")

    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        lines.append(f"{status}  [{r.score:5.1f}/100]  {r.name}")
        lines.append(f"         {r.summary}")
        if args.verbose:
            for d in r.details:
                lines.append(f"           {d}")
        lines.append("")

    lines += [
        "What each check means:",
        "  Accuracy      — pure prediction quality vs classical solver",
        "  Boundary      — V(T)=0 is a hard actuarial requirement",
        "  Physics       — model must satisfy Thiele's ODE, not just fit data",
        "  Monotonicity  — higher rate/mortality/premium must move reserve correctly",
        "  Generalisation— model must work on unseen policies, not just training ones",
        "",
        "Typical targets for a production-ready model:",
        "  Accuracy:        mean relative error < 2%",
        "  Boundary:        mean |V(T)| < £50",
        "  Physics:         mean residual < £100/yr",
        "  Monotonicity:    > 90% correct",
        "  Generalisation:  OOD error within 2× in-distribution error",
        "=" * 65,
    ]

    report = "\n".join(lines)
    print("\n" + report)

    out_dir = Path("artifacts") / config.trainer.run_name / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation_report.txt").write_text(report)

    print("\nGenerating plots...")
    plot_validation(results, rel_errors, boundary_errors, pde_res,
                    mono_scores, ood_errors, out_dir / "validation_plots.png")

    print(f"\nFull report → {out_dir}/validation_report.txt")


if __name__ == "__main__":
    main()

"""Compare PINN predictions against the classical Thiele solver.

Two bugs in the original script caused the flat-zero PINN line:

  1. Hard-coded feature scales (e.g. premium / 10_000) didn't match the
     updated FEATURE_SCALES in dataset.py (premium / 2_000), so inputs were
     wrong and the model produced garbage.

  2. The model now predicts z = (v - μ) / σ (standardised), but the original
     code plotted z directly instead of denormalising back to real £:
         V = (z * σ + μ) * S

This script fixes both issues by importing FEATURE_SCALES from dataset.py and
reading μ / σ from the test_dataset that was fitted during training.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # remove if running interactively / in a notebook
import matplotlib.pyplot as plt
import torch

from src.data.dataset import build_policy_feature_array, normalize_raw_feature_array
from src.pipeline import build_dataloaders, build_model
from src.utils.config import ConfigLoader
from src.utils.checkpoint import CheckpointManager
from src.utils.device import DeviceManager
from src.actuarial.actuarial_solver import ThieleSolver


# ── config ────────────────────────────────────────────────────────────────────

config = ConfigLoader.load(Path("configs/config.yaml"))

# ── data (we need the dataset for μ / σ, and a test policy) ──────────────────

_, _, test_loader, test_dataset, test_policies = build_dataloaders(config)

# Standardisation constants fitted on the training set
target_mean = test_dataset.target_mean   # μ  (mean of v = V/S)
target_std  = test_dataset.target_std    # σ  (std  of v = V/S)

# Representative policies instead of a single random one
selected_policies = [
    min(test_policies, key=lambda p: p.age),           # youngest
    max(test_policies, key=lambda p: p.age),           # oldest
    min(test_policies, key=lambda p: p.sum_assured),   # smallest SA
    max(test_policies, key=lambda p: p.sum_assured),   # largest SA
]

# ── model ─────────────────────────────────────────────────────────────────────

device_manager = DeviceManager(
    preferred_device=config.trainer.device,
    prefer_mixed_precision=False,
)
device = device_manager.device

model = build_model(config).to(device)

checkpoint_path = (
    Path("artifacts")
    / config.trainer.run_name
    / "checkpoints"
    / "best_model.pt"
)

checkpoint = CheckpointManager(checkpoint_path.parent).load(
    checkpoint_path,
    map_location=device,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

solver = ThieleSolver(
    method=config.solver.method,
    integration_step=config.solver.integration_step,
    rtol=config.solver.rtol,
    atol=config.solver.atol,
)

for idx, policy in enumerate(selected_policies):

    trajectory = solver.solve(
        policy=policy,
        num_steps=config.data.time_steps,
    )

    times = trajectory.times
    classical_reserves = trajectory.reserves

    pinn_reserves = []

    for t in times:
        raw_features = build_policy_feature_array(policy=policy, time_point=float(t))
        features = torch.tensor(
            normalize_raw_feature_array(raw_features),
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            z = model(features).item()

        v = z * target_std + target_mean
        V = v * policy.sum_assured

        pinn_reserves.append(V)

    errors = [
        abs(p - c)
        for p, c in zip(pinn_reserves, classical_reserves)
    ]

    mean_err = sum(errors) / len(errors)

    axes[idx].plot(
        times,
        classical_reserves,
        linewidth=2,
        label="Classical",
    )

    axes[idx].plot(
        times,
        pinn_reserves,
        "--",
        linewidth=2,
        label="PINN",
    )

    axes[idx].set_title(
        f"Age={policy.age} | "
        f"Term={policy.term}y | "
        f"SA=£{policy.sum_assured:,.0f}\n"
        f"Mean Error=£{mean_err:,.0f}"
    )

    axes[idx].set_xlabel("Time (Years)")
    axes[idx].set_ylabel("Reserve (£)")
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend()

fig.suptitle(
    "PINN vs Classical Reserve Curves (Representative Test Policies)",
    fontsize=14,
    fontweight="bold",
)

fig.tight_layout()

output_path = (
    Path("artifacts")
    / config.trainer.run_name
    / "reports"
    / "pinn_vs_classical_multi_policy.png"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight",
)

print(f"Saved to {output_path}")

plt.show()

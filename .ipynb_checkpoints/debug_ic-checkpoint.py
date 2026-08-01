from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.pipeline import (
    build_model,
    build_solver,
    build_simulator,
    build_datasets,
)

from src.utils.config import (
    ConfigLoader,
    ensure_directories,
)

from src.utils.seed import set_seed
from src.utils.device import DeviceManager
from src.utils.checkpoint import CheckpointManager

from src.data.dataset import FEATURE_SCALES


# ==========================================================
# Load configuration
# ==========================================================

config = ConfigLoader.load(Path("configs/config.yaml"))

ensure_directories(config)

set_seed(config.seed)


device_manager = DeviceManager(
    preferred_device=config.trainer.device,
    prefer_mixed_precision=False,
)


# ==========================================================
# Load trained model
# ==========================================================

model = build_model(config)

checkpoint_path = (
    Path(config.paths.checkpoints_dir)
    / "epoch_132.pt"
)

checkpoint = CheckpointManager(
    config.paths.checkpoints_dir
).load(
    checkpoint_path,
    map_location=device_manager.device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device_manager.device)
model.eval()


# ==========================================================
# Build training dataset
# (needed for target_mean / target_std)
# ==========================================================

train_dataset, _, _, _ = build_datasets(config)

solver = build_solver(config)

simulator = build_simulator(config)


# ==========================================================
# One random policy
# ==========================================================

policy = simulator.generate_random_policies(1)[0]

print("\nPolicy used\n")
print(policy)


# ==========================================================
# Interest-rate sweep
# ==========================================================

rates = np.linspace(0.01, 0.08, 30)

predicted = []
truth = []

for r in rates:

    policy.interest_rate = float(r)

    # ----------------------------------
    # Premium ratio (same as training)
    # ----------------------------------

    premium_ratio = (
        policy.premium
        / max(policy.sum_assured, 1.0)
    )

    # ----------------------------------
    # Build feature vector
    # ----------------------------------

    features = torch.tensor(
        [[
            0.0,
            float(policy.age),
            policy.interest_rate,
            premium_ratio,
            policy.sum_assured,
            policy.mortality_profile.intensity_at(0.0),
        ]],
        dtype=torch.float32,
    )

    # ----------------------------------
    # Normalise EXACTLY like dataset.py
    # ----------------------------------

    features[:, 0] /= FEATURE_SCALES["time"]

    features[:, 1] /= FEATURE_SCALES["age"]

    features[:, 2] = (
        features[:, 2]
        - train_dataset.interest_mean
    ) / train_dataset.interest_std

    features[:, 3] = (
        features[:, 3]
        - train_dataset.premium_mean
    ) / train_dataset.premium_std

    features[:, 4] /= FEATURE_SCALES["sum_assured"]

    features[:, 5] /= FEATURE_SCALES["mortality"]

    features = features.to(device_manager.device)

    # ----------------------------------
    # PINN prediction
    # ----------------------------------

    with torch.no_grad():

        z = model(features).item()
    
        print(
            f"rate={r:.4f} "
            f"z={z:.6f}"
        )
    
       

        reserve = (
            z * train_dataset.target_std
            + train_dataset.target_mean
        )

        reserve *= policy.sum_assured

    predicted.append(reserve)

    # ----------------------------------
    # Classical solver
    # ----------------------------------

    trajectory = solver.solve(
        policy=policy,
        num_steps=config.data.time_steps,
    )

    truth.append(float(trajectory.reserves[0]))

    print(
        f"Rate={r:.4f} | "
        f"PINN={reserve:.2f} | "
        f"Thiele={trajectory.reserves[0]:.2f}"
    )

# ==========================================================
# Convert to numpy
# ==========================================================

predicted = np.asarray(predicted)
truth = np.asarray(truth)


# ==========================================================
# Error metrics
# ==========================================================

mae = np.mean(np.abs(predicted - truth))

rmse = np.sqrt(
    np.mean(
        (predicted - truth) ** 2
    )
)

print("\n")
print("=" * 60)
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print("=" * 60)


# ==========================================================
# Plot
# ==========================================================

report_dir = Path(config.paths.reports_dir)

report_dir.mkdir(
    parents=True,
    exist_ok=True,
)

plot_path = (
    report_dir
    / "interest_rate_validation.png"
)

plt.figure(figsize=(8, 5))

plt.plot(
    rates * 100,
    predicted,
    linewidth=3,
    label="PINN",
)

plt.plot(
    rates * 100,
    truth,
    "--",
    linewidth=3,
    label="Thiele Solver",
)

plt.xlabel("Interest Rate (%)")
plt.ylabel("Reserve (£)")
plt.title("Interest Rate Sensitivity")

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    plot_path,
    dpi=300,
)

print(f"\nPlot saved to:\n{plot_path}")

plt.show()


print("\nNormalization")
print(f"Interest mean : {train_dataset.interest_mean:.6f}")
print(f"Interest std  : {train_dataset.interest_std:.6f}")
print(f"Premium mean  : {train_dataset.premium_mean:.6f}")
print(f"Premium std   : {train_dataset.premium_std:.6f}")
print(f"Target mean   : {train_dataset.target_mean:.8f}")
print(f"Target std    : {train_dataset.target_std:.8f}")
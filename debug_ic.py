from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.pipeline import (
    build_datasets,
    build_model,
    build_simulator,
    build_solver,
)
from src.utils.checkpoint import CheckpointManager
from src.utils.config import ConfigLoader, ensure_directories
from src.utils.device import DeviceManager
from src.utils.seed import set_seed
from src.data.dataset import FEATURE_INDEX, FEATURE_SCALES


config = ConfigLoader.load(Path("configs/config.yaml"))
ensure_directories(config)
set_seed(config.seed)

device_manager = DeviceManager(
    preferred_device=config.trainer.device,
    prefer_mixed_precision=False,
)

model = build_model(config)

checkpoint_path = Path(config.paths.checkpoints_dir) / "best_model.pt"
checkpoint = CheckpointManager(config.paths.checkpoints_dir).load(
    checkpoint_path,
    map_location=device_manager.device,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device_manager.device)
model.eval()

train_dataset, _, _, _ = build_datasets(config)
solver = build_solver(config)
simulator = build_simulator(config)

policy = simulator.generate_random_policies(1)[0]

print("\nPolicy used\n")
print(policy)

rates = np.linspace(0.01, 0.08, 30)

predicted = []
truth = []

for r in rates:
    scenario_rate = float(r)

    policy.scenario_interest_rate = scenario_rate
    policy.interest_rate = scenario_rate

    premium_ratio = policy.premium / max(policy.sum_assured, 1.0)
    mortality = policy.mortality_profile.intensity_at(0.0)

    features = torch.zeros((1, len(FEATURE_INDEX)), dtype=torch.float32)

    features[:, FEATURE_INDEX["time"]] = 0.0
    features[:, FEATURE_INDEX["age"]] = float(policy.age)
    features[:, FEATURE_INDEX["pricing_interest_rate"]] = float(policy.pricing_interest_rate)
    features[:, FEATURE_INDEX["scenario_interest_rate"]] = scenario_rate
    features[:, FEATURE_INDEX["premium_ratio"]] = premium_ratio
    features[:, FEATURE_INDEX["sum_assured"]] = float(policy.sum_assured)
    features[:, FEATURE_INDEX["mortality"]] = float(mortality)

    features[:, FEATURE_INDEX["time"]] /= FEATURE_SCALES["time"]
    features[:, FEATURE_INDEX["age"]] /= FEATURE_SCALES["age"]
    features[:, FEATURE_INDEX["pricing_interest_rate"]] /= FEATURE_SCALES["pricing_interest_rate"]

    features[:, FEATURE_INDEX["scenario_interest_rate"]] = (
        features[:, FEATURE_INDEX["scenario_interest_rate"]] - train_dataset.interest_mean
    ) / train_dataset.interest_std

    features[:, FEATURE_INDEX["premium_ratio"]] = (
        features[:, FEATURE_INDEX["premium_ratio"]] - train_dataset.premium_mean
    ) / train_dataset.premium_std

    features[:, FEATURE_INDEX["sum_assured"]] /= FEATURE_SCALES["sum_assured"]
    features[:, FEATURE_INDEX["mortality"]] /= FEATURE_SCALES["mortality"]

    features = features.to(device_manager.device)

    with torch.no_grad():
        z = model(features).item()

    reserve = (
        z * train_dataset.target_std
        + train_dataset.target_mean
    ) * policy.sum_assured

    predicted.append(float(reserve))

    trajectory = solver.solve(
        policy=policy,
        num_steps=config.data.time_steps,
    )

    thiele_reserve = float(trajectory.reserves[0])
    truth.append(thiele_reserve)

    print(
        f"Rate={scenario_rate:.4f} | "
        f"z={z:.6f} | "
        f"PINN={reserve:.2f} | "
        f"Thiele={thiele_reserve:.2f}"
    )

predicted = np.asarray(predicted)
truth = np.asarray(truth)

mae = np.mean(np.abs(predicted - truth))
rmse = np.sqrt(np.mean((predicted - truth) ** 2))

print("\n")
print("=" * 60)
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print("=" * 60)

report_dir = Path(config.paths.reports_dir)
report_dir.mkdir(parents=True, exist_ok=True)

plot_path = report_dir / "interest_rate_validation.png"

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
plt.ylabel("Reserve")
plt.title("Interest Rate Sensitivity")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(plot_path, dpi=300)

print(f"\nPlot saved to:\n{plot_path}")

plt.show()

print("\nNormalization")
print(f"Interest mean : {train_dataset.interest_mean:.6f}")
print(f"Interest std  : {train_dataset.interest_std:.6f}")
print(f"Premium mean  : {train_dataset.premium_mean:.6f}")
print(f"Premium std   : {train_dataset.premium_std:.6f}")
print(f"Target mean   : {train_dataset.target_mean:.8f}")
print(f"Target std    : {train_dataset.target_std:.8f}")
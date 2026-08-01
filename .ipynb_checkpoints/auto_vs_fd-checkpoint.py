from pathlib import Path

import torch

from src.pipeline import (
    build_model,
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


config = ConfigLoader.load(Path("configs/config.yaml"))
ensure_directories(config)
set_seed(config.seed)

device_manager = DeviceManager(
    preferred_device=config.trainer.device,
    prefer_mixed_precision=False,
)

model = build_model(config)

checkpoint = CheckpointManager(
    config.paths.checkpoints_dir
).load(
    Path(config.paths.checkpoints_dir) / "best_model.pt",
    map_location=device_manager.device,
)

model.load_state_dict(checkpoint["model_state_dict"])

model.to(device_manager.device)
model.eval()


train_dataset, _, _, _ = build_datasets(config)

simulator = build_simulator(config)

policy = simulator.generate_random_policies(1)[0]

print(policy)

target_mean = train_dataset.target_mean
target_std = train_dataset.target_std


def build_features(policy):

    x = torch.tensor(
        [[
            0.0,
            float(policy.age),
            policy.interest_rate,
            policy.premium,
            policy.sum_assured,
            policy.mortality_profile.intensity_at(0.0),
        ]],
        dtype=torch.float32,
    )

    x[:,0] /= FEATURE_SCALES["time"]
    x[:,1] /= FEATURE_SCALES["age"]
    x[:,2] /= FEATURE_SCALES["interest_rate"]
    x[:,3] /= FEATURE_SCALES["premium"]
    x[:,4] /= FEATURE_SCALES["sum_assured"]
    x[:,5] /= FEATURE_SCALES["mortality"]

    return x.to(device_manager.device)


def predict(policy):

    x = build_features(policy)

    with torch.no_grad():

        z = model(x).item()

    reserve = (
        z * target_std
        + target_mean
    )

    reserve *= policy.sum_assured

    return reserve


# ==========================================================
# AUTOGRAD
# ==========================================================

x = build_features(policy)

x.requires_grad_(True)

prediction = model(x)

reserve = (
    prediction * target_std
    + target_mean
)

reserve = reserve * policy.sum_assured

grad = torch.autograd.grad(
    reserve,
    x,
    grad_outputs=torch.ones_like(reserve),
)[0]


auto = {}

names = [
    "time",
    "age",
    "interest_rate",
    "premium",
    "sum_assured",
    "mortality",
]

for i, name in enumerate(names):

    auto[name] = (
        grad[0, i].item()
        / FEATURE_SCALES[name]
    )


# ==========================================================
# FINITE DIFFERENCE
# ==========================================================

FD_STEPS = {
    "interest_rate": 0.001,
    "premium": 100.0,
    "sum_assured": 5000.0,
    "mortality": 0.0005,
}

fd = {}

# ---------- interest ----------

policy2 = policy.__class__(**vars(policy))
policy2.interest_rate += FD_STEPS["interest_rate"]

fd["interest_rate"] = (
    predict(policy2) - predict(policy)
) / FD_STEPS["interest_rate"]


# ---------- premium ----------

policy2 = policy.__class__(**vars(policy))
policy2.premium += FD_STEPS["premium"]

fd["premium"] = (
    predict(policy2) - predict(policy)
) / FD_STEPS["premium"]


# ---------- sum assured ----------

policy2 = policy.__class__(**vars(policy))
policy2.sum_assured += FD_STEPS["sum_assured"]

fd["sum_assured"] = (
    predict(policy2) - predict(policy)
) / FD_STEPS["sum_assured"]


# ---------- mortality ----------

policy2 = policy.__class__(**vars(policy))

profile = policy2.mortality_profile

profile.intensities = (
    profile.intensities
    * (1 + FD_STEPS["mortality"])
)

fd["mortality"] = (
    predict(policy2) - predict(policy)
) / (
    profile.intensity_at(0)
    - policy.mortality_profile.intensity_at(0)
)


# ==========================================================
# PRINT COMPARISON
# ==========================================================

print()

print("=" * 70)
print("FINITE DIFFERENCE vs AUTOGRAD")
print("=" * 70)

print(
    f"{'Variable':15s}"
    f"{'FD':>18s}"
    f"{'Autograd':>18s}"
    f"{'% Diff':>15s}"
)

print("-" * 70)

for var in [
    "interest_rate",
    "premium",
    "sum_assured",
    "mortality",
]:

    fd_val = fd[var]
    ag_val = auto[var]

    pct = (
        abs(fd_val - ag_val)
        / max(abs(fd_val), 1e-8)
        * 100
    )

    print(
        f"{var:15s}"
        f"{fd_val:18.3f}"
        f"{ag_val:18.3f}"
        f"{pct:15.2f}"
    )

print("=" * 70)
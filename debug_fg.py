from pathlib import Path

import torch

from src.pipeline import (
    build_model,
    build_datasets,
)

from src.utils.config import (
    ConfigLoader,
    ensure_directories,
)

from src.utils.seed import set_seed
from src.utils.device import DeviceManager
from src.utils.checkpoint import CheckpointManager


# ==========================================================
# Load config
# ==========================================================

config = ConfigLoader.load(Path("configs/config.yaml"))
ensure_directories(config)
set_seed(config.seed)

device_manager = DeviceManager(
    preferred_device=config.trainer.device,
    prefer_mixed_precision=False,
)

device = device_manager.device


# ==========================================================
# Load model
# ==========================================================

model = build_model(config)

checkpoint = CheckpointManager(
    config.paths.checkpoints_dir
).load(
    Path(config.paths.checkpoints_dir) / "best_model.pt",
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()


# ==========================================================
# Load dataset
# ==========================================================

train_dataset, _, _, _ = build_datasets(config)

sample = train_dataset[0]

features = sample["features"].unsqueeze(0).to(device)

features.requires_grad_(True)


# ==========================================================
# Forward pass
# ==========================================================

prediction = model(features)


# ==========================================================
# Compute gradients
# ==========================================================

gradients = torch.autograd.grad(
    outputs=prediction,
    inputs=features,
    grad_outputs=torch.ones_like(prediction),
    create_graph=False,
)[0]


feature_names = [
    "Time",
    "Age",
    "Interest Rate",
    "Premium",
    "Sum Assured",
    "Mortality",
]


print("\n")
print("=" * 60)
print("FEATURE GRADIENTS")
print("=" * 60)

for i, name in enumerate(feature_names):

    print(
        f"{name:20s}: "
        f"{gradients[0, i].item(): .8f}"
    )

print("=" * 60)


# ==========================================================
# Relative importance
# ==========================================================

abs_grad = gradients.abs().cpu().numpy()[0]

print("\nRelative Importance")

total = abs_grad.sum()

for name, value in zip(feature_names, abs_grad):

    print(
        f"{name:20s}: "
        f"{100 * value / total:6.2f}%"
    )

print("=" * 60)
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
    / "best_model.pt"
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
print("\n" + "="*60)
print("FIRST LAYER WEIGHT MAGNITUDES")
print("="*60)

first = model.backbone.hidden_layers[0].weight.detach().cpu()

feature_names = [
    "time",
    "age",
    "interest_rate",
    "premium",
    "sum_assured",
    "mortality",
]

for i, name in enumerate(feature_names):
    print(f"{name:15s}: {first[:, i].abs().mean().item():.6f}")
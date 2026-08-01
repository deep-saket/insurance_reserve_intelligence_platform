from pathlib import Path
import time
import torch

from src.actuarial.actuarial_solver import ThieleSolver
from src.pipeline import build_dataloaders, build_model
from src.utils.checkpoint import CheckpointManager
from src.utils.config import ConfigLoader, ensure_directories
from src.utils.device import DeviceManager
from src.utils.seed import set_seed


config = ConfigLoader.load(Path("configs/config.yaml"))
ensure_directories(config)
set_seed(config.seed)

_, _, test_loader, test_dataset, test_policies = build_dataloaders(config)

device_manager = DeviceManager(
    preferred_device=config.trainer.device,
    prefer_mixed_precision=False,
)

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

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device_manager.device)
model.eval()
solver = ThieleSolver(
    method=config.solver.method,
    integration_step=config.solver.integration_step,
    rtol=config.solver.rtol,
    atol=config.solver.atol,
)

# Warm-up
solver.solve(
    policy=test_policies[0],
    num_steps=config.data.time_steps,
)

start = time.perf_counter()

for policy in test_policies:
    solver.solve(
        policy=policy,
        num_steps=config.data.time_steps,
    )

classical_time = time.perf_counter() - start

features = torch.stack(
    [sample["features"] for sample in test_dataset]
).to(device_manager.device)

# Warm-up
with torch.no_grad():
    model(features[:10])

if torch.cuda.is_available():
    torch.cuda.synchronize()

start = time.perf_counter()

with torch.no_grad():
    model(features)

if torch.cuda.is_available():
    torch.cuda.synchronize()

pinn_time = time.perf_counter() - start

num = len(test_policies)

print("\n===== Runtime Benchmark =====")
print(f"Policies           : {num}")
print(f"Classical Solver   : {classical_time:.6f} s")
print(f"PINN/KINN          : {pinn_time:.6f} s")
print(f"Solver / policy    : {1000*classical_time/num:.4f} ms")
print(f"PINN / policy      : {1000*pinn_time/num:.4f} ms")
print(f"Speed-up           : {classical_time/pinn_time:.2f}x")
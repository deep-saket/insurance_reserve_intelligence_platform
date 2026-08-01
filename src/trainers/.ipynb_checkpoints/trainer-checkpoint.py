"""Training loop implementations.

Created: 2026-05-31
Purpose: Train the reserve PINN with experiment logging, checkpointing, and governance controls.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    SummaryWriter = None  # type: ignore[assignment]

from src.losses.total_loss import TotalLoss
from src.utils.checkpoint import CheckpointManager
from src.utils.config import ExperimentConfig
from src.utils.device import DeviceManager
from src.utils.logger import configure_logger


@dataclass(slots=True)
class EpochMetrics:
    """Aggregate metrics for a training or validation epoch."""

    total_loss: float
    components: dict[str, float]
    weighted_components: dict[str, float]

    def to_csv_row(self) -> dict[str, float]:
        """Serialize metrics to a flat CSV row."""

        row: dict[str, float] = {"total_loss": self.total_loss}
        row.update({f"raw_{name}": value for name, value in self.components.items()})
        row.update({f"weighted_{name}": value for name, value in self.weighted_components.items()})
        return row


class BaseTrainer(ABC):
    """Abstract model trainer."""

    @abstractmethod
    def fit(self, train_loader: DataLoader, validation_loader: DataLoader) -> dict[str, list[float]]:
        """Train a model and return history."""


class PINNTrainer(BaseTrainer):
    """Full-featured trainer for the reserve PINN."""

    def __init__(self, model: nn.Module, config: ExperimentConfig, device_manager: DeviceManager | None = None) -> None:
        """Initialize the PINN trainer."""

        self.config = config
        self.device_manager = device_manager or DeviceManager(
            preferred_device=config.trainer.device,
            prefer_mixed_precision=config.trainer.mixed_precision,
        )
        self.model = self.device_manager.move_module(model)
        self.loss_fn = TotalLoss(config.losses, settings=config.loss_settings)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.trainer.learning_rate,
            weight_decay=config.trainer.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.trainer.scheduler_factor,
            patience=config.trainer.scheduler_patience,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device_manager.mixed_precision)
        self.checkpoints = CheckpointManager(config.paths.checkpoints_dir)
        self.writer = self._create_writer(
            log_dir=config.paths.tensorboard_dir,
            run_name=config.trainer.run_name,
            enabled=config.trainer.tensorboard_enabled,
        )
        self.logger = configure_logger(
            name=f"PINNTrainer[{config.trainer.run_name}]",
            log_file=str(Path(config.paths.logs_dir) / "training.log"),
        )

        self.csv_log_path = Path(config.paths.logs_dir) / "training_metrics.csv"
        self.best_validation_loss = float("inf")
        self.epochs_without_improvement = 0
        self.start_epoch = 0
        self.active_loss_names = self.loss_fn.active_loss_names
        self.base_loss_weights = {
            name: float(term.weight or 0.0)
            for name, term in self.config.losses.terms.items()
        }
        self.history: dict[str, list[float]] = {
            "train_total_loss": [],
            "validation_total_loss": [],
        }

        for name in self.active_loss_names:
            self.history[f"train_raw_{name}"] = []
            self.history[f"validation_raw_{name}"] = []
            self.history[f"train_weighted_{name}"] = []
            self.history[f"validation_weighted_{name}"] = []

        self._initialize_csv_log()
        self._log_runtime_summary()

        if config.trainer.resume_from:
            self.resume(config.trainer.resume_from)

    @staticmethod
    def _create_writer(log_dir: str, run_name: str, enabled: bool):
        """Create a TensorBoard writer or fallback no-op writer."""

        if not enabled or SummaryWriter is None:
            return _NullSummaryWriter()
        return SummaryWriter(log_dir=log_dir, comment=run_name)

    def _log_runtime_summary(self) -> None:
        """Emit run-level metadata to logs and TensorBoard."""

        self.logger.info(
            "Run %s | artifacts %s | device %s",
            self.config.trainer.run_name,
            self.config.paths.run_dir,
            self.device_manager.summary(),
        )
        self.writer.add_text("run/name", self.config.trainer.run_name, 0)
        self.writer.add_text("run/device", self.device_manager.summary(), 0)
        self.writer.add_text("run/artifacts_dir", self.config.paths.run_dir, 0)
        self.writer.add_text("run/active_losses", ", ".join(self.active_loss_names) or "none", 0)

    def _initialize_csv_log(self) -> None:
        """Initialize the CSV metrics log."""

        self.csv_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_log_path.exists():
            with self.csv_log_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "epoch",
                        "split",
                        "total_loss",
                        *[f"raw_{name}" for name in self.active_loss_names],
                        *[f"weighted_{name}" for name in self.active_loss_names],
                    ],
                )
                writer.writeheader()

    def _log_csv(self, epoch: int, split: str, metrics: EpochMetrics) -> None:
        """Append one epoch record to the CSV metrics log."""

        with self.csv_log_path.open("a", newline="", encoding="utf-8") as handle:
            row = metrics.to_csv_row()
            writer = csv.DictWriter(handle, fieldnames=["epoch", "split", *row.keys()])
            writer.writerow({"epoch": epoch, "split": split, **row})

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Move a batch to the configured device."""

        device = self.device_manager.device
        features = batch["features"].to(device)

        return {
            "features": features.requires_grad_(True),
            "raw_features": batch["raw_features"].to(device),
            "target": batch["target"].to(device),
            "sum_assured_scale": batch["sum_assured_scale"].to(device),
            "target_mean": batch["target_mean"].to(device),
            "target_std": batch["target_std"].to(device),
            "interest_mean": batch["interest_mean"].to(device),
            "interest_std": batch["interest_std"].to(device),
            "premium_mean": batch["premium_mean"].to(device),
            "premium_std": batch["premium_std"].to(device),
            "term": batch["term"].to(device),
            "terminal_mortality": batch["terminal_mortality"].to(device),
            "interest_rate_sensitivity_target": batch["interest_rate_sensitivity_target"].to(device),
            "interest_rate_shock_down_target": batch["interest_rate_shock_down_target"].to(device),
            "interest_rate_shock_up_target": batch["interest_rate_shock_up_target"].to(device),
            "interest_rate_shock_delta": batch["interest_rate_shock_delta"].to(device),
            "interest_rate_shock_down_peak_time": batch["interest_rate_shock_down_peak_time"].to(device),
            "interest_rate_shock_up_peak_time": batch["interest_rate_shock_up_peak_time"].to(device),
            "interest_rate_shock_down_peak_mortality": batch["interest_rate_shock_down_peak_mortality"].to(device),
            "interest_rate_shock_up_peak_mortality": batch["interest_rate_shock_up_peak_mortality"].to(device),
            "interest_rate_shock_down_peak_target": batch["interest_rate_shock_down_peak_target"].to(device),
            "interest_rate_shock_up_peak_target": batch["interest_rate_shock_up_peak_target"].to(device),
        }

    def _aggregate_epoch(self, loss_values: list[dict[str, Any]]) -> EpochMetrics:
        """Aggregate batch-level losses into epoch metrics."""

        if not loss_values:
            raise ValueError("No losses collected for the epoch.")

        raw_component_means = {
            name: float(torch.stack([item["components"][name].detach() for item in loss_values]).mean().item())
            for name in self.active_loss_names
        }
        weighted_component_means = {
            name: float(torch.stack([item["weighted_components"][name].detach() for item in loss_values]).mean().item())
            for name in self.active_loss_names
        }

        return EpochMetrics(
            total_loss=float(torch.stack([item["total_loss"].detach() for item in loss_values]).mean().item()),
            components=raw_component_means,
            weighted_components=weighted_component_means,
        )

    def _step(self, batch: dict[str, torch.Tensor], training: bool) -> dict[str, Any] | None:
        """Run one optimization or evaluation step."""

        prepared = self._move_batch(batch)

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(True):
            with self.device_manager.autocast_context():
                predictions = self.model(prepared["features"])
                loss_breakdown = self.loss_fn(
                    model=self.model,
                    batch=prepared,
                    predictions=predictions,
                    context={},
                )

            if training:
                total = loss_breakdown["total_loss"]
                if not torch.isfinite(total):
                    self.logger.warning(
                        "Non-finite loss detected (%.6g) - skipping batch. "
                        "Check PDE residual scale or learning rate.",
                        total.item(),
                    )
                    return None

                self.scaler.scale(total).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

        return loss_breakdown

    def _run_epoch(self, loader: DataLoader, training: bool) -> EpochMetrics:
        """Run one full training or validation epoch."""

        self.model.train(mode=training)
        losses: list[dict[str, Any]] = []
        skipped = 0

        for batch in loader:
            result = self._step(batch, training=training)
            if result is None:
                skipped += 1
                continue
            losses.append(result)

        if skipped:
            self.logger.warning("Skipped %d non-finite batch(es) this epoch.", skipped)

        if not losses:
            raise RuntimeError(
                "Every batch in this epoch produced a non-finite loss. "
                "Reduce learning_rate, pde_loss weight, or disable mixed_precision."
            )

        return self._aggregate_epoch(losses)

    def _save_checkpoint(self, epoch: int, validation_metrics: EpochMetrics) -> None:
        """Persist model and optimizer state to disk."""

        state: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_validation_loss": self.best_validation_loss,
            "validation_metrics": asdict(validation_metrics),
            "config": self.config,
        }

        self.checkpoints.save(f"epoch_{epoch:03d}.pt", state)

        if validation_metrics.total_loss <= self.best_validation_loss:
            self.checkpoints.save("best_model.pt", state)

    def resume(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint."""

        checkpoint = self.checkpoints.load(checkpoint_path, map_location=self.device_manager.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_validation_loss = float(checkpoint["best_validation_loss"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.logger.info("Resumed from checkpoint %s at epoch %s", checkpoint_path, self.start_epoch)

    _CONSTRAINT_LOSSES = {
        "pde_loss",
        "boundary_loss",
        "mortality_monotonicity_loss",
        "age_monotonicity_loss",
        "interest_rate_monotonicity_loss",
        "sum_assured_monotonicity_loss",
        "interest_rate_scenario_loss",
        "interest_rate_peak_loss",
        "solvency_loss",
        "reserve_ceiling_loss",
        "smoothness_loss",
        "portfolio_consistency_loss",
    }

    def _warmup_epochs(self) -> int:
        """Return the number of epochs used for constraint warmup."""

        return max(30, int(self.config.trainer.epochs * 0.20))

    def _curriculum_weight(self, epoch: int, name: str) -> float:
        """Return the effective weight for a loss term at a given epoch."""

        cfg_weight = self.base_loss_weights[name]

        if name not in self._CONSTRAINT_LOSSES:
            return cfg_weight

        warmup = self._warmup_epochs()
        effective_epoch = max(0, epoch)
        ramp = min(1.0, effective_epoch / warmup)
        return cfg_weight * ramp

    def fit(self, train_loader: DataLoader, validation_loader: DataLoader) -> dict[str, list[float]]:
        """Train the model and return the loss history."""

        for epoch in range(self.start_epoch, self.config.trainer.epochs):
            for name, term in self.config.losses.terms.items():
                if term.enabled and name in self.loss_fn.losses:
                    term.weight = self._curriculum_weight(epoch, name)

            warmup = self._warmup_epochs()
            if epoch <= warmup:
                ramp_pct = epoch / warmup * 100
                self.logger.info("Curriculum warmup: %.0f%% of constraint weight active", ramp_pct)

            train_metrics = self._run_epoch(train_loader, training=True)
            validation_metrics = self._run_epoch(validation_loader, training=False)
            self.scheduler.step(validation_metrics.total_loss)

            self.history["train_total_loss"].append(train_metrics.total_loss)
            self.history["validation_total_loss"].append(validation_metrics.total_loss)

            self.writer.add_scalar("loss/train_total", train_metrics.total_loss, epoch)
            self.writer.add_scalar("loss/validation_total", validation_metrics.total_loss, epoch)

            for name in self.active_loss_names:
                train_raw = train_metrics.components[name]
                validation_raw = validation_metrics.components[name]
                train_weighted = train_metrics.weighted_components[name]
                validation_weighted = validation_metrics.weighted_components[name]

                self.history[f"train_raw_{name}"].append(train_raw)
                self.history[f"validation_raw_{name}"].append(validation_raw)
                self.history[f"train_weighted_{name}"].append(train_weighted)
                self.history[f"validation_weighted_{name}"].append(validation_weighted)

                self.writer.add_scalar(f"loss_component/train/{name}", train_raw, epoch)
                self.writer.add_scalar(f"loss_component/validation/{name}", validation_raw, epoch)
                self.writer.add_scalar(f"loss_component_weighted/train/{name}", train_weighted, epoch)
                self.writer.add_scalar(f"loss_component_weighted/validation/{name}", validation_weighted, epoch)

            self._log_csv(epoch, "train", train_metrics)
            self._log_csv(epoch, "validation", validation_metrics)

            self.logger.info(
                "Epoch %s | train %.6f | validation %.6f",
                epoch,
                train_metrics.total_loss,
                validation_metrics.total_loss,
            )

            if epoch % self.config.trainer.checkpoint_every == 0:
                self._save_checkpoint(epoch, validation_metrics)

            if validation_metrics.total_loss < self.best_validation_loss:
                self.best_validation_loss = validation_metrics.total_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, validation_metrics)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.config.trainer.early_stopping_patience:
                    self.logger.info("Early stopping triggered at epoch %s", epoch)
                    break

        self.writer.flush()
        self.writer.close()
        return self.history


class _NullSummaryWriter:
    """Fallback writer used when tensorboard is unavailable."""

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """Ignore a scalar logging request."""

        del tag, scalar_value, global_step

    def add_text(self, tag: str, text_string: str, global_step: int) -> None:
        """Ignore a text logging request."""

        del tag, text_string, global_step

    def flush(self) -> None:
        """No-op flush for API compatibility."""

        return None

    def close(self) -> None:
        """No-op close for API compatibility."""

        return None
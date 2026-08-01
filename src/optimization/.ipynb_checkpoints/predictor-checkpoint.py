"""Business-facing reserve prediction interface.

Created: 2026-07-01
Purpose: Expose the trained PINN through policy objects rather than normalized tensors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.actuarial.policy import Policy
from src.data.dataset import FEATURE_INDEX, FEATURE_SCALES
from src.pipeline import build_model
from src.utils.checkpoint import CheckpointManager
from src.utils.config import ConfigLoader, ExperimentConfig
from src.utils.device import DeviceManager


@dataclass(frozen=True, slots=True)
class TrainingStatistics:
    """Normalization constants learned from the training dataset.

    Business Interpretation:
        These values are part of the trained reserve engine's contract. Keeping
        them with the predictor prevents pricing, stress, and optimization code
        from manually reproducing model-space transformations.
    """

    interest_mean: float
    interest_std: float
    premium_mean: float
    premium_std: float
    target_mean: float
    target_std: float

    @classmethod
    def from_dataset(cls, training_dataset: Any) -> "TrainingStatistics":
        source = getattr(training_dataset, "normalization", training_dataset)
    
        required = (
            "interest_mean",
            "interest_std",
            "premium_mean",
            "premium_std",
            "target_mean",
            "target_std",
        )
    
        if isinstance(source, dict) and all(name in source for name in required):
            return cls.from_mapping(source)
    
        if all(hasattr(source, name) for name in required):
            return cls.from_mapping(source)
    
        if hasattr(training_dataset, "records"):
            return cls.from_records(training_dataset.records)
    
        return cls.from_mapping(source)

    @classmethod
    def from_records(cls, records: Sequence[Any]) -> "TrainingStatistics":
        if not records:
            raise ValueError("Cannot infer training statistics from an empty dataset.")
    
        interest_values = np.asarray(
            [r.features[FEATURE_INDEX["scenario_interest_rate"]] for r in records],
            dtype=np.float32,
        )
    
        premium_values = np.asarray(
            [r.features[FEATURE_INDEX["premium_ratio"]] for r in records],
            dtype=np.float32,
        )
    
        reserve_ratios = np.asarray(
            [
                float(r.reserve) / max(float(r.features[FEATURE_INDEX["sum_assured"]]), 1.0)
                for r in records
            ],
            dtype=np.float32,
        )
    
        interest_std = float(interest_values.std())
        premium_std = float(premium_values.std())
        target_std = float(reserve_ratios.std())
    
        return cls(
            interest_mean=float(interest_values.mean()),
            interest_std=interest_std if interest_std > 1e-8 else 1.0,
            premium_mean=float(premium_values.mean()),
            premium_std=premium_std if premium_std > 1e-8 else 1.0,
            target_mean=float(reserve_ratios.mean()),
            target_std=target_std if target_std > 1e-8 else 1.0,
        )
    @classmethod
    def from_mapping(cls, normalization: Any) -> TrainingStatistics:
        """Create statistics from a mapping or object with normalization fields."""

        return cls(
            interest_mean=_read_stat(normalization, "interest_mean"),
            interest_std=_read_stat(normalization, "interest_std"),
            premium_mean=_read_stat(normalization, "premium_mean"),
            premium_std=_read_stat(normalization, "premium_std"),
            target_mean=_read_stat(normalization, "target_mean"),
            target_std=_read_stat(normalization, "target_std"),
        )

    @classmethod
    def from_records(cls, records: Sequence[Any]) -> TrainingStatistics:
        """Create statistics directly from reserve dataset records."""

        if not records:
            raise ValueError("Cannot infer training statistics from an empty dataset.")
        interest_values = np.asarray(
            [record.features[FEATURE_INDEX["scenario_interest_rate"]] for record in records],
            dtype=np.float32,
        )
        premium_values = np.asarray(
            [record.features[FEATURE_INDEX["premium_ratio"]] for record in records],
            dtype=np.float32,
        )
        reserve_ratios = np.asarray(
            [
                float(record.reserve)
                / max(float(record.features[FEATURE_INDEX["sum_assured"]]), 1.0)
                for record in records
            ],
            dtype=np.float32,
        )
        interest_std = float(interest_values.std())
        premium_std = float(premium_values.std())
        target_std = float(reserve_ratios.std())
        return cls(
            interest_mean=float(interest_values.mean()),
            interest_std=interest_std if interest_std > 1e-8 else 1.0,
            premium_mean=float(premium_values.mean()),
            premium_std=premium_std if premium_std > 1e-8 else 1.0,
            target_mean=float(reserve_ratios.mean()),
            target_std=target_std if target_std > 1e-8 else 1.0,
        )

    def as_dict(self) -> dict[str, float]:
        """Return normalization statistics as a serializable dictionary."""

        return {
            "interest_mean": self.interest_mean,
            "interest_std": self.interest_std,
            "premium_mean": self.premium_mean,
            "premium_std": self.premium_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    def validate(self) -> None:
        """Validate statistics required for invertible normalization."""

        if self.interest_std == 0.0:
            raise ValueError("interest_std must be non-zero.")
        if self.premium_std == 0.0:
            raise ValueError("premium_std must be non-zero.")
        if self.target_std == 0.0:
            raise ValueError("target_std must be non-zero.")


class ReservePredictor:
    """Convert policies into currency reserve predictions from the trained PINN.

    Scientific Context:
        The model predicts ``z = (v - target_mean) / target_std`` where
        ``v = reserve / sum_assured``. This class owns both feature normalization
        and target denormalization for inference.

    Business Interpretation:
        This is the reserve oracle used by downstream business modules. They pass
        in a policy and receive a reserve amount in currency units, without
        touching normalized tensors or z-space.
    """

    feature_names: tuple[str, ...] = (
        "time",
        "age",
        "pricing_interest_rate",
        "scenario_interest_rate",
        "premium_ratio",
        "sum_assured",
        "mortality",
    )
    input_dim: int = 7

    def __init__(
        self,
        model: torch.nn.Module,
        training_statistics: Any,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize the predictor from a model and training dataset/statistics.

        Args:
            model: Trained reserve model with six normalized input features.
            training_statistics: Training dataset or statistics object exposing
                ``interest_mean``, ``interest_std``, ``premium_mean``,
                ``premium_std``, ``target_mean``, and ``target_std``.
            device: Optional inference device.
        """

        self.statistics = _coerce_training_statistics(training_statistics)
        self.statistics.validate()
        self.interest_mean = self.statistics.interest_mean
        self.interest_std = self.statistics.interest_std
        self.premium_mean = self.statistics.premium_mean
        self.premium_std = self.statistics.premium_std
        self.target_mean = self.statistics.target_mean
        self.target_std = self.statistics.target_std
        self.normalization = self.statistics.as_dict()
        self.feature_scales = FEATURE_SCALES
        self.device = (
            torch.device(device)
            if device is not None
            else DeviceManager(prefer_mixed_precision=False).device
        )
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        training_dataset: Any,
        model: torch.nn.Module | None = None,
        config: ExperimentConfig | None = None,
        config_path: str | Path = "configs/config.yaml",
        device: torch.device | str | None = None,
    ) -> ReservePredictor:
        """Load model weights and attach training-dataset normalization statistics.

        Args:
            checkpoint_path: Checkpoint containing the trained model state.
            training_dataset: Dataset used to derive the normalization constants.
            model: Optional model instance. If omitted, the configured model is
                built before loading checkpoint weights.
            config: Optional experiment config used when ``model`` is omitted.
            config_path: Config path used when ``config`` is omitted.
            device: Optional inference device.

        Returns:
            ReservePredictor: Fully configured business reserve predictor.
        """

        resolved_device = (
            torch.device(device)
            if device is not None
            else DeviceManager(prefer_mixed_precision=False).device
        )
        if model is None:
            loaded_config = config if config is not None else ConfigLoader.load(config_path)
            reserve_model = build_model(loaded_config)
        else:
            reserve_model = model
        checkpoint = CheckpointManager(str(Path(checkpoint_path).parent)).load(
            checkpoint_path,
            map_location=resolved_device,
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        reserve_model.load_state_dict(state_dict)
        return cls(
            model=reserve_model,
            training_statistics=training_dataset,
            device=resolved_device,
        )

    def prepare_features(self, policy: Policy, time_point: float = 0.0) -> torch.Tensor:
        """Construct the normalized seven-feature model tensor for one policy.

        Args:
            policy: Policy to value.
            time_point: Elapsed policy duration for valuation.

        Returns:
            torch.Tensor: Single-row normalized feature tensor for PINN inference.
        """

        raw_features = self.raw_features_from_policy(policy=policy, time_point=time_point)
        features = self.normalize_features(raw_features)
        return features.unsqueeze(0).to(self.device)

    def predict(self, policy: Policy, time_point: float = 0.0) -> float:
        """Predict reserve in currency units for one policy.

        Args:
            policy: Policy to value.
            time_point: Elapsed policy duration for valuation.

        Returns:
            float: Reserve amount, not z-space and not reserve ratio.
        """

        features = self.prepare_features(policy, time_point=time_point)
        with torch.no_grad():
            z_value = float(self.model(features).detach().cpu().item())
        return self.denormalize_target(z_value, policy.sum_assured)

    def predict_many(
        self,
        policies: Sequence[Policy],
        time_point: float | Sequence[float] = 0.0,
    ) -> list[float]:
        """Predict currency reserves for a batch of policies.

        Args:
            policies: Policies to value.
            time_point: Single elapsed duration applied to every policy, or one
                elapsed duration per policy.

        Returns:
            list[float]: Reserve amounts in the same order as ``policies``.
        """

        if not policies:
            return []
        time_points = self._resolve_time_points(policies=policies, time_point=time_point)
        feature_rows = [
            self.prepare_features(policy, valuation_time).squeeze(0)
            for policy, valuation_time in zip(policies, time_points)
        ]
        features = torch.stack(feature_rows).to(self.device)
        with torch.no_grad():
            z_values = self.model(features).detach().cpu().numpy().reshape(-1)
        return [
            self.denormalize_target(float(z_value), policy.sum_assured)
            for z_value, policy in zip(z_values, policies)
        ]

    def raw_features_from_policy(self, policy: Policy, time_point: float = 0.0) -> torch.Tensor:
        """Build raw seven-feature tensor from a policy before normalization."""

        clipped_time = self._clip_time_point(policy=policy, time_point=time_point)
        sum_assured = float(policy.sum_assured)
        raw_features = np.empty(self.input_dim, dtype=np.float32)
        raw_features[FEATURE_INDEX["time"]] = clipped_time
        raw_features[FEATURE_INDEX["age"]] = float(policy.age)
        raw_features[FEATURE_INDEX["pricing_interest_rate"]] = float(policy.pricing_interest_rate)
        raw_features[FEATURE_INDEX["scenario_interest_rate"]] = float(policy.scenario_interest_rate)
        raw_features[FEATURE_INDEX["premium_ratio"]] = float(policy.premium) / max(sum_assured, 1.0)
        raw_features[FEATURE_INDEX["sum_assured"]] = sum_assured
        raw_features[FEATURE_INDEX["mortality"]] = float(
            policy.mortality_profile.intensity_at(clipped_time)
        )
        return torch.tensor(raw_features, dtype=torch.float32, device=self.device)

    def normalize_features(self, raw_features: torch.Tensor) -> torch.Tensor:
        """Normalize raw features using the training dataset contract."""

        features = raw_features.clone()
        features[..., FEATURE_INDEX["time"]] /= self.feature_scales["time"]
        features[..., FEATURE_INDEX["age"]] /= self.feature_scales["age"]
        features[..., FEATURE_INDEX["pricing_interest_rate"]] /= self.feature_scales["pricing_interest_rate"]
        features[..., FEATURE_INDEX["scenario_interest_rate"]] = (
            features[..., FEATURE_INDEX["scenario_interest_rate"]] - self.interest_mean
        ) / self.interest_std
        features[..., FEATURE_INDEX["premium_ratio"]] = (
            features[..., FEATURE_INDEX["premium_ratio"]] - self.premium_mean
        ) / self.premium_std
        features[..., FEATURE_INDEX["sum_assured"]] /= self.feature_scales["sum_assured"]
        features[..., FEATURE_INDEX["mortality"]] /= self.feature_scales["mortality"]
        return features

    def denormalize_target(
        self,
        z_value: torch.Tensor | float,
        sum_assured: torch.Tensor | float,
    ) -> torch.Tensor | float:
        """Convert model z-output into reserve currency units."""

        if isinstance(z_value, torch.Tensor):
            scale = torch.clamp(
                torch.as_tensor(sum_assured, dtype=z_value.dtype, device=z_value.device),
                min=1.0,
            )
            return (z_value * self.target_std + self.target_mean) * scale
        scale_value = max(float(sum_assured), 1.0)
        return (float(z_value) * self.target_std + self.target_mean) * scale_value

    @staticmethod
    def _clip_time_point(policy: Policy, time_point: float) -> float:
        """Clip valuation time into the valid policy duration."""

        return float(np.clip(float(time_point), 0.0, float(policy.term)))

    @staticmethod
    def _resolve_time_points(
        policies: Sequence[Policy],
        time_point: float | Sequence[float],
    ) -> list[float]:
        """Return one valuation time for each policy."""

        if isinstance(time_point, Sequence) and not isinstance(time_point, str):
            if len(time_point) != len(policies):
                raise ValueError("time_point sequence must match policies length.")
            return [float(value) for value in time_point]
        return [float(time_point)] * len(policies)


def _coerce_training_statistics(training_statistics: Any) -> TrainingStatistics:
    """Return a TrainingStatistics instance from a dataset or statistics object."""

    if isinstance(training_statistics, TrainingStatistics):
        return training_statistics
    return TrainingStatistics.from_dataset(training_statistics)


def _has_required_stats(source: Any) -> bool:
    """Return whether a mapping/object exposes all required normalization stats."""

    names = (
        "interest_mean",
        "interest_std",
        "premium_mean",
        "premium_std",
        "target_mean",
        "target_std",
    )
    if isinstance(source, dict):
        return all(name in source for name in names)
    return all(hasattr(source, name) for name in names)


def _read_stat(source: Any, name: str) -> float:
    """Read a named training statistic from common dataset container shapes."""

    if hasattr(source, name):
        return float(getattr(source, name))
    if isinstance(source, dict) and name in source:
        return float(source[name])
    for container_name in ("training_statistics", "statistics", "normalization_stats"):
        if not hasattr(source, container_name):
            continue
        container = getattr(source, container_name)
        if hasattr(container, name):
            return float(getattr(container, name))
        if isinstance(container, dict) and name in container:
            return float(container[name])
    raise AttributeError(f"Training dataset is missing required statistic '{name}'.")




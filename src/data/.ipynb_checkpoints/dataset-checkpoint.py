"""Datasets for reserve learning.

Created: 2026-05-31  Revised: 2026-07-02
Purpose: Build supervised and knowledge-informed reserve records with
scenario-rate features and solver-derived rate-shock targets.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.actuarial.actuarial_solver import BaseActuarialSolver
from src.actuarial.policy import Policy


FEATURE_INDEX: dict[str, int] = {
    "time": 0,
    "age": 1,
    "pricing_interest_rate": 2,
    "scenario_interest_rate": 3,
    "premium_ratio": 4,
    "sum_assured": 5,
    "mortality": 6,
}

FEATURE_SCALES: dict[str, float] = {
    "time": 30.0,
    "age": 45.0,
    "pricing_interest_rate": 0.07,
    "scenario_interest_rate": 0.07,
    "premium_ratio": 0.02,
    "sum_assured": 950_000.0,
    "mortality": 0.05,
}

_INTEREST_RATE_SENSITIVITY_DELTA = 0.005
_MIN_INTEREST_RATE = 1.0e-4


@dataclass(slots=True)
class ReserveRecord:
    """One reserve-learning record with optional scenario supervision targets."""

    policy_id: str
    features: np.ndarray
    reserve: float
    term: float
    terminal_mortality: float
    interest_rate_sensitivity: float
    interest_rate_shock_down_reserve_ratio: float
    interest_rate_shock_up_reserve_ratio: float
    interest_rate_shock_down_peak_time: float
    interest_rate_shock_up_peak_time: float
    interest_rate_shock_down_peak_mortality: float
    interest_rate_shock_up_peak_mortality: float
    interest_rate_shock_down_peak_reserve_ratio: float
    interest_rate_shock_up_peak_reserve_ratio: float


def build_raw_feature_array(
    *,
    time_point: float,
    age: float,
    pricing_interest_rate: float,
    scenario_interest_rate: float,
    premium_ratio: float,
    sum_assured: float,
    mortality: float,
) -> np.ndarray:
    """Build one raw seven-feature vector aligned to FEATURE_INDEX."""

    return np.asarray(
        [
            time_point,
            age,
            pricing_interest_rate,
            scenario_interest_rate,
            premium_ratio,
            sum_assured,
            mortality,
        ],
        dtype=np.float32,
    )


def build_policy_feature_array(
    policy: Policy,
    time_point: float,
    *,
    scenario_interest_rate: float | None = None,
    mortality: float | None = None,
) -> np.ndarray:
    """Build one raw feature vector from a policy and time point."""

    resolved_mortality = (
        policy.mortality_profile.intensity_at(time_point)
        if mortality is None
        else mortality
    )
    resolved_scenario_rate = (
        policy.scenario_interest_rate
        if scenario_interest_rate is None
        else scenario_interest_rate
    )
    sum_assured = float(policy.sum_assured)
    premium_ratio = float(policy.premium) / max(sum_assured, 1.0)

    return build_raw_feature_array(
        time_point=float(time_point),
        age=float(policy.age),
        pricing_interest_rate=float(policy.pricing_interest_rate),
        scenario_interest_rate=float(resolved_scenario_rate),
        premium_ratio=premium_ratio,
        sum_assured=sum_assured,
        mortality=float(resolved_mortality),
    )



def normalize_raw_feature_array(raw_features: np.ndarray) -> np.ndarray:
    """Normalize one raw seven-feature vector using static feature scales."""

    normalized = raw_features.astype(np.float32, copy=True)
    for name, index in FEATURE_INDEX.items():
        normalized[index] /= FEATURE_SCALES[name]
    return normalized
class ReserveDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset of normalized PINN inputs and z-scored reserve-ratio targets."""

    def __init__(
        self,
        policies: Iterable[Policy],
        solver: BaseActuarialSolver,
        time_steps: int,
        target_mean: float | None = None,
        target_std: float | None = None,
        interest_mean: float | None = None,
        interest_std: float | None = None,
        premium_mean: float | None = None,
        premium_std: float | None = None,
    ) -> None:
        self.records = self._build_records(
            list(policies),
            solver=solver,
            time_steps=time_steps,
        )

        if not self.records:
            raise ValueError("ReserveDataset cannot be built from zero records.")

        premium_values = np.asarray(
            [record.features[FEATURE_INDEX["premium_ratio"]] for record in self.records],
            dtype=np.float32,
        )
        self.premium_mean = float(premium_values.mean()) if premium_mean is None else float(premium_mean)
        if premium_std is None:
            std = float(premium_values.std())
            self.premium_std = std if std > 1e-8 else 1.0
        else:
            self.premium_std = float(premium_std) if float(premium_std) > 1e-8 else 1.0
        
        interest_values = np.asarray(
            [record.features[FEATURE_INDEX["scenario_interest_rate"]] for record in self.records],
            dtype=np.float32,
        )
        self.interest_mean = float(interest_values.mean()) if interest_mean is None else float(interest_mean)
        if interest_std is None:
            std = float(interest_values.std())
            self.interest_std = std if std > 1e-8 else 1.0
        else:
            self.interest_std = float(interest_std) if float(interest_std) > 1e-8 else 1.0

        reserves = np.asarray([record.reserve for record in self.records], dtype=np.float64)
        print("\n")
        print("=" * 60)
        print("DATASET RESERVE STATISTICS")
        print("=" * 60)
        print(f"Minimum Reserve : {reserves.min():,.2f}")
        print(f"Maximum Reserve : {reserves.max():,.2f}")
        print(f"Mean Reserve    : {reserves.mean():,.2f}")
        print(f"Std Reserve     : {reserves.std():,.2f}")
        print("=" * 60)

        reserve_ratios = np.asarray(
            [
                record.reserve / max(float(record.features[FEATURE_INDEX["sum_assured"]]), 1.0)
                for record in self.records
            ],
            dtype=np.float32,
        )
        self.target_mean = float(reserve_ratios.mean()) if target_mean is None else float(target_mean)
        if target_std is None:
            std = float(reserve_ratios.std())
            self.target_std = std if std > 1e-8 else 1.0
        else:
            self.target_std = float(target_std) if float(target_std) > 1e-8 else 1.0

        print("\nNormalization")
        print(f"interest_mean = {self.interest_mean:.8f}")
        print(f"interest_std  = {self.interest_std:.8f}")
        print(f"premium_mean  = {self.premium_mean:.8f}")
        print(f"premium_std   = {self.premium_std:.8f}")
        print(f"target_mean   = {self.target_mean:.8f}")
        print(f"target_std    = {self.target_std:.8f}")

        self.normalization = {
            "interest_mean": self.interest_mean,
            "interest_std": self.interest_std,
            "premium_mean": self.premium_mean,
            "premium_std": self.premium_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    @staticmethod
    def _build_records(
        policies: list[Policy],
        solver: BaseActuarialSolver,
        time_steps: int,
    ) -> list[ReserveRecord]:
        records: list[ReserveRecord] = []

        for policy in policies:
            trajectory = solver.solve(policy=policy, num_steps=time_steps)
            base_rate = float(policy.scenario_interest_rate)
            low_rate = max(_MIN_INTEREST_RATE, base_rate - _INTEREST_RATE_SENSITIVITY_DELTA)
            high_rate = max(low_rate + 1.0e-6, base_rate + _INTEREST_RATE_SENSITIVITY_DELTA)

            low_rate_policy = replace(policy, scenario_interest_rate=low_rate, interest_rate=low_rate)
            high_rate_policy = replace(policy, scenario_interest_rate=high_rate, interest_rate=high_rate)
            low_rate_trajectory = solver.solve(policy=low_rate_policy, num_steps=time_steps)
            high_rate_trajectory = solver.solve(policy=high_rate_policy, num_steps=time_steps)

            rate_span = high_rate - low_rate
            sum_assured_scale = max(float(policy.sum_assured), 1.0)
            terminal_mortality = policy.mortality_profile.intensity_at(float(policy.term))
            low_peak_index = int(np.argmax(low_rate_trajectory.reserves))
            high_peak_index = int(np.argmax(high_rate_trajectory.reserves))
            low_peak_time = float(low_rate_trajectory.times[low_peak_index])
            high_peak_time = float(high_rate_trajectory.times[high_peak_index])
            low_peak_mortality = float(policy.mortality_profile.intensity_at(low_peak_time))
            high_peak_mortality = float(policy.mortality_profile.intensity_at(high_peak_time))
            low_peak_reserve_ratio = float(low_rate_trajectory.reserves[low_peak_index]) / sum_assured_scale
            high_peak_reserve_ratio = float(high_rate_trajectory.reserves[high_peak_index]) / sum_assured_scale

            for time_point, reserve, reserve_low, reserve_high in zip(
                trajectory.times,
                trajectory.reserves,
                low_rate_trajectory.reserves,
                high_rate_trajectory.reserves,
            ):
                mortality = policy.mortality_profile.intensity_at(time_point)
                interest_rate_sensitivity = ((float(reserve_high) - float(reserve_low)) / rate_span) / sum_assured_scale
                records.append(
                    ReserveRecord(
                        policy_id=policy.policy_id,
                        features=build_policy_feature_array(
                            policy=policy,
                            time_point=float(time_point),
                            mortality=float(mortality),
                        ),
                        reserve=float(reserve),
                        term=float(policy.term),
                        terminal_mortality=float(terminal_mortality),
                        interest_rate_sensitivity=float(interest_rate_sensitivity),
                        interest_rate_shock_down_reserve_ratio=float(reserve_low) / sum_assured_scale,
                        interest_rate_shock_up_reserve_ratio=float(reserve_high) / sum_assured_scale,
                        interest_rate_shock_down_peak_time=low_peak_time,
                        interest_rate_shock_up_peak_time=high_peak_time,
                        interest_rate_shock_down_peak_mortality=low_peak_mortality,
                        interest_rate_shock_up_peak_mortality=high_peak_mortality,
                        interest_rate_shock_down_peak_reserve_ratio=low_peak_reserve_ratio,
                        interest_rate_shock_up_peak_reserve_ratio=high_peak_reserve_ratio,
                    )
                )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def normalize_features(self, raw_features: torch.Tensor) -> torch.Tensor:
        """Normalize raw PINN features using the exact training contract."""

        features = raw_features.clone()
        features[..., FEATURE_INDEX["time"]] /= FEATURE_SCALES["time"]
        features[..., FEATURE_INDEX["age"]] /= FEATURE_SCALES["age"]
        features[..., FEATURE_INDEX["pricing_interest_rate"]] /= FEATURE_SCALES["pricing_interest_rate"]
        features[..., FEATURE_INDEX["scenario_interest_rate"]] = (
            features[..., FEATURE_INDEX["scenario_interest_rate"]] - self.interest_mean
        ) / self.interest_std
        features[..., FEATURE_INDEX["premium_ratio"]] = (
            features[..., FEATURE_INDEX["premium_ratio"]] - self.premium_mean
        ) / self.premium_std
        features[..., FEATURE_INDEX["sum_assured"]] /= FEATURE_SCALES["sum_assured"]
        features[..., FEATURE_INDEX["mortality"]] /= FEATURE_SCALES["mortality"]
        return features

    def denormalize_target(
        self,
        z_value: torch.Tensor | float,
        sum_assured: torch.Tensor | float,
    ) -> torch.Tensor | float:
        """Convert model z-output back into reserve currency units."""

        if isinstance(z_value, torch.Tensor):
            scale = torch.clamp(
                torch.as_tensor(sum_assured, dtype=z_value.dtype, device=z_value.device),
                min=1.0,
            )
            return (z_value * self.target_std + self.target_mean) * scale
        scale_value = max(float(sum_assured), 1.0)
        return (float(z_value) * self.target_std + self.target_mean) * scale_value

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        raw_features = torch.tensor(record.features, dtype=torch.float32)
        features = self.normalize_features(raw_features)
        sum_assured_value = float(raw_features[FEATURE_INDEX["sum_assured"]].item())
        sum_assured_scale = max(sum_assured_value, 1.0)
        reserve_ratio = float(record.reserve) / sum_assured_scale
        z_target = (reserve_ratio - self.target_mean) / self.target_std

        return {
            "features": features,
            "raw_features": raw_features,
            "target": torch.tensor([z_target], dtype=torch.float32),
            "sum_assured_scale": torch.tensor([sum_assured_scale], dtype=torch.float32),
            "target_mean": torch.tensor([self.target_mean], dtype=torch.float32),
            "target_std": torch.tensor([self.target_std], dtype=torch.float32),
            "term": torch.tensor([record.term], dtype=torch.float32),
            "terminal_mortality": torch.tensor([record.terminal_mortality], dtype=torch.float32),
            "interest_mean": torch.tensor([self.interest_mean], dtype=torch.float32),
            "interest_std": torch.tensor([self.interest_std], dtype=torch.float32),
            "premium_mean": torch.tensor([self.premium_mean], dtype=torch.float32),
            "premium_std": torch.tensor([self.premium_std], dtype=torch.float32),
            "interest_rate_sensitivity_target": torch.tensor([record.interest_rate_sensitivity], dtype=torch.float32),
            "interest_rate_shock_down_target": torch.tensor([record.interest_rate_shock_down_reserve_ratio], dtype=torch.float32),
            "interest_rate_shock_up_target": torch.tensor([record.interest_rate_shock_up_reserve_ratio], dtype=torch.float32),
            "interest_rate_shock_delta": torch.tensor([_INTEREST_RATE_SENSITIVITY_DELTA], dtype=torch.float32),
            "interest_rate_shock_down_peak_time": torch.tensor([record.interest_rate_shock_down_peak_time], dtype=torch.float32),
            "interest_rate_shock_up_peak_time": torch.tensor([record.interest_rate_shock_up_peak_time], dtype=torch.float32),
            "interest_rate_shock_down_peak_mortality": torch.tensor([record.interest_rate_shock_down_peak_mortality], dtype=torch.float32),
            "interest_rate_shock_up_peak_mortality": torch.tensor([record.interest_rate_shock_up_peak_mortality], dtype=torch.float32),
            "interest_rate_shock_down_peak_target": torch.tensor([record.interest_rate_shock_down_peak_reserve_ratio], dtype=torch.float32),
            "interest_rate_shock_up_peak_target": torch.tensor([record.interest_rate_shock_up_peak_reserve_ratio], dtype=torch.float32),
        }






"""Datasets for reserve learning.

Created: 2026-05-31  Revised: 2026-07-02
Purpose: Build supervised and knowledge-informed training records for reserve
surrogate experiments, including explicit pricing-rate vs scenario-rate
features and solver-derived rate-shock curve targets.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Iterable
import numpy as np
import torch
from torch.utils.data import Dataset
from src.actuarial.actuarial_solver import BaseActuarialSolver
from src.actuarial.policy import Policy

FEATURE_INDEX = {
    "time": 0,
    "age": 1,
    "pricing_interest_rate": 2,
    "scenario_interest_rate": 3,
    "premium": 4,
    "sum_assured": 5,
    "mortality": 6,
}

FEATURE_SCALES = {
    "time":          30.0,
    "age":           45.0,
    "pricing_interest_rate": 0.07,
    "scenario_interest_rate": 0.07,
    "premium":       2_000.0,
    "sum_assured":   950_000.0,
    "mortality":     0.05,
}

_INTEREST_RATE_SENSITIVITY_DELTA = 0.005
_MIN_INTEREST_RATE = 1.0e-4


@dataclass(slots=True)
class ReserveRecord:
    """One reserve-observation row used for model training or evaluation.

    Attributes:
        policy_id: Policy identifier associated with the row.
        features: Raw actuarial features in business units.
        reserve: Classical reserve value at the time point.
        term: Policy term in years.
        terminal_mortality: Mortality intensity at maturity for boundary loss.
        interest_rate_sensitivity: Central-difference approximation of
            ``d(V/S)/dr`` at the same time point, where ``V`` is reserve,
            ``S`` is sum assured, and ``r`` is the interest-rate assumption.
        interest_rate_shock_down_reserve_ratio: Classical shocked reserve ratio
            under a downward interest-rate perturbation with all other policy
            attributes fixed.
        interest_rate_shock_up_reserve_ratio: Classical shocked reserve ratio
            under an upward interest-rate perturbation with all other policy
            attributes fixed.
        interest_rate_shock_down_peak_time: Time location of the classical peak
            reserve under the downward rate shock.
        interest_rate_shock_up_peak_time: Time location of the classical peak
            reserve under the upward rate shock.
        interest_rate_shock_down_peak_mortality: Mortality intensity at the
            downward-shock peak time.
        interest_rate_shock_up_peak_mortality: Mortality intensity at the
            upward-shock peak time.
        interest_rate_shock_down_peak_reserve_ratio: Peak reserve ratio
            ``max_t V(t; r-Δr) / S`` under the downward rate shock.
        interest_rate_shock_up_peak_reserve_ratio: Peak reserve ratio
            ``max_t V(t; r+Δr) / S`` under the upward rate shock.

    Scientific Context:
        ``interest_rate_sensitivity`` is computed by solving the same policy
        under ``r - Δr`` and ``r + Δr`` while keeping premium and all other
        contract features fixed. This turns a pure cross-sectional dataset into
        a local counterfactual dataset that can supervise rate sensitivities.

    Business Interpretation:
        This record stores not only "what the reserve is" but also "what the
        reserve curve should look like if rates are shocked up or down", which
        is exactly what stress testing and what-if dashboards need to get right.
    """

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
    premium: float,
    sum_assured: float,
    mortality: float,
) -> np.ndarray:
    """Build one raw feature vector in business units.

    Args:
        time_point: Elapsed time since policy inception.
        age: Issue age.
        pricing_interest_rate: Issue-time pricing rate used to derive premium.
        scenario_interest_rate: Valuation/stress-testing rate used in reserve
            dynamics.
        premium: Contractual premium cash flow.
        sum_assured: Death benefit amount.
        mortality: Pointwise mortality intensity.

    Returns:
        np.ndarray: Raw feature vector aligned to ``FEATURE_INDEX``.

    Business Interpretation:
        This is the canonical contract-plus-state representation used
        consistently across training, validation, plotting, and stress testing.
    """

    return np.asarray(
        [
            time_point,
            age,
            pricing_interest_rate,
            scenario_interest_rate,
            premium,
            sum_assured,
            mortality,
        ],
        dtype=np.float32,
    )


def normalize_raw_feature_array(raw_features: np.ndarray) -> np.ndarray:
    """Normalize a raw feature vector using configured feature scales.

    Args:
        raw_features: Raw feature vector aligned to ``FEATURE_INDEX``.

    Returns:
        np.ndarray: Normalized feature vector suitable for model input.
    """

    normalized = raw_features.astype(np.float32, copy=True)
    for name, index in FEATURE_INDEX.items():
        normalized[index] /= FEATURE_SCALES[name]
    return normalized


def build_policy_feature_array(
    policy: Policy,
    time_point: float,
    *,
    scenario_interest_rate: float | None = None,
    mortality: float | None = None,
) -> np.ndarray:
    """Build one raw feature vector from a policy and time point.

    Args:
        policy: Policy supplying contract features.
        time_point: Elapsed policy time.
        scenario_interest_rate: Optional valuation-rate override.
        mortality: Optional mortality override at ``time_point``.

    Returns:
        np.ndarray: Raw feature vector aligned to ``FEATURE_INDEX``.
    """

    resolved_mortality = policy.mortality_profile.intensity_at(time_point) if mortality is None else mortality
    resolved_scenario_rate = policy.scenario_interest_rate if scenario_interest_rate is None else scenario_interest_rate
    return build_raw_feature_array(
        time_point=float(time_point),
        age=float(policy.age),
        pricing_interest_rate=float(policy.pricing_interest_rate),
        scenario_interest_rate=float(resolved_scenario_rate),
        premium=float(policy.premium),
        sum_assured=float(policy.sum_assured),
        mortality=float(resolved_mortality),
    )


class ReserveDataset(Dataset):
    """Reserve-learning dataset backed by synthetic policies and a classical solver.

    Scientific Context:
        Each policy is expanded into a time-indexed reserve trajectory. The
        dataset fits target standardization on reserve-to-sum-assured ratios so
        the neural network learns a scale-stabilized liability surface.

    Business Interpretation:
        This dataset is the actuarial training book used to teach the surrogate
        both reserve levels and selected local behaviors such as rate
        sensitivity and scenario response.
    """

    def __init__(self, policies, solver, time_steps,
                 target_mean=None, target_std=None):
        self.records = self._build_records(list(policies), solver=solver, time_steps=time_steps)

        v_values = np.array([
            r.reserve / max(r.features[FEATURE_INDEX["sum_assured"]], 1.0)
            for r in self.records
        ], dtype=np.float32)

        if target_mean is None:
            self.target_mean = float(v_values.mean())
        else:
            self.target_mean = float(target_mean)

        if target_std is None:
            std = float(v_values.std())
            self.target_std = std if std > 1e-8 else 1.0
        else:
            self.target_std = float(target_std)

    @staticmethod
    def _build_records(policies, solver, time_steps):
        records = []
        for policy in policies:
            trajectory = solver.solve(policy=policy, num_steps=time_steps)
            low_rate = max(_MIN_INTEREST_RATE, float(policy.scenario_interest_rate) - _INTEREST_RATE_SENSITIVITY_DELTA)
            high_rate = max(low_rate + 1.0e-6, float(policy.scenario_interest_rate) + _INTEREST_RATE_SENSITIVITY_DELTA)
            low_rate_policy = replace(policy, scenario_interest_rate=low_rate)
            high_rate_policy = replace(policy, scenario_interest_rate=high_rate)
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
                interest_rate_sensitivity = ((reserve_high - reserve_low) / rate_span) / sum_assured_scale
                reserve_ratio_low = float(reserve_low) / sum_assured_scale
                reserve_ratio_high = float(reserve_high) / sum_assured_scale
                features = build_policy_feature_array(
                    policy=policy,
                    time_point=float(time_point),
                    mortality=float(mortality),
                )
                records.append(ReserveRecord(
                    policy_id=policy.policy_id,
                    features=features,
                    reserve=float(reserve),
                    term=float(policy.term),
                    terminal_mortality=float(terminal_mortality),
                    interest_rate_sensitivity=float(interest_rate_sensitivity),
                    interest_rate_shock_down_reserve_ratio=reserve_ratio_low,
                    interest_rate_shock_up_reserve_ratio=reserve_ratio_high,
                    interest_rate_shock_down_peak_time=low_peak_time,
                    interest_rate_shock_up_peak_time=high_peak_time,
                    interest_rate_shock_down_peak_mortality=low_peak_mortality,
                    interest_rate_shock_up_peak_mortality=high_peak_mortality,
                    interest_rate_shock_down_peak_reserve_ratio=low_peak_reserve_ratio,
                    interest_rate_shock_up_peak_reserve_ratio=high_peak_reserve_ratio,
                ))
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        raw_features = torch.tensor(record.features, dtype=torch.float32)
        features = torch.tensor(normalize_raw_feature_array(record.features), dtype=torch.float32)

        sum_assured_val = float(raw_features[FEATURE_INDEX["sum_assured"]].item())
        scale = max(sum_assured_val, 1.0)
        v = record.reserve / scale
        z = (v - self.target_mean) / self.target_std

        return {
            "features":        features,
            "raw_features":    raw_features,
            "target":          torch.tensor([z], dtype=torch.float32),
            "sum_assured_scale": torch.tensor([scale], dtype=torch.float32),
            "target_mean":     torch.tensor([self.target_mean], dtype=torch.float32),
            "target_std":      torch.tensor([self.target_std], dtype=torch.float32),
            "term":            torch.tensor([record.term], dtype=torch.float32),
            "terminal_mortality": torch.tensor([record.terminal_mortality], dtype=torch.float32),
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

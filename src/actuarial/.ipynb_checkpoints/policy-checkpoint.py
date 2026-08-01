"""Policy domain models.

Created: 2026-05-31
Purpose: Define policy and mortality containers used throughout the platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class MortalityProfile:
    """Discrete mortality profile for a policy."""

    times: np.ndarray
    intensities: np.ndarray
    source: str = "synthetic"

    def intensity_at(self, time_point: float) -> float:
        """Interpolate mortality intensity at an arbitrary time point."""

        return float(np.interp(time_point, self.times, self.intensities))


@dataclass(init=False, slots=True)
class Policy:
    """Term-life policy description with pricing and scenario rate semantics.

    ``interest_rate`` is kept as a backward-compatible alias for
    ``scenario_interest_rate``. New code should use ``pricing_interest_rate``
    for issue-time premium pricing and ``scenario_interest_rate`` for reserve
    valuation and stress testing.
    """

    policy_id: str
    age: int
    term: int
    premium: float
    interest_rate: float
    sum_assured: float
    mortality_profile: MortalityProfile
    pricing_interest_rate: float
    scenario_interest_rate: float
    metadata: dict[str, float | int | str]

    def __init__(
        self,
        policy_id: str,
        age: int,
        term: int,
        premium: float,
        interest_rate: float | None = None,
        sum_assured: float | None = None,
        mortality_profile: MortalityProfile | None = None,
        pricing_interest_rate: float | None = None,
        scenario_interest_rate: float | None = None,
        metadata: dict[str, float | int | str] | None = None,
    ) -> None:
        if sum_assured is None:
            raise ValueError("sum_assured must be provided.")
        if mortality_profile is None:
            raise ValueError("mortality_profile must be provided.")

        resolved_pricing_rate = pricing_interest_rate
        if resolved_pricing_rate is None:
            resolved_pricing_rate = interest_rate
        if resolved_pricing_rate is None:
            resolved_pricing_rate = scenario_interest_rate
        if resolved_pricing_rate is None:
            raise ValueError(
                "Policy requires pricing_interest_rate, scenario_interest_rate, or interest_rate."
            )

        resolved_scenario_rate = scenario_interest_rate
        if resolved_scenario_rate is None:
            resolved_scenario_rate = interest_rate
        if resolved_scenario_rate is None:
            resolved_scenario_rate = resolved_pricing_rate

        self.policy_id = policy_id
        self.age = int(age)
        self.term = int(term)
        self.premium = float(premium)
        self.interest_rate = float(resolved_scenario_rate)
        self.sum_assured = float(sum_assured)
        self.mortality_profile = mortality_profile
        self.pricing_interest_rate = float(resolved_pricing_rate)
        self.scenario_interest_rate = float(resolved_scenario_rate)
        self.metadata = {} if metadata is None else metadata

    def times(self, num_steps: int) -> np.ndarray:
        """Return evenly spaced time points over the policy term."""

        return np.linspace(0.0, float(self.term), num_steps, dtype=float)


def build_mortality_profile(times: Sequence[float], intensities: Sequence[float], source: str) -> MortalityProfile:
    """Construct a mortality profile from iterable inputs."""

    return MortalityProfile(
        times=np.asarray(times, dtype=float),
        intensities=np.asarray(intensities, dtype=float),
        source=source,
    )

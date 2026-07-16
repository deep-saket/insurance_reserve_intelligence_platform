"""Policy simulation engine.

Created: 2026-05-31
Purpose: Generate synthetic term-life policies and scenario-adjusted portfolios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.actuarial.policy import MortalityProfile, Policy
from src.data.mortality_loader import MortalityDataSource


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate using the newest available NumPy trapezoid implementation.

    Args:
        y: Function values to integrate.
        x: Grid points for the integration.

    Returns:
        float: Numerical integral result.

    Business Interpretation:
        This keeps premium generation portable across NumPy versions so reserve
        research does not fail because of a local runtime mismatch.
    """

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


@dataclass(frozen=True, slots=True)
class RiskProfile:
    """Underwriting risk categories and their combined adjustment factor.

    Business Interpretation:
        This is a compact synthetic underwriting classification. The combined
        factor scales mortality. Premiums are then calculated from the adjusted
        mortality curve so higher-risk policies feed more realistic assumptions
        into reserve valuation.
    """

    smoker_status: str
    health_tier: str
    occupation_risk: str
    gender: str
    smoker_factor: float
    health_factor: float
    occupation_factor: float
    gender_factor: float

    @property
    def combined_factor(self) -> float:
        """Return the multiplicative underwriting adjustment factor."""

        return (
            self.smoker_factor
            * self.health_factor
            * self.occupation_factor
            * self.gender_factor
        )

    def as_metadata(self) -> dict[str, float | str]:
        """Return policy metadata for reporting and scenario cloning."""

        return {
            "smoker_status": self.smoker_status,
            "health_tier": self.health_tier,
            "occupation_risk": self.occupation_risk,
            "gender": self.gender,
            "smoker_factor": self.smoker_factor,
            "health_factor": self.health_factor,
            "occupation_factor": self.occupation_factor,
            "gender_factor": self.gender_factor,
            "risk_adjustment_factor": self.combined_factor,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, float | int | str]) -> RiskProfile:
        """Reconstruct a risk profile stored on an existing policy."""

        return cls(
            smoker_status=str(metadata["smoker_status"]),
            health_tier=str(metadata["health_tier"]),
            occupation_risk=str(metadata["occupation_risk"]),
            gender=str(metadata["gender"]),
            smoker_factor=float(metadata["smoker_factor"]),
            health_factor=float(metadata["health_factor"]),
            occupation_factor=float(metadata["occupation_factor"]),
            gender_factor=float(metadata["gender_factor"]),
        )


@dataclass(slots=True)
class ScenarioDefinition:
    """Scenario overrides for synthetic policy generation.

    Attributes:
        interest_rate_shift: Additive interest-rate adjustment.
        mortality_multiplier: Multiplicative mortality adjustment.
        premium_multiplier: Multiplicative premium adjustment.
        sum_assured_multiplier: Multiplicative sum-assured adjustment.
    """

    interest_rate_shift: float = 0.0
    mortality_multiplier: float = 1.0
    premium_multiplier: float = 1.0
    sum_assured_multiplier: float = 1.0


class PolicySimulator:
    """Generate synthetic term-life policies.

    Underwriting assumptions:
        - Smoker status: 20% smoker, with a 30% mortality increase.
        - Health tier depends on smoking status. Smokers are more likely to be
          substandard and less likely to receive a preferred classification.
        - Occupation risk: 70% low, 25% moderate, and 5% high.
        - Gender: 52% male and 48% female, with a lower female mortality factor.

    Category factors are multiplied together. These are transparent research
    assumptions, not calibrated insurer pricing or underwriting tables. Premiums
    are calculated using the actuarial equivalence principle with a loading.
    """

    SMOKER_FACTORS: dict[str, float] = {"non_smoker": 1.0, "smoker": 1.30}
    HEALTH_FACTORS: dict[str, float] = {"preferred": 0.90, "standard": 1.0, "substandard": 1.25}
    OCCUPATION_FACTORS: dict[str, float] = {"low": 0.95, "moderate": 1.0, "high": 1.20}
    GENDER_FACTORS: dict[str, float] = {"male": 1.0, "female": 0.85}

    def __init__(
        self,
        age_range: tuple[int, int],
        term_range: tuple[int, int],
        interest_rate_range: tuple[float, float],
        sum_assured_range: tuple[float, float],
        mortality_source: MortalityDataSource | None = None,
        mortality_scale: float = 0.0005,
        mortality_shape: float = 1.08,
        mortality_reference_age: int = 25,
        premium_loading: float = 1.10,
        max_expiry_age: int = 80,
        sum_assured_rounding: float = 50_000.0,
        sum_assured_age_decay: float = 0.02,
        seed: int = 42,
    ) -> None:
        """Initialize the policy simulator.

        Args:
            age_range: Inclusive age sampling bounds.
            term_range: Inclusive term sampling bounds.
            interest_rate_range: Interest-rate sampling bounds.
            sum_assured_range: Sum-assured sampling bounds.
            mortality_source: Optional external mortality source.
            mortality_scale: Synthetic mortality intensity at the reference age.
            mortality_shape: Growth factor for synthetic mortality.
            mortality_reference_age: Age corresponding to the mortality scale.
            premium_loading: Multiplicative loading applied to the net premium.
            max_expiry_age: Maximum permitted age when term coverage expires.
            sum_assured_rounding: Increment used to round sampled death benefits.
            sum_assured_age_decay: Exponential decay in maximum cover by issue age.
            seed: Random seed for reproducibility.
        """
        self.age_range = age_range
        self.term_range = term_range
        self.interest_rate_range = interest_rate_range
        self.sum_assured_range = sum_assured_range
        self.mortality_source = mortality_source
        self.mortality_scale = mortality_scale
        self.mortality_shape = mortality_shape
        self.mortality_reference_age = mortality_reference_age
        self.premium_loading = premium_loading
        self.max_expiry_age = max_expiry_age
        self.sum_assured_rounding = sum_assured_rounding
        self.sum_assured_age_decay = sum_assured_age_decay
        self.rng = np.random.default_rng(seed)

    def _sample_risk_profile(self) -> RiskProfile:
        """Sample a synthetic underwriting classification with health correlation."""

        smoker_status = str(self.rng.choice(["non_smoker", "smoker"], p=[0.80, 0.20]))
        health_probabilities = (
            [0.05, 0.65, 0.30] if smoker_status == "smoker" else [0.25, 0.65, 0.10]
        )
        health_tier = str(
            self.rng.choice(["preferred", "standard", "substandard"], p=health_probabilities)
        )
        occupation_risk = str(self.rng.choice(["low", "moderate", "high"], p=[0.70, 0.25, 0.05]))
        gender = str(self.rng.choice(["male", "female"], p=[0.52, 0.48]))
        return RiskProfile(
            smoker_status=smoker_status,
            health_tier=health_tier,
            occupation_risk=occupation_risk,
            gender=gender,
            smoker_factor=self.SMOKER_FACTORS[smoker_status],
            health_factor=self.HEALTH_FACTORS[health_tier],
            occupation_factor=self.OCCUPATION_FACTORS[occupation_risk],
            gender_factor=self.GENDER_FACTORS[gender],
        )

    def _sample_term(self, age: int) -> int:
        """Sample a valid term without allowing coverage beyond the expiry age."""

        term_max = min(self.term_range[1], self.max_expiry_age - age)
        if term_max < self.term_range[0]:
            raise ValueError(
                f"No valid term for issue age {age}: minimum term {self.term_range[0]} "
                f"exceeds maximum expiry age {self.max_expiry_age}."
            )
        return int(self.rng.integers(self.term_range[0], term_max + 1))

    def _sum_assured_upper_bound(self, age: int) -> float:
        """Return the age-adjusted maximum death benefit."""

        years_above_minimum = max(0, age - self.age_range[0])
        age_adjusted_maximum = self.sum_assured_range[1] * np.exp(
            -self.sum_assured_age_decay * years_above_minimum
        )
        rounded_maximum = np.floor(age_adjusted_maximum / self.sum_assured_rounding)
        rounded_maximum *= self.sum_assured_rounding
        return float(max(self.sum_assured_range[0], rounded_maximum))

    def _sample_sum_assured(self, age: int) -> float:
        """Sample a right-skewed, age-sensitive, rounded death benefit."""

        minimum = self.sum_assured_range[0]
        maximum = self._sum_assured_upper_bound(age)
        median = max(minimum, maximum * 0.35)
        sampled = float(self.rng.lognormal(mean=np.log(median), sigma=0.60))
        bounded = float(np.clip(sampled, minimum, maximum))
        rounded = round(bounded / self.sum_assured_rounding) * self.sum_assured_rounding
        return float(np.clip(rounded, minimum, maximum))

    def _synthetic_mortality_profile(
        self, age: int, term: int, multiplier: float = 1.0
    ) -> MortalityProfile:
        """Create a synthetic mortality profile.

        Args:
            age: Inception age for the policy.
            term: Policy term in years.
            multiplier: Scenario multiplier applied to intensities.

        Returns:
            MortalityProfile: Synthetic mortality profile.
        """
        times = np.linspace(0.0, float(term), term + 1, dtype=float)
        ages = age + times
        rates = (
            self.mortality_scale
            * np.power(self.mortality_shape, ages - self.mortality_reference_age)
            * multiplier
        )
        return MortalityProfile(times=times, intensities=rates, source="synthetic")

    def _mortality_profile(self, age: int, term: int, multiplier: float = 1.0) -> MortalityProfile:
        """Resolve either external or synthetic mortality assumptions.

        Args:
            age: Inception age for the policy.
            term: Policy term in years.
            multiplier: Scenario multiplier applied to intensities.

        Returns:
            MortalityProfile: Policy mortality profile.
        """
        if self.mortality_source is not None:
            profile = self.mortality_source.load(age=age, term=term)
            return MortalityProfile(
                times=profile.times,
                intensities=profile.intensities * multiplier,
                source=profile.source,
            )
        return self._synthetic_mortality_profile(age=age, term=term, multiplier=multiplier)

    def _calculate_premium_rate(
        self,
        mortality_profile: MortalityProfile,
        term: int,
        interest_rate: float,
        sum_assured: float,
    ) -> tuple[float, float]:
        """Calculate a loaded continuous premium rate using the equivalence principle.

        Premiums are paid continuously while the insured is alive. Death benefits
        are paid at death. The net premium rate equates the expected present value
        of premiums and benefits, matching the continuous-time Thiele equation.
        """

        times = np.linspace(0.0, float(term), term * 12 + 1, dtype=float)
        mortality = np.asarray(
            [mortality_profile.intensity_at(time_point) for time_point in times],
            dtype=float,
        )
        intervals = np.diff(times)
        cumulative_hazard = np.concatenate(
            ([0.0], np.cumsum(0.5 * (mortality[:-1] + mortality[1:]) * intervals))
        )
        survival_probability = np.exp(-cumulative_hazard)
        discount_factor = np.exp(-interest_rate * times)
        premium_annuity_epv = _trapezoid(survival_probability * discount_factor, times)
        benefit_epv = _trapezoid(
            survival_probability * mortality * sum_assured * discount_factor,
            times,
        )
        net_premium = benefit_epv / premium_annuity_epv
        return net_premium, net_premium * self.premium_loading

    def _build_policy(
        self,
        policy_id: str,
        age: int,
        term: int,
        pricing_interest_rate: float | None = None,
        sum_assured: float | None = None,
        mortality_multiplier: float = 1.0,
        premium_multiplier: float = 1.0,
        risk_profile: RiskProfile | None = None,
        scenario_interest_rate: float | None = None,
        interest_rate: float | None = None,
    ) -> Policy:
        """Build a single policy object.

        Args:
            policy_id: Unique policy identifier.
            age: Age at inception.
            term: Policy term in years.
            pricing_interest_rate: Interest-rate assumption used to price the
                policy premium at issue.
            sum_assured: Death benefit amount.
            mortality_multiplier: Scenario multiplier for mortality.
            premium_multiplier: Scenario multiplier applied after pricing.
            risk_profile: Optional underwriting risk categories. A new synthetic
                profile is sampled when this is omitted.
            scenario_interest_rate: Optional reserve-valuation/stress-testing
                rate. Defaults to the pricing rate when omitted.
            interest_rate: Backward-compatible alias for
                ``pricing_interest_rate`` retained for older internal callers
                and tests.

        Returns:
            Policy: Structured policy instance.
        """
        resolved_pricing_rate = pricing_interest_rate if pricing_interest_rate is not None else interest_rate
        if resolved_pricing_rate is None:
            raise ValueError("Either pricing_interest_rate or interest_rate must be provided.")
        if sum_assured is None:
            raise ValueError("sum_assured must be provided.")
        profile = risk_profile or self._sample_risk_profile()
        adjusted_mortality_multiplier = mortality_multiplier * profile.combined_factor
        mortality_profile = self._mortality_profile(
            age=age,
            term=term,
            multiplier=adjusted_mortality_multiplier,
        )
        net_premium, loaded_premium = self._calculate_premium_rate(
            mortality_profile=mortality_profile,
            term=term,
            interest_rate=resolved_pricing_rate,
            sum_assured=sum_assured,
        )
        valuation_rate = resolved_pricing_rate if scenario_interest_rate is None else scenario_interest_rate
        return Policy(
            policy_id=policy_id,
            age=age,
            term=term,
            premium=loaded_premium * premium_multiplier,
            pricing_interest_rate=resolved_pricing_rate,
            scenario_interest_rate=valuation_rate,
            sum_assured=sum_assured,
            mortality_profile=mortality_profile,
            metadata={
                **profile.as_metadata(),
                "net_premium": net_premium,
                "premium_loading": self.premium_loading,
                "premium_multiplier": premium_multiplier,
                "pricing_interest_rate": resolved_pricing_rate,
                "scenario_interest_rate": valuation_rate,
            },
        )

    def generate_random_policies(self, count: int) -> list[Policy]:
        """Generate randomly sampled policies.

        Args:
            count: Number of policies to sample.

        Returns:
            list[Policy]: Randomly sampled policies.
        """

        policies: list[Policy] = []
        for index in range(count):
            age = int(self.rng.integers(self.age_range[0], self.age_range[1] + 1))
            term = self._sample_term(age)
            interest_rate = float(self.rng.uniform(*self.interest_rate_range))
            sum_assured = self._sample_sum_assured(age)
            policies.append(
                self._build_policy(
                    f"policy_{index:05d}",
                    age,
                    term,
                    interest_rate,
                    sum_assured,
                )
            )
        return policies

    def generate_stratified_policies(
        self, counts_by_age_band: dict[tuple[int, int], int]
    ) -> list[Policy]:
        """Generate policies by age strata.

        Args:
            counts_by_age_band: Mapping from age band to desired policy count.

        Returns:
            list[Policy]: Stratified policies across the requested age bands.
        """

        policies: list[Policy] = []
        start_index = 0
        for (age_min, age_max), count in counts_by_age_band.items():
            for offset in range(count):
                age = int(self.rng.integers(age_min, age_max + 1))
                term = self._sample_term(age)
                interest_rate = float(self.rng.uniform(*self.interest_rate_range))
                sum_assured = self._sample_sum_assured(age)
                policies.append(
                    self._build_policy(
                        f"stratified_{start_index + offset:05d}",
                        age,
                        term,
                        interest_rate,
                        sum_assured,
                    )
                )
            start_index += count
        return policies

    def generate_scenario_policies(
        self, base_policies: Iterable[Policy], scenario: ScenarioDefinition
    ) -> list[Policy]:
        """Clone policies under a user-specified scenario.

        Args:
            base_policies: Baseline policies to clone.
            scenario: Scenario overrides to apply.

        Returns:
            list[Policy]: Scenario-adjusted policies.
        """

        stressed: list[Policy] = []
        for policy in base_policies:
            risk_profile = RiskProfile.from_metadata(policy.metadata)
            adjusted_mortality_multiplier = scenario.mortality_multiplier * risk_profile.combined_factor
            mortality_profile = self._mortality_profile(
                age=policy.age,
                term=policy.term,
                multiplier=adjusted_mortality_multiplier,
            )
            stressed.append(
                Policy(
                    policy_id=f"{policy.policy_id}_scenario",
                    age=policy.age,
                    term=policy.term,
                    premium=policy.premium * scenario.premium_multiplier,
                    pricing_interest_rate=policy.pricing_interest_rate,
                    scenario_interest_rate=policy.scenario_interest_rate + scenario.interest_rate_shift,
                    sum_assured=policy.sum_assured * scenario.sum_assured_multiplier,
                    mortality_profile=mortality_profile,
                    metadata={
                        **policy.metadata,
                        "premium_multiplier": float(policy.metadata.get("premium_multiplier", 1.0)) * scenario.premium_multiplier,
                        "pricing_interest_rate": policy.pricing_interest_rate,
                        "scenario_interest_rate": policy.scenario_interest_rate + scenario.interest_rate_shift,
                    },
                )
            )
        return stressed

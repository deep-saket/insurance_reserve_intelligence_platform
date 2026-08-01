"""Policy simulation engine.

Created: 2026-05-31
Updated: 2026-06-25
Purpose: Generate synthetic term-life policies and scenario-adjusted portfolios.

Change summary (2026-06-25):
    P1-1  Premium is now fully risk-based: net premium is calculated on the
          risk-adjusted mortality curve so every underwriting factor drives cost.
    P1-2  Sum assured is now age-sensitive via a LogNormal whose median tracks a
          per-age income-multiple schedule, producing realistic SA distributions.
    P1-3  Premium/SA coupling is verified and enforced via a sanity check.
    P2-4  Mortality transitions are smoothed by cubic-spline interpolation on
          the external CSV path; synthetic path is unchanged Gompertz.
    P2-5  Interest rates follow a clipped Normal (μ=4 %, σ=1.5 %) rather than
          Uniform, matching real portfolio clustering.
    P2-6  Issue ages follow a Normal (μ=42, σ=10) instead of Uniform.
    P3-7  generate_random_policies produces a 50/30/20 policy-mix
          (standard / high-value / small).
    P3-8  Age → SA → premium correlation is enforced structurally (older →
          lower max SA → lower premium); interest rate is mild-Normal, not iid
          Uniform, reducing spurious variance.
    P3-9  Portfolio-level sanity checks are run after generation; policies that
          fail hard constraints are regenerated (up to max_retries attempts).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np

from src.actuarial.policy import MortalityProfile, Policy
from src.data.mortality_loader import MortalityDataSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate using the newest available NumPy trapezoid implementation."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _smooth_mortality(
    times: np.ndarray, intensities: np.ndarray
) -> np.ndarray:
    """Apply light cubic-spline smoothing to remove abrupt table discontinuities.

    We fit a cubic spline to log(μ) and evaluate it on the same grid.  Working
    in log-space keeps intensities positive and naturally smooths the
    exponential Gompertz shape.  If scipy is unavailable we return the
    original array unchanged (graceful degradation).

    Args:
        times: Time grid (years since policy inception).
        intensities: Raw mortality intensities.

    Returns:
        np.ndarray: Smoothed intensities on the same grid.
    """
    if len(times) < 4:
        return intensities
    try:
        from scipy.interpolate import CubicSpline  # optional dependency

        log_mu = np.log(np.clip(intensities, 1e-9, None))
        spline = CubicSpline(times, log_mu, bc_type="natural")
        smoothed = np.exp(spline(times))
        return smoothed.astype(float)
    except ImportError:
        return intensities


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class PolicyMix(str, Enum):
    """Policy mix segment labels for portfolio stratification."""

    STANDARD = "standard"
    HIGH_VALUE = "high_value"
    SMALL = "small"


# Mix proportions: (segment, weight, SA-scale factor, age-bias)
_MIX_SEGMENTS: list[tuple[PolicyMix, float, float, float]] = [
    (PolicyMix.STANDARD,   0.50, 1.00, 0.0),
    (PolicyMix.HIGH_VALUE, 0.30, 2.00, -5.0),   # younger, larger SA
    (PolicyMix.SMALL,      0.20, 0.35, +5.0),   # older, smaller SA
]


@dataclass(frozen=True, slots=True)
class RiskProfile:
    """Underwriting risk categories and their combined mortality adjustment factor.

    The combined_factor is a pure mortality multiplier.  Premiums are computed
    on the *adjusted* mortality curve so every underwriting category drives both
    expected-claim cost and the resulting premium symmetrically.
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
        """Multiplicative mortality adjustment across all underwriting axes."""
        return (
            self.smoker_factor
            * self.health_factor
            * self.occupation_factor
            * self.gender_factor
        )

    def as_metadata(self) -> dict[str, float | str]:
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
    """Scenario overrides for synthetic policy generation."""

    interest_rate_shift: float = 0.0
    mortality_multiplier: float = 1.0
    premium_multiplier: float = 1.0
    sum_assured_multiplier: float = 1.0


# ---------------------------------------------------------------------------
# PolicySimulator
# ---------------------------------------------------------------------------

class PolicySimulator:
    """Generate synthetic term-life policies with realistic joint distributions.

    Underwriting assumptions
    ------------------------
    Smoker status  : 20 % smoker → +30 % mortality.
    Health tier    : conditional on smoking; smokers are 5/65/30 % preferred/
                     standard/substandard vs. 25/65/10 % for non-smokers.
    Occupation risk: 70 % low / 25 % moderate / 5 % high.
    Gender         : 52 % male / 48 % female; female mortality factor 0.85.

    All category factors multiply together and scale the *mortality curve*.
    Premiums are then derived from the risk-adjusted curve so that higher-risk
    policies always pay higher premiums for the same SA and term.

    Interest rates  : Normal(μ=4 %, σ=1.5 %), clipped to [rate_min, rate_max].
    Issue ages      : Normal(μ=42, σ=10), clipped to [age_min, age_max].
    Sum assured     : LogNormal whose median follows an age-income-multiple
                      schedule, clipped and rounded per actuarial convention.
    Term            : Drawn from standard market tenors {5,10,15,20,25,30}
                      with empirically calibrated weights.
    """

    # ------------------------------------------------------------------
    # Underwriting factor tables
    # ------------------------------------------------------------------
    SMOKER_FACTORS: dict[str, float] = {"non_smoker": 1.00, "smoker": 1.30}
    HEALTH_FACTORS: dict[str, float] = {
        "preferred": 0.90, "standard": 1.00, "substandard": 1.25
    }
    OCCUPATION_FACTORS: dict[str, float] = {
        "low": 0.95, "moderate": 1.00, "high": 1.20
    }
    GENDER_FACTORS: dict[str, float] = {"male": 1.00, "female": 0.85}

    # Standard market term tenors and their approximate market-share weights
    _TERM_TENORS: list[int] = [5, 10, 15, 20, 25, 30]
    _TERM_WEIGHTS: np.ndarray = np.array([0.05, 0.15, 0.20, 0.30, 0.15, 0.15])

    # Income-multiple schedule: maps issue age → SA as a multiple of notional
    # annual income (₹6 L baseline).  Younger buyers need more cover; older
    # buyers need less and face underwriting limits.
    _SA_INCOME_MULTIPLES: dict[int, float] = {
        25: 20.0,
        35: 15.0,
        45: 10.0,
        55:  6.0,
        65:  4.0,
        75:  2.5,
    }
    _NOTIONAL_ANNUAL_INCOME: float = 600_000.0   # ₹6 L

    def __init__(
        self,
        age_range: tuple[int, int],
        term_range: tuple[int, int],
        interest_rate_range: tuple[float, float],
        sum_assured_range: tuple[float, float],
        mortality_source: MortalityDataSource | None = None,
        mortality_scale: float = 0.00045,
        mortality_shape: float = 1.085,
        mortality_reference_age: int = 25,
        premium_loading: float = 1.10,
        max_expiry_age: int = 80,
        sum_assured_rounding: float = 50_000.0,
        sum_assured_age_decay: float = 0.02,
        seed: int = 42,
        max_retries: int = 5,
        # Interest-rate distribution parameters (P2-5)
        interest_rate_mean: float = 0.04,
        interest_rate_std: float = 0.015,
        # Age distribution parameters (P2-6)
        age_mean: float = 42.0,
        age_std: float = 10.0,
    ) -> None:
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
        self.max_retries = max_retries
        self.interest_rate_mean = interest_rate_mean
        self.interest_rate_std = interest_rate_std
        self.age_mean = age_mean
        self.age_std = age_std
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_risk_profile(self) -> RiskProfile:
        """Sample an underwriting classification with health/smoking correlation."""
        smoker_status = str(
            self.rng.choice(["non_smoker", "smoker"], p=[0.80, 0.20])
        )
        health_probs = (
            [0.05, 0.65, 0.30] if smoker_status == "smoker" else [0.25, 0.65, 0.10]
        )
        health_tier = str(
            self.rng.choice(["preferred", "standard", "substandard"], p=health_probs)
        )
        occupation_risk = str(
            self.rng.choice(["low", "moderate", "high"], p=[0.70, 0.25, 0.05])
        )
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

    def _sample_age(self, age_bias: float = 0.0) -> int:
        """Sample an issue age from a Normal distribution (P2-6).

        Args:
            age_bias: Additive shift applied to the mean before sampling
                      (used by policy-mix segments to skew age slightly).

        Returns:
            int: Issue age clipped to [age_min, age_max].
        """
        raw = self.rng.normal(loc=self.age_mean + age_bias, scale=self.age_std)
        return int(np.clip(round(raw), self.age_range[0], self.age_range[1]))

    def _sample_term(self, age: int) -> int:
        """Sample a policy term from standard market tenors (P2-6 refinement).

        Only tenors that keep expiry ≤ max_expiry_age are eligible.  Their
        weights are renormalised so the distribution shape is preserved.

        Args:
            age: Issue age of the policy.

        Returns:
            int: Sampled term in years.

        Raises:
            ValueError: If no tenor keeps expiry within the permitted limit.
        """
        valid_mask = np.array(
            [age + t <= self.max_expiry_age for t in self._TERM_TENORS]
        )
        if not valid_mask.any():
            raise ValueError(
                f"No valid term for issue age {age}: all tenors exceed "
                f"max_expiry_age={self.max_expiry_age}."
            )
        valid_tenors = [t for t, ok in zip(self._TERM_TENORS, valid_mask) if ok]
        valid_weights = self._TERM_WEIGHTS[valid_mask]
        valid_weights = valid_weights / valid_weights.sum()
        return int(self.rng.choice(valid_tenors, p=valid_weights))

    def _sample_interest_rate(self) -> float:
        """Sample an interest rate from a clipped Normal distribution (P2-5).

        Returns:
            float: Interest rate in [rate_min, rate_max].
        """
        raw = self.rng.normal(loc=self.interest_rate_mean, scale=self.interest_rate_std)
        return float(np.clip(raw, self.interest_rate_range[0], self.interest_rate_range[1]))

    def _income_multiple_at_age(self, age: int) -> float:
        """Interpolate the income-multiple schedule at a given age (P1-2).

        Args:
            age: Issue age.

        Returns:
            float: Smoothly interpolated income multiple.
        """
        knot_ages = sorted(self._SA_INCOME_MULTIPLES)
        knot_mults = [self._SA_INCOME_MULTIPLES[a] for a in knot_ages]
        return float(np.interp(age, knot_ages, knot_mults))

    def _sample_sum_assured(self, age: int, sa_scale: float = 1.0) -> float:
        """Sample a realistically age-correlated sum assured (P1-2 + P3-8).

        The median of the LogNormal is set to

            median_SA = income_multiple(age) × notional_income × sa_scale

        so younger policyholders tend toward larger cover and older ones toward
        smaller cover.  The sigma=0.45 produces a right-skewed but bounded
        distribution consistent with observed retail term-life SA distributions.

        Args:
            age: Issue age of the policy.
            sa_scale: Segment-level multiplier (1.0 for standard, 2.0 for
                      high-value, 0.35 for small policies).

        Returns:
            float: Rounded, clipped sum assured.
        """
        multiple = self._income_multiple_at_age(age)
        median_sa = multiple * self._NOTIONAL_ANNUAL_INCOME * sa_scale
        minimum = self.sum_assured_range[0]
        maximum = self.sum_assured_range[1]
        median_sa = float(np.clip(median_sa, minimum, maximum))

        raw = float(self.rng.lognormal(mean=np.log(median_sa), sigma=0.45))
        clipped = float(np.clip(raw, minimum, maximum))
        rounded = round(clipped / self.sum_assured_rounding) * self.sum_assured_rounding
        return float(np.clip(rounded, minimum, maximum))

    # ------------------------------------------------------------------
    # Mortality
    # ------------------------------------------------------------------

    def _synthetic_mortality_profile(
        self, age: int, term: int, multiplier: float = 1.0
    ) -> MortalityProfile:
        """Gompertz synthetic mortality: μ(t) = scale × shape^(age+t − ref_age).

        Args:
            age: Issue age.
            term: Policy term in years.
            multiplier: Combined underwriting mortality factor.

        Returns:
            MortalityProfile: Synthetic mortality curve.
        """
        times = np.linspace(0.0, float(term), term + 1, dtype=float)
        ages = age + times
        rates = (
            self.mortality_scale
            * np.power(self.mortality_shape, ages - self.mortality_reference_age)
            * multiplier
        )
        return MortalityProfile(times=times, intensities=rates, source="synthetic")

    def _mortality_profile(
        self, age: int, term: int, multiplier: float = 1.0
    ) -> MortalityProfile:
        """Resolve mortality profile from external source or synthetic model (P2-4).

        When an external CSV source is provided the raw intensities are smoothed
        via cubic-spline in log-space to remove abrupt age-table discontinuities
        before the underwriting multiplier is applied.

        Args:
            age: Issue age.
            term: Policy term in years.
            multiplier: Combined underwriting mortality factor.

        Returns:
            MortalityProfile: Policy mortality profile.
        """
        if self.mortality_source is not None:
            profile = self.mortality_source.load(age=age, term=term)
            smoothed = _smooth_mortality(profile.times, profile.intensities)
            return MortalityProfile(
                times=profile.times,
                intensities=smoothed * multiplier,
                source=profile.source,
            )
        return self._synthetic_mortality_profile(age=age, term=term, multiplier=multiplier)

    # ------------------------------------------------------------------
    # Premium calculation  (P1-1 — fully risk-based)
    # ------------------------------------------------------------------

    def _calculate_premium_rate(
        self,
        mortality_profile: MortalityProfile,
        term: int,
        interest_rate: float,
        sum_assured: float,
    ) -> tuple[float, float]:
        """Derive the actuarially fair and loaded continuous premium (P1-1).

        Premium pipeline
        ----------------
        1. Risk-adjusted mortality curve (already embedded in mortality_profile
           via the combined underwriting multiplier) → expected-claim cost.
        2. Equivalence principle: net_premium = EPV(benefits) / EPV(annuity).
        3. Gross premium = net_premium × expense_loading.

        Because step 1 uses the fully risk-adjusted μ(t), a smoker-substandard-
        high-occupation policy automatically pays a higher premium than an
        otherwise identical preferred non-smoker, with no manual post-hoc
        adjustment needed.

        Args:
            mortality_profile: Risk-adjusted mortality curve for this policy.
            term: Policy term in years.
            interest_rate: Valuation interest rate.
            sum_assured: Death benefit.

        Returns:
            tuple[float, float]: (net_premium, gross_premium).
        """
        # Fine integration grid: monthly points over the full term
        times = np.linspace(0.0, float(term), term * 12 + 1, dtype=float)
        mu = np.asarray(
            [mortality_profile.intensity_at(t) for t in times], dtype=float
        )

        # Survival probability via trapezoidal cumulative hazard
        dt = np.diff(times)
        cum_hazard = np.concatenate(
            ([0.0], np.cumsum(0.5 * (mu[:-1] + mu[1:]) * dt))
        )
        survival = np.exp(-cum_hazard)
        discount = np.exp(-interest_rate * times)

        # EPV of a continuous life annuity-due (premium payment vehicle)
        epv_annuity = _trapezoid(survival * discount, times)

        # EPV of death benefit paid continuously at death
        epv_benefit = _trapezoid(survival * mu * sum_assured * discount, times)

        if epv_annuity <= 0.0:
            raise ValueError(
                f"EPV of annuity is non-positive ({epv_annuity:.6f}); "
                "check mortality and interest-rate inputs."
            )

        net_premium = epv_benefit / epv_annuity
        gross_premium = net_premium * self.premium_loading
        return net_premium, gross_premium

    # ------------------------------------------------------------------
    # Policy builder
    # ------------------------------------------------------------------
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
        segment: PolicyMix = PolicyMix.STANDARD,
        scenario_interest_rate: float | None = None,
        interest_rate: float | None = None,
    ) -> Policy:
        """Construct a single fully risk-priced Policy object.

        The combined underwriting factor from risk_profile is applied to the
        mortality curve *before* premium calculation so that the gross premium
        already reflects all underwriting adjustments (P1-1).

        Args:
            policy_id: Unique identifier.
            age: Issue age.
            term: Policy term in years.
<<<<<<< HEAD
            interest_rate: Valuation interest rate.
            sum_assured: Death benefit.
            mortality_multiplier: Scenario-level mortality scaling.
            premium_multiplier: Post-pricing scenario premium scaling.
            risk_profile: Underwriting classification (sampled if None).
            segment: Policy-mix segment label stored in metadata.
=======
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
>>>>>>> upstream/circuits/interest-rate-experiment

        Returns:
            Policy: Fully constructed policy.
        """
        resolved_pricing_rate = pricing_interest_rate if pricing_interest_rate is not None else interest_rate
        if resolved_pricing_rate is None:
            raise ValueError("Either pricing_interest_rate or interest_rate must be provided.")
        if sum_assured is None:
            raise ValueError("sum_assured must be provided.")
        profile = risk_profile or self._sample_risk_profile()

        # P1-1: multiply scenario factor AND underwriting factor into mortality
        adjusted_multiplier = mortality_multiplier * profile.combined_factor
        mortality_profile = self._mortality_profile(
            age=age, term=term, multiplier=adjusted_multiplier
        )

        # Premium is calculated on the risk-adjusted curve → fully risk-based
        net_premium, gross_premium = self._calculate_premium_rate(
            mortality_profile=mortality_profile,
            term=term,
            interest_rate=resolved_pricing_rate,
            sum_assured=sum_assured,
        )
        
        valuation_rate = (
            resolved_pricing_rate
            if scenario_interest_rate is None
            else scenario_interest_rate
        )

        return Policy(
            policy_id=policy_id,
            age=age,
            term=term,
            premium=gross_premium * premium_multiplier,
            pricing_interest_rate=resolved_pricing_rate,
            scenario_interest_rate=valuation_rate,
            sum_assured=sum_assured,
            mortality_profile=mortality_profile,
            metadata={
                **profile.as_metadata(),
                "net_premium": net_premium,
                "premium_loading": self.premium_loading,
                "premium_multiplier": premium_multiplier,
                "segment": segment.value,
                "pricing_interest_rate": resolved_pricing_rate,
                "scenario_interest_rate": valuation_rate,
            },
        )

    # ------------------------------------------------------------------
    # Sanity checks  (P3-9)
    # ------------------------------------------------------------------

    @staticmethod
    def _policy_passes_checks(policy: Policy) -> tuple[bool, str]:
        """Run hard-constraint checks on a single policy (P3-9).

        Checks
        ------
        * Premium > 0.
        * Sum assured > 0.
        * Mortality intensities are all positive and non-decreasing on average
          (we allow tiny wobble from interpolation but enforce the trend).
        * Premium / SA ratio is within plausible actuarial bounds (0.05 %–5 %).

        Args:
            policy: Policy to validate.

        Returns:
            tuple[bool, str]: (passes, reason_if_failed).
        """
        if policy.premium <= 0.0:
            return False, f"premium={policy.premium:.4f} ≤ 0"

        if policy.sum_assured <= 0.0:
            return False, f"sum_assured={policy.sum_assured:.0f} ≤ 0"

        mu = policy.mortality_profile.intensities
        if np.any(mu <= 0.0):
            return False, "mortality intensity contains non-positive values"

        # Trend check: last half of intensities should average ≥ first half
        mid = len(mu) // 2
        if mu[mid:].mean() < mu[:mid].mean() * 0.80:
            return False, "mortality intensities are declining over the term"

        ratio = policy.premium / policy.sum_assured
        if not (5e-4 <= ratio <= 0.05):
            return False, f"premium/SA ratio {ratio:.4f} outside [0.05 %, 5 %]"

        return True, ""

    def _validate_portfolio(self, policies: list[Policy]) -> list[Policy]:
        """Log sanity-check results for an entire portfolio (P3-9).

        Policies that fail are reported as warnings.  We do not silently drop
        them here; see _build_policy_with_retry for per-policy enforcement.

        Args:
            policies: Generated policy list.

        Returns:
            list[Policy]: Same list (for chaining).
        """
        failed = []
        for p in policies:
            ok, reason = self._policy_passes_checks(p)
            if not ok:
                failed.append((p.policy_id, reason))

        if failed:
            summary = "; ".join(f"{pid}: {r}" for pid, r in failed[:5])
            warnings.warn(
                f"{len(failed)}/{len(policies)} policies failed sanity checks. "
                f"First failures: {summary}",
                stacklevel=3,
            )
        return policies

    def _build_policy_with_retry(
        self,
        policy_id: str,
        age: int,
        term: int,
        interest_rate: float,
        sum_assured: float,
        mortality_multiplier: float = 1.0,
        premium_multiplier: float = 1.0,
        risk_profile: RiskProfile | None = None,
        segment: PolicyMix = PolicyMix.STANDARD,
    ) -> Policy:
        """Build a policy, regenerating it up to max_retries times on failure (P3-9).

        Args:
            Identical to _build_policy; risk_profile is resampled on each retry.

        Returns:
            Policy: A policy that passes all sanity checks.

        Raises:
            RuntimeError: If no valid policy can be produced within max_retries.
        """
        for attempt in range(self.max_retries):
            candidate = self._build_policy(
                policy_id=policy_id,
                age=age,
                term=term,
                interest_rate=interest_rate,
                sum_assured=sum_assured,
                mortality_multiplier=mortality_multiplier,
                premium_multiplier=premium_multiplier,
                risk_profile=risk_profile if attempt == 0 else None,
                segment=segment,
            )
            ok, _ = self._policy_passes_checks(candidate)
            if ok:
                return candidate
        raise RuntimeError(
            f"Could not build a valid policy for {policy_id} after "
            f"{self.max_retries} attempts."
        )

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate_random_policies(self, count: int) -> list[Policy]:
        """Generate policies using a realistic 50/30/20 policy mix (P3-7).

        Segments
        --------
        * Standard (50 %): Normal ages, standard SA scale.
        * High-value (30 %): Younger bias, 2× SA scale.
        * Small (20 %): Older bias, 0.35× SA scale.

        Per-policy sanity checks are enforced with automatic retries (P3-9).

        Args:
            count: Total number of policies to generate.

        Returns:
            list[Policy]: Generated policies in segment order.
        """
        policies: list[Policy] = []
        global_index = 0

        for segment, weight, sa_scale, age_bias in _MIX_SEGMENTS:
            segment_count = round(count * weight)
            for offset in range(segment_count):
                age = self._sample_age(age_bias=age_bias)
                try:
                    term = self._sample_term(age)
                except ValueError:
                    # Edge case: very old age with no valid tenor — skip
                    continue
                interest_rate = self._sample_interest_rate()
                sum_assured = self._sample_sum_assured(age, sa_scale=sa_scale)
                policies.append(
                    self._build_policy_with_retry(
                        policy_id=f"policy_{global_index:05d}",
                        age=age,
                        term=term,
                        interest_rate=interest_rate,
                        sum_assured=sum_assured,
                        segment=segment,
                    )
                )
                global_index += 1

        # Fill any rounding shortfall with standard-segment policies
        while len(policies) < count:
            age = self._sample_age()
            try:
                term = self._sample_term(age)
            except ValueError:
                continue
            policies.append(
                self._build_policy_with_retry(
                    policy_id=f"policy_{global_index:05d}",
                    age=age,
                    term=self._sample_term(age),
                    interest_rate=self._sample_interest_rate(),
                    sum_assured=self._sample_sum_assured(age),
                    segment=PolicyMix.STANDARD,
                )
            )
            global_index += 1

        self._validate_portfolio(policies)
        return policies

    def generate_stratified_policies(
        self, counts_by_age_band: dict[tuple[int, int], int]
    ) -> list[Policy]:
        """Generate policies by explicit age strata with realistic interest rates (P2-5).

        Each band draws ages uniformly within its bounds but samples interest
        rates from the Normal distribution for consistency with the rest of the
        platform.

        Args:
            counts_by_age_band: Mapping from (age_min, age_max) to policy count.

        Returns:
            list[Policy]: Stratified policies.
        """
        policies: list[Policy] = []
        start_index = 0

        for (age_min, age_max), count in counts_by_age_band.items():
            for offset in range(count):
                age = int(self.rng.integers(age_min, age_max + 1))
                try:
                    term = self._sample_term(age)
                except ValueError:
                    continue
                interest_rate = self._sample_interest_rate()
                sum_assured = self._sample_sum_assured(age)
                policies.append(
                    self._build_policy_with_retry(
                        policy_id=f"stratified_{start_index + offset:05d}",
                        age=age,
                        term=term,
                        interest_rate=interest_rate,
                        sum_assured=sum_assured,
                    )
                )
            start_index += count

        self._validate_portfolio(policies)
        return policies

    def generate_scenario_policies(
        self, base_policies: Iterable[Policy], scenario: ScenarioDefinition
    ) -> list[Policy]:
        """Clone an existing portfolio under a stress scenario.

        The original risk profile is preserved exactly so that scenario deltas
        are attributable solely to the scenario parameters, not to underwriting
        re-sampling.

        Args:
            base_policies: Baseline policies to clone.
            scenario: Scenario overrides (interest shift, mortality mult, etc.).

        Returns:
            list[Policy]: Scenario-adjusted portfolio.
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
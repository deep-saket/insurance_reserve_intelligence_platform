"""Policy simulator tests.

Created: 2026-05-31
Purpose: Validate policy simulation behavior used by the research pipeline.
"""

from __future__ import annotations

import pytest

from src.data.simulator import PolicySimulator, RiskProfile, ScenarioDefinition


def _standard_profile(gender: str = "male") -> RiskProfile:
    """Return a neutral underwriting profile for relationship tests."""

    gender_factor = 1.0 if gender == "male" else 0.85
    return RiskProfile(
        "non_smoker",
        "standard",
        "moderate",
        gender,
        1.0,
        1.0,
        1.0,
        gender_factor,
    )


def test_policy_simulator_generates_requested_count() -> None:
    """Verify that the simulator returns the requested number of valid policies."""
    simulator = PolicySimulator(
        age_range=(25, 60),
        term_range=(5, 20),
        interest_rate_range=(0.01, 0.05),
        sum_assured_range=(100000.0, 500000.0),
        seed=7,
    )
    policies = simulator.generate_random_policies(5)
    assert len(policies) == 5
    assert all(policy.term >= 5 for policy in policies)
    assert all(policy.age + policy.term <= 80 for policy in policies)
    assert all("risk_adjustment_factor" in policy.metadata for policy in policies)
    assert all("gender" in policy.metadata for policy in policies)


def test_risk_profile_adjusts_mortality_and_actuarial_premium() -> None:
    """Verify that underwriting factors feed mortality and actuarial pricing."""
    simulator = PolicySimulator(
        age_range=(40, 40),
        term_range=(10, 10),
        interest_rate_range=(0.03, 0.03),
        sum_assured_range=(100000.0, 100000.0),
        mortality_scale=0.001,
        mortality_shape=1.0,
        premium_loading=1.0,
        seed=1,
    )
    risk_profile = RiskProfile(
        smoker_status="smoker",
        health_tier="substandard",
        occupation_risk="high",
        gender="male",
        smoker_factor=1.30,
        health_factor=1.25,
        occupation_factor=1.20,
        gender_factor=1.0,
    )

    high_risk_policy = simulator._build_policy(
        policy_id="high_risk",
        age=40,
        term=10,
        interest_rate=0.03,
        sum_assured=100000.0,
        risk_profile=risk_profile,
    )
    standard_policy = simulator._build_policy(
        policy_id="standard",
        age=40,
        term=10,
        interest_rate=0.03,
        sum_assured=100000.0,
        risk_profile=_standard_profile(),
    )

    assert risk_profile.combined_factor == pytest.approx(1.95)
    assert high_risk_policy.mortality_profile.intensity_at(0.0) == pytest.approx(0.00195)
    assert high_risk_policy.premium > standard_policy.premium


def test_issue_age_drives_fallback_mortality() -> None:
    """Verify that fallback mortality is based on attained age, not duration alone."""
    simulator = PolicySimulator(
        age_range=(30, 60),
        term_range=(10, 10),
        interest_rate_range=(0.03, 0.03),
        sum_assured_range=(100000.0, 100000.0),
        mortality_scale=0.001,
        mortality_shape=1.08,
        mortality_reference_age=25,
        seed=1,
    )
    standard = _standard_profile()

    younger = simulator._build_policy("younger", 30, 10, 0.03, 100000.0, risk_profile=standard)
    older = simulator._build_policy("older", 60, 10, 0.03, 100000.0, risk_profile=standard)

    assert older.mortality_profile.intensity_at(0.0) > younger.mortality_profile.intensity_at(0.0)
    assert older.premium > younger.premium


def test_sum_assured_and_term_drive_premium() -> None:
    """Verify core coverage characteristics are reflected in actuarial pricing."""
    simulator = PolicySimulator(
        age_range=(40, 40),
        term_range=(10, 20),
        interest_rate_range=(0.03, 0.03),
        sum_assured_range=(100000.0, 200000.0),
        mortality_scale=0.001,
        mortality_shape=1.08,
        premium_loading=1.0,
        seed=1,
    )
    standard = _standard_profile()

    base = simulator._build_policy("base", 40, 10, 0.03, 100000.0, risk_profile=standard)
    higher_cover = simulator._build_policy(
        "higher_cover", 40, 10, 0.03, 200000.0, risk_profile=standard
    )
    longer_term = simulator._build_policy(
        "longer_term", 40, 20, 0.03, 100000.0, risk_profile=standard
    )

    assert higher_cover.premium == pytest.approx(base.premium * 2.0)
    assert longer_term.premium > base.premium


def test_gender_drives_mortality_and_premium() -> None:
    """Verify that gender classification affects mortality and pricing."""
    simulator = PolicySimulator(
        age_range=(40, 40),
        term_range=(10, 10),
        interest_rate_range=(0.03, 0.03),
        sum_assured_range=(100000.0, 100000.0),
        mortality_scale=0.001,
        mortality_shape=1.0,
        premium_loading=1.0,
        seed=1,
    )

    male = simulator._build_policy("male", 40, 10, 0.03, 100000.0, risk_profile=_standard_profile())
    female = simulator._build_policy(
        "female", 40, 10, 0.03, 100000.0, risk_profile=_standard_profile("female")
    )

    assert female.mortality_profile.intensity_at(0.0) == pytest.approx(0.00085)
    assert female.premium < male.premium


def test_smokers_are_more_likely_to_receive_substandard_health_tier() -> None:
    """Verify that health-tier sampling is correlated with smoker status."""
    simulator = PolicySimulator(
        age_range=(40, 40),
        term_range=(10, 10),
        interest_rate_range=(0.03, 0.03),
        sum_assured_range=(100000.0, 100000.0),
        seed=9,
    )
    profiles = [simulator._sample_risk_profile() for _ in range(10000)]
    smokers = [profile for profile in profiles if profile.smoker_status == "smoker"]
    non_smokers = [profile for profile in profiles if profile.smoker_status == "non_smoker"]

    smoker_substandard_rate = (
        sum(profile.health_tier == "substandard" for profile in smokers) / len(smokers)
    )
    non_smoker_substandard_rate = (
        sum(profile.health_tier == "substandard" for profile in non_smokers) / len(non_smokers)
    )
    assert smoker_substandard_rate > non_smoker_substandard_rate


def test_term_and_sum_assured_sampling_depend_on_issue_age() -> None:
    """Verify that older applicants receive shorter terms and lower cover limits."""
    simulator = PolicySimulator(
        age_range=(25, 70),
        term_range=(5, 30),
        interest_rate_range=(0.03, 0.03),
        sum_assured_range=(50000.0, 1000000.0),
        max_expiry_age=80,
        sum_assured_rounding=50000.0,
        sum_assured_age_decay=0.02,
        seed=3,
    )

    assert max(simulator._sample_term(70) for _ in range(100)) <= 10
    assert simulator._sum_assured_upper_bound(65) < simulator._sum_assured_upper_bound(25)
    assert all(simulator._sample_sum_assured(65) % 50000.0 == 0.0 for _ in range(100))


def test_scenario_cloning_does_not_double_apply_underwriting_factor() -> None:
    """Verify that a scenario shock is layered on the existing risk profile once."""
    simulator = PolicySimulator(
        age_range=(40, 40),
        term_range=(10, 10),
        interest_rate_range=(0.03, 0.03),
        sum_assured_range=(100000.0, 100000.0),
        mortality_scale=0.001,
        mortality_shape=1.0,
        premium_loading=1.0,
        seed=1,
    )
    risk_profile = RiskProfile(
        smoker_status="smoker",
        health_tier="standard",
        occupation_risk="moderate",
        gender="male",
        smoker_factor=1.30,
        health_factor=1.0,
        occupation_factor=1.0,
        gender_factor=1.0,
    )
    policy = simulator._build_policy(
        policy_id="smoker",
        age=40,
        term=10,
        interest_rate=0.03,
        sum_assured=100000.0,
        risk_profile=risk_profile,
    )

    scenario_policy = simulator.generate_scenario_policies(
        [policy],
        ScenarioDefinition(mortality_multiplier=1.10, premium_multiplier=1.05),
    )[0]

    assert scenario_policy.mortality_profile.intensity_at(0.0) == pytest.approx(0.00143)
    assert scenario_policy.premium == pytest.approx(policy.premium * 1.05)

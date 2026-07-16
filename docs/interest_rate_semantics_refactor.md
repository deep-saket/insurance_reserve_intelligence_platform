"""Interest-rate semantics refactor summary.

Created: 2026-07-02
Purpose: Explain the pricing-rate vs scenario-rate refactor, the reason for the
change, and its expected impact on training, evaluation, and reserve-vs-rate
behavior.
"""

# Interest-Rate Semantics Refactor

## Executive Summary

The project previously used one field, `interest_rate`, for two different business meanings:

1. pricing-time discount rate used to compute the contract premium at issue
2. valuation-time or scenario-time interest rate used later in Thiele reserve calculations

That overload created a mathematical mismatch. During data generation, the premium was produced using the sampled interest rate. During stress testing, evaluation, and interest-rate curve analysis, the same contract was then shocked to a different rate while the premium stayed fixed. That is valid as an in-force stress scenario, but the code and feature semantics did not represent it explicitly.

The refactor separates those meanings into:

- `pricing_interest_rate`
- `scenario_interest_rate`

This makes the training data, reserve equation, stress workflows, and debugging plots consistent with the business story.

## Why The Old Setup Was Problematic

### Mathematical issue

For a term-life contract, the reserve equation is:

```text
dV/dt = rV + P - μ(S - V)
```

where:

- `r` is the valuation interest rate used inside the reserve dynamics
- `P` is the premium already embedded in the in-force contract

If a policy is priced at rate `r_price`, then `P = P(r_price)`. If we later perform a scenario stress at `r_scn`, the correct in-force scenario is:

```text
premium stays fixed at P(r_price)
reserve dynamics use r = r_scn
```

The old implementation effectively treated the same single variable as both `r_price` and `r_scn` in different parts of the stack. That made the model input semantics ambiguous and made debugging reserve-vs-interest behavior harder than it should have been.

### Business issue

In layman terms:

- the premium is the amount the customer is already paying
- the scenario rate is the market environment we are testing today

Those are not the same concept. Mixing them makes the digital twin less trustworthy because the scenario engine is not clearly separated from the original pricing assumptions.

## What Changed

### Contract model

`/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/actuarial/policy.py`

- Added `pricing_interest_rate`
- Added `scenario_interest_rate`
- Kept a compatibility alias `interest_rate -> scenario_interest_rate`

### Classical reserve solver

`/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/actuarial/actuarial_solver.py`

- Thiele dynamics now use `policy.scenario_interest_rate`

This is the correct reserve-valuation interpretation.

### Simulator

`/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/simulator.py`

- Premium generation still uses `pricing_interest_rate`
- Scenario generation now keeps premium fixed and only shocks `scenario_interest_rate`

This is the core business correction. We now represent an in-force policy stress instead of silently repricing the contract when only a reserve scenario was intended.

### Dataset and model features

`/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/dataset.py`

The model feature vector changed from 6 inputs to 7 inputs:

```text
[time, age, pricing_interest_rate, scenario_interest_rate, premium, sum_assured, mortality]
```

Why this matters:

- `pricing_interest_rate` explains where the premium came from
- `scenario_interest_rate` explains the reserve environment being evaluated

That extra separation gives the network a cleaner causal view of the problem.

### Losses and sensitivities

Updated modules:

- `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/pde_loss.py`
- `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/interest_rate_monotonicity_loss.py`
- `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/interest_rate_scenario_loss.py`
- `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/interest_rate_peak_loss.py`
- `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/evaluators/evaluator.py`

All interest-rate derivatives and scenario targets now explicitly refer to `scenario_interest_rate`.

That is the correct variable for:

- `dV/dr`
- rate stress curves
- peak reserve under rate shocks
- PDE residual calculations

## Secondary Fixes Required By The Refactor

The interest-rate split exposed another issue: some downstream engines were still assuming old 6-feature inputs or were operating directly in normalized model space without properly reconstructing monetary reserves.

Updated modules:

- `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/stress/stress_tester.py`
- `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/optimization/optimizer_engine.py`
- `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/digital_twin/engine.py`

Fixes made:

- all three now build features through the shared dataset helper
- all three now use the 7-feature schema
- reserve outputs are denormalized back into currency units before reporting

Without this, scenario comparisons can look wrong even if the model itself is fine.

## Expected Impact

### What should improve

- clearer reserve-vs-interest interpretation
- better supervision for interest-rate sensitivity losses
- cleaner separation of pricing assumptions and scenario assumptions
- more defensible stress testing and optimization behavior

### What may not fully disappear

This refactor fixes semantics. It does not automatically guarantee perfect reserve-vs-rate shape learning. Remaining error can still come from:

- insufficient rate-shock supervision weight
- limited model capacity
- poor balance between data loss and rate-specific losses
- training instability from large loss weights
- undercoverage of difficult policy regions

## How To Explain This To The Team

Use this wording:

> We separated the issue-rate used to price premium from the scenario-rate used to value reserves. Previously the project used one interest-rate field for both meanings, which made rate-stress learning ambiguous. The refactor makes the contract economics explicit, keeps premium fixed during in-force stress scenarios, and aligns the solver, dataset, losses, and downstream analytics with the same business semantics.

## Practical Outcome

After this refactor, a reserve-vs-interest experiment should be interpreted as:

- same policy
- same premium
- same mortality profile unless explicitly shocked
- only valuation rate changes

That is the correct actuarial stress experiment for the question:

“How does the reserve move when the interest-rate environment changes for an already-issued contract?”

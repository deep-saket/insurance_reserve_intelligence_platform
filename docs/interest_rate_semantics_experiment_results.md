"""Interest-rate semantics refactor experiment results.

Created: 2026-07-02
Purpose: Record the first train/evaluate/validate outcome after separating
pricing-time and scenario-time interest-rate semantics.
"""

# Interest-Rate Semantics Refactor: Experiment Results

## Run Information

- Run name: `run_15_rate_semantics_split`
- Artifact root: `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_15_rate_semantics_split`
- Best validation epoch: `6`
- Best validation total loss: `0.155345`
- Training status: completed without NaN, early stopped at epoch `16`

## What Improved

### 1. Training stability improved

The run trained cleanly through 16 epochs and did not reproduce the earlier “loss becomes NaN after a few epochs” failure mode.

Business meaning:

- the refactor did not destabilize the pipeline
- the new semantics are at least numerically trainable
- we can now debug model quality instead of fighting silent semantic confusion and exploding loss

### 2. PDE consistency is strong

Validation reported:

- mean PDE residual: `£5.4/yr`
- maximum residual: `£826.8/yr`

Interpretation:

The model is learning a surface that is locally compatible with the Thiele equation much better than before. In other words, the network is behaving like a plausible reserve manifold in differential form.

## What Did Not Improve Enough

### 1. Overall predictive quality is still not production-ready

Evaluation:

- MSE: `304324.81`
- MAE: `254.59`
- RMSE: `551.66`
- R²: `0.95594`

Validation:

- mean relative error: `17.52%`
- boundary mean `|V(T)|`: `£712.9`
- monotonicity sign correctness: `44.5%`
- OOD mean relative error: `56.76%`

Interpretation:

The model is learning something meaningful, but it is still not accurate or robust enough for reserve analytics that need to be trusted policy by policy.

### 2. The reserve-vs-interest curve issue still remains

Direct fixed-policy rate-curve comparison results:

- overall mean percentage error across sampled policies and rates: `121.60%`

Per-policy summary:

| Policy | Mean % Error | Classical Peak-Reserve Range | PINN Peak-Reserve Range |
|---|---:|---:|---:|
| `policy_00013` | `33.33%` | `34.27` | `213.40` |
| `policy_00043` | `145.90%` | `24.32` | `601.46` |
| `policy_00059` | `185.56%` | `13.09` | `453.41` |

Interpretation:

The PINN is still dramatically overreacting to interest-rate changes. The classical solver shows relatively small peak-reserve movement across the tested rate grid, while the neural model produces much larger swings. So the semantic split fixed an important correctness issue, but it did not by itself solve the shape-learning problem.

Business meaning:

- the current digital twin should not yet be trusted for interest-rate scenario analytics
- scenario-direction intuition may look right in some local cases, but magnitude control is still poor
- the model can satisfy the PDE while still learning the wrong global scenario geometry

## Why This Happened

The refactor addressed:

- feature semantics
- solver semantics
- stress/optimization/digital-twin consistency

It did **not** automatically guarantee:

- strong global reserve-vs-rate supervision
- correct monotonicity across the full domain
- good out-of-distribution behavior
- realistic peak-reserve sensitivity magnitude

That remaining gap is consistent with the current numbers:

- PDE residual is excellent
- but monotonicity and direct reserve-fit are weak

This means the model is learning a smooth differential surface that is locally lawful, but still globally miscalibrated for rate-scenario behavior.

## Recommended Next Actions

### Priority 1: strengthen direct rate-curve supervision

The semantic fix should now be paired with stronger supervision for:

- `interest_rate_peak_loss`
- `interest_rate_scenario_loss`

Likely actions:

- turn on `interest_rate_scenario_loss`
- increase coverage of fixed-policy shocked-rate targets in the dataset
- consider weighting rate-curve supervision earlier in training instead of only relying on later curriculum exposure

### Priority 2: improve hard boundary behavior

Boundary error is still too high. That suggests the model is not respecting the maturity reserve condition strongly enough in practice.

Likely actions:

- revisit `boundary_loss` weight
- inspect whether boundary examples are sufficiently represented and stable under normalization

### Priority 3: improve monotonicity and OOD robustness

Likely actions:

- re-enable or rebalance interest-rate monotonicity supervision
- broaden scenario coverage in synthetic data
- test whether model capacity and depth are causing oversensitivity

## Key Artifact Paths

- Reserve trajectory comparison:
  `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_15_rate_semantics_split/reports/reserve_trajectory_comparison.png`
- Multi-policy reserve comparison:
  `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_15_rate_semantics_split/reports/pinn_vs_classical_multi_policy.png`
- Interest-rate curve comparison:
  `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_15_rate_semantics_split/reports/interest_rate_curve_comparison.png`
- Interest-rate curve detailed data:
  `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_15_rate_semantics_split/reports/interest_rate_curve_comparison.csv`
- Interest-rate curve summary:
  `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_15_rate_semantics_split/reports/interest_rate_curve_summary.csv`
- Validation plots:
  `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_15_rate_semantics_split/reports/validation_plots.png`
- Validation text report:
  `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_15_rate_semantics_split/reports/validation_report.txt`

## Bottom Line

The pricing-rate vs scenario-rate split was the correct change and it removed a genuine semantic defect. It also made the run numerically stable.

But the result is:

- **semantic correctness improved**
- **training stability improved**
- **interest-rate curve fidelity is still poor**

So the next phase should focus on **stronger direct scenario-shape supervision**, not on reverting this refactor.

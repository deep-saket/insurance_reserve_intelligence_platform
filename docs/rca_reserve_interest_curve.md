# Reserve vs Interest Curve RCA

Created: 2026-07-02  
Purpose: Record the root-cause analysis for the incorrect reserve-versus-interest-rate plot, the related training defects discovered during investigation, and the current status after local fixes and reruns.

## Executive Summary

The original reserve-vs-interest plot was wrong for two separate reasons:

1. The **EDA plot itself was not an interest sensitivity plot**. It compared different policies with different ages, terms, premiums, mortality curves, and sum assured values, then regressed those cross-sectional outcomes against interest rate. That is not the same thing as holding a contract fixed and varying only `r`.
2. The **training stack had multiple issues** that made the learned model less trustworthy than the physics residual alone suggested:
   - constraint weights were being zeroed out by the curriculum schedule
   - the PDE residual in `z`-space was mathematically incorrect
   - the boundary loss used the wrong mortality input at maturity
   - the sum-assured monotonicity loss enforced monotonicity on `z`, not on reserve `V`

After fixing the plot logic, the new interest-rate plot now shows the expected Thiele-style negative relationship for fixed contracts.  
After fixing the trainer/math bugs, training behavior is more honest, but model quality is still not production-ready: physics consistency is strong, while accuracy, monotonicity, and OOD generalisation remain weak.

## Investigation Scope

The investigation covered:

- classical EDA plots under `artifacts/eda/`
- trained-model comparison plots under `artifacts/run_08/reports/` and `artifacts/run_09_tuned/reports/`
- training logs and per-loss CSV metrics
- the current merged code in:
  - [generate_and_eda.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/generate_and_eda.py)
  - [src/trainers/trainer.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/trainers/trainer.py)
  - [src/losses/pde_loss.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/pde_loss.py)
  - [src/losses/boundary_loss.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/boundary_loss.py)
  - [src/losses/sum_assured_monotonicity_loss.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/sum_assured_monotonicity_loss.py)
  - [src/data/dataset.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/dataset.py)
  - [configs/config.yaml](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/configs/config.yaml)

## Symptom 1: Reserve vs Interest Rate Plot Looked Wrong

### Observation

The original `17_reserve_vs_interest_rate.png` did not show a clear decreasing reserve profile as interest rates rose. That contradicted the expected present-value effect from Thiele-style reserve mechanics.

### Root Cause

The previous plotting logic was cross-sectional:

- each point was a different policy
- each policy had a different:
  - age
  - term
  - mortality profile
  - sum assured
  - premium
- premium itself was generated using the same issue-rate assumption, so interest rate and premium moved together

That means the plot was confounded before the model was even involved.

### Fix

The plot was rewritten to hold the contract fixed and vary only interest rate.

Reference: [generate_and_eda.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/generate_and_eda.py:409)

```python
# generate_and_eda.py:409-497
def plot_reserve_vs_interest_rate(
    policies: list,
    solver: ThieleSolver,
    rate_min: float,
    rate_max: float,
    time_steps: int,
) -> None:
    """Controlled reserve sensitivity to interest rate."""

    representative_policies = [
        min(policies, key=lambda p: p.age),
        max(policies, key=lambda p: p.age),
        min(policies, key=lambda p: p.sum_assured),
        max(policies, key=lambda p: p.sum_assured),
    ]
    interest_grid = np.linspace(rate_min, rate_max, 8)

    for policy in representative_policies:
        peak_reserves = []
        for rate in interest_grid:
            shocked_policy = replace(policy, interest_rate=float(rate))
            trajectory = solver.solve(shocked_policy, num_steps=time_steps)
            peak_reserves.append(float(np.max(trajectory.reserves)))
```

### Evidence

Updated plot:

- [artifacts/eda/17_reserve_vs_interest_rate.png](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/eda/17_reserve_vs_interest_rate.png)

Result:

- fixed-contract reserve curves now slope downward as `r` increases
- the normalised peak reserve curve across a sample of fixed policies also slopes downward

### Conclusion

The original chart was not showing a Thiele sensitivity failure. It was showing a **plot design error**.

## Symptom 2: Enabled Constraint Losses Were Not Actually Active

### Observation

In the first local rerun after merge, validation monotonicity remained poor even though the monotonicity losses were marked `enabled: true` in YAML.

Inspection of the training metrics showed that by later epochs the weighted constraint losses were all effectively `0.0`, while only data loss remained active.

### Root Cause

The trainer mutated `term.weight` in place during curriculum warmup and then reused that mutated value as the future base weight. Because epoch 0 starts with 0% warmup, the stored config weight became zero and never recovered.

### Fix

Store immutable base weights at trainer initialization and compute effective weights from those frozen originals.

Reference: [src/trainers/trainer.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/trainers/trainer.py:141)

```python
# src/trainers/trainer.py:141-144
self.base_loss_weights = {
    name: float(term.weight or 0.0)
    for name, term in self.config.losses.terms.items()
}
```

And use those base weights in the curriculum function:

Reference: [src/trainers/trainer.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/trainers/trainer.py:391)

```python
# src/trainers/trainer.py:391-409
def _curriculum_weight(self, epoch: int, name: str) -> float:
    cfg_weight = self.base_loss_weights[name]
    if name not in self._CONSTRAINT_LOSSES:
        return cfg_weight
    warmup = self._warmup_epochs()
    effective_epoch = max(0, epoch - self.start_epoch)
    ramp = min(1.0, effective_epoch / warmup)
    return cfg_weight * ramp
```

### Related Bug

The logged warmup percentage and the actual warmup schedule were inconsistent. The code now uses one shared helper:

Reference: [src/trainers/trainer.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/trainers/trainer.py:386)

```python
# src/trainers/trainer.py:386-389
def _warmup_epochs(self) -> int:
    """Return the number of epochs used for constraint warmup."""
    return max(30, int(self.config.trainer.epochs * 0.20))
```

### Evidence

Before the fix, a later row in `artifacts/pr3_review_run/logs/training_metrics.csv` showed:

- `weighted_pde_loss = 0.0`
- `weighted_boundary_loss = 0.0`
- `weighted_interest_rate_monotonicity_loss = 0.0`
- `weighted_mortality_monotonicity_loss = 0.0`

That was impossible under the configured YAML unless the curriculum logic had effectively disabled them.

## Symptom 3: PDE Residual Was Wrong in z-Space

### Observation

The merged code trains on standardised `z = (v - μ) / σ`, where `v = V / S`. The PDE loss was ported into this space, but the transformation was incomplete.

### Root Cause

The death-benefit term in the transformed Thiele equation was missing a `1 / σ` scaling factor.

Correct derivation:

- `v = V / S`
- `z = (v - μ_target) / σ_target`
- `v = z * σ_target + μ_target`
- `dv/dt = r v + P/S - μ_mort (1 - v)`
- therefore  
  `dz/dt = r(z + μ_target/σ_target) + P/(Sσ_target) - μ_mort(((1 - μ_target)/σ_target) - z)`

### Fix

Reference: [src/losses/pde_loss.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/pde_loss.py:17)

```python
# src/losses/pde_loss.py:17-22
t_mean  = batch["target_mean"].to(predictions.device)
t_std   = batch["target_std"].to(predictions.device)
z_offset = predictions + t_mean / t_std
survival_gap = ((1.0 - t_mean) / t_std) - predictions
residual = dz_dt - r * z_offset - P / (S * t_std) + mu_mort * survival_gap
```

### Conclusion

Before this fix, the physics loss looked numerically small but was not enforcing the correct transformed Thiele equation.

## Symptom 4: Boundary Loss Used an Off-Manifold Terminal State

### Observation

At maturity `T`, the boundary loss changed only the time input to `T`, but it left the mortality feature at its original time-point value.

### Root Cause

For term insurance, the terminal state should be evaluated at:

- `t = T`
- `μ = μ(T)`

The previous implementation enforced:

- `t = T`
- `μ = μ(t_current)`

That is not the same state.

### Fix

The dataset now stores terminal mortality explicitly:

Reference: [src/data/dataset.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/dataset.py:63)

```python
# src/data/dataset.py:63-82
terminal_mortality = policy.mortality_profile.intensity_at(float(policy.term))
...
records.append(ReserveRecord(
    policy_id=policy.policy_id,
    features=features,
    reserve=float(reserve),
    term=float(policy.term),
    terminal_mortality=float(terminal_mortality),
))
```

And it is passed through the batch:

Reference: [src/data/dataset.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/dataset.py:104)

```python
# src/data/dataset.py:104-112
return {
    "features": features,
    ...
    "term": torch.tensor([record.term], dtype=torch.float32),
    "terminal_mortality": torch.tensor([record.terminal_mortality], dtype=torch.float32),
}
```

Then boundary loss uses both terminal time and terminal mortality:

Reference: [src/losses/boundary_loss.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/boundary_loss.py:13)

```python
# src/losses/boundary_loss.py:13-20
terminal_mortality = self.require_batch_tensor(batch, "terminal_mortality")
bf = features.clone()
bf[:, FEATURE_INDEX["time"]:FEATURE_INDEX["time"]+1] = terms / FEATURE_SCALES["time"]
bf[:, FEATURE_INDEX["mortality"]:FEATURE_INDEX["mortality"]+1] = (
    terminal_mortality / FEATURE_SCALES["mortality"]
)
```

## Symptom 5: Sum-Assured Monotonicity Was Enforced on the Wrong Quantity

### Observation

The model predicts `z`, not `V`. The original sum-assured monotonicity loss compared `pred_high - predictions` directly in `z`-space.

### Root Cause

For `r` and mortality, the sign in `z`-space is aligned with `V` because the output scaling is constant for the sample.  
For `S`, that is not true:

- `V = (zσ + μ) * S`

So `dV/dS > 0` is not equivalent to `dz/dS > 0`.

### Fix

Reconstruct reserve `V` for both the base and perturbed sum assured values, then compare those reserves directly.

Reference: [src/losses/sum_assured_monotonicity_loss.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/sum_assured_monotonicity_loss.py:40)

```python
# src/losses/sum_assured_monotonicity_loss.py:40-55
raw_sum_assured = batch["raw_features"][:, _S_IDX : _S_IDX + 1]
target_mean = batch["target_mean"].to(predictions.device)
target_std = batch["target_std"].to(predictions.device)
...
reserve_base = (predictions * target_std + target_mean) * raw_sum_assured
reserve_high = (pred_high * target_std + target_mean) * sum_assured_high
violation = reserve_base - reserve_high
```

## Tuned Configuration Used After Fixes

Reference: [configs/config.yaml](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/configs/config.yaml:52)

```yaml
losses:
  data_loss:
    enabled: true
    weight: 4.0
  pde_loss:
    enabled: true
    weight: 0.2
  boundary_loss:
    enabled: true
    weight: 0.01
  mortality_monotonicity_loss:
    enabled: true
    weight: 0.005
  interest_rate_monotonicity_loss:
    enabled: true
    weight: 0.02
  solvency_loss:
    enabled: true
    weight: 0.1
  reserve_ceiling_loss:
    enabled: true
    weight: 0.05
  smoothness_loss:
    enabled: true
    weight: 0.01
  sum_assured_monotonicity_loss:
    enabled: true
    weight: 0.0001
```

The stale `resume_from` path was also cleared and the default run name moved to `run_09_tuned`.

## Results After Fixes

### Plot-Level Outcome

The interest-rate EDA plot is now semantically correct:

- [artifacts/eda/17_reserve_vs_interest_rate.png](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/eda/17_reserve_vs_interest_rate.png)

This now shows:

- a downward-sloping reserve response for fixed representative contracts
- a downward-sloping mean normalised peak reserve across a sample of fixed policies

### Training Outcome

Two key reruns were examined:

1. `pr3_review_run`
   - constraint-weight bug still present
   - best validation approximately `0.0419`
   - looked numerically good, but monotonicity remained poor because constraints were effectively off

2. `run_09_tuned`
   - curriculum, PDE, boundary, and sum-assured fixes applied
   - best validation approximately `0.1518` at epoch 8
   - training remained finite and honest under active constraints, but overall predictive quality did not yet improve enough

### Validation Outcome

Current tuned validation report:

- [artifacts/run_09_tuned/reports/validation_report.txt](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_09_tuned/reports/validation_report.txt)
- [artifacts/run_09_tuned/reports/validation_plots.png](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_09_tuned/reports/validation_plots.png)

Summary from the tuned run:

- Accuracy: fail
- Boundary: fail
- Physics: pass
- Monotonicity: fail
- Generalisation: fail

Interpretation:

- physics residual is now strong and credible
- model shape and risk sensitivities are still not good enough
- the system is no longer silently cheating by turning off the constraint losses

## What Is Fixed vs What Is Still Open

### Fixed

- reserve-vs-interest EDA plot semantics
- curriculum weight mutation bug
- curriculum warmup/log mismatch
- PDE residual transform bug in `z`-space
- boundary terminal mortality mismatch
- sum-assured monotonicity objective mismatch

### Still Open

- interest-rate monotonicity remains poor in the learned model
- mortality monotonicity remains poor
- absolute reserve accuracy is still too weak on some cohorts
- OOD generalisation is poor
- the current loss balance still favors physics consistency more than predictive shape quality in difficult regions

## Recommended Next Steps

1. Move all monotonicity checks to reconstructed reserve `V`, not just the sum-assured loss.
2. Revisit the interest-rate and mortality finite-difference step sizes and penalty multipliers.
3. Add per-loss validation gating so best checkpoint selection is not based on total loss alone.
4. Add a dedicated fixed-policy sensitivity validation suite for:
   - `r`
   - mortality
   - premium
   - sum assured
5. Consider a two-stage training schedule:
   - stage 1: data-only / light PDE fit
   - stage 2: controlled constraint fine-tuning

## Follow-Up Experiment: Reserve-Space Interest Loss

On 2026-07-02, the interest-rate monotonicity loss was changed from a latent
`z`-space output comparison to a reserve-space derivative constraint in:

- [src/losses/interest_rate_monotonicity_loss.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/interest_rate_monotonicity_loss.py)

The experiment used:

- `run_name = run_10_interest_rate_reserve_space`
- `interest_rate_monotonicity_loss.weight = 5.0e-5`
- best validation loss observed: approximately `0.2808` at epoch 6

Artifacts:

- [artifacts/run_10_interest_rate_reserve_space/reports/interest_rate_thiele_vs_pinn.png](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_10_interest_rate_reserve_space/reports/interest_rate_thiele_vs_pinn.png)
- [artifacts/run_10_interest_rate_reserve_space/reports/interest_rate_thiele_vs_pinn_summary.csv](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_10_interest_rate_reserve_space/reports/interest_rate_thiele_vs_pinn_summary.csv)
- [artifacts/run_10_interest_rate_reserve_space/reports/interest_rate_sign_check_40_policies_summary.csv](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_10_interest_rate_reserve_space/reports/interest_rate_sign_check_40_policies_summary.csv)

Outcome:

- sign agreement improved directionally on a broader 40-policy check:
  - `run_09_tuned`: `10%`
  - `run_10_interest_rate_reserve_space`: `52.5%`
- but slope magnitude became pathological:
  - median absolute slope gap remained very large at approximately `6416`
  - representative policies showed massively over-steep negative PINN slopes
  - one representative policy still had the wrong sign
- overall reserve fit worsened materially versus `run_09_tuned`

Interpretation:

- moving the constraint into reserve space was the right conceptual direction
- but a one-sided `ReLU(dV/dr)` penalty mainly teaches the model to make the
  slope negative, not to make it actuarially realistic
- the current formulation can therefore improve sign frequency while still
  destroying slope magnitude and level fit

## Follow-Up Experiment: Teacher Sensitivity Loss

On 2026-07-02, the rate-sensitivity experiment was upgraded again:

- [src/data/dataset.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/dataset.py)
  now precomputes local classical targets for
  `d(V/S)/dr` using central differences around each policy's interest rate
- [src/losses/interest_rate_monotonicity_loss.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/interest_rate_monotonicity_loss.py)
  now applies a Huber penalty between PINN autodiff sensitivity and the
  classical teacher sensitivity instead of using a one-sided sign penalty

Two runs were tested:

1. `run_11_interest_rate_teacher_slope`
   - scratch retrain
   - configured interest-rate teacher weight: `50.0`
   - best validation total loss: approximately `0.1826`

2. `run_12_interest_rate_teacher_finetune`
   - warm-start from `run_09_tuned` model weights only
   - lower learning rate fine-tune
   - configured interest-rate teacher weight: `10.0`
   - early-stopped at epoch 10

Artifacts:

- [artifacts/run_11_interest_rate_teacher_slope/reports/interest_rate_thiele_vs_pinn_summary.csv](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_11_interest_rate_teacher_slope/reports/interest_rate_thiele_vs_pinn_summary.csv)
- [artifacts/run_11_interest_rate_teacher_slope/reports/interest_rate_sign_check_40_policies_summary.csv](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_11_interest_rate_teacher_slope/reports/interest_rate_sign_check_40_policies_summary.csv)
- [artifacts/run_12_interest_rate_teacher_finetune/reports/checkpoint_interest_rate_screen.csv](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_12_interest_rate_teacher_finetune/reports/checkpoint_interest_rate_screen.csv)

Outcome:

- `run_11_interest_rate_teacher_slope` was worse than both `run_09_tuned` and
  `run_10_interest_rate_reserve_space` on the actual reserve-vs-interest metric
  used for diagnosis
- representative fixed-policy plots still had the wrong sign for all three
  sampled policies
- 40-policy sign-match rate fell to approximately `2.5%`
- `run_12_interest_rate_teacher_finetune` did not improve the benchmark either
  - the best total-loss checkpoint was epoch 0, which is effectively the
    loaded `run_09_tuned` model before the new loss had any influence
  - checkpoint screening across saved epochs did not find a better
    rate-sensitivity tradeoff

Interpretation:

- the local supervised target `d(V/S)/dr` is more principled than the previous
  sign-only penalty
- but optimizing local time-point sensitivity did not translate into a better
  peak-reserve-vs-interest curve
- the current training objective is still too indirect relative to the
  business-facing diagnostic, which is based on fixed-policy peak reserve under
  scenario shifts

## Follow-Up Experiment: Scenario-Curve Loss

On branch `circuits/interest-rate-experiment`, the interest-rate loss was moved
one step closer to the business plot:

- [src/data/dataset.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/dataset.py)
  now stores classical reserve-ratio targets under `r - Δr` and `r + Δr`
- [src/losses/interest_rate_scenario_loss.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/interest_rate_scenario_loss.py)
  adds a direct Huber loss between PINN shocked reserve curves and Thiele
  shocked reserve curves at the same policy time point
- [src/losses/registry.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/registry.py),
  [src/utils/config.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/utils/config.py),
  and [configs/config.yaml](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/configs/config.yaml)
  were updated to support the new configurable loss

Experiment:

- `run_13_interest_rate_scenario_curve`
- warm-started from `run_09_tuned`
- learning rate reduced to `3e-4`
- `interest_rate_scenario_loss.weight = 50000.0`

Artifacts:

- [artifacts/run_13_interest_rate_scenario_curve/reports/checkpoint_interest_rate_screen.csv](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_13_interest_rate_scenario_curve/reports/checkpoint_interest_rate_screen.csv)
- [artifacts/run_13_interest_rate_scenario_curve/reports/benchmark_comparison.csv](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_13_interest_rate_scenario_curve/reports/benchmark_comparison.csv)

Benchmark comparison on the same 40-policy screen:

- `run_09_tuned`: sign match `10.0%`, median slope gap `14239`
- `run_10_interest_rate_reserve_space`: sign match `52.5%`, median slope gap `6416`
- `run_11_interest_rate_teacher_slope`: sign match `2.5%`, median slope gap `20761`
- `run_13_interest_rate_scenario_curve`: sign match `10.0%`, median slope gap `15797`

Interpretation:

- the scenario-curve loss did **not** improve the fixed-policy
  reserve-vs-interest benchmark relative to the baseline tuned run
- in training logs, the weighted scenario term reached a meaningful share of the
  total loss, so this was not simply "ignored by the optimizer"
- however, matching small `±50 bp` shocked reserve ratios at individual time
  points still did not fix the larger business diagnostic based on peak reserve
  across the full term and a wider `1%` to `8%` rate sweep

Current conclusion:

- local slope matching was too indirect
- small-shock curve matching was still too local
- the next credible step is a policy-level loss that optimizes the actual
  trajectory or peak-reserve scenario metric over complete shocked policy
  curves, not random independent time-point rows

## Follow-Up Experiment: Peak-Reserve Loss

The next experiment moved directly onto the business metric:

- [src/data/dataset.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/dataset.py)
  now stores classical shocked peak times, peak mortalities, and peak reserve
  ratios for `r - Δr` and `r + Δr`
- [src/losses/interest_rate_peak_loss.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/interest_rate_peak_loss.py)
  adds a direct Huber penalty on those classical shocked peak reserve ratios
- [configs/config.yaml](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/configs/config.yaml)
  enables `interest_rate_peak_loss` in `run_14_interest_rate_peak_curve`

Experiment:

- `run_14_interest_rate_peak_curve`
- warm-started from `run_09_tuned`
- `interest_rate_peak_loss.weight = 50000.0`

Important selection note:

- the total-loss-selected checkpoint (`best_model.pt`) did not improve the
  interest-rate benchmark
- the best checkpoint on the actual business metric was
  `epoch_002.pt`, identified by a dedicated checkpoint screen

Artifacts:

- [artifacts/run_14_interest_rate_peak_curve/reports/checkpoint_interest_rate_screen.csv](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_14_interest_rate_peak_curve/reports/checkpoint_interest_rate_screen.csv)
- [artifacts/run_14_interest_rate_peak_curve/reports/baseline_vs_peak_loss_benchmark.csv](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_14_interest_rate_peak_curve/reports/baseline_vs_peak_loss_benchmark.csv)
- [artifacts/run_14_interest_rate_peak_curve/reports/baseline_vs_peak_loss_interest_rate.png](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_14_interest_rate_peak_curve/reports/baseline_vs_peak_loss_interest_rate.png)

Benchmark comparison:

- `run_09_tuned`:
  - sign match `10.0%`
  - median slope gap `14239`
  - mean slope gap `41078`
- `run_14_interest_rate_peak_curve` at `epoch_002.pt`:
  - sign match `10.0%`
  - median slope gap `12301`
  - mean slope gap `40370`

Interpretation:

- this is the first experiment that improved the slope-gap metric while staying
  on a policy-level target tied to the plot
- the improvement is modest:
  - median slope gap improved by roughly `13.6%`
  - mean slope gap improved by roughly `1.7%`
- sign behavior did not improve at all
- representative policies remained poor, so this is not yet a satisfactory fix

Current conclusion:

- policy-level peak supervision is more promising than local derivative or
  pointwise shocked-curve supervision
- but checkpoint selection must use the interest-rate benchmark itself, not
  total validation loss
- the next likely improvement would be combining:
  - policy-level peak loss for magnitude
  - a weak sign/ordering constraint for monotonic direction

## Bottom Line

The original “terrible” reserve-vs-interest curve was first and foremost a **bad chart design problem**.  
The deeper investigation then uncovered real training/math bugs that were masking the true model behavior.

After local fixes:

- the **plot is now correct**
- the **training stack is more mathematically honest**
- the **model itself still needs further tuning before the reserve curves and sensitivities are trustworthy enough for production claims**

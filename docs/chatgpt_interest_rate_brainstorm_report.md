"""Interest-rate experiment brainstorming report.

Created: 2026-07-02
Purpose: Provide a copy-paste-ready diagnostic brief for external reasoning
about the reserve-vs-interest-rate failure mode.
"""

# Interest-Rate Experiment Brainstorming Report

## How to use this file

If you want a compact prompt for ChatGPT, copy the section:

- `## Paste Into ChatGPT`

If you want the supporting detail first, read the rest of this file.

## Executive Summary

I am debugging a PINN-based insurance reserve model for term life insurance.

The core business-facing issue is:

- for a fixed policy, the classical Thiele solver shows that peak reserve should
  decrease as interest rate increases
- the neural model often predicts the wrong sign, or the right sign with
  completely wrong magnitude

I have already tried four increasingly direct interest-rate losses:

1. local monotonicity / sign-style loss
2. local supervised derivative loss
3. pointwise shocked-curve loss
4. policy-level shocked peak reserve loss

Only the policy-level peak reserve loss helped at all, and even that helped
only modestly. It improved slope-gap metrics, but did not improve sign-match.

The current situation is:

- the model can fit some reserve levels
- the model can satisfy some physics constraints
- the model still does not reliably learn economically correct
  interest-rate sensitivity

## Business Metric That Actually Matters

The real diagnostic is not total loss. It is this:

- hold one policy fixed
- vary only interest rate from `1%` to `8%`
- compute classical Thiele peak reserve
- compute PINN peak reserve
- compare slope sign and slope magnitude

That is the metric that looks bad in plots and makes the model unusable for
scenario analysis.

## Baseline Mathematical Setting

Term life reserve equation:

```text
dV/dt = rV + P - μ(S - V)
```

Where:

- `V(t)`: reserve
- `r`: interest rate
- `P`: premium
- `μ`: mortality intensity
- `S`: sum assured

Expected qualitative sensitivity for standard term insurance:

- reserve should generally decrease as interest rate increases
- i.e. for fixed policy inputs, `dV/dr <= 0`

## Important Code Design

The model does **not** predict raw reserve directly.

It predicts a standardized reserve ratio target:

```text
v = V / S
z = (v - mean) / std
```

So the model output is `z`, not `V`.

Relevant code:

- reserve standardization in [dataset.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/dataset.py:204)
- total loss composition in [total_loss.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/total_loss.py:84)

This means any monotonicity or sensitivity loss can easily become misaligned if
it is enforced in latent `z` space rather than business reserve space.

## Important Training Design

The trainer:

- composes all losses additively
- selects `best_model.pt` using **validation total loss**
- ramps constraint losses linearly during warmup

Relevant code:

- checkpoint selection by total validation loss in [trainer.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/trainers/trainer.py:349)
- constraint warmup in [trainer.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/trainers/trainer.py:386)

This is important because I found that the checkpoint that is best on
`validation_total_loss` is **not** necessarily best on the interest-rate
business metric.

## What Was Already Fixed Before These New Experiments

These were genuine bugs or design mismatches already discovered and corrected:

1. The original EDA reserve-vs-interest plot was wrong.
   It was cross-sectional across different policies, so it confounded:
   - age
   - term
   - premium
   - mortality
   - sum assured

2. Curriculum warmup was previously mutating configured loss weights and
   effectively zeroing some constraints after early epochs.

3. PDE loss in normalized `z` space had a transformation bug.

4. Boundary loss was using the wrong terminal mortality state.

5. Sum-assured monotonicity was being enforced in the wrong space.

Those fixes made the training more mathematically honest, but they did **not**
solve the reserve-vs-interest-rate behavior.

## Interest-Rate Loss Experiments Already Tried

### Experiment 1: Local derivative supervision

Loss type:

- supervise `d(V/S)/dr` against a central-difference Thiele target

Relevant code:

- [interest_rate_monotonicity_loss.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/interest_rate_monotonicity_loss.py:19)

Problem:

- too local
- matches point sensitivity at a time point, not the policy-level peak reserve
  curve that the business sees

Outcome:

- did not improve the real plot

### Experiment 2: Pointwise shocked-curve supervision

Loss type:

- supervise reserve ratio at `r - Δr` and `r + Δr` for each row

Relevant code:

- dataset shock targets in [dataset.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/dataset.py:132)
- loss in [interest_rate_scenario_loss.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/interest_rate_scenario_loss.py:22)

Problem:

- still local in time
- still not directly optimizing the peak reserve summary used in the plot

Outcome:

- no useful improvement on the business metric

### Experiment 3: Policy-level shocked peak reserve supervision

Loss type:

- dataset stores classical shocked peak time, peak mortality, and peak reserve
  ratio under `r - Δr` and `r + Δr`
- loss evaluates the model at those peak coordinates and matches the Thiele
  shocked peak reserve ratio

Relevant code:

- peak target generation in [dataset.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/dataset.py:146)
- loss in [interest_rate_peak_loss.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/interest_rate_peak_loss.py:24)

Outcome:

- this was the **first** experiment that improved the slope-gap metric
- but the improvement was modest
- sign behavior still did not improve

## Benchmark Numbers

All numbers below come from the same 40-policy fixed-policy benchmark.

### Benchmark definition

For each policy:

1. hold the policy fixed
2. sweep interest rate from `1%` to `8%`
3. compute peak reserve under Thiele
4. compute peak reserve under PINN
5. compare:
   - sign of slope
   - magnitude of slope error

### Results

`run_09_tuned` baseline:

- sign match rate: `10.0%`
- median absolute slope gap: `14239`
- mean absolute slope gap: `41078`

`run_10_interest_rate_reserve_space`:

- sign match rate: `52.5%`
- median absolute slope gap: `6416`
- mean absolute slope gap: `19491`

Important caveat:

- this run improved benchmark metrics, but produced pathological curve shapes
  and very unrealistic reserve magnitudes
- it was not a usable fix

`run_11_interest_rate_teacher_slope`:

- sign match rate: `2.5%`
- median absolute slope gap: `20761`
- mean absolute slope gap: `44837`

`run_13_interest_rate_scenario_curve`:

- sign match rate: `10.0%`
- median absolute slope gap: `15797`
- mean absolute slope gap: `45016`

`run_14_interest_rate_peak_curve`, best by the actual business metric
(`epoch_002.pt`, not `best_model.pt`):

- sign match rate: `10.0%`
- median absolute slope gap: `12301`
- mean absolute slope gap: `40370`

Supporting files:

- benchmark summary:
  [baseline_vs_peak_loss_benchmark.csv](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_14_interest_rate_peak_curve/reports/baseline_vs_peak_loss_benchmark.csv)
- representative policy summary:
  [baseline_vs_peak_loss_interest_rate_summary.csv](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_14_interest_rate_peak_curve/reports/baseline_vs_peak_loss_interest_rate_summary.csv)

## Representative Policies

Selected-policy comparison from the latest peak-loss experiment:

`policy_00059`

- Thiele slope: `-187.22`
- baseline PINN slope: `+1460.70`
- peak-loss checkpoint slope: `+620.43`

Interpretation:

- still wrong sign
- magnitude improved somewhat

`policy_00013`

- Thiele slope: `-489.42`
- baseline PINN slope: `-46.03`
- peak-loss checkpoint slope: `+1198.26`

Interpretation:

- baseline had correct sign but weak magnitude
- peak-loss checkpoint made it worse

`policy_00043`

- Thiele slope: `-346.84`
- baseline PINN slope: `+2470.21`
- peak-loss checkpoint slope: `+2334.78`

Interpretation:

- still wrong sign
- only marginal magnitude improvement

## Strong Hypotheses About Why This Is Happening

### Hypothesis 1: The training unit of work is still wrong

The dataset is row-based, not policy-trajectory-based.

Even after adding peak targets, the optimizer still sees many independent rows
from the same policy rather than treating the whole shocked policy curve as one
structured object.

Consequence:

- the model may fit local or semi-local targets
- but it is not directly forced to preserve global ordering of peak reserve over
  a rate sweep

### Hypothesis 2: The loss that matters is too small a slice of the total objective

Even when the weighted interest-rate loss becomes nontrivial, it is still just
one term inside a larger loss:

- data fit
- PDE
- boundary
- solvency
- reserve ceiling
- smoothness
- sum-assured monotonicity
- other constraints depending on config

See [total_loss.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/total_loss.py:103).

Consequence:

- the model can improve total loss while still failing badly on the specific
  interest-rate metric

### Hypothesis 3: Checkpoint selection is misaligned with the business metric

`best_model.pt` is currently chosen by:

- lowest validation total loss

See [trainer.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/trainers/trainer.py:367).

But the best interest-rate checkpoint in `run_14` was:

- `epoch_002.pt`

not:

- `best_model.pt`

Consequence:

- the training loop is optimizing and selecting for the wrong KPI

### Hypothesis 4: Peak reserve depends on a non-smooth argmax operation

The business metric is:

- `max_t V(t; r)`

But the losses I added approximate this by supervising:

- peak point coordinates found by the classical solver

This may still be unstable because:

- small changes in the curve can move the argmax time
- the PINN is not directly optimized over a dense predicted trajectory and then
  an internal differentiable max surrogate

Consequence:

- local target matching may not transfer robustly to peak ordering behavior

### Hypothesis 5: The model architecture may be too entangled

The same MLP must jointly represent:

- time evolution
- age effects
- mortality effects
- premium effects
- sum assured effects
- interest-rate effects

with no special structure for interest-rate sensitivity.

Consequence:

- the rate dimension may be getting represented as a weak entangled latent
  direction instead of a clean monotone/scenario-aware factor

### Hypothesis 6: The simulator may create a semantic mismatch for `r`

In synthetic generation:

- premium is created using issue assumptions that include interest rate

But in the business diagnostic:

- premium is held fixed while only `r` changes

This means the same variable is playing two roles:

1. issue-time pricing assumption
2. scenario-time reserve shock assumption

Consequence:

- the model may learn a confused mapping because in training data the meaning of
  `r` is partly entangled with how the contract was originally priced

This is one of the strongest hypotheses.

### Hypothesis 7: The loss operates on reserve ratio, but the business plot is in currency

Training targets are based on:

- `V / S`

Business plots are based on:

- `V`

This normalization is useful numerically, but it may distort the gradient
priorities across contracts with different sum assured values.

Consequence:

- the model may look okay in normalized space while still looking poor in raw
  reserve-space sensitivity

## What Seems Most Likely Overall

If I had to prioritize likely causes, I would rank them as:

1. **objective mismatch**:
   total loss and selected losses are still not perfectly aligned with the exact
   business benchmark
2. **training sample structure mismatch**:
   row-based training is too weak for a policy-level scenario curve problem
3. **semantic mismatch in the meaning of interest rate**:
   issue-rate and scenario-shock rate are conflated
4. **checkpoint selection mismatch**:
   best checkpoint is chosen by total loss, not the business metric
5. **model entanglement**:
   plain MLP may not separate interest effects cleanly

## What I Think Should Be Tried Next

### Option A: Policy-batch training

Instead of sampling independent rows, create batches of complete policy
trajectories:

- one batch element = one policy
- inside it, include full time grid
- include shocked policy trajectories in the batch

Then define losses directly on:

- full shocked curves
- peak reserve over shocked curves
- ordering across interest-rate grid

This is the most principled next step.

### Option B: Multi-rate ranking / ordering loss

For each policy, evaluate the model on a small rate grid:

- `r1 < r2 < r3 < ...`

Then penalize violations of:

```text
peak_reserve(r1) >= peak_reserve(r2) >= peak_reserve(r3) ...
```

This would directly teach monotone ordering of the actual plot.

### Option C: Separate issue rate from scenario shock rate

Introduce two distinct inputs:

- `pricing_interest_rate`
- `scenario_interest_rate`

Then:

- premium generation uses pricing rate
- reserve scenario analysis uses scenario rate

This may remove a major semantic confusion in the learning task.

### Option D: Validation and checkpointing on the real KPI

Add a dedicated validation suite that computes:

- sign match rate
- median slope gap
- representative policy plots

Then select checkpoints on that metric, not on total validation loss.

### Option E: Structured architecture

Potentially use an architecture where:

- base reserve level is modeled separately
- rate sensitivity head or monotone correction head is modeled separately

For example:

```text
V_pred = base_head(features_except_r) + rate_correction(features, r)
```

or another structured decomposition.

## Exact Questions I Want Help Brainstorming

1. Is the main failure here fundamentally an **objective mismatch**, a
   **data-design mismatch**, or a **model-architecture mismatch**?
2. Is the biggest conceptual bug that I am using a row-level MLP for what is
   actually a policy-trajectory scenario problem?
3. Does conflating issue-rate and shocked scenario rate make the learning task
   ill-posed?
4. Should I stop trying to fix this with scalar auxiliary losses and instead
   redesign the training batch structure around complete policy trajectories?
5. What is the best mathematically clean way to impose:
   - correct sign
   - correct magnitude
   - correct peak ordering
   all at once?
6. Would a pairwise ranking loss across rate scenarios be better than matching
   shocked values directly?
7. Would it be better to train directly on a small fixed policy-rate scenario
   grid per policy rather than on random independent time rows?

## Paste Into ChatGPT

```text
I am debugging a PINN-based insurance reserve model for term life insurance, and I want your help brainstorming why the interest-rate sensitivity is still bad even after several increasingly direct loss-function experiments.

Business problem:
For a fixed policy, the classical Thiele solver shows that peak reserve should generally DECREASE as interest rate increases. But my neural model often predicts the wrong sign, or the right sign with very wrong magnitude.

Mathematical setting:
dV/dt = rV + P - μ(S - V)

Expected qualitative behavior:
For fixed policy features, reserve should generally decrease as r increases.

Important implementation detail:
The model does NOT predict raw reserve V directly.
It predicts a standardized reserve ratio target:
v = V / S
z = (v - mean) / std
So many constraints can become misaligned if enforced in the wrong space.

Current training design:
- total loss is additive across many terms
- best_model checkpoint is selected by validation total loss
- constraint losses are warmed up gradually
- dataset is row-based, not policy-trajectory-based

Important code behaviors:
- best checkpoint is picked by validation total loss, not by the interest-rate business metric
- the actual business metric is a fixed-policy reserve-vs-interest-rate benchmark

What I already fixed earlier:
1. the original reserve-vs-interest plot was wrong because it was cross-sectional across different policies
2. curriculum loss-weight mutation bug
3. PDE loss transformation bug in normalized space
4. boundary loss terminal-mortality mismatch
5. sum-assured monotonicity loss mismatch

Those fixes made training more honest, but did not solve the rate-sensitivity issue.

Experiments I tried:

1. Local derivative supervision:
- supervised d(V/S)/dr against classical central-difference target
- result: did not improve the real reserve-vs-interest plot

2. Pointwise shocked-curve supervision:
- supervised reserve ratio at r-Δr and r+Δr for each row
- result: no useful improvement on the business metric

3. Policy-level shocked peak reserve supervision:
- for each policy I stored the classical peak time, peak mortality, and peak reserve ratio under r-Δr and r+Δr
- loss evaluates the model at those peak coordinates and matches those classical peak reserve ratios
- this was the first experiment that helped at all, but only modestly

Benchmark definition:
For each fixed policy:
1. hold everything fixed
2. sweep interest rate from 1% to 8%
3. compute Thiele peak reserve
4. compute model peak reserve
5. compare slope sign and slope magnitude

Benchmark results:

Baseline run_09_tuned:
- sign match rate: 10.0%
- median absolute slope gap: 14239
- mean absolute slope gap: 41078

Reserve-space sign-style run_10:
- sign match rate: 52.5%
- median absolute slope gap: 6416
- mean absolute slope gap: 19491
But this run produced pathological curve shapes and unrealistic magnitudes, so it was not usable.

Teacher slope run_11:
- sign match rate: 2.5%
- median absolute slope gap: 20761
- mean absolute slope gap: 44837

Scenario-curve run_13:
- sign match rate: 10.0%
- median absolute slope gap: 15797
- mean absolute slope gap: 45016

Peak-loss run_14, selecting the checkpoint by the actual business metric rather than validation total loss:
- sign match rate: 10.0%
- median absolute slope gap: 12301
- mean absolute slope gap: 40370

So the policy-level peak-reserve loss is the first thing that improved the benchmark at all, but the improvement is still modest and sign behavior did not improve.

Representative policies:

Policy 00059:
- Thiele slope: -187.22
- baseline PINN slope: +1460.70
- peak-loss checkpoint slope: +620.43
- still wrong sign, but magnitude improved somewhat

Policy 00013:
- Thiele slope: -489.42
- baseline PINN slope: -46.03
- peak-loss checkpoint slope: +1198.26
- got worse

Policy 00043:
- Thiele slope: -346.84
- baseline PINN slope: +2470.21
- peak-loss checkpoint slope: +2334.78
- still wrong sign, only marginally better magnitude

My strongest current hypotheses are:
1. objective mismatch: my training losses are still not aligned enough with the exact business metric
2. row-based training is wrong for a policy-trajectory scenario problem
3. issue-rate and scenario-rate semantics may be conflated
4. checkpoint selection by total validation loss is misaligned with the business metric
5. plain MLP architecture may entangle rate sensitivity too much

One especially important hypothesis:
In synthetic generation, premium is created using an issue-time interest-rate assumption, but in the business diagnostic premium is held fixed while only r changes. So the same variable may be playing two roles:
- issue/pricing rate
- shocked reserve scenario rate
This may make the learning problem ill-posed.

What I want from you:
1. Give me your best root-cause analysis of why the model still cannot learn realistic reserve-vs-interest behavior.
2. Tell me which hypotheses above are most likely.
3. Tell me whether the biggest problem is objective design, batch/data design, architecture, or semantic confounding of inputs.
4. Propose the next 3 most promising redesigns, ordered by expected payoff.
5. Specifically comment on whether I should move from row-level training to policy-trajectory training.
6. Specifically comment on whether I should separate pricing_interest_rate from scenario_interest_rate.
7. If you were redesigning this from scratch for correct interest-rate sensitivity, what would you build?
```

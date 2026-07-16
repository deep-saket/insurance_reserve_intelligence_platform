"""Variable semantics audit.

Created: 2026-07-02
Purpose: Audit whether key variables keep the same business meaning across data
generation, training, validation, and scenario evaluation.
"""

# Variable Semantics Audit

## Bottom Line

Yes. There is at least one **definite semantic mismatch**:

- `interest_rate` means **issue/pricing rate** during policy generation
- but is treated as **scenario shock rate with premium held fixed** in several
  training/evaluation experiments

There is also a second, separate problem:

- the stress tester appears to pass **raw, unnormalized features** directly into
  the model and reads the output without denormalizing it

That second point is more of a preprocessing/units bug than a semantic mismatch,
but it is still important.

## 1. Interest Rate

### During data generation

`interest_rate` is used in two roles at once:

1. discount/valuation rate
2. pricing rate used to derive premium

Code:

- premium pricing uses `interest_rate` in
  [simulator.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/simulator.py:306)
- policy creation stores that same `interest_rate` on the policy in
  [simulator.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/simulator.py:349)

So at generation time:

```text
interest_rate -> affects premium
interest_rate -> also becomes model input later
```

### During training dataset construction

The dataset stores:

- `policy.interest_rate`
- `policy.premium`

as separate inputs on each row.

Code:

- [dataset.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/dataset.py:164)

So the model sees:

```text
features = [t, age, interest_rate, premium, sum_assured, mortality]
```

But that premium was already generated *using the same interest rate* earlier.

### During controlled rate-sensitivity evaluation

In the fixed-policy reserve-vs-interest plots, only `interest_rate` is changed:

- `policy = replace(policy, interest_rate=new_rate)`
- `premium` is left unchanged

Code:

- [generate_and_eda.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/generate_and_eda.py:441)

This means the plot is testing:

```text
same policy
same premium
same mortality
only interest rate changes
```

That is **not** the same semantic regime used during generation, where premium
was originally priced from the interest rate.

### During training of rate-related auxiliary losses

The newer rate losses also do the same thing:

- they shock `interest_rate`
- they keep premium fixed

Code:

- shock target construction in
  [dataset.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/dataset.py:137)
- scenario loss in
  [interest_rate_scenario_loss.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/interest_rate_scenario_loss.py:87)
- peak loss in
  [interest_rate_peak_loss.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/losses/interest_rate_peak_loss.py:97)

### Conclusion for `interest_rate`

This is a **real semantic mismatch**.

The variable means:

- **pricing + valuation rate** at generation time

but later becomes:

- **scenario-only valuation shock rate with fixed premium**

in evaluation and some training targets.

This is likely one of the main reasons the model struggles with
reserve-vs-interest behavior.

## 2. Premium

### During data generation

Premium is not sampled independently.

It is *derived* from:

- mortality profile
- term
- interest rate
- sum assured

using the equivalence principle.

Code:

- [simulator.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/simulator.py:288)
- [simulator.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/simulator.py:349)

### During training and validation

Premium is treated as just another independent feature.

Code:

- [dataset.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/dataset.py:164)
- [validate.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/validate.py:250)

In monotonicity checks, premium is bumped by `+10%` while other features stay
fixed.

Code:

- [validate.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/validate.py:267)

### Conclusion for `premium`

This is not necessarily a bug, but it is a **semantic shift**:

- at generation time, premium is an implied actuarial quantity
- at evaluation/training diagnostics, premium is treated as an independent
  controllable input

That can be acceptable if intentional, but it means the model is being asked to
learn off-manifold feature combinations that did not arise naturally during
generation.

## 3. Scenario Generator vs Rate-Sensitivity Plots

There is a second inconsistency inside the codebase itself.

### Scenario generator behavior

When `generate_scenario_policies()` applies an interest-rate shift, it rebuilds
the policy through `_build_policy()`, which recalculates premium using the new
rate.

Code:

- [simulator.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/simulator.py:430)

So this path interprets an interest-rate shock as:

```text
rate changes
premium may change because policy is rebuilt/repriced
```

### Rate-curve evaluation behavior

The fixed-policy rate plots interpret it as:

```text
rate changes
premium stays fixed
```

Code:

- [generate_and_eda.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/generate_and_eda.py:441)

### Conclusion

There are currently **two different semantics for an interest-rate shock**
inside the project:

1. repriced scenario
2. fixed-premium in-force scenario

That ambiguity should be removed.

## 4. Mortality

### During generation

Mortality is a full policy-level curve built from:

- age
- term
- risk profile
- mortality multipliers

Code:

- [simulator.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/simulator.py:243)
- [simulator.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/simulator.py:343)

### During dataset/training

Only the pointwise mortality intensity `μ(t)` is fed into the model at each row.

Code:

- [dataset.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/dataset.py:160)

### Conclusion for `mortality`

This is mostly **consistent**, but note that the model only sees the scalar
`μ(t)`, not the full mortality curve object. So the curve-level structure is
collapsed into pointwise values.

That is a modeling simplification, not a semantic contradiction.

## 5. Age and Time

### During generation

The policy stores:

- issue age `age`
- term

### During training/eval

The model input is:

- `age` as issue age
- `time` as elapsed time

not explicit current age `x+t`.

Code:

- [dataset.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/dataset.py:164)

### Conclusion for `age` and `time`

This is semantically consistent, provided the model is expected to infer current
age from `age + time`.

## 6. Sum Assured

### During generation

`sum_assured` is generated as a contract benefit amount.

### During training/eval

It remains a contract benefit amount and is also used for reserve
denormalization:

```text
V = (z * std + mean) * S
```

Code:

- [dataset.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/data/dataset.py:204)

### Conclusion for `sum_assured`

This is mostly semantically consistent.

## 7. Stress Tester Bug

The stress tester looks inconsistent with training preprocessing.

It builds features in raw units:

- `[time, age, interest_rate, premium, sum_assured, mortality]`

Code:

- [stress_tester.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/stress/stress_tester.py:81)

But `_predict()` passes those raw values directly to the model:

- no normalization
- no output denormalization

Code:

- [stress_tester.py](/Users/saketm10/Projects/insurance_reserve_intelligence_platform/src/stress/stress_tester.py:89)

This is inconsistent with training, where inputs are normalized and outputs are
later mapped back to reserve space.

So stress testing is likely numerically wrong at the moment.

This is a **preprocessing/units bug**, not just a semantics issue.

## Final Assessment

### Definite semantic mismatch

- `interest_rate`

because it means:

- issue/pricing rate during generation
- scenario shock rate with fixed premium during evaluation/training experiments

### Likely semantic shift that may matter

- `premium`

because it is:

- generated as an implied actuarial quantity
- later treated as an independently bumpable feature

### Mostly consistent

- `mortality` as pointwise `μ(t)`
- `age`
- `time`
- `sum_assured`

### Separate operational bug

- stress tester uses raw model inputs without normalization or reserve
  denormalization

## Recommendation

The clean fix is to split the concept of rate into two variables:

1. `pricing_interest_rate`
2. `scenario_interest_rate`

Then define clearly whether premium is:

- fixed as an in-force contractual cashflow

or:

- recomputed under repricing scenarios

Right now the codebase mixes both interpretations.

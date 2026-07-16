"""Run-16 interest-rate checkpoint progression summary.

Created: 2026-07-02
Purpose: Summarize the longer run with rate-focused losses enabled and explain
what checkpoint progression revealed about reserve-vs-interest-rate learning.
"""

# Run 16: Longer Interest-Rate Experiment

## Objective

This run was designed to answer one specific question:

> If we train longer and let the rate-specific losses fully warm up, does the
> reserve-vs-interest-rate curve start to resemble the classical Thiele curve?

## Configuration Changes

Compared with the prior run:

- `interest_rate_monotonicity_loss` enabled at weight `5.0`
- `interest_rate_scenario_loss` enabled at weight `25000.0`
- `interest_rate_peak_loss` retained at weight `30000.0`
- training extended to `100` epochs
- early stopping patience increased to `35`
- checkpoint cadence changed to every `5` epochs

Run name:

- `run_16_rate_curve_longer`

## Why This Experiment Was Necessary

In the prior run, the interest-rate losses were still in warmup when training
stopped. That meant the model never experienced the full intended supervision
for rate sensitivity.

This run gave the model enough time to:

1. complete the 30-epoch constraint warmup
2. continue training under the full rate-loss regime
3. expose whether later checkpoints improve reserve-vs-rate behavior even if
   the aggregate validation loss does not

## Key Result

The best checkpoint for interest-rate curve quality was **not** the checkpoint
with the lowest total validation loss.

This is the central finding of the run.

## Progression Metrics

Source:

- `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_16_rate_curve_longer/reports/checkpoint_interest_curve_progression_metrics.csv`

Best checkpoints by curve metric:

- Best overall combined curve quality: epoch `25`
- Best peak-reserve curve: epoch `20`
- Best fixed-time reserve curve (`t = T/2`): epoch `25`

Selected results:

| Epoch | Mid Reserve Error | Reserve at Baseline Peak Time Error | Peak Reserve Error | Validation Total Loss |
|---|---:|---:|---:|---:|
| 3  | 127.49% | 87.24% | 97.77% | 0.2289 |
| 10 | 306.87% | 224.78% | 207.54% | 0.2212 |
| 20 | 157.10% | 52.49% | 43.28% | 0.3464 |
| 25 | 67.88% | 65.41% | 84.99% | 0.2938 |
| 45 | 73.51% | 92.26% | 158.59% | 0.3153 |

Interpretation:

- early training improved the global fit
- mid-training around epochs `20–25` improved the interest-rate curve materially
- continuing beyond that did **not** monotonically improve the rate curve
- later checkpoints under full constraint weight can drift again

## Business Interpretation

The platform is currently optimizing two partially conflicting objectives:

1. global reserve prediction quality across the whole synthetic book
2. realistic reserve-vs-interest-rate scenario behavior for fixed policies

The checkpoint analysis shows that a lower overall validation loss does not
guarantee better rate stress behavior. In practical terms:

- a model that looks better on aggregate may still be worse for ALM or
  what-if interest-rate analysis
- checkpoint selection should eventually consider scenario-quality metrics, not
  only overall loss

## Main Takeaways

### 1. Longer training was useful

It was worth doing. The curve quality improved significantly compared with many
earlier checkpoints, especially around epochs `20–25`.

### 2. The curve problem is not fully solved

Even the best checkpoint still has large percentage error on the reserve-vs-rate
curve. The model is better, but not yet good enough to trust blindly.

### 3. Total validation loss is not the right model-selection metric

For this use case, we likely need one of:

- a secondary checkpoint-selection criterion based on rate-curve quality
- a multi-objective early-stopping rule
- a post-training checkpoint sweep that picks the best scenario-behavior model

## Useful Artifacts

- Progression metrics:
  `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_16_rate_curve_longer/reports/checkpoint_interest_curve_progression_metrics.csv`
- Progression plot:
  `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_16_rate_curve_longer/reports/checkpoint_interest_curve_progression.png`
- Best overall curve checkpoint visual:
  `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_16_rate_curve_longer/reports/interest_rate_curve_epoch_025.png`

## Recommended Next Step

Use checkpoint `epoch_025.pt` as the current best candidate for rate-curve
inspection, even though it is not the lowest-total-loss checkpoint.

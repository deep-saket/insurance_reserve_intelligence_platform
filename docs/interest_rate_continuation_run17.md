"""Run-17 continuation summary.

Created: 2026-07-03
Purpose: Record the outcome of resuming run 16 for additional epochs under the
full-strength rate-constraint regime.
"""

# Run 17: Longer Continuation After Full Warmup

## Objective

Continue training beyond run 16 to answer:

> Does additional optimization after the full constraint warmup improve the
> reserve-vs-interest-rate curve?

## Important Technical Fix

Before resuming, the trainer was corrected so that constraint warmup does not
restart from zero after loading a checkpoint.

Without that fix, a resumed run would have been misleading because the
interest-rate losses would have been artificially weakened again.

## Continuation Setup

- Source checkpoint: `run_16_rate_curve_longer/checkpoints/epoch_045.pt`
- Continuation run: `run_17_rate_curve_resume_longer`
- Additional epochs evaluated: `50` through `89`

## Result Summary

### Fixed-time reserve curve improved further

From:

- `/Users/saketm10/Projects/insurance_reserve_intelligence_platform/artifacts/run_17_rate_curve_resume_longer/reports/checkpoint_interest_curve_progression_metrics.csv`

Best run-17 checkpoints:

| Epoch | Mid Reserve Error | Reserve at Baseline Peak Time Error | Peak Reserve Error | Validation Total Loss |
|---|---:|---:|---:|---:|
| 50 | 37.03% | 42.57% | 115.64% | 0.3060 |
| 70 | 34.70% | 44.14% | 118.91% | 0.3089 |
| 55 | 39.51% | 57.60% | 117.52% | 0.2998 |

Interpretation:

- fixed-time reserve-vs-rate behavior got better than in run 16
- the best mid-curve error dropped from roughly `67.9%` in run 16 to `34.7%` in run 17
- reserve at baseline peak time also improved, reaching roughly `42–44%`

### Peak-reserve curve did not improve

This is the key limitation of the continuation.

Run 16 still had the best peak-reserve checkpoint:

- epoch `20` peak-reserve error: `43.28%`

Run 17 peak-reserve checkpoints were worse:

- epoch `50`: `115.64%`
- epoch `70`: `118.91%`

Interpretation:

Longer training helped the ordinary reserve-vs-rate curve more than the
peak-reserve curve.

## Business Interpretation

The surrogate appears to improve on:

- reserve sensitivity at a fixed point in the policy life

but not on:

- the more complex policy-level peak-liability scenario metric

That suggests the model is learning local reserve-rate relationships better than
it is learning the global shape of the full reserve trajectory across rate
scenarios.

## Main Takeaway

Longer training **did** help, but only for part of the problem.

Best observed outcomes now are:

- best fixed-time reserve curve: run 17 around epoch `70`
- best peak-reserve curve: run 16 at epoch `20`

So the conclusion is:

- more training after warmup can improve fixed-time rate behavior
- more training alone is not enough to solve peak-reserve behavior

## Recommended Next Step

Use different checkpoint-selection logic for different objectives:

- fixed-time reserve-vs-rate diagnostics: prefer run 17 epoch `70`
- peak-reserve scenario diagnostics: prefer run 16 epoch `20`

If the business objective is mainly interest-rate what-if analysis at a chosen
valuation horizon, the continuation helped. If the objective is peak reserve
under stressed scenarios, a different loss design is still needed.

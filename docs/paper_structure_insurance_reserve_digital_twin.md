"""Paper structure for the insurance reserve digital twin use case.

Created: 2026-07-08
Purpose: Adapt the section logic of the reference paper
`Constraint-Driven Model Optimization: An Industry Framework for Selecting
Compression and Acceleration Techniques in Modern Machine Learning Systems`
to the ActuaryTwin / insurance reserve digital twin research narrative.
"""

# Paper Structure for Insurance Liability Digital Twin

## Recommended Title Options

1. **Toward Differentiable Actuarial Digital Twins: A PINN/KINN Framework for Insurance Reserve Analytics**
2. **Insurance Liability Digital Twin: Physics-Informed and Knowledge-Informed Neural Surrogates for Reserve Estimation, Stress Testing, and Optimization**
3. **From Classical Reserve Solvers to Differentiable Liability Analytics: A Scenario-Aware Framework for Insurance Reserve Digital Twins**
4. **Scenario-Aware Differentiable Surrogates for Life Insurance Reserves: A PINN/KINN Digital Twin Framework**

## Why This Structure Fits

The reference paper is not a pure algorithm paper. Its real structure is:

1. motivate an industry problem
2. explain why the old framing is too narrow
3. build a taxonomy
4. organize the solution space
5. propose a decision framework
6. map it to real scenarios
7. discuss trade-offs and limitations

That logic fits your use case very well.

Your paper should therefore **not** be written as:

- “we used a PINN to solve Thiele’s equation”

It should be written as:

- “we propose a differentiable reserve-surrogate framework for insurance liability digital twins, and we show why scenario-aware evaluation and semantically correct data generation are essential”

---

# Recommended Section-by-Section Outline

## Title Page

**Title**

Use one of the options above.

**Authors**

Add affiliations and corresponding author details.

---

## Abstract

### Purpose of the abstract

The abstract should answer five things:

1. what business problem exists
2. why classical reserve engines are insufficient for digital-twin-style use
3. what framework you propose
4. what experiments/evaluation you ran
5. what the main conclusion is

### Suggested abstract structure

1. **Problem context**
   Insurance reserve analytics increasingly require not only point estimates, but also real-time sensitivity analysis, stress testing, inverse solving, and scenario exploration across large portfolios.

2. **Method**
   We formulate term-life reserve estimation as a differentiable surrogate learning problem using PINN/KINN-style neural models constrained by Thiele’s reserve equation and actuarial knowledge-based losses.

3. **System contribution**
   We introduce an insurance liability digital twin framework that separates pricing-time assumptions from scenario-time valuation assumptions and supports reserve estimation, stress testing, optimization, and what-if simulation.

4. **Evaluation**
   We evaluate pointwise reserve accuracy, PDE residual compliance, monotonicity, and scenario-aware reserve-vs-interest-rate behavior.

5. **Finding**
   We show that good aggregate loss or PDE consistency does not necessarily imply correct scenario behavior, making scenario-aware evaluation essential for differentiable actuarial surrogates.

6. **Conclusion**
   Differentiable reserve surrogates are promising for digital-twin analytics, but their practical usefulness depends on semantically correct data generation and direct supervision of scenario behavior.

---

## 1. Introduction

### 1.1 Why Insurance Liability Digital Twins Matter

Explain:

- reserves are not only accounting quantities
- they drive capital planning, stress testing, pricing, solvency, and strategy
- institutions increasingly need real-time analytics across many policies and scenarios

Business framing:

- reserve forecasting
- sensitivity analysis
- scenario simulation
- capital efficiency analysis
- what-if dashboards

### 1.2 Evolution from Classical Reserve Solvers to Differentiable Surrogates

Explain the progression:

- classical actuarial formulas
- ODE-based reserve solvers
- nested scenario loops
- computational bottlenecks
- surrogate and ML approaches
- PINN/KINN as a scientific-ML alternative

### 1.3 Research Gap

This should be one of the strongest sections.

State that current gaps include:

- most reserve studies focus on pointwise reserve prediction, not digital-twin usage
- standard PINN success metrics may miss scenario failure modes
- reserve modeling often does not separate pricing assumptions from scenario assumptions
- there is little scenario-aware evaluation of differentiable actuarial surrogates

### 1.4 Contributions

Write this explicitly as a numbered list.

Suggested contribution list:

1. We formulate an insurance reserve digital twin framework using PINN/KINN-style differentiable surrogates.
2. We introduce a semantically explicit policy representation that separates `pricing_interest_rate` from `scenario_interest_rate`.
3. We define a scenario-aware evaluation protocol beyond pointwise reserve accuracy, including fixed-time reserve-vs-interest-rate and peak-reserve-vs-interest-rate diagnostics.
4. We implement a configurable loss framework combining data, physics, and actuarial knowledge constraints.
5. We show empirically that aggregate validation loss and PDE compliance do not guarantee correct scenario behavior.

---

## 2. From Classical Reserve Computation to Differentiable Liability Analytics

This section should mirror the role of Section 2 in the reference paper.

### Purpose

Explain that the field should move from:

- solving one reserve query at a time

to:

- learning a differentiable liability manifold that can support multiple downstream tasks

### 2.1 Classical Actuarial Reserve Solving

Cover:

- Thiele equation
- numerical ODE solution
- reserve trajectory generation

### 2.2 PINN-Based Reserve Surrogates

Cover:

- learning `V(t, x, r, μ, P, S)`
- using data loss + PDE loss + boundary loss
- why this is attractive for repeated evaluation

### 2.3 Knowledge-Informed Extensions

Cover:

- monotonicity
- solvency constraints
- reserve ceilings
- smoothness

### 2.4 Why Digital Twin Use Cases Change the Evaluation Problem

This subsection is important.

Explain:

- pointwise reserve fit is not enough
- stakeholders care about scenario curves, sensitivities, and optimization consistency

### 2.5 Synthesis and Motivation

End with:

- therefore the key research problem is not merely solving Thiele’s equation
- it is constructing a reserve surrogate that is useful for digital-twin analytics

---

## 3. Insurance Liability Digital Twin Requirement Taxonomy

This is the analog of the operational taxonomy in the reference paper.

### Purpose

Define what the system must satisfy if it is to be useful in practice.

### 3.1 Reserve Accuracy Requirements

- pointwise reserve accuracy
- terminal condition correctness
- robustness across ages, terms, and sum assured ranges

### 3.2 Scenario Fidelity Requirements

- correct reserve-vs-interest-rate behavior
- correct reserve-vs-mortality behavior
- stable stress responses

### 3.3 Sensitivity and Differentiability Requirements

- `dV/dr`
- `dV/dμ`
- `dV/dP`
- `dV/dS`
- second derivatives where useful

### 3.4 Business Operational Requirements

- large portfolio throughput
- real-time evaluation
- explainability
- integration with risk workflows

### 3.5 Modeling Governance Requirements

- no negative reserves when inappropriate
- scenario consistency
- semantic correctness in data generation
- stable checkpoint selection

### Discussion

Conclude that a useful reserve digital twin must satisfy multiple requirements simultaneously and cannot be judged on a single ML metric.

---

## 4. Methodological Building Blocks of the Reserve Digital Twin

This is the analog of the “optimization family taxonomy” section in the reference paper.

### Purpose

Organize the modeling components by what they contribute.

### 4.1 Classical Reserve Generation

- synthetic policy simulator
- Thiele solver
- target generation

### 4.2 Feature Representation

- time
- issue age
- pricing interest rate
- scenario interest rate
- premium
- sum assured
- mortality

### 4.3 PINN Loss Components

- data loss
- PDE residual loss
- boundary loss

### 4.4 KINN / Knowledge Constraint Components

- mortality monotonicity
- interest-rate monotonicity
- solvency
- reserve ceiling
- smoothness
- scenario and peak losses

### 4.5 Scenario-Aware Diagnostics

- fixed-time reserve-vs-rate curve
- peak-reserve-vs-rate curve
- checkpoint progression analysis

### 4.6 Digital Twin Functional Modules

- reserve estimation
- stress testing
- sensitivity analysis
- optimization
- portfolio simulation
- inverse problem solving

### 4.7 Comparative Role of Components

Explain which components are responsible for:

- pointwise fit
- local physics consistency
- scenario realism
- business usability

---

## 5. Scenario-Aware Modeling and Evaluation Framework

This is the central section of the paper.

It should play the same role that the decision framework plays in the reference paper.

### 5.1 Semantic Policy-State Mapping

Describe the key design choice:

- separate pricing-time and scenario-time interest-rate semantics

This section should explain why this matters mathematically and operationally.

### 5.2 Training Objective Design

Describe:

- weighted config-driven loss system
- why each loss exists
- how physics and business constraints are combined

### 5.3 Evaluation Matrix

Define explicit evaluation axes:

- reserve accuracy
- boundary compliance
- PDE residual
- monotonicity
- OOD generalization
- fixed-time rate-curve error
- peak-reserve rate-curve error

### 5.4 Engineering Guidelines

Mirror the style of the reference paper.

Suggested guideline headings:

#### Guideline 1: Evaluate According to Business Use Cases, Not Only Aggregate Loss

#### Guideline 2: Separate Pricing Assumptions from Scenario Assumptions

#### Guideline 3: Use Scenario-Aware Checkpoint Selection

#### Guideline 4: Treat Digital Twin Readiness as a Multi-Metric Property

### 5.5 Discussion

Conclude that differentiable reserve models require evaluation aligned with how they are used in actuarial decision-making.

---

## 6. Insurance Liability Digital Twin Use Cases

This section maps the framework to actual scenarios, just like the deployment scenarios in the reference paper.

### 6.1 Real-Time Reserve Estimation

- many policies
- rapid lookup
- reserve surface as a learned manifold

### 6.2 Interest-Rate Stress Testing

- fixed-policy scenario sweeps
- reserve-vs-rate behavior
- ALM interpretation

### 6.3 Mortality Stress Testing

- mortality shock
- longevity shock
- solvency interpretation

### 6.4 Sensitivity and Exposure Analysis

- automatic differentiation
- fast first- and second-order sensitivities

### 6.5 Product / Premium Optimization

- premium design
- target reserve calibration
- solvency-constrained pricing

### 6.6 Inverse Problem Solving

- implied assumptions from observed reserves
- diagnostic use

### 6.7 Portfolio Simulation

- large portfolio evaluation
- capital efficiency
- scenario comparisons

### 6.8 Digital Twin Dashboard / What-If Analytics

- sliders for rate, mortality, inflation, lapse
- interactive simulation use case

### 6.9 Cross-Use-Case Analysis

Explain:

- some use cases require pointwise accuracy
- some require derivative accuracy
- some require scenario-curve accuracy
- one model may perform differently across these objectives

---

## 7. Experimental Design and Empirical Results

The reference paper seems more conceptual and scenario-driven. For your paper, this section should be stronger and more empirical.

### 7.1 Experimental Setup

- policy simulation setup
- solver setup
- model architecture
- training configuration
- loss configurations

### 7.2 Baselines and Ablations

Strongly recommended:

- supervised MLP baseline
- PINN baseline
- PINN + KINN losses
- semantic split ablation
- scenario-loss ablation

### 7.3 Pointwise Reserve Results

- MAE
- RMSE
- R²

### 7.4 Physics and Boundary Results

- PDE residual
- `V(T)=0` error

### 7.5 Scenario-Aware Results

- fixed-time reserve-vs-interest-rate
- peak-reserve-vs-interest-rate
- best vs final checkpoint

### 7.6 Checkpoint Progression and Training Dynamics

This should be one of your signature results.

Explain:

- better total loss does not imply better scenario curve
- checkpoint selection depends on business objective

### 7.7 Summary of Main Empirical Findings

State clearly:

- what improved
- what failed
- what remained difficult

---

## 8. Discussion

This section should mirror the tone of the reference paper and be intellectually honest.

### 8.1 Understanding the Trade-offs

- pointwise fit vs scenario fidelity
- PDE compliance vs business usefulness
- local derivative correctness vs global trajectory correctness

### 8.2 When PINN/KINN Is Justified and When It Is Not

Very important.

Say clearly:

- for simple term-reserve calculation alone, PINN may be unnecessary
- for digital twin tasks such as sensitivities, optimization, inverse solving, and multi-scenario exploration, the framework becomes more compelling

### 8.3 Limitations

- synthetic data only
- peak-reserve scenario behavior still difficult
- not yet production-ready
- limited product scope

### 8.4 Future Research Directions

- richer scenario-grid supervision
- multi-product extension
- stochastic extensions
- portfolio-level constraints
- better checkpoint-selection criteria

---

## 9. Conclusion

The conclusion should not oversell.

Suggested message:

- differentiable actuarial surrogates are promising for insurance digital twins
- but correct scenario behavior requires more than pointwise reserve fitting
- semantically correct data generation and scenario-aware evaluation are essential

---

## References

Organize references around:

- PINNs / scientific ML
- actuarial reserve theory
- mortality modeling
- proxy / surrogate modeling in insurance
- digital twin literature

---

## Appendices

### Appendix A. Reserve Equation and Variable Definitions

- Thiele equation
- variable definitions
- boundary conditions

### Appendix B. Loss Function Definitions

- full mathematical definitions of each loss
- PINN vs KINN classification

### Appendix C. Experimental Configurations

- YAML settings
- data-generation ranges
- training runs

### Appendix D. Additional Scenario Curves

- reserve-vs-rate plots
- checkpoint progression plots
- stress-testing examples

---

# Figures and Tables You Should Plan Upfront

## Suggested Figures

1. System architecture of the insurance liability digital twin
2. Classical Thiele reserve trajectory vs neural reserve trajectory
3. Fixed-time reserve-vs-interest-rate curve
4. Peak-reserve-vs-interest-rate curve
5. Checkpoint progression of scenario-aware error
6. Digital twin capability map

## Suggested Tables

1. Variable definitions and actuarial interpretation
2. Loss functions and business interpretation
3. Evaluation metric matrix
4. Ablation study results
5. Checkpoint comparison by business objective
6. Use-case-to-capability mapping

---

# Recommended Writing Strategy

Write the paper in this order:

1. Section 3 taxonomy
2. Section 4 method
3. Section 7 experiments
4. Section 8 discussion
5. Section 1 introduction
6. Abstract and conclusion last

This usually leads to a much tighter paper because the introduction is written after the actual evidence is clear.

---

# Bottom-Line Positioning

If you use this structure, the paper will read as:

- a **framework paper**
- with a real **scientific-ML implementation**
- plus a **scenario-aware diagnostic contribution**

That is a much stronger framing than a narrow “PINN solves reserve ODE” paper.

"""Draft paper manuscript for the insurance reserve digital twin project.

Created: 2026-07-08
Purpose: Provide a paper-style draft based on the project's current scope,
results, and scenario-aware evaluation findings.
"""

# Toward Differentiable Actuarial Digital Twins: A PINN/KINN Framework for Insurance Reserve Analytics

## Abstract

Insurance reserve analytics increasingly require more than pointwise liability estimates. Modern actuarial and risk workflows demand real-time reserve estimation, stress testing, sensitivity analysis, optimization, scenario exploration, and portfolio-level what-if simulation. Classical reserve engines based on differential equations remain mathematically rigorous, but they are not naturally designed for repeated interactive evaluation across large policy populations and scenario grids. In this work, we propose an insurance liability digital twin framework built around a differentiable reserve surrogate for term life insurance. The framework combines a classical Thiele-equation solver with a configurable Physics-Informed Neural Network (PINN) and Knowledge-Informed Neural Network (KINN) loss system, enabling reserve learning under data, physics, and actuarial consistency constraints.

Our implementation introduces an explicit separation between pricing-time and scenario-time interest-rate semantics, which proved necessary for consistent data generation, reserve valuation, and scenario analysis. We further define a scenario-aware evaluation protocol that goes beyond aggregate validation loss and pointwise reserve metrics by explicitly measuring reserve-versus-interest-rate behavior at fixed times and at peak-reserve levels. Empirically, we show that strong PDE residual performance and low aggregate validation loss do not guarantee correct scenario behavior. In particular, checkpoint progression experiments reveal that the best model for overall validation loss is not necessarily the best model for business-relevant scenario diagnostics. Additional training after full constraint warmup improves fixed-time reserve-versus-rate behavior but does not fully resolve peak-reserve scenario fidelity.

These findings suggest that differentiable actuarial surrogates are promising for insurance liability digital twins, but their practical usefulness depends on semantically correct data generation, direct supervision of scenario behavior, and evaluation protocols aligned with downstream business use cases rather than conventional aggregate machine learning loss alone.

## 1. Introduction

### 1.1 Why insurance liability digital twins matter

Insurance reserves are not merely accounting quantities. They are central to pricing, solvency management, capital planning, sensitivity analysis, asset-liability management, and executive risk oversight. In practical settings, reserve users do not ask only one question such as “what is the reserve for this policy today?” They ask families of related questions: how does reserve change under mortality stress, how sensitive is liability to interest rates, what premium is required to hit a target reserve, and how does an entire portfolio behave across a scenario surface. These use cases motivate the idea of an insurance liability digital twin: a computational representation of policy liabilities that can be queried interactively under changing assumptions.

Classical actuarial reserve methods remain the benchmark for rigor. For term life insurance, Thiele’s differential equation provides a principled description of reserve evolution under premiums, interest, mortality, and benefits. However, direct numerical solving is fundamentally query-by-query. When millions of policies and large scenario sets are involved, repeated ODE evaluation becomes a computational bottleneck, especially when combined with stress grids, sensitivities, optimization loops, or inverse calibration tasks.

### 1.2 From classical reserve solvers to differentiable surrogates

Scientific machine learning offers an alternative perspective. Rather than solving a reserve equation from scratch for each query, one may learn a differentiable surrogate for the reserve manifold itself. Physics-Informed Neural Networks (PINNs) provide a natural template for this idea by combining data supervision with equation residual penalties [Raissi et al., 2019; Karniadakis et al., 2021]. Knowledge-informed extensions further allow business and actuarial constraints, such as monotonicity and solvency, to be enforced during training. In principle, such a surrogate can support not only reserve estimation, but also automatic differentiation for sensitivities, direct optimization over assumptions, inverse solving, and real-time scenario analytics.

### 1.3 Research gap

Despite this promise, several gaps remain. First, many reserve-surrogate efforts are implicitly evaluated as prediction tasks rather than as digital-twin systems. A model can fit pointwise reserves reasonably well while still failing under scenario sweeps that matter to treasury, ALM, and actuarial governance. Second, conventional PINN success criteria such as low PDE residual do not necessarily imply correct business behavior under assumption changes. Third, reserve data generation itself may embed semantic inconsistencies. In our case, a critical issue emerged when a single interest-rate variable was used both for premium pricing and for scenario-time reserve valuation. Finally, there is little scenario-aware evaluation of differentiable actuarial surrogates that explicitly checks reserve-versus-interest-rate behavior, peak reserve behavior, and checkpoint selection by business metric rather than aggregate loss alone.

### 1.4 Contributions

This paper makes five main contributions.

1. We formulate a differentiable insurance reserve framework oriented toward insurance liability digital twins rather than narrow reserve prediction alone.
2. We introduce an explicit semantic split between `pricing_interest_rate` and `scenario_interest_rate`, and show why this matters for data generation and scenario analysis.
3. We define a scenario-aware evaluation protocol that includes pointwise accuracy, PDE residual compliance, boundary behavior, monotonicity, fixed-time reserve-versus-rate curves, peak-reserve-versus-rate curves, and checkpoint progression.
4. We implement a configurable PINN/KINN loss system that combines data loss, physics loss, boundary loss, and actuarial knowledge constraints.
5. We show empirically that low aggregate validation loss and strong PDE compliance do not guarantee correct scenario behavior, motivating scenario-aware checkpoint selection and richer supervisory signals.

## 2. From classical reserve computation to differentiable liability analytics

### 2.1 Classical actuarial reserve solving

Our initial product scope is term life insurance. For a policy issued at age \(x\), with elapsed time \(t\), interest rate \(r\), premium \(P\), mortality intensity \(\mu\), and sum assured \(S\), the reserve \(V(t)\) is governed by Thiele’s equation:

\[
\frac{dV}{dt} = rV + P - \mu(S - V).
\]

We use a classical solver based on numerical ODE integration as the actuarial reference engine. This solver generates reserve trajectories used both as training labels and as evaluation benchmarks. In the current platform, it plays the role of the ground-truth reserve oracle against which the neural surrogate is compared.

### 2.2 PINN-based reserve surrogates

The neural model is trained to approximate the reserve surface as a function of time, age, pricing assumptions, scenario assumptions, premium, sum assured, and mortality. In practice, the model learns a normalized reserve ratio representation rather than raw reserve directly, which improves scale stability across heterogeneous policy sizes. The PINN portion of the training objective includes a supervised data term, a PDE residual term derived from Thiele’s equation, and a term-insurance boundary condition enforcing \(V(T)=0\) at maturity.

This formulation is attractive because it transforms reserve learning into a differentiable function approximation problem. Once trained, the model can be queried repeatedly with simple forward passes rather than repeated ODE solves. It also supports automatic differentiation, enabling direct computation of reserve sensitivities such as \(\partial V/\partial r\), \(\partial V/\partial \mu\), \(\partial V/\partial P\), and \(\partial V/\partial S\) [Baydin et al., 2018].

### 2.3 Knowledge-informed extensions

Pointwise fitting and PDE compliance alone are not sufficient for business usefulness. We therefore augment the PINN objective with knowledge-informed losses representing actuarial expectations: reserves should generally remain non-negative, should not exceed the sum assured in ordinary term settings, and should satisfy domain-informed monotonicity relations such as reserve sensitivity to mortality and interest rates. We also add scenario-oriented losses designed to align the model with shocked interest-rate behavior, including both fixed-rate-shock reserve targets and peak-reserve targets.

### 2.4 Why digital twin use cases change the evaluation problem

This shift in objective changes how model quality must be judged. For a reserve digital twin, the question is not only “does the model predict reserves accurately at sampled points?” but also “does the model behave correctly as an interactive scenario engine?” A model may display good aggregate validation loss and low PDE residual while still producing the wrong sign or magnitude of reserve changes when interest rates are varied across a fixed in-force contract. This is precisely the type of failure that conventional ML evaluation can miss.

## 3. Insurance liability digital twin requirement taxonomy

For a differentiable reserve surrogate to be useful as an insurance liability digital twin, it must satisfy several requirement classes simultaneously.

### 3.1 Reserve accuracy requirements

The system must approximate classical reserve levels with acceptable error across ages, terms, premiums, mortality states, and sum assured ranges. It must also respect terminal behavior, especially the term-insurance boundary condition at maturity.

### 3.2 Scenario fidelity requirements

The model must respond plausibly under assumption shocks. In this paper, we emphasize interest-rate scenario fidelity, but the same logic extends to mortality, inflation, lapse, and longevity shocks. Scenario fidelity is not just about sign direction; it is also about the magnitude and shape of response curves.

### 3.3 Sensitivity and differentiability requirements

One of the primary motivations for differentiable surrogates is rapid exposure analysis. The model should yield stable and interpretable first- and second-order sensitivities. In business terms, this is essential for rate sensitivity, mortality sensitivity, and nonlinear reserve exposure assessment.

### 3.4 Operational and governance requirements

The system must scale to repeated queries across portfolios and scenarios, integrate with stress-testing and optimization workflows, and remain semantically interpretable. A reserve engine that is mathematically sophisticated but semantically ambiguous is not suitable for governance-heavy actuarial environments.

## 4. Methodological building blocks

### 4.1 Synthetic policy generation and actuarial targets

The platform includes a policy simulator that generates synthetic term-life policies over configurable age, term, sum assured, and interest-rate ranges. Mortality is generated either from synthetic curves or from external mortality sources when available. The classical solver then produces reserve trajectories for each policy.

### 4.2 Semantic feature representation

The key representational change in this work is the explicit split between pricing-time and scenario-time interest rates. The model input vector now includes:

\[
[t,\; \text{age},\; r_{\text{price}},\; r_{\text{scenario}},\; P,\; S,\; \mu].
\]

This separation matters because the contract premium is determined at issue under the pricing rate, whereas reserve valuation and scenario analysis may occur later under a different market-rate environment. Treating these as a single variable causes semantic leakage between pricing and valuation.

### 4.3 Loss system

The loss framework is configuration-driven and includes:

- Data loss for supervised reserve fit
- PDE residual loss from Thiele’s equation
- Boundary loss enforcing \(V(T)=0\)
- Interest-rate monotonicity loss
- Interest-rate scenario loss for shocked reserve behavior
- Interest-rate peak loss for peak-liability behavior
- Solvency, reserve ceiling, smoothness, and other knowledge-informed terms

This design allows experiments to move between supervised, PINN, and PINN+KINN settings by editing configuration rather than rewriting training code.

### 4.4 Scenario-aware diagnostics

We extend evaluation beyond ordinary regression metrics. Specifically, we measure reserve-versus-interest-rate behavior in three forms:

1. reserve at a fixed time \(t\)
2. reserve at a baseline peak time
3. peak reserve over the trajectory

We also track how these diagnostics evolve across checkpoints during training.

## 5. Scenario-aware modeling and evaluation framework

### 5.1 The pricing-rate versus scenario-rate issue

The most important data-creation issue uncovered in this project was semantic overloading of the interest-rate variable. Initially, one interest-rate field implicitly served two incompatible roles:

1. pricing-time rate used to compute the contractual premium
2. scenario-time valuation rate used later inside the reserve equation

For an in-force contract, the premium should remain fixed unless explicitly shocked, while the scenario valuation rate may change independently. In other words, if \(P = P(r_{\text{price}})\), then under a scenario rate \(r_{\text{scenario}}\) the correct reserve experiment is:

\[
P \text{ fixed}, \qquad r = r_{\text{scenario}}.
\]

We refactored the simulator, dataset, solver, and downstream analytics to make this distinction explicit. This was not a cosmetic change; it corrected a genuine mismatch between contract economics and scenario semantics.

### 5.2 Evaluation matrix

We evaluate models using the following matrix:

- Mean squared error, mean absolute error, RMSE, and \(R^2\)
- Boundary-condition error at maturity
- PDE residual magnitude
- Monotonicity correctness
- Out-of-distribution generalization
- Fixed-time reserve-versus-rate error
- Peak-reserve-versus-rate error
- Checkpoint progression under scenario-aware metrics

This evaluation matrix is designed to reflect digital-twin readiness rather than reserve prediction alone.

## 6. Experimental results

### 6.1 Initial semantic-refactor run

After the semantic split and related pipeline fixes, the first major retraining run (`run_15_rate_semantics_split`) trained cleanly and removed the earlier numerical instability. This confirmed that the semantic correction did not destabilize training and improved interpretability. However, it did not by itself solve the scenario behavior problem. The evaluation showed good PDE residual performance but poor scenario-curve fidelity.

### 6.2 Longer rate-focused run

We then enabled interest-rate monotonicity, scenario, and peak losses more aggressively and extended training (`run_16_rate_curve_longer`). This run revealed a key empirical fact: the checkpoint with the best aggregate validation loss was not the checkpoint with the best reserve-versus-interest-rate behavior. The best overall curve checkpoint occurred around epoch 25, while the best peak-reserve checkpoint occurred around epoch 20. This implies that standard early stopping and model selection based on total validation loss alone are insufficient for digital-twin objectives.

### 6.3 Continuation after full warmup

Finally, we resumed training beyond the full constraint warmup regime (`run_17_rate_curve_resume_longer`) after correcting the trainer so that warmup would not restart on resume. Additional training improved the fixed-time reserve-versus-rate curve substantially. The best fixed-time mid-curve error improved from roughly 67.9% in run 16 to roughly 34.7% in run 17. However, the peak-reserve curve did not improve correspondingly; the best peak-reserve checkpoint remained in run 16. This suggests that ordinary reserve-rate behavior and peak-liability scenario behavior are related but not identical learning objectives.

### 6.4 Main empirical lesson

The central empirical lesson is that strong local physics compliance does not guarantee correct global scenario geometry. Our models can satisfy Thiele residual behavior reasonably well while still producing reserve-versus-rate curves that are too steep, sign-inverted, or otherwise inconsistent with the classical solver. This is especially important for business use cases such as interest-rate what-if analysis and peak-liability stress testing.

## 7. Discussion

### 7.1 What the current results show

The current system demonstrates that a differentiable actuarial reserve surrogate is feasible and useful as a research platform. It supports reserve estimation, sensitivities, stress testing, optimization, and digital twin simulation in a unified architecture. The semantically explicit representation of pricing versus scenario assumptions is a meaningful methodological improvement. Moreover, the scenario-aware evaluation protocol exposes behaviors that standard PINN evaluation would likely miss.

### 7.2 What the current results do not show

The present results do not justify the claim that the model fully replaces the classical reserve solver for scenario-critical actuarial work. In particular, peak-reserve-versus-rate behavior remains difficult, and even the improved fixed-time reserve-versus-rate behavior still exhibits material error. Therefore, the framework is best interpreted as a promising differentiable reserve analytics platform rather than a production-ready actuarial replacement engine.

### 7.3 When PINN/KINN is justified

For a simple term-insurance reserve equation alone, a PINN may not be justified. Classical methods are mathematically mature, precise, and easy to interpret. The case for differentiable surrogates becomes stronger when the use case expands to include:

- real-time multi-scenario exploration
- automatic sensitivity analysis
- direct optimization and inverse solving
- repeated portfolio-level evaluation
- digital twin simulation and executive what-if tooling

In these settings, the value proposition is not merely solving Thiele’s equation, but learning a differentiable liability surface that can be reused across many decision tasks.

### 7.4 Limitations and next steps

This work remains limited to term life insurance, synthetic policy generation, and a still-imperfect scenario-learning regime. The next improvements should likely come from richer scenario-grid supervision, explicit fixed-time curve losses, peak-time supervision, and checkpoint selection rules based on business metrics rather than aggregate loss only. These are natural next steps for turning the current framework into a more robust actuarial digital twin.

## 8. Conclusion

We presented a PINN/KINN-based framework for an insurance liability digital twin centered on term-life reserve analytics. Beyond the core reserve-learning problem, the project addressed a critical semantic issue in actuarial data creation by separating pricing-time interest assumptions from scenario-time valuation assumptions. This change aligned the simulator, solver, dataset, losses, and downstream analytics with the real business meaning of in-force reserve stress testing.

Our experiments show that differentiable reserve surrogates are promising, but they must be evaluated according to the downstream tasks they are meant to support. Low aggregate validation loss and strong PDE residual performance are not enough. Scenario-aware evaluation, especially reserve-versus-interest-rate curve analysis and checkpoint progression analysis, is necessary to determine whether a reserve model is actually suitable for digital-twin-style actuarial use. We therefore argue that the most important contribution of this work is not only the surrogate itself, but the combination of semantic data correctness, configurable actuarial constraints, and business-aligned evaluation required to make differentiable reserve analytics scientifically and operationally credible.

## References

The following references should anchor the first submission draft and BibTeX file.

- Baydin, A. G., Pearlmutter, B. A., Radul, A. A., and Siskind, J. M. (2018). *Automatic Differentiation in Machine Learning: a Survey*.
- Dickson, D. C. M., Hardy, M. R., and Waters, H. R. (2019). *Actuarial Mathematics for Life Contingent Risks*.
- Fernandez-Arjona, L. (2020). *A neural network model for solvency calculations in life insurance*.
- Gerber, H. U. (1997). *Life Insurance Mathematics*.
- Gompertz, B. (1825). *On the nature of the function expressive of the law of human mortality*.
- Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., and Yang, L. (2021). *Physics-informed machine learning*.
- Krishnapriyan, A., Gholami, A., Zhe, S., Kirby, R., and Mahoney, M. W. (2021). *Characterizing possible failure modes in physics-informed neural networks*.
- Krah, A.-S., Nikolić, U., and Korn, R. (2020). *Least-Squares Monte Carlo for Proxy Modeling in Life Insurance: Neural Networks*.
- Norberg, R. (1993). *On the application of Thiele’s differential equation in life insurance*.
- Raissi, M., Perdikaris, P., and Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*.
- Wang, S., Sankaran, S., Wang, H., and Perdikaris, P. (2023). *An Expert’s Guide to Training Physics-informed Neural Networks*.

## Appendix A. Core equation and variables

For term insurance, the governing reserve equation is:

\[
\frac{dV}{dt} = rV + P - \mu(S - V),
\]

where:

- \(V(t)\): reserve at time \(t\)
- \(t\): elapsed policy duration
- \(x\): issue age
- \(r\): scenario-time valuation interest rate
- \(P\): premium
- \(\mu\): mortality intensity
- \(S\): sum assured

## Appendix B. Key project artifacts

The following project artifacts support the empirical claims in this draft:

- scenario semantics and refactor note
- interest-rate checkpoint progression analysis
- continuation-run analysis
- reserve-versus-interest-rate plots
- validation reports and checkpoint-wise scenario metrics

# Final Dissertation Analysis Specification

## Document status

**Status:** FROZEN ANALYSIS SPECIFICATION  
**Freeze date:** 22 August 2026  
**Valuation year:** 2024  

This document defines the final planned empirical analysis for the
dissertation.

The purpose of freezing the analysis specification is to prevent
changes to the experimental design being made solely in response to
favourable or unfavourable model results.

Changes after this date should only be made where:

1. a coding or methodological error is identified;
2. an analysis proves technically infeasible;
3. the dissertation supervisor recommends a material change; or
4. an additional sensitivity analysis is clearly identified as
   exploratory.

Any such changes must be documented in the amendment log at the end
of this file.


# 1. Research objective

The dissertation investigates the robustness of actuarial and
predictive reserving methods under claims inflation, settlement
pattern change and excess-of-loss reinsurance.

The study uses simulated claim-level pseudo-data so that the true
future reserve is known and reserving-model accuracy can therefore
be evaluated directly.


# 2. Primary research question

The primary research question is:

> How do claims inflation, settlement-pattern change and
> excess-of-loss reinsurance affect the accuracy, bias and
> operational robustness of traditional and predictive reserving
> methods?


# 3. Secondary research questions

The empirical analysis will additionally investigate:

1. How does structural change in settlement behaviour affect
   reserving-model accuracy?

2. Does the application of excess-of-loss reinsurance materially
   change model applicability and performance relative to gross
   claims?

3. How do prior-based methods such as Expected Loss and
   Bornhuetter-Ferguson perform when historical development is
   unreliable or sparse?

4. Can predictive models with structural-break interactions improve
   upon models that assume a stable relationship over time?

5. Does a Tweedie specification improve modelling of sparse and
   highly variable insurance payments relative to Poisson
   regression?

6. How does the amount of historical experience used for estimation
   affect reserve accuracy under stable and changing environments?

7. How sensitive are Expected Loss and Bornhuetter-Ferguson estimates
   to prior misspecification?

8. How sensitive are ceded Bornhuetter-Ferguson estimates to the
   assumed development pattern?


# 4. Primary simulation scenarios

The primary experiment contains nine scenarios.

| Scenario ID | Tail type | Inflation | Structural break |
| --- | --- | --- | --- |
| short_stable_no_break | Short | Stable | No |
| short_emerging_no_break | Short | Emerging | No |
| short_shock_no_break | Short | Shock | No |
| long_stable_no_break | Long | Stable | No |
| long_emerging_no_break | Long | Emerging | No |
| long_shock_no_break | Long | Shock | No |
| long_stable_break | Long | Stable | Yes |
| long_emerging_break | Long | Emerging | Yes |
| long_shock_break | Long | Shock | Yes |

These nine scenarios form the primary scenario set.

Additional scenarios introduced later must be labelled as sensitivity
or exploratory analyses rather than silently added to the primary
experiment.


# 5. Reserving bases

Every model will be evaluated, where technically applicable, on:

- gross payments; and
- ceded payments after application of the excess-of-loss treaty.

Gross and ceded results will be reported separately.

Ceded results will not be assumed to possess the same statistical or
development properties as gross results.


# 6. Final model family

The final planned model family is:

| Model | Role |
| --- | --- |
| Chain Ladder | Traditional actuarial benchmark |
| Inflation-Adjusted Chain Ladder | Inflation-adjusted actuarial benchmark |
| Cashflow Uplift | Explicit inflation/cashflow adjustment |
| Expected Loss | Prior-based benchmark |
| Standard Bornhuetter-Ferguson | Prior plus standard development assumption |
| Break-Aware Bornhuetter-Ferguson | Alternative structural-break development assumption |
| Regularized Poisson | Statistical-learning benchmark |
| Regularized Poisson with break interactions | Structural-change extension |
| Regularized Tweedie | Alternative statistical-learning model |

No Random Forest, neural network or other additional machine-learning
model will be introduced unless requested by the supervisor or needed
to resolve a specific research question.

The original versions of all models will be preserved.

Later extensions must not overwrite baseline results.


# 7. Bornhuetter-Ferguson interpretation

The Standard and Break-Aware BF models will both be retained.

The Break-Aware BF model is not assumed to be the objectively correct
model.

Step 22 diagnostics showed that the accelerated benchmark development
pattern can differ materially from realised aggregate development.

The Break-Aware model will therefore be interpreted as an alternative
development-pattern specification and structural-change sensitivity.

BF performance will be analysed in terms of both:

- expected-loss prior adequacy; and
- development-pattern adequacy.


# 8. Valuation and evaluation framework

The primary valuation year is:

\[
2024.
\]

The main simulated history contains accident years:

\[
2010,\ldots,2024.
\]

Only information observable by the valuation date may enter a
reserving estimator.

Future simulated payments and true reserves may only be used after
estimation for evaluation and diagnostic purposes.

The primary baseline reserve evaluation uses the full portfolio
available under the frozen experiment.

Historical-window sensitivity analysis will use a separate fixed
evaluation target as defined below.


# 9. Simulation policy

The same evaluation portfolios should be used across competing models
wherever possible.

A common simulation ID within a scenario should represent the same
underlying simulated portfolio for every model.

This allows paired model comparisons and reduces unnecessary Monte
Carlo noise when comparing methods.

Evaluation seed schedules must not be altered because of model
performance.


# 10. Seed and information-leakage policy

The following rules are frozen:

1. Expected-loss prior calibration seeds are separate from evaluation
   seeds.

2. Evaluation seeds are shared across models where technically
   possible.

3. Hyperparameter tuning must use only information available at the
   valuation date.

4. Future reserve truth must never be used to select model
   parameters.

5. True future payments may be used only after estimation for
   performance evaluation.

6. Sensitivity experiments should reuse the same evaluation seed
   schedule wherever possible.

7. Seeds must not be rerun selectively to remove unfavourable
   outcomes.


# 11. Primary performance measures

The primary model-performance measures are:

## 11.1 Successful-fit rate

\[
\text{Success Rate}
=
\frac{\text{Successful Fits}}
{\text{Attempted Fits}}.
\]

Operational applicability is treated as an important model outcome.

## 11.2 Mean Percentage Error

\[
MPE
=
\frac{1}{n}
\sum_{s=1}^{n}
100
\frac{\hat R_s-R_s}{R_s}.
\]

MPE measures systematic reserve bias.

Positive values indicate over-reserving.

Negative values indicate under-reserving.

## 11.3 Mean Absolute Percentage Error

\[
MAPE
=
\frac{1}{n}
\sum_{s=1}^{n}
\left|
100
\frac{\hat R_s-R_s}{R_s}
\right|.
\]

MAPE is the primary conditional accuracy measure.


# 12. Secondary performance measures

Secondary metrics are:

- median percentage error;
- root mean squared error;
- standard deviation of percentage error;
- 5th percentile of percentage error;
- 95th percentile of percentage error;
- Monte Carlo standard error of mean percentage error;
- runtime where useful.

No model will be ranked purely using a single metric.


# 13. Treatment of model failures

Model failure is itself an empirical result.

Failed model fits will not be silently replaced, removed or assigned
artificial reserve estimates.

For every model/scenario/basis combination, the dissertation will
report:

1. successful-fit rate; and
2. accuracy conditional on successful estimation.

Conditional MAPE or RMSE must not be interpreted independently of
model success rate.

This is particularly important for sparse ceded triangles.


# 14. Baseline analyses

The following analyses form part of the core model-comparison study:

1. nine-scenario comparison;
2. gross versus ceded comparison;
3. inflation-regime comparison;
4. structural-break versus no-break comparison;
5. model success-rate comparison;
6. reserve bias and absolute-error comparison;
7. distributional diagnostics;
8. Expected Loss analysis;
9. Standard BF analysis;
10. Break-Aware BF analysis;
11. BF prior/development error decomposition;
12. Regularized Poisson analysis;
13. Regularized Poisson break-interaction analysis;
14. Regularized Tweedie analysis.


# 15. Historical-data-window sensitivity

A dedicated sensitivity analysis will investigate the effect of using
different amounts of historical data.

The primary history windows are:

| History length | Earliest AY | Latest AY |
| ---: | ---: | ---: |
| 15 years | 2010 | 2024 |
| 10 years | 2015 | 2024 |
| 7 years | 2018 | 2024 |
| 5 years | 2020 | 2024 |

The valuation year remains 2024.

The primary comparison target will be held fixed at:

\[
AY2020-2024.
\]

Changing the history window must therefore change the information used
for estimation without changing the reserve portfolio being evaluated.

The historical-window experiment will primarily be performed for
models whose estimates depend on observed historical experience:

- Chain Ladder;
- Inflation-Adjusted Chain Ladder;
- Cashflow Uplift;
- Regularized Poisson;
- Regularized Poisson with break interactions;
- Regularized Tweedie.

Expected Loss and Bornhuetter-Ferguson may be shown as fixed reference
benchmarks but will not be treated as equivalent historical-window
models.

The primary question is whether the benefit of greater historical
sample size outweighs the disadvantage of using potentially stale
pre-break experience.


# 16. Prior-misspecification sensitivity

Expected Loss and BF will be tested under deliberate prior
misspecification.

Primary prior multipliers are:

\[
0.8,\quad
0.9,\quad
1.0,\quad
1.1,\quad
1.2.
\]

The analysis may additionally test regime misspecification, including:

- stable-inflation prior applied to emerging inflation;
- stable-inflation prior applied to shock inflation; and
- no-break prior applied to structural-break portfolios.

The correctly specified independent prior remains the baseline.

Misspecified priors are sensitivity analyses.


# 17. Ceded-development sensitivity

The Step 22 diagnostic analysis identified material differences
between realised gross and ceded development.

A later sensitivity analysis may therefore compare:

1. BF using the existing gross-derived benchmark development pattern;
   and

2. BF using a ceded-specific development pattern independently
   calibrated from calibration simulations.

The evaluation portfolios must not be used to calibrate this
development pattern.

This analysis is classified as a sensitivity analysis rather than a
replacement of the existing BF results.


# 18. Additional simulation sensitivities

The following analyses are secondary and should only be completed if
time permits after the core analyses are finished:

- indexed versus fixed treaty terms;
- increased claim frequency / portfolio volume;
- stochastic claim-level payment patterns;
- gradual rather than abrupt structural change;
- attachment-point sensitivity.

These must be labelled as sensitivities and must not replace the
primary nine-scenario experiment.


# 19. Simulation-count and convergence analysis

The current evaluation experiment uses 50 simulations per scenario.

Final robustness analysis will investigate whether key conclusions
stabilise as the number of simulations increases.

Candidate simulation counts are:

\[
10,\quad25,\quad50,\quad75,\quad100.
\]

If computationally feasible, 100 simulations per scenario will be
used for the final robustness run.

Selected scenarios may be extended to 200 simulations if Monte Carlo
uncertainty remains material.

The principal conclusion will not be altered solely because a larger
simulation count produces a more favourable result.


# 20. Paired model comparison

Where competing models are evaluated on the same simulated portfolio,
paired error comparisons will be used.

For models \(A\) and \(B\),

\[
D_s
=
APE_{A,s}
-
APE_{B,s}.
\]

The paired analysis may report:

- mean paired difference;
- median paired difference;
- proportion of simulations in which one model has lower APE;
- 95% bootstrap confidence interval.

Accuracy comparisons will only use simulations where both models
successfully produced estimates.

Success rates will be reported separately.


# 21. Interpretation rules

The following principles will be used when interpreting results:

1. No model will be described as universally superior solely because
   it has the lowest overall MAPE.

2. Model performance must be interpreted conditional on scenario,
   basis and model applicability.

3. Numerical failure is treated as a substantive result where it
   arises from the structure of the simulated data.

4. Unexpected results will first be diagnosed rather than modified
   away.

5. A model estimate that is accurate because of offsetting
   misspecification will not automatically be interpreted as a
   correctly specified model.

6. Gross and ceded development are not assumed to be identical.

7. Structural-break awareness does not automatically imply improved
   model accuracy if the assumed response to the break is itself
   misspecified.


# 22. Reproducibility requirements

All final analyses must satisfy the following requirements:

- deterministic seed schedules are documented;
- baseline results remain frozen;
- new analyses use separate output directories;
- unit tests pass before final analysis;
- acceptance checks accompany major experiments;
- frozen baseline checksums remain valid;
- Python dependencies are recorded;
- code and methodological notes are stored in the project repository.


# 23. Scope control

No additional reserving model will be added after this specification
is frozen unless:

1. it directly answers an unresolved research question;
2. the supervisor specifically recommends it; or
3. a flaw in an existing model makes an alternative necessary.

The priority after completing the planned model and sensitivity
experiments is interpretation and dissertation writing rather than
continually increasing model complexity.


# 24. Amendment log

Any methodological change after the freeze date must be recorded
below.

| Date | Change | Reason | Classification |
| --- | --- | --- | --- |
| 22 Aug 2026 | Initial analysis specification frozen | Prevent result-driven changes to experimental design | Initial specification |
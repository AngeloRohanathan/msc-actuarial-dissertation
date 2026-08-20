# Step 18 — Distributional Diagnostics

## Run information

- Created: 2026-08-03T22:37:59.250677+01:00
- Input results: `data/final/baseline_step16/results.csv`
- Distributional summary rows: 72
- Boxplots created: 36
- Conditional-accuracy rows: 9
- No-success rows: 9

## Purpose

This analysis supplements mean model errors with distributional
diagnostics. It distinguishes systematic bias, ordinary simulation
variability and rare extreme outcomes.

The frozen Step 16 baseline results were read without being modified.

## Metric definitions

### Signed error

Signed error is:

$$
e_s = \widehat R_s - R_s^{\mathrm{true}}.
$$

A negative value indicates reserve underestimation. A positive value
indicates reserve overestimation.

### Percentage error

Percentage error is:

$$
e_{s,\%}
=
100
\frac{
\widehat R_s-R_s^{\mathrm{true}}
}{
R_s^{\mathrm{true}}
}.
$$

### Mean absolute percentage error

Mean absolute percentage error is:

$$
\operatorname{MAPE}
=
\frac{1}{n}
\sum_{s=1}^n
\left|
e_{s,\%}
\right|.
$$

### Root mean squared error

Root mean squared error is:

$$
\operatorname{RMSE}
=
\sqrt{
\frac{1}{n}
\sum_{s=1}^n e_s^2
}.
$$

### Monte Carlo standard error

For a sample metric $X_s$, the Monte Carlo standard error of its
sample mean is:

$$
\operatorname{MCSE}(\overline X)
=
\frac{s_X}{\sqrt n},
$$

where $s_X$ is the sample standard deviation and $n$ is the number of
successful simulation estimates.

## Interpretation of accuracy scope

`all_attempts_successful` means that the accuracy statistics use every
simulation attempted for that model-scenario combination.

`conditional_on_successful_fits` means that accuracy is calculated
only from the subset of simulations in which the model produced an
estimate. These results must be interpreted alongside the success
rate.

`not_available_no_successful_fits` means that no accuracy statistic
can be calculated because the model did not produce any successful
estimates.

## Boxplot interpretation

The signed-percentage-error boxplots show model bias:

- values below zero indicate underestimation;
- values above zero indicate overestimation;
- values near zero indicate low bias.

The absolute-percentage-error boxplots show accuracy:

- lower values are better;
- a wide box indicates high variability;
- distant points indicate unusually extreme simulation outcomes.

## Important limitation

For classical long-tail ceded models with low success rates, error
statistics are conditional on the rare portfolios for which a
development factor could be estimated. They are not unconditional
measures of model performance.

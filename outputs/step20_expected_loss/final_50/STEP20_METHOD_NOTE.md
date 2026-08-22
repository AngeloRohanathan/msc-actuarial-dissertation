# Step 20 — Expected-Loss Reserving

## Method

Expected-Loss reserves were calculated separately for each accident
year and for the gross and ceded bases.

For accident year \(i\),

\[
\widehat R_i^{\mathrm{EL}}
=
\max
\left(
U_i^{\mathrm{EL}}
-
P_i,
0
\right),
\]

where:

- \(U_i^{\mathrm{EL}}\) is the independently calibrated expected
  ultimate from Step 19;
- \(P_i\) is cumulative nominal paid claims observable at the valuation
  date;
- \(\widehat R_i^{\mathrm{EL}}\) is the estimated outstanding reserve.

The total portfolio reserve is

\[
\widehat R^{\mathrm{EL}}
=
\sum_i
\widehat R_i^{\mathrm{EL}}.
\]

## Prior

The expected ultimate was taken from the independent pricing Monte
Carlo calibration completed in Step 19.

The final prior used 2,000 independent calibration simulations per
scenario and is identified by:

`pricing_mc_v1`

The calibration simulations used a separate seed range from the
reserve-evaluation portfolios.

## Information available to the estimator

The Expected-Loss estimator used only:

1. the independently calibrated expected ultimate;
2. payments observed up to and including the valuation year.

Future payments and realised true reserves were not supplied to the
estimator.

True future payments were calculated only after the Expected-Loss
estimate had been produced and were used solely to evaluate model
accuracy.

## Non-negative reserve constraint

If cumulative paid claims exceeded the prior expected ultimate, the
estimated reserve was floored at zero:

\[
\widehat R_i^{\mathrm{EL}} = 0.
\]

This avoids economically meaningless negative outstanding reserves.

## Evaluation

The Expected-Loss method was applied to:

- nine simulation scenarios;
- 50 evaluation portfolios per scenario;
- gross and ceded claims.

Its performance was compared with the frozen Chain Ladder and
Regularized Poisson results using the same evaluation portfolios.

Performance measures include:

- success rate;
- mean and median percentage error;
- mean absolute percentage error;
- root mean squared error;
- error dispersion and percentiles;
- Monte Carlo standard error.
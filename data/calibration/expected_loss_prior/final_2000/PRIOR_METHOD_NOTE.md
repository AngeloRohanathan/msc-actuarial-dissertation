# Step 19 — Independent Expected-Loss Prior

## Purpose

The expected-loss prior was constructed independently of the
reserve-evaluation simulations. It will be used by the Expected Loss
and Bornhuetter–Ferguson reserving methods.

## Calibration method

The prior was estimated using an independent pricing Monte Carlo
sample consisting of 2,000 simulations per scenario.

For each calibration simulation:

1. Claim frequency and severity were simulated using the fixed
   scenario assumptions.
2. The complete lifetime nominal gross payment stream was generated.
3. The fixed per-claim excess-of-loss treaty was applied to the
   complete payment stream.
4. Ultimate gross, ceded and retained payments were aggregated by
   accident year.
5. The expected ultimate for each scenario and accident year was
   calculated as the mean across the independent calibration sample.

For basis \(b\), scenario \(q\) and accident year \(i\), the prior was:

\[
U^{\mathrm{EL}}_{q,i,b}
=
\frac{1}{M}
\sum_{m=1}^{M}
U^{(m)}_{q,i,b},
\]

where \(M=2{,}000\) is the number of independent pricing
simulations.

## Independence from evaluation data

The pricing calibration used a separate seed base from the
reserve-evaluation experiment.

No Step 16 results, realised evaluation ultimates or evaluation
reserve errors were read when constructing the prior.

The calibration and evaluation seed ranges were verified to be
disjoint.

## Exposure measure

The exposure measure is the mean simulated number of claims for the
relevant accident year and scenario across the pricing calibration
sample.

## Correctly specified prior

This first prior uses the same scenario assumptions as the
data-generating scenario. It therefore represents a correctly
specified pricing prior, but it does not use the realised outcome of
any evaluation portfolio.

A later sensitivity analysis will deliberately use misspecified
inflation or development assumptions to test how Expected Loss and
Bornhuetter–Ferguson respond to prior error.

## Monte Carlo precision

The calibration output records the Monte Carlo standard error of each
expected ultimate estimate.

The relative MCSE is:

\[
\mathrm{Relative\ MCSE}
=
\frac{
\mathrm{MCSE}(\widehat U)
}{
\widehat U
}.
\]

The intended precision targets were 2% for gross expected ultimates
and 5% for ceded expected ultimates.
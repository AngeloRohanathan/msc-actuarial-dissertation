# Step 28 — Independent Ceded Development Calibration

## Objective

This calibration regenerates the frozen Step 19 independent pricing
portfolios solely to estimate ceded paid-development patterns. It does
not alter the Step 19 expected-loss prior or use evaluation portfolios.

## Frozen simulation design

The calibration uses the original nine scenarios, 2,000 simulations
per scenario, accident years 2010–2024, calibration seed range
91,000,001–91,002,000, and the frozen £5m xs £2m per-claim XoL treaty.

## Development age

The dissertation convention is retained exactly:

\[
j = \text{payment calendar year} - \text{accident year} + 1.
\]

Reporting delay therefore contributes to development age. The
calibration retains the complete natural payment horizon and does not
truncate or renormalise at development year 10.

## Volume-weighted calibration

For calibration stratum (h) and development age (j):

\[
q^{ceded}_{h,j}
=
\frac{\sum_{m,i} C^{ceded}_{m,i,j}}
{\sum_{m,i} U^{ceded}_{m,i}},
\qquad
p^{ceded}_{h,j}
=
\sum_{k\leq j} q^{ceded}_{h,k}.
\]

Zero-ceded accident years remain in the calibration population and
contribute zero volume. No individual AY ratios are averaged.

## Strata

Short-tail and long-tail business are separate. Stable, emerging and
shock inflation scenarios remain separate because nominal inflation
and cumulative-paid XoL attachment can affect ceded payment timing.

No-break patterns use all 2010–2024 AYs in the corresponding independent
no-break scenario. Post-break patterns use only AY2018–2024 from the
corresponding independently simulated structural-break scenario.
Pre-break evaluation cohorts use the matched no-break calibration.

## Stability

Every usable pattern must have at least 1,000 calibration simulations,
5,000 AY observations, positive ceded volume and a maximum first-half
versus second-half cumulative-pattern difference no greater than 0.02.
All patterns must reconcile to their full lifetime ceded ultimates and
reach cumulative proportion one at natural terminal maturity.

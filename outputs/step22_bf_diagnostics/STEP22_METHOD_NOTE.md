# Step 22 — Bornhuetter–Ferguson Diagnostic Analysis

## Objective

The purpose of this analysis is to investigate why the standard
Bornhuetter–Ferguson specification outperformed the break-aware
specification in the structural-break scenarios.

No reserving model parameters are changed in this step.

## BF reserve estimator

For accident year \(i\),

\[
\hat{R}^{BF}_i
=
U^{prior}_i
(1-p^{assumed}_i),
\]

where \(U^{prior}_i\) is the independently calibrated Expected Loss
prior ultimate and \(p^{assumed}_i\) is the assumed cumulative paid
proportion.

The realised outstanding reserve is

\[
R^{true}_i
=
U^{true}_i
(1-p^{true}_i).
\]

## Error decomposition

The BF reserve error can be written exactly as

\[
\hat{R}^{BF}_i-R^{true}_i
=
(U^{prior}_i-U^{true}_i)
(1-p^{assumed}_i)
+
U^{true}_i
(p^{true}_i-p^{assumed}_i).
\]

The first component measures the effect of prior ultimate
misspecification.

The second component measures the effect of development-pattern
misspecification.

## True ultimate

True ultimate is calculated only for diagnostic evaluation after
the BF estimate has already been produced:

\[
U^{true}_i
=
P_i + R^{true}_i.
\]

Therefore realised future information is not used in the BF
estimator itself.

## Comparisons

Diagnostics are produced by:

- scenario;
- gross and ceded basis;
- standard and break-aware BF;
- accident year;
- development age;
- pre-break and post-break period.

The purpose is explanatory rather than model calibration.
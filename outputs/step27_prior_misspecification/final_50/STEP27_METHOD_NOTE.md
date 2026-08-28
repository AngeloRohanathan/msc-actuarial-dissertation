# Step 27 — Prior Misspecification Sensitivity

## Objective

This experiment measures how Expected Loss and paid
Bornhuetter–Ferguson reserves respond when the frozen independent
expected-ultimate prior is systematically too low or too high. Model
accuracy is a research result rather than a technical acceptance
criterion.

## Frozen prior and multiplier design

The baseline prior is the independently calibrated Step 19 pricing
Monte Carlo prior `pricing_mc_v1`. Calibration used a
seed range separate from the evaluation experiment and did not use
evaluation reserve truth.

For each basis and accident year, the sensitivity prior is

\[
U^{sensitivity}_i = m U^{baseline}_i.
\]

The configured grid is `0.80, 0.90, 1.00, 1.10, 1.20`. This run uses:
`0.80, 0.90, 1.00, 1.10, 1.20`. The same multiplier is applied to gross, ceded and
retained prior ultimates. No multiplier is calibrated from observed
evaluation errors.

## Frozen models

Expected Loss uses

\[
\widehat R_i = \max(U_i - P_i, 0).
\]

Standard and Break-Aware paid BF use

\[
\widehat R_i = U_i(1-p_i).
\]

The existing Step 20 and Step 21 implementations are called directly.
The BF pattern definitions and the structural-break accident year are
unchanged. No regime-mismatch sensitivity is included.

## Reused evaluation data and leakage safeguards

No portfolio is resimulated. The experiment reuses the frozen Step 20
accident-year rows, preserving scenario, simulation ID, basis,
paid-to-date and true reserve. BF assumptions are checked against the
frozen Step 21 detail rows. Estimator inputs are constructed without
truth; truth is merged only after each reserve estimate exists.

The source prior SHA-256 digest is checked before and after the run.
Multiplier 1.00 is compared row-by-row with the frozen Step 20 Expected
Loss and Step 21 BF portfolio results.

## Experimental design

This run contains 9 scenarios × 50 simulations ×
2 bases × 3 models × 5 multiplier value(s), giving
13,500 model attempts. Gross and ceded results remain separate.
Model failures, if any, remain explicit and accuracy is summarised only
for successful estimates.

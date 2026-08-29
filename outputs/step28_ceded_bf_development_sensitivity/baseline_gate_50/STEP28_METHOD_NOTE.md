# Step 28 — Ceded BF Development Sensitivity

## Run type

`baseline-gate`

The evaluation reuses frozen Step 21 ceded paid-to-date, independent
Step 19 expected-loss prior ultimates and true reserves. No evaluation
portfolio is resimulated.

The paid BF formula remains:

\[
\widehat R_i = U_i^{prior}(1-p_i).
\]

The baseline gate recomputes Standard and Break-Aware BF using the
original frozen Step 21 patterns. The ceded-specific final sensitivity
is not permitted until this gate is approved.

For any later ceded-specific evaluation, only (p_i) changes. Prior,
paid-to-date, truth, valuation year and accident year remain fixed.

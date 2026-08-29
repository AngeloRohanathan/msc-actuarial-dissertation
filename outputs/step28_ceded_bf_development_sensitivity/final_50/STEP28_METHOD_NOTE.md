# Step 28 — Ceded BF Development Sensitivity

## Run type

`final`

## Motivation and independent calibration

Step 22 identified development-pattern misspecification as the main
driver of the Break-Aware BF ceded under-reserving result. Step 28 tests
that diagnosis by replacing the gross-derived BF paid pattern with a
ceded-specific pattern calibrated independently from the regenerated
Step 19 pricing portfolios. Calibration seeds 91,000,001–91,002,000 are
disjoint from the frozen Step 21 evaluation seeds.

The independent pattern is volume weighted: ceded payments at each
development age are divided by total lifetime ceded ultimate for the
same calibration population. Development age remains payment calendar
year minus accident year plus one. Short-tail patterns reach natural
maturity at DY6 and long-tail patterns at DY14. Nothing is truncated or
renormalised at DY10.

Stable, emerging and shock inflation strata remain separate. Long-tail
post-break AY2018–2024 cohorts have independently calibrated post-break
patterns. Pre-break cohorts use the corresponding no-break pattern.

## Frozen evaluation

The evaluation reuses frozen Step 21 ceded paid-to-date, independent
Step 19 expected-loss prior ultimates and true reserves. No calibration
or evaluation portfolio is resimulated. The Expected Loss prior and BF
formula are frozen.

The paid BF formula remains:

\[
\widehat R_i = U_i^{prior}(1-p_i).
\]

Standard and Break-Aware BF retain the original Step 21 patterns.
Ceded-Specific BF uses the matched independent no-break pattern.
Break-Aware Ceded-Specific BF uses the matched independent post-break
pattern only for post-break AYs in structural-break scenarios. In
no-break scenarios the two ceded-specific variants are identical by
design. For valuation ages beyond natural maturity, paid proportion is
one; no extrapolation is performed.

Only (p_i) changes between corresponding variants. Prior, paid-to-date,
truth, valuation year and accident year remain fixed. Pattern calibration
does not use evaluation truth. Accuracy and paired APE comparisons are
descriptive only; formal paired inference is deferred to Step 31.

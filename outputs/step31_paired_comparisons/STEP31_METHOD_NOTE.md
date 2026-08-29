# Step 31 — Paired Model Comparisons

## Scope and pairing

Step 31 is analysis-only. It reads frozen Step 20, 21 and 24–29 result
tables and does not run simulation, reserving, calibration or model-fitting
code. Reserving comparisons pair only exact `scenario_id`, `simulation_id`
and `basis` matches, with model-, history-window- or prior-multiplier
dimensions added as required. Evaluation truth must match within fixed
absolute/relative tolerances before a pair is accepted.

For each successful reserving pair,

    D_s = APE_A,s - APE_B,s.

Negative values favour the pre-specified model or sensitivity A. Only pairs
where both estimators are structurally applicable, successfully fitted and
have finite APE enter conditional accuracy. Missing, failed and structurally
inapplicable rows remain represented in the applicability output. The APE
tie tolerance is fixed in advance at `1e-12`.

## Bootstrap interval

The 95% percentile bootstrap interval resamples the paired differences
directly with replacement. Each group uses `10000` resamples
and a deterministic group seed derived from base seed `20260831`.
An interval below zero favours A descriptively; one above zero favours B;
an interval containing zero provides no clear paired difference under this
interval. It is not a p-value. No t-test, Wilcoxon test, multiple-testing
correction or post-result comparison is added.

## Pre-specified comparisons

The analysis covers Step 24 interaction versus baseline Poisson, Step 25
Tweedie versus Poisson, Step 21 Break-Aware versus Standard BF, both Step 28
ceded-specific BF comparisons, the Step 26 10-versus-15, 7-versus-15 and
7-versus-10 history windows, and each Step 27 misspecified multiplier versus
1.00. Shorter history is A. Structurally inapplicable ML 7-year rows and all
excluded fit failures receive no accuracy statistic.

Step 29 is kept separate: fully indexed minus fixed nominal differences are
calculated for treaty mechanics on the same gross portfolio. No APE is
calculated because the treaty variants change the insured/ceded outcome
rather than estimate the same reserve truth.

All interpretations must retain success/applicability alongside conditional
accuracy. A realised APE improvement under prior misspecification does not
establish that the misspecified prior is preferable or optimally calibrated.

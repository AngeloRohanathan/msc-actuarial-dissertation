# Step 26: Historical Data-Window Sensitivity

## Objective

Step 26 investigates how the amount of historical accident-year experience
used for fitting affects reserve accuracy, responsiveness, stability, and
applicability, particularly under inflation and structural change.

## Frozen design

The four history windows are fixed at AY2010--2024 (15 years), AY2015--2024
(10 years), AY2018--2024 (7 years), and AY2020--2024 (5 years). Regardless of
the fitting window, every successful estimate and its independent truth are
evaluated only for AY2020--2024.

The six frozen models are Chain Ladder, inflation-adjusted Chain Ladder,
Cashflow Uplift, Regularized Poisson, Regularized Poisson with the existing
calendar-regime by development interaction, and Regularized Tweedie. Existing
model features, hyperparameter grids, validation requirements, links,
estimators, scenario definitions, and seeds are unchanged.

For each scenario and simulation, one portfolio is simulated and one gross and
ceded truth package is built. Every model and history window therefore uses the
same simulated portfolio, triangle basis, and AY2020--2024 evaluation truth.
Training-window filtering copies the triangle and supplies only accident years
at or after the frozen window start. Observed upper-triangle cells are retained;
future cells remain missing and never enter fitting or hyperparameter selection.

## Applicability and fitting status

With five minimum training diagonals and three minimum validation folds, a
7-year window supplies only two potential validation folds and a 5-year window
supplies none. The 7- and 5-year combinations are consequently marked
`not_applicable_by_design` for Regularized Poisson, Regularized Poisson with
break-development interactions, and Regularized Tweedie. Their estimator is
not invoked, but every required experiment row and its fixed evaluation truth
is retained. No estimate or accuracy statistic is calculated for these rows.

All classical combinations and all 10- and 15-year ML combinations invoke the
frozen estimator. Their failures are data-dependent and remain explicitly
reported. Applicability rate, unconditional success rate, and conditional fit
success rate are reported separately. Accuracy statistics use successful fits
only and are never used to redefine applicability or tune a model.

## Leakage safeguards and experiment size

No evaluation reserve truth enters model fitting or hyperparameter selection.
Rolling validation uses only historical observed cells within the requested
history window. All estimates and truths independently reconcile over the
fixed AY2020--2024 target.

The final design contains 9 scenarios x 50 simulations x 2 bases x 4 history
windows x 6 models = 21,600 required result rows. Of these, 5,400 rows are the
pre-specified structurally inapplicable ML 5- and 7-year combinations described
above. Accuracy improvement is a research result, not a technical acceptance
criterion.

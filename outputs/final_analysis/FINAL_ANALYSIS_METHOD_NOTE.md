# Step 32 — Consolidated Final Analysis Dataset

## Purpose

Step 32 is an analysis-only provenance and harmonisation layer over frozen
Step 16–31 outputs. It does not run simulations, refit reserving models,
recalibrate priors, rerun bootstrap resampling, or change any source result.
CSV is the authoritative output format for Step 33 and dissertation writing.

## Source hierarchy and authoritative baseline decisions

The original Step 16 archive is authoritative for Chain Ladder,
Inflation-Adjusted Chain Ladder, Cashflow Uplift and Regularized Poisson.
Step 20 is authoritative for Expected Loss and Step 21 for Standard and
Break-Aware Paid BF. Step 24 contributes only the Poisson break-interaction
model; its reproduced baseline Poisson rows are excluded. Step 25 contributes
only Regularized Tweedie; its Poisson comparator rows are excluded. This gives
one baseline row per scenario, simulation, basis and canonical model.

Step 26, Step 27 and Step 28 remain explicitly separate history, prior and
development sensitivities. Their baseline-like settings are retained because
they are necessary members of the frozen sensitivity designs, not substituted
for the authoritative baseline results. Step 29 is kept as treaty mechanics
and has no estimator APE. Step 31 frozen paired summaries are copied and
labelled; confidence intervals are not recomputed.

## Naming, nulls, failures and applicability

Model labels use one documented canonical mapping in `model_dictionary.csv`.
Unavailable or inapplicable fields are null: there is no zero filling,
hyperparameter invention, prior-multiplier inference, runtime imputation or
truth imputation. Successful, failed and structurally not-applicable rows are
distinct. Conditional success is calculated over attempted applicable fits;
unconditional success retains the full scheduled denominator.

AY detail is included only for compatible Expected Loss/BF result families.
Its status is joined from the matching frozen portfolio row rather than
inferred from the estimate. Step 29 index factors are described by the
minimum, maximum and mean of its own frozen AY schedule for each portfolio
variant.

## Paired comparisons and treaty mechanics

`master_paired_comparisons.csv` harmonises the three frozen Step 31 accuracy
summary tables. `inference_direction` is a deterministic label: an interval
below zero favours A, an interval above zero favours B, an interval containing
zero is `includes_zero`, and a missing interval is `not_available`. Treaty
mechanics remain in `master_treaty_sensitivity.csv` because indexation changes
the ceded outcome rather than estimates a common reserve truth.

## Provenance and validation

Every input has a repository-relative path, declared primary key, row count,
purpose, analytical level and SHA-256 digest in `source_inventory.csv`.
Digests are recorded before consolidation and checked again afterwards. The
known frozen hashes of the Step 16 results, Step 19 prior and Step 28 calibrated
ceded pattern are checked explicitly.

Validation covers source existence and stability; canonical scenarios, bases
and models; unique master and baseline keys; authoritative baseline counts;
truth consistency; reserve-error, PE and APE reconciliation; finite and
nonnegative successful estimates; failure metadata; structural applicability;
sensitivity-field scope; treaty/paired separation; and full data-dictionary
coverage. `validation_report.csv` is the machine-readable acceptance record.

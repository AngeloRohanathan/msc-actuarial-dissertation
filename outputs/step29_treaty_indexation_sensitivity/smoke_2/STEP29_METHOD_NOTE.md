# Step 29 — Simplified Treaty Indexation Sensitivity

## Scope

This `smoke` run is an optional reinsurance-mechanics sensitivity. It
does not alter the frozen baseline treaty, simulation scenarios, payment
mechanisms, reserving models, or Steps 16–28 outputs. No reserving model
is fitted in Step 29.

## Monetary and timing convention

Claim ultimate severity is generated in real 2010-base money. Each real
payment is converted to nominal money using the existing scenario-specific
claims-inflation index for its payment calendar year. XoL recovery is then
calculated from cumulative nominal paid amounts for each claim.

The simulation specification assigns treaty terms by claim accident year.
The terms are fixed for that claim's subsequent lifetime. This run therefore
uses accident-year indexation rather than payment-year or report-year
indexation. It requires no future payment or evaluation-truth information.

For accident year i, with reference year 2010:

    A_i = £2,000,000 × I_i / I_2010
    L_i = £5,000,000 × I_i / I_2010

Both terms use the same factor, preserving the real layer shape. The fixed
nominal comparator retains £2m attachment and £5m limit in every accident
year. This is a simplified stabilisation/indexation sensitivity, not a full
legal implementation of every market stabilisation clause.

## Pairing and evaluation

The run contains 2 simulations per scenario. Each gross portfolio
is simulated once per scenario/simulation and both treaty variants are applied
to that exact payment stream. Primary outputs describe ceded volume, claim
penetration, limit exhaustion, reserve truth, accident-year emergence, and
development-age payments. The Step 16 archive is read only for the fixed
nominal reproduction check.

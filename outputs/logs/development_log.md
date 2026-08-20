## Step 11 — Scenario engine validation

**Completed:** 1–2 August 2026

The simulation engine was extended to support nine permitted
combinations of portfolio type, claims-inflation scenario and
structural-change assumption.

The scenario engine supports:

- short-tail and long-tail portfolios;
- stable, emerging and temporary-shock inflation;
- an accelerated long-tail payment pattern following the
  configured structural-break accident year.

Short-tail structural-break scenarios are deliberately excluded
because the structural-change mechanism is defined specifically
for long-tail settlement behaviour.

All scenario outputs contain complete scenario labels. A separate
metadata table records the simulation ID, random seed, tail type,
frequency scenario, inflation scenario, structural-break setting,
structural-break year, Pareto parameters and simulated claim count.

The same master seed is used across alternative scenarios to
support paired comparisons. Consequently, changes in nominal
payments can be attributed to inflation or settlement assumptions
rather than unrelated changes in the underlying random portfolio.

Automated tests confirm:

- reproducibility under a fixed seed;
- variation under different seeds;
- payment-pattern reconciliation;
- correct inflation compounding;
- correct structural-break assignment;
- complete inflation-year coverage;
- no payments before reporting;
- complete scenario labelling;
- correct scenario metadata.


## Step 12 — Basic Excess-of-Loss reinsurance

A fixed, unindexed, per-claim Excess-of-Loss treaty was
implemented using pilot terms of £5 million excess of
£2 million.

Treaty recoveries were calculated on a cumulative nominal
paid basis. For each claim, cumulative ceded recoveries at
payment period k were defined as:

$$
C_{i,k}
=
\min\left[
\max\left(S_{i,k}-A,0\right),
L
\right],
$$

where $S_{i,k}$ is cumulative nominal gross paid, $A$ is the
attachment and $L$ is the treaty limit.

Incremental ceded payments were calculated as:

$$
Y_{i,k}
=
C_{i,k}-C_{i,k-1}.
$$

This approach applies the attachment and limit once per claim
rather than separately to each payment. The baseline treaty
terms remain fixed in nominal monetary amounts. Indexation and
stability clauses are introduced in a later extension.

Automated tests confirmed:

- correct hand-worked treaty outcomes;
- zero recoveries below the attachment;
- correct partial recoveries above the attachment;
- correct application of the treaty limit;
- payment-level gross, ceded and retained reconciliation;
- claim-level reconciliation with the XoL formula;
- integration with the simulated payment data.


## Step 13 — Observed triangles and true reserves

Paid triangles were constructed at the configured valuation year
of 2024. Payments with calendar year less than or equal to the
valuation year were classified as observed. Payments after the
valuation year were withheld from the reserving datasets and used
only to calculate the true outstanding reserve.

Incremental paid triangles were defined by accident year and
development year, where:

$$
j=t-i+1,
$$

with $i$ denoting accident year, $t$ payment calendar year and
$j$ development year.

Observed cells with no payment were recorded as zero. Cells
corresponding to future, unobserved calendar periods were retained
as missing values rather than being replaced with zero.

The true outstanding reserve for accident year $i$ was calculated
as:

$$
R_i^{\mathrm{true}}
=
\sum_{t>v} P_{i,t},
$$

where $v$ is the valuation year.

Separate gross and ceded incremental and cumulative paid triangles
were produced. Automated tests confirmed that:

- cumulative triangles equal cumulative sums of incremental values;
- future payments do not enter observed triangles;
- observed triangle values reconcile with observed payments;
- future payments reconcile with true reserves;
- observed and future payments reconcile with ultimate payments;
- gross and ceded triangles have identical dimensions;
- future lower-triangle cells remain missing;
- observed zero-payment cells remain zero.



## Step 14 — Classical reserving models

Three deterministic reserving methods were implemented:

1. Standard paid Chain Ladder.
2. Inflation-Adjusted Chain Ladder.
3. A stylised Cashflow Uplift benchmark.

### Standard Chain Ladder

Volume-weighted age-to-age factors were calculated as:

$$
\widehat f_j
=
\frac{\sum_i C_{i,j+1}}
{\sum_i C_{i,j}},
$$

using only accident years for which both adjacent cumulative
development observations were available.

For the gross paid triangle, all development factors were finite
and greater than or equal to one. The factors generally approached
one at later development ages.

The estimated gross reserve was approximately:

$$
£648.793\text{ million}.
$$

The true simulated gross reserve was approximately:

$$
£660.701\text{ million}.
$$

The resulting signed error was approximately:

$$
-£11.908\text{ million},
$$

equivalent to an underestimation of approximately 1.80%.

### Inflation-Adjusted Chain Ladder

Observed nominal incremental payments were restated in
valuation-year prices:

$$
P^{(v)}_{i,j}
=
P_{i,j}
\frac{I_v}{I_{i+j-1}}.
$$

Chain Ladder was fitted to the real-terms cumulative triangle.
Projected future real cashflows were then converted back to
nominal values using the forecast inflation index.

The estimated gross reserve was approximately:

$$
£649.090\text{ million}.
$$

This represented an underestimation of approximately 1.76% relative
to the true simulated reserve.

### Cashflow Uplift

Standard Chain Ladder future cashflows were adjusted using:

$$
U_t
=
\frac{I_t/I_v}
{(1+\pi_{\mathrm{embedded}})^{t-v}}.
$$

Both forecast inflation and embedded inflation were set equal to
4% in the stable pilot. The resulting uplift factor was one, so the
Cashflow Uplift estimate matched standard Chain Ladder.

### Ceded paid triangle limitation

Standard paid Chain Ladder could not be fitted to the ceded triangle.
The cumulative ceded amount at development year 1 was zero for all
accident years available for the first age-to-age factor, while
positive ceded payments emerged at development year 2.

The first ceded development factor would therefore require:

$$
\widehat f_1
=
\frac{\text{positive amount}}{0},
$$

which is undefined.

The same limitation affected Inflation-Adjusted Chain Ladder because
deflation does not change structural zeros. Cashflow Uplift also
failed because it requires an initial standard Chain Ladder
projection.

The failures were retained and documented rather than replacing
the zero denominator with an arbitrary value. This is interpreted
as a limitation of direct multiplicative paid Chain Ladder for the
simulated sparse XoL ceded triangle, rather than evidence that the
£5 million excess of £2 million treaty is invalid.

The successful gross estimates were saved to separate accident-year
result files. All six attempted model-basis combinations were
recorded in `model_status.csv`, including the three documented ceded
failures.

## Step 15 — Regularised Poisson reserving model

A regularised Poisson regression model was implemented as the
primary statistical-learning reserving method.

The response variable was incremental nominal paid claims at the
accident-year and development-year cell level. Monetary responses
were scaled into millions of pounds during model fitting and
converted back to pounds for reserve estimation.

Separate models were fitted to gross and ceded incremental paid
triangles.

The conditional mean was represented using a log link:

$$
\log\left(
\operatorname{E}[Y_{i,j}]
\right)
=
\beta_0
+
\beta_{\mathrm{AY}}x_i
+
\beta_{\pi}z_{i,j}
+
\beta_B B_{i,j}
+
\gamma_j,
$$

where:

- $x_i$ is centred accident year;
- $z_{i,j}$ is the relative log inflation index;
- $B_{i,j}$ is the structural-break indicator;
- $\gamma_j$ is a categorical development-year effect.

The model used L2 regularisation. The regularisation parameter was
selected from a predefined grid using rolling calendar-year
diagonal validation.

For validation year $v$, the training data contained only cells
with calendar year less than $v$, while validation was conducted
on the cells from calendar year $v$:

$$
\mathcal{D}_{\mathrm{train},v}
=
\{(i,j):i+j-1<v\},
$$

$$
\mathcal{D}_{\mathrm{validation},v}
=
\{(i,j):i+j-1=v\}.
$$

Random cross-validation was not used because it could place future
calendar-year cells in the training set and earlier cells in the
validation set.

All preprocessing was fitted within a pipeline separately for each
training fold, preventing validation information from influencing
feature scaling or categorical encoding.

The final model was refitted to all cells observable at the 2024
valuation date after selecting the regularisation parameter.
Predictions were then generated only for the unobserved lower
triangle.

Unlike multiplicative paid Chain Ladder, the regularised Poisson
model could accommodate zero incremental ceded cells. This enabled
a ceded reserve estimate to be produced even where the first
cumulative ceded Chain Ladder denominator was zero.

The model remains a stylised aggregate-cell model. The suitability
of the Poisson mean-variance assumption for continuous aggregate
payment amounts will be discussed as a limitation, with a Tweedie
specification identified as a possible sensitivity extension.
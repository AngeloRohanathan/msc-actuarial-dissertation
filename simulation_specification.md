# Simulation Specification

## Working dissertation title

Inflation-Aware Reserving for Excess-of-Loss Reinsurance:
A Monte Carlo Comparison of Classical and Machine-Learning Methods
under Structural Change

## Author

Angelo Rohanathan

## Version

Version 0.1 — 28 July 2026

## Status

Provisional specification. Parameters and model scope remain subject
to validation, literature review and supervisor feedback.

## 1. Research question

How do volatile claims inflation, structural changes and reinsurance indexation clauses affect the accuracy of gross and ceded reserve estimates, and when do locally reproducible machine-learning models improve upon classical reserving methods?

### Supporting research questions

1. Does standard Chain Ladder become biased when the future inflation
   environment differs from the historical inflation embedded in the
   observed claims triangle?

2. Do Inflation-Adjusted Chain Ladder and Cashflow Uplift reduce reserve
   error under emerging and shock-inflation scenarios?

3. How do excess-of-loss attachment points, limits and indexation clauses
   affect ceded claim payments and ceded reserves?

4. Are the consequences of inflation and indexation more material for
   long-tail claims than for short-tail claims?

5. How do the selected classical methods compare with a regularised
   Poisson GLM and a compact neural network?

6. How does a structural break in the settlement pattern affect the
   accuracy of each reserving method?


## 2. Unit of simulation

Each Monte Carlo simulation represents one complete hypothetical
insurance portfolio observed over multiple accident years.

Within each simulation:

- each accident year contains a random number of claims;
- each claim has an ultimate real claim amount;
- each claim has a reporting date;
- each claim may produce one or more payments;
- each payment has a development year and calendar year;
- calendar-year inflation converts real payments into nominal payments;
- the excess-of-loss treaty determines gross, ceded and retained
  payments;
- claim payments are aggregated into development triangles.

## 3. Claim-frequency distribution

### Claim Frequency Assumptions

The number of claims in accident year \(i\) is assumed to follow a Poisson distribution:

\[
N_i \sim \text{Poisson}(\lambda_i).
\]

The baseline claim frequency is assumed to be

\[
\lambda_i = 50
\]

claims per accident year.

### Frequency Scenarios

Let $i_{\min}$ denote the earliest accident year in the simulation. Define:

$$
r_i = i-i_{\min},
$$

where $r_i=0$ for the first accident year, $r_i=1$ for the second
accident year, and so on.

Three frequency scenarios are considered:

1. **Constant frequency**

   $$
   \lambda_i = 50.
   $$

2. **Decreasing frequency**

   $$
   \lambda_i = 50(0.95)^{r_i}.
   $$

3. **Increasing frequency**

   $$
   \lambda_i = 50(1.05)^{r_i}.
   $$

This definition ensures that the expected claim frequency is 50 in the
first accident year under all three scenarios.

### Purpose

The frequency scenarios test whether reserving methods designed to
address claims-severity inflation can distinguish inflation from changes
in the number of claims.

### Reference

Creedon, C., Bargate, E., Lenney, S., Schofield, M. and Stock, R. (2024), Claims Inflation Estimation: A Practical Guide for Historical Data.

Where in the paper to look

In the 2024 paper:

Pages 10–11: pseudo-data assumptions, including Poisson mean 50 and Pareto severity.
Pages 12–14: inflation and frequency scenarios.
Page 13: the ±5% frequency trends for Scenarios D1 and D2.
Appendix 1: explanation of the claims-generation tool.


## 4. Claim-severity distribution

### Baseline specification

Individual real ultimate claim amounts will follow a Pareto
distribution.

Initial parameters:

- scale or observation point: x_m = £1,000,000;
- shape parameter: alpha = 2.5.

### Rationale

A Pareto distribution allows occasional very large claims and is
therefore suitable for examining excess-of-loss reinsurance.

### Sensitivity analysis

A later sensitivity test may vary alpha to examine whether the results
change when the severity tail becomes heavier or lighter.

### Status

The parameters are provisional. Their use will be justified using the
pseudo-data literature and sensitivity testing.


## 5. Reporting-delay model

A reporting delay will be generated for each claim as a non-negative
integer number of years between the accident year and the report year.

### Short-tail reporting probabilities

| Reporting delay | Probability |
|---|---:|
| 0 years | 0.85 |
| 1 year  | 0.12 |
| 2 years | 0.03 |

### Long-tail reporting probabilities

| Reporting delay | Probability |
|---|---:|
| 0 years | 0.55 |
| 1 year  | 0.25 |
| 2 years | 0.12 |
| 3 years | 0.05 |
| 4 years | 0.03 |

### Calculation

Let $D_{i,k}$ denote the reporting delay for claim $k$ arising from
accident year $i$.

The report year is:

$$
r_{i,k}=i+D_{i,k}.
$$

Let $j=1,\ldots,J$ denote the payment period measured from the claim's
report year. The payment calendar year is:

$$
c(i,k,j)=r_{i,k}+j-1.
$$

The corresponding development year measured from the accident year is:

$$
d(i,k,j)=D_{i,k}+j.
$$

Under this convention, no payment can occur before the claim has been
reported.

### Purpose
Reporting delays allow the simulation to produce reported claim-count
information and distinguish reporting development from payment
development.

### Validation requirement

A sensitivity analysis will test whether moderate changes to the
reporting-delay probabilities materially affect the principal reserve
comparisons.

## 6. Short-tail payment pattern

The provisional short-tail incremental payment pattern is:

| Development year | Incremental percentage | Cumulative percentage |
|---|---:|---:|
| 1 | 60% | 60% |
| 2 | 25% | 85% |
| 3 | 10% | 95% |
| 4 | 5%  | 100% |

Let $X_{i,k}$ denote the real ultimate severity of claim $k$ from
accident year $i$, and let $w_j^{\mathrm{short}}$ denote the proportion
paid in payment period $j$ after reporting.

The real payment is:

$$
P_{i,k,j}^{\mathrm{real}}
=
X_{i,k}w_j^{\mathrm{short}},
$$

where:

$$
\sum_{j=1}^{4}w_j^{\mathrm{short}}=1.
$$

The payment periods are measured from the report year. Therefore, the
payment calendar year is determined using:

$$
c(i,k,j)=r_{i,k}+j-1.
$$

This pattern is a designed scenario intended to represent relatively
rapid claim settlement. It is not presented as an empirical estimate
for every short-tail line of business.

The initial simulation uses deterministic payment proportions. Every
short-tail claim follows the same expected payment pattern. This
simplification isolates the effects of inflation, reinsurance and
structural change. Stochastic variation around the payment proportions
may be considered later as a sensitivity analysis.


## 7. Long-tail payment pattern

The long-tail incremental payment pattern is:

| Development year | Incremental percentage | Cumulative percentage |
|---|---:|---:|
| 1  | 4%  | 4% |
| 2  | 20% | 24% |
| 3  | 19% | 43% |
| 4  | 14% | 57% |
| 5  | 9%  | 66% |
| 6  | 7%  | 73% |
| 7  | 6%  | 79% |
| 8  | 5%  | 84% |
| 9  | 6%  | 90% |
| 10 | 10% | 100% |

Let $w_j^{\mathrm{long}}$ denote the proportion of a long-tail claim paid
in payment period $j$ after reporting.

For a claim with real ultimate severity $X_{i,k}$:

$$
P_{i,k,j}^{\mathrm{real}}
=
X_{i,k}w_j^{\mathrm{long}},
$$

where:

$$
\sum_{j=1}^{10}w_j^{\mathrm{long}}=1.
$$

The payment periods are measured from the report year, so that:

$$
c(i,k,j)=r_{i,k}+j-1.
$$

The short-tail and long-tail scenarios use the same underlying Pareto
claim-severity distribution. They differ in their reporting-delay and
payment-timing assumptions rather than in the heaviness of the severity
tail.

The Pareto distribution governs the size of ultimate claims, while the
reporting-delay and payment-pattern assumptions govern the timing of
recognition and settlement.


## 8. Inflation scenarios

### Scenario A: Stable inflation

Annual claims inflation remains at 4% throughout the observed and
future periods:

$$
\pi_t=0.04.
$$

### Scenario B: Emerging inflation

Annual claims inflation is initially 4% and increases permanently to
6% from a specified calendar year $T_E$:

$$
\pi_t
=
\begin{cases}
0.04, & t<T_E, \\[4pt]
0.06, & t\geq T_E.
\end{cases}
$$

### Scenario C: Inflation shock and reversion

Claims inflation follows:

- 4% during the initial stable period;
- 6% during the emerging-inflation period;
- 12% in the first shock year;
- 10% in the second shock year; and
- gradual reversion towards 6% thereafter.

The exact shock years and subsequent reversion path will be fixed after
the accident-year range and valuation date have been selected.

These scenarios follow the structure of the existing volatile-inflation
scenarios.

Let $b$ denote the base calendar year for the claims-inflation index.
The index is normalised so that:

$$
I_b=1.
$$

Let $I_t$ denote the claims-inflation index in calendar year $t$. For
calendar years after the base year, the index is defined recursively by:

$$
I_t=I_{t-1}(1+\pi_t),
$$

where $\pi_t$ is the annual claims-inflation rate in calendar year $t$.

Let $P_{i,k,j}^{\mathrm{real}}$ denote the real payment for claim $k$
from accident year $i$ in payment period $j$. The payment occurs in
calendar year $c(i,k,j)$. Its corresponding nominal value is:

$$
P_{i,k,j}^{\mathrm{nominal}}
=
P_{i,k,j}^{\mathrm{real}}
\frac{I_{c(i,k,j)}}{I_b}.
$$

Inflation will be applied according to the calendar year in which each
payment is made. This means that payments from several accident years
may be affected simultaneously by the same calendar-year inflation
shock.

## 9. Structural-break scenario

A structural break will occur from a known accident year onwards.

Before the break:
- claims follow the original long-tail payment pattern.

After the break:
- claims settle more rapidly, with a greater proportion paid in the
  early development years.

The purpose is to test whether reserving methods calibrated on the old
settlement pattern misestimate claims from the new settlement regime.

| Development Year | Original Long-Tail Pattern | Accelerated Settlement Pattern |
|:----------------:|:--------------------------:|:------------------------------:|
| 1 | 4%  | 8%  |
| 2 | 20% | 27% |
| 3 | 19% | 23% |
| 4 | 14% | 16% |
| 5 | 9%  | 10% |
| 6 | 7%  | 6%  |
| 7 | 6%  | 4%  |
| 8 | 5%  | 3%  |
| 9 | 6%  | 2%  |
| 10 | 10% | 1% |
| **Total** | **100%** | **100%** |


The accelerated pattern is a deliberately designed scenario rather
than an empirical estimate. The principal requirement is that the
pattern sums to 100% and visibly moves settlement towards earlier
development years.

## 10. Excess-of-loss treaty structures

For excess-of-loss reinsurance, the attachment point and treaty limit
are applied to the cumulative amount of an individual claim rather than
separately to each individual payment.

Let $P_{i,k,j}^{\mathrm{gross}}$ denote the nominal gross payment for
claim $k$ from accident year $i$ in payment period $j$.

The cumulative gross amount paid by payment period $j$ is:

$$
S_{i,k,j}
=
\sum_{h=1}^{j}P_{i,k,h}^{\mathrm{gross}}.
$$

Let $A_{i,k}$ denote the attachment point applicable to the claim and
let $M_{i,k}$ denote the applicable treaty limit.

The cumulative ceded recovery by payment period $j$ is:

$$
G_{i,k,j}
=
\min\left\{
\max\left(S_{i,k,j}-A_{i,k},0\right),
M_{i,k}
\right\}.
$$

The incremental ceded payment in payment period $j$ is:

$$
P_{i,k,j}^{\mathrm{ceded}}
=
G_{i,k,j}-G_{i,k,j-1},
$$

where:

$$
G_{i,k,0}=0.
$$

The retained payment is:

$$
P_{i,k,j}^{\mathrm{retained}}
=
P_{i,k,j}^{\mathrm{gross}}
-
P_{i,k,j}^{\mathrm{ceded}}.
$$

For the initial model, the attachment point and treaty limit applicable
to a claim will be determined using the treaty terms associated with the
claim's accident year. Once assigned, those terms remain fixed for the
lifetime of the claim.

### Treaty calibration approach

Attachment points will be selected after generating a large baseline
sample of claim severities.

The primary treaty should:

- attach for approximately 5% to 15% of claims;
- have a non-negligible probability of limit exhaustion;
- produce enough ceded observations for reserving methods to be tested.

### Proposed treaty structures

- Treaty 1: moderate attachment, moderate limit.
- Treaty 2: higher attachment, higher limit.

The exact monetary values will be selected using baseline severity
quantiles and recorded before the final simulations are run.

The selected treaty parameters must be frozen before the final Monte
Carlo experiment and must not be adjusted after viewing model
performance.


## 11. Indexation Clause Variants

Three indexation clause structures will be considered for the excess-of-loss
reinsurance treaty.

### Clause A: No Indexation

Under this clause, the attachment point and treaty limit remain fixed
throughout the lifetime of the treaty:

$$
A_t = A_0,
\qquad
M_t = M_0.
$$

where:

- $A_0$ is the original attachment point.
- $M_0$ is the original treaty limit.

The nominal attachment point and limit do not change as claims inflation
increases. Consequently, inflation causes a greater proportion of claims
to exceed the attachment point over time, increasing reinsurance
recoveries.

---

### Clause B: Full Indexation

Under full indexation, both the attachment point and treaty limit increase
in line with the claims inflation index:

$$
A_t
=
A_0
\frac{I_t}{I_b},
$$

$$
M_t
=
M_0
\frac{I_t}{I_b},
$$

where:

- $I_t$ is the claims inflation index in calendar year $t$.
- $I_b$ is the inflation index at the base year.

Under this structure, the treaty retains approximately the same real
economic value over time because both thresholds increase with inflation.

---

### Clause C: Single-Trigger Threshold Indexation

Under threshold indexation, the treaty terms remain unchanged until
cumulative inflation from the clause base year exceeds a predefined
threshold.

Let:

$$
q_t=\frac{I_t}{I_b}
$$

denote the cumulative inflation factor from the clause base year to
calendar year $t$.

Let $\tau$ denote the threshold. The attachment point is:

$$
A_t
=
\begin{cases}
A_0, & q_t\leq 1+\tau, \\[6pt]
A_0q_t, & q_t>1+\tau.
\end{cases}
$$

The treaty limit is:

$$
M_t
=
\begin{cases}
M_0, & q_t\leq 1+\tau, \\[6pt]
M_0q_t, & q_t>1+\tau.
\end{cases}
$$

This is a stylised single-trigger clause. No adjustment is made before
the threshold is exceeded. Once the threshold is exceeded, the treaty
terms are adjusted using cumulative inflation.

For an individual claim, the applicable attachment point and limit are
determined using the indexation position in the claim's accident year.
The selected terms then remain fixed throughout the subsequent payment
development of that claim.

---

### Comparison of Clause Types

| Clause | Attachment Point | Treaty Limit | Expected Effect |
|:-------|:-----------------|:-------------|:----------------|
| No Indexation | Fixed | Fixed | Recoveries increase as inflation pushes more claims above the attachment point. |
| Full Indexation | Inflation-adjusted annually | Inflation-adjusted annually | Treaty maintains approximately constant real value. |
| Threshold Indexation | Adjusted only after cumulative inflation exceeds a threshold | Adjusted only after cumulative inflation exceeds a threshold | Intermediate behaviour between the two extremes. |


## 12. Output datasets

### A. Claim-level table

One row per simulated claim:

- simulation_id;
- claim_id;
- accident_year;
- ultimate_real_severity;
- report_delay;
- report_year;
- tail_type;
- frequency_scenario;
- inflation_scenario;
- structural_break_indicator;
- clause_type;
- treaty_id;
- clause_reference_year;
- applicable_attachment;
- applicable_limit.

### B. Payment-level table

One row per claim payment:

- simulation_id;
- claim_id;
- accident_year;
- report_delay;
- report_year;
- payment_sequence;
- development_year;
- payment_calendar_year;
- tail_type;
- frequency_scenario;
- inflation_scenario;
- structural_break_indicator;
- real_payment;
- inflation_index;
- nominal_gross_payment;
- clause_type;
- treaty_id;
- clause_reference_year;
- indexed_attachment;
- indexed_limit;
- cumulative_nominal_gross_payment;
- cumulative_ceded_recovery;
- ceded_payment;
- retained_payment.

### C. Triangle datasets

For each simulation and scenario:

- gross incremental paid triangle;
- gross cumulative paid triangle;
- ceded incremental paid triangle;
- ceded cumulative paid triangle;
- retained incremental paid triangle;
- retained cumulative paid triangle;
- reported claim-count triangle.

### D. Truth datasets

- true gross ultimate by accident year;
- true ceded ultimate by accident year;
- true gross IBNR;
- true ceded IBNR;
- true future lower-triangle payments.

### E. Observed modelling datasets

- observed gross upper triangle;
- observed ceded upper triangle;
- held-out validation diagonals.

### F. Model-results table

- simulation_id;
- scenario;
- tail_type;
- clause_type;
- treaty_id;
- reserving_method;
- true_ibnr;
- estimated_ibnr;
- percentage_error;
- absolute_error;
- squared_error;
- under_reserve_indicator;
- runtime;
- random_seed.

## 13. Validation checks

### Simulation checks

- Claim counts are non-negative integers.
- Claim severities are at least the Pareto scale value.
- Reporting delays are non-negative.
- Payment proportions sum to 100%.
- The nominal ultimate amount of a claim is defined as the sum of all
  nominal payments:

  $$
  U_{i,k}^{\mathrm{nominal}}
  =
  \sum_{j=1}^{J}P_{i,k,j}^{\mathrm{nominal}}.
  $$

- The sum of nominal payments for each claim equals its nominal ultimate
  amount.
- No payment occurs before the claim's report year.
- Payment years are no earlier than the accident year.
- The same random seed reproduces the same data.

### Reinsurance checks

- Incremental ceded payments are non-negative.
- Cumulative ceded recovery is non-decreasing.
- Cumulative ceded recovery never exceeds the claim-specific treaty
  limit.
- A claim whose cumulative gross amount remains below the attachment
  produces zero recovery.
- For every payment:

  $$
  P_{i,k,j}^{\mathrm{gross}}
  =
  P_{i,k,j}^{\mathrm{ceded}}
  +
  P_{i,k,j}^{\mathrm{retained}}.
  $$

- For every claim:

  $$
  \sum_{j=1}^{J}P_{i,k,j}^{\mathrm{ceded}}
  =
  G_{i,k,J}.
  $$

- Incremental ceded payment never exceeds the corresponding gross
  payment.
- Treaty terms remain fixed for an individual claim after the applicable
  attachment point and limit have been determined.
- Full indexation moves the attachment point and limit in the expected
  direction.
- No-indexation results remain unchanged when only the clause index is
  altered.

### Triangle checks

- Incremental triangle cells sum to total claim payments.
- Cumulative triangle rows are non-decreasing.
- The latest observed diagonal is identified correctly.
- The true lower triangle is never supplied to a reserving model.
- True IBNR equals total future lower-triangle payments.

### Experiment checks

- Every method is fitted to the same simulations.
- Hyperparameters are selected without using the true lower triangle.
- Scenario labels and random seeds are preserved.
- Final treaty and scenario parameters are frozen before final results
  are generated.


## 14. Model-comparison measures

Primary measures:

- mean percentage error, measuring systematic bias;
- median percentage error;
- mean absolute error;
- root mean squared error;
- standard deviation of reserve error;
- 5th and 95th percentiles of reserve error;
- probability of under-reserving;
- performance by accident year;
- performance for gross and ceded reserves;
- performance under short-tail and long-tail patterns;
- performance before and after the structural break;
- runtime and model convergence failures.

### Treatment of zero or very small reserves

Percentage reserve error will be calculated as:

$$
E^{\mathrm{percentage}}
=
100
\frac{\widehat{R}-R}{R},
$$

where $R$ is the true reserve and $\widehat{R}$ is the estimated reserve.

This measure will only be calculated when the true reserve exceeds a
predefined positive tolerance. It is undefined when $R=0$ and can be
unstable when $R$ is very small.

Where the true reserve is zero or below the tolerance, absolute error
and a portfolio-normalised error measure will be used.

The portfolio-normalised reserve error is:

$$
E^{\mathrm{normalised}}
=
\frac{\widehat{R}-R}
{U^{\mathrm{gross}}},
$$

where $U^{\mathrm{gross}}$ is the corresponding gross ultimate claim
amount.


Because every model will be applied to the same simulated portfolio,
model errors will be compared on a paired basis.

For example:

absolute_error_model_A - absolute_error_model_B

will be calculated for each simulation.

## 15. Assumptions requiring references

| Assumption or Method | Reference Required? | Proposed Source Type |
|-----------------------|---------------------|----------------------|
| Poisson claim frequency | Yes | Actuarial textbook or pseudo-data paper |
| Pareto severity | Yes | Actuarial severity reference |
| Baseline parameters | Yes (if retained from seminar) | Seminar framework or original paper |
| Long-tail payment pattern | Yes | Volatile-inflation study |
| Short-tail payment pattern | Explain as designed assumption | Sensitivity analysis |
| Claims-inflation scenarios | Yes | Claims Inflation Working Party |
| Chain Ladder | Yes | Academic reserving paper or textbook |
| Inflation-Adjusted Chain Ladder (IACL) | Yes | Reserving manual or original paper |
| Cashflow Uplift | Yes | Volatile-inflation study |
| Excess-of-Loss (XoL) payment formula | Yes | Reinsurance textbook or paper |
| Stability / indexation clauses | Yes | LMA, IFoA Working Party, or market sources |
| Regularised Poisson GLM | Yes | Academic paper or IFoA Working Party |
| Neural network | Yes | Academic paper or IFoA Working Party |
| Diagonal validation | Yes | Machine learning reserving paper |
| Solvency / regulatory motivation | Yes | PRA or official regulatory guidance |
| Generated observations | No | State that the data are simulated |
| Code written entirely by you | No | Document the implementation |
| Adapted seminar code or design | Yes | Acknowledge the seminar or original source |

Although the final observations will be generated by my own program,
the statistical methods, scenario designs and parameter choices must
still be referenced where they were informed by external work.

## 16. Decisions still required

- Exact accident-year range.
- Exact valuation date.
- Number of Monte Carlo simulations.
- Exact reporting-delay probabilities.
- Final short-tail payment pattern.
- Structural-break accident year.
- Final accelerated settlement pattern.
- Treaty attachment points and limits.
- Contractual index base date.
- Threshold percentage for the stability clause.
- Whether Cashflow Uplift remains a core method.
- Whether the neural network is sufficiently stable for the main
  analysis.
- Whether the reported claim-count triangle will be actively modelled or
  retained only as supporting data.
- Earliest accident year used to define $i_{\min}$.
- Claims-inflation base calendar year.
- Contractual clause-index base calendar year.
- Whether the claims-inflation base year and clause-index base year are
  identical.
- Exact emerging-inflation break year $T_E$.
- Exact inflation-shock calendar years and reversion path.
- Positive tolerance used when calculating percentage reserve errors.
- Minimum acceptable number of ceded claims per accident year.
- Maximum acceptable proportion of zero cells in the ceded triangle.
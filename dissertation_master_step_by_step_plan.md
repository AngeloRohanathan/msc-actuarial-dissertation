# Dissertation Master Step-by-Step Plan

## Purpose of this document

This is the master reference roadmap for Angelo Rohanathan's MSc dissertation project. It reconstructs the detailed workflow developed in the chat **“Dissertation Topic and Plan”**, incorporates the later implementation progress, and separates completed work from remaining work.

Use this document in future chats by stating:

> Continue from Step X of my Dissertation Master Step-by-Step Plan. First inspect the current code and outputs, then complete only that step and its acceptance checks.

Do not treat a step as complete merely because a script runs. A step is complete only when its stated outputs and acceptance checks have been satisfied and the relevant result has been documented.

---

## 1. Current project definition

### Active working title

**Inflation-Aware Reserving for Excess-of-Loss Reinsurance: A Monte Carlo Comparison of Classical and Machine-Learning Methods under Structural Change**

### Active research question

How do volatile claims inflation, structural changes and reinsurance indexation clauses affect the accuracy of gross and ceded reserve estimates, and when do locally reproducible machine-learning models improve upon classical reserving methods?

### Supporting research questions

1. Does standard Chain Ladder become biased when the future inflation environment differs from the historical inflation embedded in the observed claims triangle?
2. Do Inflation-Adjusted Chain Ladder and Cashflow Uplift reduce reserve error under emerging and shock-inflation scenarios?
3. How do XoL attachment points, limits and indexation clauses affect ceded claim payments and ceded reserves?
4. Are inflation and indexation effects more material for long-tail claims than short-tail claims?
5. How do classical methods compare with a regularised Poisson model and, only if sufficiently stable, a compact neural network?
6. How does a structural break in the settlement pattern affect each reserving method?

### Important scope history

The earlier title, **“Climate-Aware Pricing and Capital Modelling of Catastrophe Reinsurance using Extreme Value Theory and Monte Carlo Simulation,”** and its NOAA/EVT/pricing/capital plan were superseded by the reserving project above. Do not reintroduce catastrophe-event data, climate covariates, EVT pricing or capital modelling unless the dissertation scope is deliberately changed again.

### Core model set currently represented in the work

- Standard Chain Ladder.
- Inflation-Adjusted Chain Ladder (IACL).
- Cashflow Uplift.
- Regularised Poisson regression with rolling calendar-year diagonal validation.
- Expected Loss model.
- Standard Bornhuetter-Ferguson (BF).
- Break-aware Bornhuetter-Ferguson.

A compact neural network was initially proposed, but it is optional and should be included only if it is stable, adds a genuine comparison, and does not displace essential validation and writing.

---

## 2. Status legend

| Status | Meaning |
|---|---|
| Complete | Code/output exists and the principal acceptance checks have passed. |
| Complete, document | Technical work is substantially complete but dissertation prose, tables or citations still need to be written. |
| Partial / revisit | Some implementation exists, but a material issue remains. |
| Pending | Not yet evidenced as complete. |
| Optional | Include only after the core dissertation is secure. |

---

# Part A — Project framing and reproducible setup

## Step 1 — Create the project workspace and dissertation skeleton

**Status: Complete, document**

### Actions

1. Create separate folders for source code, configuration, tests, data outputs, figures, tables, notebooks or exploratory work, and the LaTeX dissertation.
2. Create a `README` explaining how to install dependencies and run the pipeline.
3. Create a pinned or recorded dependency file.
4. Create the Overleaf/LaTeX master file and chapter files.
5. Add bibliography management, cross-referencing, figure/table directories, appendices and a declaration page.
6. Keep generated data and large outputs outside the main source-code folders where practical.

### Outputs

- Reproducible Python project structure.
- Compiling LaTeX dissertation skeleton.
- README and requirements/configuration files.

### Acceptance checks

- A fresh environment can install the required packages.
- The LaTeX project compiles without missing-file errors.
- One documented command or sequence runs the code in the intended order.

---

## Step 2 — Freeze the active research scope and contribution

**Status: Complete, document**

### Actions

1. State the active title and research question.
2. Define the principal contribution as a controlled Monte Carlo comparison of reserving methods under inflation volatility, settlement-pattern change and XoL reinsurance.
3. Define the main outcomes as gross and ceded reserve accuracy, not catastrophe pricing or regulatory capital.
4. Record the distinction between simulated observations and externally sourced assumptions/methods.
5. Keep the neural network and extra frequency scenarios subordinate to the core contribution.

### Outputs

- Final scope paragraph for the introduction.
- A short contribution statement.
- Explicit inclusions and exclusions.

### Acceptance checks

- Every code module and planned dissertation chapter supports at least one research question.
- The title, methods and final results describe the same project.

---

## Step 3 — Build and maintain the literature and assumptions log

**Status: Partial / revisit**

### Actions

1. Record complete references for:
   - Poisson claim frequency;
   - Pareto severity;
   - volatile claims-inflation scenarios;
   - Chain Ladder and Bornhuetter-Ferguson;
   - IACL and Cashflow Uplift;
   - XoL reinsurance;
   - stabilisation/indexation clauses;
   - regularised Poisson regression;
   - calendar-year/diagonal validation;
   - settlement-pattern change;
   - any regulatory or market motivation.
2. Use the IFoA Claims Inflation Working Party material and the reserving seminar material as starting points, while preferring original or authoritative references in the final dissertation.
3. Record which numerical assumptions are copied, adapted or deliberately designed.
4. Add a citation in the text everywhere an external method, parameter or claim is used.

### Outputs

- Literature matrix: source, topic, method, assumption, page/section, planned chapter.
- Clean BibTeX file.
- Assumptions table distinguishing sourced assumptions from designed scenarios.

### Acceptance checks

- Every methodological and externally informed assumption has a traceable source.
- Simulated observations are described as generated by the project code and are not falsely presented as real data.
- Referencing is complete and consistent.

---

## Step 4 — Finalise the simulation specification

**Status: Complete, but parameters must be frozen before final runs**

### Actions

1. Define the simulation unit as one complete hypothetical portfolio over multiple accident years.
2. Define claim-level, payment-level, triangle, truth, observed-modelling and model-results datasets.
3. Define the valuation diagonal separating observed and future payments.
4. Record every scenario parameter in configuration rather than scattering constants through the code.
5. Maintain a specification version and change log.

### Current principal assumptions

- Claim count: Poisson, baseline mean 50 claims per accident year.
- Frequency extensions: constant, 5% annual decrease and 5% annual increase.
- Real ultimate severity: Pareto with provisional scale £1,000,000 and shape 2.5.
- Short-tail reporting delays: probabilities 0.85, 0.12 and 0.03 for delays 0, 1 and 2 years.
- Long-tail reporting delays: probabilities 0.55, 0.25, 0.12, 0.05 and 0.03 for delays 0–4 years.
- Short-tail payment pattern: 60%, 25%, 10%, 5% across development years 1–4.
- Long-tail payment pattern: 4%, 20%, 19%, 14%, 9%, 7%, 6%, 5%, 6%, 10% across development years 1–10.
- Accelerated long-tail pattern: 8%, 27%, 23%, 16%, 10%, 6%, 4%, 3%, 2%, 1%.
- Inflation: stable 4%; emerging 4% to 6%; shock path including 12%, 10% and reversion towards 6%.

### Acceptance checks

- The specification matches the actual configuration and code.
- All probability and payment vectors sum to one.
- The accident-year range, valuation date, structural-break year and simulation count are stated explicitly before final production runs.

---

# Part B — Synthetic claims and scenario engine

## Step 5 — Implement central configuration and random-seed control

**Status: Complete**

### Actions

1. Store all frequency, severity, delay, payment, inflation, break and treaty parameters centrally.
2. Use a controlled random-number generator.
3. Allocate and retain seeds by simulation and scenario.
4. Ensure model comparison uses the same simulated portfolio for every method.

### Outputs

- Configuration objects/files.
- Reproducible seed map.

### Acceptance checks

- The same seed reproduces identical claim and payment data.
- A changed seed changes the simulated portfolio.
- Scenario and seed identifiers survive through to the final results table.

---

## Step 6 — Generate accident-year claim counts and claim severities

**Status: Complete**

### Actions

1. Generate `N_i ~ Poisson(lambda_i)` by accident year.
2. Apply the selected constant or frequency-trend scenario.
3. Generate a Pareto real ultimate severity for every claim.
4. Assign unique claim and simulation identifiers.
5. Retain real severities before calendar-year inflation is applied.

### Outputs

- Claim-level dataset containing simulation, scenario, claim, accident year and real ultimate severity.

### Acceptance checks

- Claim counts are non-negative integers.
- Severities are no lower than the Pareto scale.
- Empirical claim-count means and severity quantiles are plausible for the configured distributions.
- No duplicate claim identifiers exist within a simulation.

---

## Step 7 — Generate reporting delays and real payment cashflows

**Status: Complete**

### Actions

1. Draw the reporting delay from the appropriate short- or long-tail distribution.
2. Calculate report year.
3. Allocate each claim's real ultimate severity across development years using the selected payment pattern.
4. For break scenarios, apply the accelerated pattern only to the defined post-break accident years.
5. Expand the claim table into one row per payment.

### Outputs

- Payment-level real cashflow dataset.
- Reporting information and payment-pattern labels.

### Acceptance checks

- Reporting delays are non-negative.
- Payment development and calendar years are correct.
- Claim-level payment proportions sum to 100%.
- Real payments for each claim sum to its real ultimate severity.
- Pre-break claims retain the original pattern and post-break claims use the accelerated pattern.

---

## Step 8 — Apply calendar-year inflation and structural-change logic

**Status: Complete**

### Actions

1. Construct the inflation index recursively: `I_t = I_(t-1)(1 + pi_t)`.
2. Map every payment to its calendar year.
3. Convert real payments to nominal payments with the appropriate calendar-year index.
4. Implement stable, emerging and shock/reversion paths.
5. Tag every row with inflation, tail and break information.

### Outputs

- Nominal gross payment dataset.
- Inflation index table.

### Acceptance checks

- Inflation indices are positive and follow the configured path.
- All payments in the same calendar year use the same scenario index.
- Stable inflation produces the expected compound growth.
- The structural break changes settlement timing, not claim severity.

---

## Step 9 — Build and validate the full scenario engine

**Status: Complete**

### Active core scenario structure

The implemented core engine contains nine main scenarios:

- Short-tail × stable/emerging/shock, without a structural break.
- Long-tail × stable/emerging/shock × with/without a structural break.

Frequency trend, treaty and clause variants can be layered on as sensitivity dimensions rather than multiplying the main table prematurely.

### Actions

1. Generate every required scenario from the same engine.
2. Preserve scenario names and component labels.
3. Concatenate claims and payments without collisions.
4. Produce counts and nominal-total summaries by scenario.
5. Run scenario-level validation.

### Recorded milestone evidence

- Nine scenarios were generated.
- One reported run contained 6,282 claim rows and 50,256 payment rows.
- Typical scenario claim count was 698.
- Long-tail scenarios generated 6,980 payment rows; short-tail scenarios generated 2,792.
- Break scenarios contained both original and accelerated long-tail patterns.

### Acceptance checks

- All expected scenario identifiers appear once in the scenario registry.
- Row counts agree with claim counts multiplied by the relevant number of payment periods.
- Scenario totals differ for explicable reasons.
- Every invariant from Steps 5–8 passes.

---

# Part C — Excess-of-loss reinsurance and triangles

## Step 10 — Implement claim-level XoL reinsurance allocation

**Status: Complete for the base treaty; indexation extension remains in Step 23**

### Actions

1. Compute ultimate ceded loss for claim amount `X`, attachment `A` and limit `M`:
   `C(X) = min(max(X - A, 0), M)`.
2. Calculate retained ultimate as gross less ceded.
3. Allocate ceded recovery over the claim's payment path consistently with a cumulative per-claim treaty.
4. Cap cumulative ceded recovery at the limit.
5. Flag attachment breaches and limit exhaustion.

### Recorded milestone evidence

For the pilot treaty £5m xs £2m:

- 698 claims.
- Total gross approximately £1.833bn.
- Total ceded approximately £471.95m, or 25.74%.
- 404 claims breached the attachment.
- 14 claims exhausted the limit.
- A claim with cumulative gross increasing from £1m to £9.5m produced cumulative ceded recovery from £0 to the £5m cap.

### Acceptance checks

- Ceded amounts are non-negative.
- Ceded never exceeds gross or the applicable limit.
- Gross equals retained plus ceded at payment and claim levels.
- Claims below attachment have zero recovery.
- Cumulative recovery follows the treaty correctly rather than reapplying a full layer independently to each payment.

---

## Step 11 — Construct triangles, observation masks and truth datasets

**Status: Complete**

### Actions

1. Aggregate payment rows by accident year and development year.
2. Produce gross, ceded and retained incremental paid triangles.
3. Convert each incremental triangle to cumulative form.
4. Define the valuation calendar year and observed upper triangle.
5. Store the future lower triangle separately as truth.
6. Calculate true ultimate and true IBNR by accident year and in total.
7. Optionally construct a reported claim-count triangle.

### Recorded milestone evidence

- Gross incremental, gross cumulative and ceded incremental triangles were produced.
- A deterministic test gave development factors 1.5, 1.2 and approximately 1.1111, with CDFs 2.0, 1.3333, 1.1111 and 1.0.

### Acceptance checks

- Triangle cell totals reconcile to payment-level totals.
- Cumulative triangle rows are non-decreasing.
- Gross equals ceded plus retained by cell.
- The latest observed diagonal is correct.
- True IBNR equals the sum of future lower-triangle payments.
- No modelling function can access truth rows during fitting or hyperparameter selection.

---

# Part D — Reserving models

## Step 12 — Implement standard Chain Ladder

**Status: Complete, with ceded-triangle failure behaviour requiring explicit reporting**

### Actions

1. Validate the incremental triangle and convert it to cumulative form when required.
2. Calculate volume-weighted age-to-age development factors.
3. Calculate CDFs to ultimate.
4. Project the lower triangle.
5. Calculate ultimate and IBNR by accident year and total.
6. Return diagnostics, fitted factors, convergence/failure status and runtime.

### Known issue encountered

Some ceded triangles have zero denominators for development transitions, especially sparse long-tail XoL data. Earlier runs raised `denominator is zero for development 1 to 2`, followed by an `UnboundLocalError` when code attempted to use a result that had not been created.

### Required handling

- Detect zero or invalid denominators before division.
- Do not silently replace a failed actuarial fit with a fabricated factor.
- Return a structured failure record and continue the Monte Carlo experiment.
- Ensure downstream code never references an unassigned result.
- Report success rates as substantive model-performance information.

### Acceptance checks

- The deterministic triangle reproduces known factors and CDFs.
- Stable gross scenarios produce plausible reserves.
- Sparse ceded failures are counted and surfaced.
- A failed fit does not terminate the full simulation run.

---

## Step 13 — Implement Inflation-Adjusted Chain Ladder

**Status: Complete, document and retain denominator diagnostics**

### Actions

1. Start from the observed incremental nominal triangle.
2. Deflate each observed cell using its calendar-year inflation index.
3. Reconstruct the deflated cumulative triangle.
4. Estimate development factors on the deflated triangle.
5. Project future real/deflated incremental cells.
6. Re-inflate each projected future incremental cell using its payment calendar year's forecast inflation.
7. Sum projected nominal lower-triangle cells to obtain IBNR.
8. Store both deflated and nominal diagnostics.

### Acceptance checks

- Under stable inflation, IACL and Chain Ladder are reasonably close.
- Future cells use future calendar-year indices; there is no accident-year/calendar-year mapping error.
- Deflation followed by reinflation is consistent on deterministic examples.
- No observed or future cell is double-inflated.
- Sparse ceded failures are handled as in Step 12.

---

## Step 14 — Implement Cashflow Uplift

**Status: Complete, document**

### Actions

1. Produce a baseline Chain Ladder or BF future cashflow projection by calendar year.
2. Estimate or specify the implicit historical inflation embedded in the baseline projection.
3. Apply the ratio of forecast future inflation to embedded historical inflation to each future cashflow.
4. Sum uplifted future payments to obtain the adjusted IBNR.
5. Keep the base and uplifted cashflows for auditability.

### Acceptance checks

- Under stable inflation matching the historical assumption, Cashflow Uplift equals or closely matches the baseline.
- Under emerging/shock inflation, only future cashflows receive the intended adjustment.
- The uplift is transparent by calendar year and reconciles to total adjusted IBNR.

---

## Step 15 — Implement regularised Poisson regression

**Status: Complete, document**

### Model design already represented in the code

- Response: incremental paid triangle cell.
- Numeric features: centred accident year, relative log inflation and structural-break indicator.
- Categorical feature: development year.
- Model: Poisson regression with regularisation.
- Preprocessing: scaling of numeric features and one-hot encoding of development year.
- Selection: rolling calendar-year diagonal validation rather than random train/test splitting.

### Actions

1. Convert the observed triangle into long cell-level data.
2. Create accident-year, development-year and calendar-year variables.
3. Add inflation and structural-break features without using future truth.
4. Fit candidate regularisation strengths.
5. For each validation year, train only on earlier calendar years and validate on the next diagonal.
6. Select hyperparameters using validation Poisson deviance, supported by MAE.
7. Refit on the full observed upper triangle.
8. Predict future lower-triangle cells and aggregate to IBNR.
9. Store selected hyperparameters, fold diagnostics, convergence and runtime.

### Acceptance checks

- No random row splitting is used.
- Each validation fold respects calendar time.
- Predictions are finite and non-negative.
- Hyperparameters never use the true future lower triangle.
- Gross and ceded fits have complete, inspectable diagnostics.

---

## Step 16 — Implement the Expected Loss benchmark

**Status: Complete, document**

### Actions

1. Define how the expected ultimate/prior is estimated without lower-triangle leakage.
2. Trend or inflate prior experience to the target accident year where required.
3. Combine the prior ultimate with the appropriate unpaid proportion to obtain reserve.
4. Run on gross and ceded bases.
5. Retain prior estimates and assumptions in the results.

### Acceptance checks

- The prior is reproducible and external to the future truth for the target simulation.
- The method returns results for sparse ceded triangles where appropriate.
- The dissertation explains why the benchmark may be biased under poor prior calibration.

---

## Step 17 — Implement standard Bornhuetter-Ferguson

**Status: Complete**

### Actions

1. Obtain the observed paid amount and a percentage-paid estimate by accident year.
2. Estimate an a priori ultimate using only admissible information.
3. Calculate `BF ultimate = paid to date + unpaid proportion × prior ultimate`.
4. Calculate reserve and store its prior and development components.
5. Run on gross and ceded bases.

### Acceptance checks

- Mature accident years converge towards observed/Chain Ladder experience.
- Immature accident years place more weight on the prior.
- BF reserve reconciles exactly to its component formula.
- No lower-triangle truth is used to create the operational prior.

---

## Step 18 — Implement break-aware Bornhuetter-Ferguson

**Status: Complete**

### Actions

1. Define the known structural-break year.
2. Estimate the paid development pattern for post-break cohorts using only information available at valuation.
3. Apply the break-aware unpaid proportion to affected accident years.
4. Use the same underlying truth and prior basis as the standard BF comparison.
5. Label pre-break/no-break and post-break results.

### Acceptance checks

- Standard and break-aware BF use identical simulated truth in paired comparisons.
- They are identical in no-break or unaffected periods unless the specification deliberately says otherwise.
- Differences arise from the estimated development pattern, not from different simulated portfolios.

---

## Step 19 — Complete BF decomposition diagnostics

**Status: Complete**

### Actions

1. Decompose BF reserve error into:
   - prior-error component;
   - development/unpaid-proportion error component.
2. Reconcile the components to total BF reserve error.
3. Summarise by scenario, basis, model and break period.
4. Compare standard and break-aware variants on paired rows.

### Recorded milestone evidence

The Step 22 acceptance report in the original implementation history passed all listed checks:

- decomposition reconciles;
- true paid/unpaid proportions valid;
- both BF variants present;
- variants use the same truth;
- true ultimates non-negative.

The paired diagnostic dataset contained 13,500 rows.

### Acceptance checks

- Maximum decomposition residual is only floating-point noise.
- The dissertation uses the decomposition to explain results rather than merely ranking methods.

---

## Step 20 — Treat the neural network as an optional extension

**Status: Optional / not required for the core dissertation**

### Only proceed if

- all core models and diagnostics are complete;
- the network can be trained reproducibly;
- time-respecting diagonal validation is used;
- it materially improves the research comparison;
- there is sufficient time to explain and validate it properly.

### Minimum design if retained

1. Use a compact feedforward architecture with accident year, development year and calendar year/inflation inputs.
2. Use a non-negative output transformation suitable for incremental payments.
3. Tune only through rolling diagonal validation.
4. Compare with the regularised Poisson model to determine whether non-linearity adds value.
5. Report training instability and failure rates.

### Decision rule

Exclude the network if it produces an unstable or weakly explained extra result. A rigorous smaller model set is preferable to an under-validated model added for breadth.

---

# Part E — Monte Carlo experiment and diagnostics

## Step 21 — Build the unified experiment runner

**Status: Complete**

### Actions

1. Loop through simulation seeds and the nine core scenarios.
2. Generate one portfolio per simulation/scenario.
3. Build gross and ceded observed triangles and truth datasets.
4. Apply every reserving method to the same data.
5. Catch model-level failures without losing the remaining experiment.
6. Record estimates, truth, errors, runtime, hyperparameters and status.
7. Save checkpoint outputs so a long run can resume safely.

### Acceptance checks

- Each method is paired to the same simulation/scenario/basis truth.
- There is one clear result row per attempted fit.
- Failed fits have error metadata rather than missing silently.
- Re-running a seed reproduces the same result.

---

## Step 22 — Produce comparison metrics and paired results

**Status: Complete, document**

### Required metrics

- Mean and median signed error.
- Mean and median percentage error.
- Mean and median absolute percentage error.
- Mean absolute error.
- Root mean squared error.
- Standard deviation of reserve error.
- 5th, 25th, 75th and 95th error percentiles.
- Probability of under-reserving.
- Monte Carlo standard error where useful.
- Runtime.
- Successful fits and success rate.
- Accident-year-level performance.
- Pre-/post-break performance.
- Gross versus ceded performance.
- Paired model-error differences.

### Recorded milestone evidence

- Later result tables contain 50 attempts per scenario/method/basis in the main runs.
- Gross fits generally have high success rates.
- Classical ceded Chain Ladder/IACL/Cashflow methods can have very low success rates on sparse long-tail ceded triangles, while the regularised Poisson method and prior-based methods often still fit.

### Acceptance checks

- Percentage errors use a clearly documented denominator and handle zero truth safely.
- Metrics are calculated only over successful fits, with success rates shown beside them.
- Method rankings are not compared without acknowledging different success populations.
- Paired differences use only truly paired simulations.

---

## Step 23 — Diagnose sparse ceded-triangle failure as a result, not just a bug

**Status: Partial / high priority**

### Actions

1. Quantify the proportion of zero cells and zero link-ratio denominators by scenario/development age.
2. Explain why claim-level XoL produces sparse and volatile aggregate ceded triangles.
3. Separate numerical/code failures from legitimate method non-applicability.
4. Compare success rates across methods, tail types and treaty structures.
5. Decide whether to:
   - retain strict failure as the primary classical result;
   - add a clearly labelled robustness variant with factor selection or a prior-based method;
   - aggregate or recalibrate the treaty only as a pre-specified sensitivity, not after seeing rankings.
6. Never conceal unsuccessful fits by dropping them from headline tables.

### Outputs

- Failure-rate table.
- Zero-cell/denominator diagnostic figure or table.
- Dissertation subsection on the limits of traditional aggregate methods for sparse XoL recoveries.

### Acceptance checks

- Every failure category has a count and explanation.
- The main conclusions distinguish accuracy conditional on fit from operational reliability.

---

## Step 24 — Implement and compare reinsurance indexation clauses

**Status: Pending or not yet evidenced as complete in the main results**

### Clause variants

1. **No indexation:** fixed attachment and limit.
2. **Full indexation:** attachment and limit move with the selected inflation index.
3. **Threshold/stability clause:** adjustment occurs only after the defined cumulative inflation threshold is breached, according to a precisely specified reset rule.

### Actions

1. Define the index base date and index series.
2. Define whether indexation applies by payment date, claim occurrence, notification, settlement or another contractual trigger; justify the choice.
3. Define the threshold and whether the base resets after an adjustment.
4. Apply the indexed attachment and limit consistently to cumulative claim recovery.
5. Recreate ceded and retained payments and triangles by clause.
6. Compare claim breach rates, limit exhaustion, ceded ultimate and ceded reserve accuracy.

### Acceptance checks

- No-indexation results do not change when only an unused clause index changes.
- Full indexation moves attachment and limit in the expected direction.
- Stable inflation gives interpretable ordering across clause types.
- Gross results are identical across clauses; only ceded/retained results change.
- The exact clause formula matches the code, including threshold reset behaviour.

---

## Step 25 — Calibrate and freeze treaty structures

**Status: Partial**

### Actions

1. Generate a large baseline severity sample before the final experiment.
2. Choose a primary attachment producing approximately 5%–15% claim penetration.
3. Choose a limit with non-negligible exhaustion probability.
4. Define at least one sensitivity treaty, such as a higher attachment/higher limit layer.
5. Record breach rate, exhaustion rate and ceded share.
6. Freeze monetary parameters before inspecting final method comparisons.

### Acceptance checks

- The primary treaty produces enough ceded observations to test the methods while remaining recognisably excess-of-loss.
- Treaty selections are justified using severity quantiles, not chosen to make a preferred method perform well.
- The £5m xs £2m pilot is either formally adopted and justified or clearly labelled as superseded.

---

## Step 26 — Run sensitivity and robustness analysis

**Status: Pending / partial**

### Minimum sensitivities

1. Pareto shape: heavier and lighter severity tails.
2. Reporting-delay probabilities.
3. Treaty attachment and limit.
4. Indexation clause and threshold.
5. Simulation count/Monte Carlo error.
6. Frequency trend: constant, decreasing 5% and increasing 5% if retained.
7. Structural-break year or magnitude of acceleration.
8. Alternative BF prior or development-pattern window.
9. Regularisation-grid stability for the Poisson model.

### Acceptance checks

- Sensitivities are pre-specified and limited to research-relevant questions.
- The same random seeds are used where paired comparisons are possible.
- Results distinguish conclusions that are robust from those driven by one calibration.
- Monte Carlo uncertainty is small enough for the reported method differences.

---

## Step 27 — Freeze the final experimental design

**Status: Pending final confirmation**

### Freeze all of the following

- Accident-year range and valuation date.
- Number of simulations.
- Random-seed policy.
- Frequency and severity parameters.
- Reporting and payment patterns.
- Inflation paths.
- Structural-break year and accelerated pattern.
- Treaty attachment/limit structures.
- Clause index, base date, threshold and reset rule.
- Model list and hyperparameter grids.
- Failure-handling rules.
- Primary metrics and sensitivity set.

### Acceptance checks

- A dated configuration snapshot exists.
- The final run is generated only after the snapshot.
- No parameter is changed after seeing final comparative results unless the entire affected analysis is rerun and the change is documented.

---

## Step 28 — Run the final production experiment

**Status: Partial; 50-simulation result runs exist, but final-design freeze must be confirmed**

### Actions

1. Run automated unit and integration checks first.
2. Run the full frozen scenario grid.
3. Save raw result rows before summarisation.
4. Save model diagnostics and failure records.
5. Produce a run manifest containing configuration version, timestamp, code revision and seeds.
6. Do not overwrite earlier pilot outputs without preserving a clear archive.

### Acceptance checks

- Expected row counts reconcile across scenarios, simulations, bases and models.
- All scenarios have truth and result records.
- The run can be reproduced from the manifest.
- No leakage check fails.

---

## Step 29 — Create final tables and figures

**Status: Pending / partial**

### Essential tables

1. Simulation assumptions and scenarios.
2. Treaty and clause definitions.
3. Model definitions and validation schemes.
4. Fit success rates.
5. Primary gross-reserve metrics.
6. Primary ceded-reserve metrics.
7. Paired model comparisons.
8. Structural-break and BF decomposition results.
9. Sensitivity results.

### Essential figures

1. Inflation paths by calendar year.
2. Original and accelerated payment patterns.
3. Example gross/ceded triangles.
4. Reserve-error distributions by method and scenario.
5. Bias/MAE or RMSE comparison across scenarios.
6. Success-rate or zero-cell diagnostic for ceded triangles.
7. Pre-/post-break accident-year errors.
8. Clause effect on ceded payments/reserves if Step 24 is retained.

### Acceptance checks

- Every table and figure answers a research question.
- Captions are self-contained and state basis, scenario and units.
- Scales and colours are consistent.
- Failed fits and differing sample sizes are visible.
- Excess output is moved to appendices rather than overwhelming the results chapter.

---

## Step 30 — Interpret results and form conclusions

**Status: Pending final synthesis**

### Required interpretation sequence

1. Begin with the stable/no-break baseline as a validity check.
2. Isolate the effect of changing inflation without a break.
3. Isolate the structural-break effect within each inflation path.
4. Compare short- and long-tail business.
5. Compare gross and ceded results.
6. Discuss both conditional accuracy and probability of successful application.
7. Use the BF decomposition to explain whether bias comes from the prior or development pattern.
8. Evaluate whether the regularised Poisson model adds robustness or accuracy.
9. Evaluate the economic effect of indexation clauses if implemented.
10. State limitations without weakening valid conclusions.

### Acceptance checks

- Claims are supported by tables/figures and uncertainty measures.
- No causal claim is made beyond the controlled simulation design.
- Results are compared on the same basis and same paired simulations.
- The conclusion answers each research question directly.

---

# Part F — Dissertation writing

Writing should proceed in parallel with technical work. Do not wait for every result before drafting methods and background.

## Step 31 — Write Chapter 1: Introduction

**Status: Partial / revisit**

### Content

1. Claims inflation and reserving motivation.
2. Why long-tail claims and XoL reinsurance are sensitive to inflation and timing.
3. Problem with implicit historical assumptions in classical reserving.
4. Structural-break motivation.
5. Research gap and contribution.
6. Research questions.
7. Scope and chapter outline.

### Acceptance checks

- The introduction matches the active reserving title.
- It does not promise catastrophe/EVT pricing or capital modelling.
- Contributions are specific and testable.

---

## Step 32 — Write Chapter 2: Literature Review and Background

**Status: Partial / revisit**

### Content

1. Claims inflation: economic, social and superimposed inflation.
2. Paid triangles and calendar/development/accident-year structure.
3. Chain Ladder and its implicit stability assumptions.
4. Inflation-adjustment approaches, focusing on IACL and Cashflow Uplift.
5. Expected Loss and Bornhuetter-Ferguson.
6. Settlement-pattern change and structural breaks.
7. XoL reinsurance and sparse ceded data.
8. Stabilisation/indexation clauses.
9. Statistical learning in reserving and time-respecting validation.
10. Gap addressed by the study.

### Acceptance checks

- The chapter synthesises sources rather than listing them.
- Each method's assumptions and limitations lead into the methodology.
- The literature gap is credible and not overstated.

---

## Step 33 — Write Chapter 3: Methodology

**Status: Partial / substantial drafting required**

### Content

1. Simulation unit and data flow.
2. Frequency and severity models.
3. Reporting and payment processes.
4. Inflation index and scenarios.
5. Structural break.
6. XoL recovery and indexation clauses.
7. Triangle construction and valuation cut-off.
8. Truth definition.
9. Every reserving method, with equations.
10. Rolling diagonal validation.
11. Monte Carlo design and pairing.
12. Metrics, failure handling and sensitivity analysis.

### Acceptance checks

- A reader could independently reproduce the study.
- Equations match the implementation.
- The distinction between observed data, validation diagonals and hidden truth is unambiguous.
- Long code listings are left to the appendix/repository; the chapter explains algorithms.

---

## Step 34 — Write Chapter 4: Simulation and Model Validation

**Status: Pending / partial**

### Content

1. Distribution and parameter checks.
2. Payment and inflation reconciliation.
3. XoL deterministic examples.
4. Triangle reconstruction checks.
5. Known-factor Chain Ladder test.
6. IACL deflation/reinflation test.
7. Poisson validation-fold diagnostics.
8. BF decomposition reconciliation.
9. Failure-handling and reproducibility checks.

### Acceptance checks

- Validation is shown before performance results.
- Deterministic examples establish that the code implements the intended formulas.
- Important failures are documented, not hidden.

---

## Step 35 — Write Chapter 5: Results

**Status: Partial / final synthesis pending**

### Recommended order

1. Baseline stable/no-break results.
2. Emerging and shock inflation.
3. Structural-break results.
4. Short-tail versus long-tail.
5. Gross versus ceded.
6. Method reliability/success rates.
7. Expected Loss and BF comparisons.
8. BF error decomposition.
9. Reinsurance clause analysis.
10. Sensitivity analysis.

### Acceptance checks

- The chapter reports results before giving broader interpretation.
- All metrics have units and sample sizes.
- Tables are not truncated in LaTeX and figures remain legible.

---

## Step 36 — Write Chapter 6: Discussion

**Status: Pending**

### Content

1. Why each method behaved as observed.
2. Practical significance for reserving actuaries and reinsurers.
3. Accuracy versus operational reliability.
4. Effect of inflation and settlement-pattern misspecification.
5. Consequences of sparse ceded data.
6. Value and limits of regularisation and prior-based methods.
7. Implications of indexation clauses.
8. Limitations of synthetic data and simplified assumptions.
9. Future research.

---

## Step 37 — Write Chapter 7: Conclusion

**Status: Pending**

### Content

1. One direct answer to each research question.
2. The most important numerical findings.
3. The main methodological contribution.
4. Practical implications.
5. Limitations and concise future-work statement.

### Acceptance checks

- No new evidence or literature appears in the conclusion.
- Conclusions match the statistical results and acknowledge fit failures.

---

## Step 38 — Complete appendices and reproducibility documentation

**Status: Pending / partial**

### Content

- Full parameter tables.
- Additional validation and sensitivity tables.
- Model hyperparameters.
- Pseudocode or selected code excerpts.
- Run manifest and seed policy.
- Supplementary results.
- Repository/file guide.

### Acceptance checks

- Main chapters remain readable without losing reproducibility.
- Every appendix item is referenced in the main text.

---

## Step 39 — Perform technical and academic quality assurance

**Status: Pending**

### Code QA

1. Run all tests.
2. Run the full pipeline from a clean environment.
3. Check for hard-coded local paths.
4. Confirm no truth leakage.
5. Confirm configuration and final results agree.
6. Check seeds and result row counts.
7. Inspect warnings and failure logs.

### Dissertation QA

1. Compile LaTeX from scratch.
2. Resolve overfull/underfull boxes, truncated tables and broken references.
3. Check equations against code.
4. Check all figures, captions, labels and units.
5. Check bibliography completeness and consistency.
6. Check page numbering, title page, abstract and signed declaration.
7. Keep the main body near the expected 30–40 pages of text/formulas and the full submission within the applicable limit.
8. Ensure all work and assistance are handled in accordance with QMUL academic-integrity rules.

---

## Step 40 — Final submission package

**Status: Pending**

### Actions

1. Freeze the final PDF and code version.
2. Verify the submitted PDF opens and every page renders correctly.
3. Preserve the exact final configuration, raw results, generated tables/figures and code revision.
4. Submit before the official deadline using the required platform and naming convention.
5. Retain submission confirmation.

---

# 3. Immediate next-step priority order

Based on the recovered progress, the recommended continuation order is:

1. **Step 23:** fully diagnose and document sparse ceded-triangle failures.
2. **Steps 24–25:** implement/finalise indexation clauses and freeze treaty calibration, or formally remove clause comparison from the research question if it will not be completed.
3. **Step 26:** run a limited, pre-specified robustness set.
4. **Step 27:** freeze the complete experimental design.
5. **Step 28:** run the final production experiment.
6. **Steps 29–30:** create final outputs and synthesize the results.
7. **Steps 31–40:** finish, integrate and quality-check the dissertation.

The neural network in Step 20 should remain optional until these items are secure.

---

# 4. Master data flow

1. **Configuration and seed**
2. **Accident-year claim count**
3. **Claim-level real ultimate severity**
4. **Reporting delay and payment pattern**
5. **Payment calendar year**
6. **Calendar-year inflation**
7. **Nominal gross payments**
8. **XoL treaty and indexation clause**
9. **Ceded and retained payments**
10. **Gross/ceded/retained triangles**
11. **Observed upper triangle and hidden lower-triangle truth**
12. **Reserving models and diagonal validation**
13. **Estimated IBNR versus true IBNR**
14. **Paired metrics, failure diagnostics and sensitivities**
15. **Tables, figures and dissertation conclusions**

---

# 5. Non-negotiable rules for future coding chats

1. Inspect the current code and configuration before proposing replacements.
2. Preserve user code and unrelated changes.
3. Work on one named roadmap step at a time.
4. State the input, output and acceptance tests before changing code.
5. Do not use the true lower triangle for training, tuning, priors or model selection.
6. Do not use random train/test splits for triangle cells; use calendar-year diagonal validation.
7. Apply all models to the same simulations and preserve pairing.
8. Treat model failure rates as results.
9. Do not tune treaty/scenario assumptions after viewing final performance.
10. Keep code, equations, parameter tables and dissertation prose consistent.
11. Add or update tests whenever a bug is corrected.
12. Record the exact command and outputs used to declare a step complete.

---

# 6. Completion snapshot recovered from the original work

| Area | Recovered status |
|---|---|
| Project/LaTeX structure | Created; writing and formatting remain. |
| Simulation specification | Version 0.1 completed; final parameters still need freezing. |
| Claims/payment generation | Completed and validated. |
| Nine-scenario engine | Completed through the earlier Step 9 milestone. |
| Base XoL pilot | Completed and reconciled. |
| Triangle construction/truth split | Completed. |
| Chain Ladder | Implemented; sparse ceded failure behaviour identified. |
| IACL | Implemented; denominator handling/interpretation remains important. |
| Cashflow Uplift | Implemented. |
| Regularised Poisson | Implemented with rolling diagonal validation. |
| Monte Carlo summaries | 50-attempt scenario/model tables exist. |
| Expected Loss | Present in later results. |
| Standard and break-aware BF | Implemented. |
| BF diagnostics | Completed and passed the recorded acceptance report. |
| Clause comparison | Defined in specification; completion not evidenced in the recovered main result tables. |
| Treaty/sensitivity freeze | Not yet confirmed. |
| Final production run | Existing runs may be pilots; final-design freeze is not yet confirmed. |
| Final tables/figures and chapters | Incomplete. |

This snapshot should be updated whenever a step is completed so future chats begin from the actual project state rather than recreating earlier work.

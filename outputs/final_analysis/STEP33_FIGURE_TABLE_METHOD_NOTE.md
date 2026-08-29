# Step 33 — Final Dissertation Figures and Tables

## Scope and source

Step 33 is presentation-only. Its sole analytical source is the frozen Step 32
directory `outputs/final_analysis/`. No simulation, reserving fit, calibration,
bootstrap resampling, paired comparison or treaty experiment is rerun. The
Step 32 source files are hashed before and after generation.

Eight main figures and seven concise main tables were selected to address the
research questions without placing every available result in the dissertation.
Six detailed descriptive tables are retained as appendix candidates. Every
main output has a CSV backing file and a metadata entry in
`step33_output_index.csv`.

## Accuracy, failure and applicability

MPE and MAPE are calculated from successful frozen estimator rows only and are
therefore conditional accuracy measures. Attempts, structural applicability
and unconditional success are presented separately. A failed or structurally
not-applicable estimator is never assigned an accuracy value, and a model is
not presented as superior solely because failures are excluded.

## Paired evidence

Figure/Table 4 select only pre-existing Step 31 comparisons for ceded
structural-break scenarios. The plotted quantity is frozen mean paired APE
difference A minus B, in percentage points, with the frozen 95% bootstrap
interval. Below zero is labelled “Favours A”, above zero “Favours B”, and an
interval containing zero “Includes zero”. No new inference or significance
language is introduced.

## Ordering and labels

The canonical nine-model order from Step 32 is retained. Scenario labels use:

- `short_stable_no_break` → Short / Stable
- `short_emerging_no_break` → Short / Emerging
- `short_shock_no_break` → Short / Shock
- `long_stable_no_break` → Long / Stable
- `long_emerging_no_break` → Long / Emerging
- `long_shock_no_break` → Long / Shock
- `long_stable_break` → Long / Stable + Break
- `long_emerging_break` → Long / Emerging + Break
- `long_shock_break` → Long / Shock + Break

## Units and rounding

Reserve and ultimate amounts are GBP; table presentation may express them in
GBP millions. MPE/MAPE are percentages. Differences between APE, MAPE, ceded
shares, or rates are percentage points. Backing CSVs retain numerical precision;
rounding to two decimals is a dissertation typesetting recommendation rather
than a mutation of source data.

## Treaty limitation and appendix policy

Step 32 intentionally provides Step 29 treaty mechanics at portfolio/scenario
level, not ceded share by accident year. Figure 8 therefore compares fixed and
indexed scenario-level mean ceded shares. No Step 29 folder is consulted and no
AY trajectory is invented. Appendix tables remain candidates and need not all
appear in the submitted dissertation.

# Codebase Cleanup Report

## A. Branch

- Source branch: `codex-dissertation-work`
- Cleanup branch: `dissertation-code-cleanup`
- Starting commit: `22f552c` (`Add final dissertation figures and tables`)
- Commit, merge, and push status: none performed

The source worktree initially contained two untracked cleanup candidates. A
repository-wide dependency search found no external references, so they were
removed before the cleanup branch was created, as explicitly authorized:

- `outputs/step25_regularized_tweedie/final_50/STEP25_FINAL_SUMMARY.xlsx`
- `outputs/step27_prior_misspecification/baseline_gate_1_00/`

## B. Code audit

The audit covered `config.py`, all 20 `src` modules present at the starting
commit, all 22 script modules, all 23 test modules, the three specifications,
the master step-by-step plan, README, requirements, 101 tracked data files, and
290 tracked output files.

Checks included repository-wide fixed-string and symbol searches, Python AST
inspection, import-use analysis, public-docstring coverage, large-function and
duplicate-body reports, source-inventory and Step 33 index reconciliation,
checkpoint hashing, and Git diff review.

Findings:

- `src/figures.py` was an empty, unimported legacy placeholder referenced only
  by the stale README.
- Fifteen imports had no load/reference in their modules.
- No production function or class had strong evidence of being dead. None was
  removed.
- Exact duplicate helpers exist in the Step 16-style and Step 24 runners, but
  consolidation was rejected as an unnecessary methodological risk.
- Several core modelling and validation functions are large. They were left
  intact because restructuring would require expensive scientific reruns to
  provide proportionate assurance.

Classification policy:

- KEEP: frozen sources, final experiment outputs, final-analysis outputs,
  required gate evidence, tests, diagnostics, and uncertain artifacts.
- REFACTOR: analysis-only copy handling, unused imports, documentation, and
  README.
- REMOVE: the empty module, verified-unused smoke artifacts, and byte-identical
  completed-run checkpoints.
- UNCERTAIN: retained without modification.

## C. Code removed

- `src/figures.py`: zero-byte placeholder; no import, test, CLI, Step 32, or
  Step 33 dependency. README was corrected to point to the actual reporting
  modules and final-analysis locations.
- Fifteen unused imports were removed across source, runners, and two tests.
- No model, calibration, simulation, estimator, validation function, class, or
  test was deleted.

## D. Code refactored

- `src/final_results_consolidation.py`
  - Existing columns are no longer copied by `_column_or_na`; callers only read
    or assign the returned Series.
  - `harmonise_accident_year_results` no longer copies each large frozen AY
    source before building a separate output frame.
  - The portfolio status lookup no longer makes an unused defensive copy.
  - Risk: low and confined to Step 32 analysis. Exact Step 32/33 CSV equality
    and the full test suite verified behaviour.
- Import cleanup affected no executable expressions or constants.
- No architectural rewrite was undertaken.

## E. Documentation improvements

- Replaced the pilot-stage README with a concise technical guide covering the
  completed workflow, repository structure, tests, Step 32/33 entry points,
  dissertation artifact locations, and reproducibility safeguards.
- Added package docstrings to `src` and `scripts`.
- Added concise purpose/convention docstrings to previously undocumented public
  experiment helpers and runners.
- Added focused documentation to Step 32 consolidation/validation helpers and
  Step 33 figure/table generation helpers.
- Production AST audit after cleanup: no missing module docstrings and no
  undocumented public functions/classes.

## F. Efficiency improvements

Previous issue: Step 32 made full or partial defensive copies of already-frozen
DataFrames that were not mutated.

Change: removed the three unnecessary copy operations described in Section D.

Expected benefit: lower transient memory allocation during accident-year
harmonisation, especially for the 270,000-row Step 32 AY output. No performance
number is claimed because no isolated benchmark was run.

No changes were made to randomness, floating-point calculation order,
optimisation, model grids, tolerance values, algorithms, or parallelisation.

## G. Output cleanup

`cleanup_output_inventory.csv` is the file-level decision ledger with all
required columns. It records 220 KEEP and 72 REMOVE decisions, including the
two authorized untracked candidates.

Removed tracked outputs:

- 70 files / 16,962,790 bytes (about 16.96 MB, or 16.18 MiB).
- Eight generated smoke directories:
  - `outputs/step20_expected_loss/smoke_test/`
  - `outputs/step21_paid_bf/smoke_break/`
  - `outputs/step21_paid_bf/smoke_short/`
  - `outputs/step24_poisson_break_interaction/smoke_break/`
  - `outputs/step24_poisson_break_interaction/smoke_no_break/`
  - `outputs/step25_regularized_tweedie/smoke_all_9_scenarios/`
  - `outputs/step26_history_window_sensitivity/smoke_1/`
  - `outputs/step29_treaty_indexation_sensitivity/smoke_2/`
- Three completed-run checkpoints whose SHA-256 digests were identical to the
  retained `results.csv` in the same directory:
  - `outputs/step24_poisson_break_interaction/final_50/results_checkpoint.csv`
  - `outputs/step25_regularized_tweedie/final_50/results_checkpoint.csv`
  - `outputs/step26_history_window_sensitivity/final_50/results_checkpoint.csv`

The two initially untracked candidates were also removed. Their size was not
included in the measured tracked-byte total.

The 70 exact tracked output paths and the two untracked candidate paths are
listed individually as REMOVE rows in `cleanup_output_inventory.csv`.

The Step 28 `baseline_gate_50` and Step 29 `baseline_gate_2` directories were
explicitly retained: their acceptance reports are required default inputs to
the corresponding final-run reproducibility workflows.

## H. Outputs explicitly retained for the dissertation

Step 32 datasets retained:

- `outputs/final_analysis/master_results.csv`
- `outputs/final_analysis/baseline_model_results.csv`
- `outputs/final_analysis/master_by_accident_year.csv`
- `outputs/final_analysis/master_treaty_sensitivity.csv`
- `outputs/final_analysis/master_paired_comparisons.csv`
- `outputs/final_analysis/master_applicability.csv`

Step 33 retained in full:

- `outputs/final_analysis/step33_figures/`: 8 PNGs, 8 PDFs, and 11 backing CSVs
- `outputs/final_analysis/step33_tables/`: 7 main tables
- `outputs/final_analysis/step33_tables/appendix/`: 6 appendix tables
- `outputs/final_analysis/step33_output_index.csv`
- `outputs/final_analysis/STEP33_MANIFEST.json`
- `outputs/final_analysis/STEP33_VALIDATION_REPORT.csv`
- `outputs/final_analysis/STEP33_FIGURE_TABLE_METHOD_NOTE.md`

All 19 frozen sources in `source_inventory.csv`, all final experiment outputs
used by Step 32, the Step 28 final-2000 calibration, manifests, method notes,
acceptance reports, and analytical Step 18/22 diagnostics were retained.

## I. Numerical verification

Before cleanup, the complete `outputs/final_analysis` tree and SHA-256 digests
for all frozen sources/final artifacts were copied to:

`C:/Users/oumac/AppData/Local/Temp/msc-actuarial-dissertation-cleanup-22f552c/`

Step 32 and Step 33 were rebuilt from scratch into the repository and compared
with the frozen copy:

- CSV path sets: identical
- CSV files compared: 36
- schemas/dtypes: 36/36 exact
- row order and values: 36/36 exact
- CSV bytes: 36/36 exact
- PNGs generated: 8/8; dimensions identical
- PDFs generated: 8/8; page dimensions identical and all non-empty
- Main tables: 7/7
- Appendix tables: 6/6

Transient byte differences were limited to the two timestamp/elapsed-time JSON
manifests and PDF metadata. The original frozen final-analysis tree was restored
after comparison, leaving no Git differences in `outputs/final_analysis/`.

Critical hashes before and after:

- Step 16: `bf1069b9d85516611851d36f09fa2b3dece7d1f887c0280f8afcef84f1e9fb42`
- Step 19: `06c23adae7ab9d96e0fd657d3af173a0bf05f91bd00e55d911c08d8459023912`
- Step 28: `a953f05567a417fc287c0cc8263b7668876e1490e42737b8409deb0e9c19468e`

No floating-point differences occurred; maximum absolute and relative
differences are both zero across regenerated CSV outputs.

## J. Tests and static validation

- Focused Step 32/33 tests: 35 passed in 1.46 seconds
- Full suite: 248 passed in 11.25 seconds
- `python -m compileall -q src scripts tests`: PASS
- `git diff --check`: PASS
- Step 32 validation: 41/41 PASS
- Step 33 validation: 23/23 PASS
- Static unused-import recheck: no candidates after excluding intentional
  `from __future__ import annotations`
- Protected paths missing: 0
- Deleted paths referenced by Step 32/33: 0

## K. Remaining cleanup opportunities

Deliberately not performed:

- Consolidation of duplicated scenario/inflation/reconciliation helpers across
  historical experiment runners.
- Decomposition of large simulation, model-fitting, calibration, and acceptance
  functions.
- Removal of pilot data, Step 18/22 diagnostics, baseline gates, or any artifact
  whose dissertation/reproducibility value was uncertain.
- Reruns of the 2,000-simulation calibration or other multi-hour experiments.
- Formatting-only rewrites of the mature modelling modules.

These opportunities carry more review or numerical-revalidation cost than is
justified in a no-methodology-change cleanup.

## L. Exact files modified or deleted

Modified/new non-output files:

- `README.md`
- `CODEBASE_CLEANUP_REPORT.md`
- `cleanup_output_inventory.csv`
- `scripts/__init__.py`
- `scripts/build_final_analysis_dataset.py`
- `scripts/build_final_figures_tables.py`
- `scripts/run_bf_diagnostics.py`
- `scripts/run_ceded_bf_development_sensitivity.py`
- `scripts/run_end_to_end_pilot.py`
- `scripts/run_expected_loss_experiment.py`
- `scripts/run_history_window_sensitivity.py`
- `scripts/run_paid_bf_experiment.py`
- `scripts/run_paired_comparisons.py`
- `scripts/run_poisson_break_interaction_experiment.py`
- `scripts/run_reinsurance_pilot.py`
- `scripts/run_treaty_indexation_sensitivity.py`
- `src/__init__.py`
- `src/bornhuetter_ferguson.py`
- `src/final_figures_tables.py`
- `src/final_results_consolidation.py`
- `src/treaty_indexation_sensitivity.py`
- `tests/test_bornhuetter_ferguson.py`
- `tests/test_ml_models.py`

Deleted code file:

- `src/figures.py`

Deleted outputs are exactly the three checkpoint paths and every tracked file
under the eight smoke directories in Section G, plus the two initially
untracked candidates. The file-level list and rationale are in
`cleanup_output_inventory.csv`.

At report creation, the change set contains 94 repository paths: 92 tracked
modifications/deletions plus the two new audit/report files. Python physical
lines changed from 35,205 to 35,249 (+44), reflecting documentation additions
net of import cleanup and removal of the empty module.

## M. Material commands run

All commands were run from the repository root. Read-only AST and PowerShell
inventory scripts are summarized by purpose above; the material repository and
validation commands were:

```powershell
git branch --show-current
git status --short
git log --oneline -5
rg -n -a -uuu --hidden --glob '!**/.git/**' -F 'STEP25_FINAL_SUMMARY.xlsx' .
rg -n -a -uuu --hidden --glob '!**/.git/**' -F 'baseline_gate_1_00' .
git switch -c dissertation-code-cleanup
git branch --show-current
git status --short
git log --oneline -3
Get-FileHash -Algorithm SHA256 <authoritative-path>
git ls-files
rg --files src scripts tests -g '*.py'
rg -n 'pd\.read_csv|read_text|open\(' src scripts
rg -n '^def |^class ' src scripts
$env:MPLBACKEND = 'Agg'
$env:LOKY_MAX_CPU_COUNT = '1'
.\.venv\Scripts\python.exe -m pytest -q tests\test_final_results_consolidation.py tests\test_final_figures_tables.py
.\.venv\Scripts\python.exe -m scripts.build_final_analysis_dataset
.\.venv\Scripts\python.exe -m scripts.build_final_figures_tables
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
git diff --check
git status --short
git diff --stat
git diff
```

## N. Technical result and warnings

Technical result: **PASS**.

Warnings:

- Git reports the repository's normal LF-to-CRLF working-copy warning on edited
  text files; `git diff --check` passes.
- The removed untracked workbook/directory were not included in the measured
  tracked-byte total.
- Performance benefit is reasoned from eliminated allocations, not benchmarked.

No commit, merge, push, branch deletion, or switch back to the source branch was
performed.

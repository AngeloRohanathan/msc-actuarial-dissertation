# Inflation-Aware Reserving for Excess-of-Loss Reinsurance

This repository contains the simulation, reserving, sensitivity-analysis, and
reporting code for an MSc actuarial dissertation. The study examines how claims
inflation, structural change, and excess-of-loss (XoL) treaty indexation affect
gross and ceded reserve estimates.

The main modelling pipeline is complete through Step 31. Step 32 consolidates
the frozen experiment results, and Step 33 produces the dissertation-facing
figures and tables. Those two steps are analysis and presentation layers only;
they do not rerun simulations, calibration, model fitting, or bootstrap
resampling.

## Repository structure

- `config.py` — frozen assumptions, scenarios, seeds, and experiment settings.
- `src/` — simulation, XoL, triangle, reserving, model, sensitivity, and
  reporting libraries.
- `scripts/` — command-line experiment runners and the Step 32/33 builders.
- `tests/` — automated behavioural and scientific-assumption checks.
- `data/` — pilot inputs, frozen Step 16 results, and calibration sources.
- `outputs/step*/` — final experiment results, manifests, method notes, and
  validation evidence.
- `outputs/final_analysis/` — authoritative Step 32 datasets and Step 33
  dissertation artifacts.
- `simulation_specification.md` and `analysis_specification.md` — simulation
  and analysis definitions.
- `dissertation_master_step_by_step_plan.md` — end-to-end project record.

## Core workflow

1. Simulate claims, reporting delays, and payment schedules under the frozen
   scenario definitions.
2. Apply calendar-year claims inflation and the specified XoL treaty terms.
3. Aggregate gross, ceded, and retained payments into reserving triangles.
4. Estimate reserves using the frozen classical, expected-loss, BF, Poisson,
   and Tweedie methods.
5. Evaluate estimates against known simulation truth and run the approved
   sensitivity and paired-comparison analyses.
6. Consolidate frozen Step 16–31 outputs with Step 32.
7. Generate dissertation figures, tables, backing CSVs, and validation records
   with Step 33.

## Environment and tests

Create and activate a virtual environment, then install the requirements:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Run the full test suite from the repository root:

```bash
python -m pytest -q
```

For headless or resource-constrained runs, use `MPLBACKEND=Agg` and
`LOKY_MAX_CPU_COUNT=1`. In PowerShell:

```powershell
$env:MPLBACKEND = "Agg"
$env:LOKY_MAX_CPU_COUNT = "1"
.\.venv\Scripts\python.exe -m pytest -q
```

## Final analysis entry points

Step 32 reads only the frozen sources declared in
`outputs/final_analysis/source_inventory.csv`:

```bash
python -m scripts.build_final_analysis_dataset
```

Step 33 reads only the consolidated Step 32 directory:

```bash
python -m scripts.build_final_figures_tables
```

Both builders refuse to overwrite existing outputs. To verify a rebuild,
first copy or move the existing `outputs/final_analysis/` directory to a safe
location, run Step 32 and then Step 33, and reconcile the regenerated CSVs with
the frozen copy. Do not rerun the multi-hour experiments merely to rebuild the
final analysis.

The principal dissertation artifacts are:

- Step 32 master datasets in `outputs/final_analysis/*.csv`.
- Eight figures, their PDFs, and backing CSVs in
  `outputs/final_analysis/step33_figures/`.
- Seven main tables and appendix candidates in
  `outputs/final_analysis/step33_tables/`.
- Step 32/33 manifests, validation reports, method notes, and
  `step33_output_index.csv` in `outputs/final_analysis/`.

## Reproducibility safeguards

- Scenario definitions, valuation year, accident years, treaty design, model
  grids, and random seeds are controlled in the frozen configuration and
  experiment runners.
- The Step 19 expected-loss prior is calibrated independently of evaluation
  simulations.
- The Step 28 ceded development pattern and Step 31 bootstrap design are frozen.
- `source_inventory.csv` records paths, keys, row counts, and SHA-256 hashes for
  every Step 32 source.
- `validation_report.csv` and `STEP33_VALIDATION_REPORT.csv` must remain fully
  passing after any code-quality change.
- CSV backing data, rather than image-file hashes, is authoritative when figure
  metadata makes PNG or PDF bytes nondeterministic.

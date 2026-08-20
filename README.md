# Inflation-Aware Reserving for Excess-of-Loss Reinsurance

This repository contains the simulation, reserving and evaluation code
for an MSc dissertation examining the effects of claims inflation,
structural change and reinsurance indexation on gross and ceded reserve
estimates.

## Current status

The project is currently in the pilot implementation stage.

The simulation specification is contained in:

`simulation_specification.md`

## Core workflow

1. Generate claim counts and real claim severities.
2. Generate reporting delays and payment schedules.
3. Apply calendar-year claims inflation.
4. Apply excess-of-loss reinsurance terms.
5. Aggregate payments into gross, ceded and retained triangles.
6. Estimate reserves using classical and machine-learning methods.
7. Compare estimated reserves against the known simulated truth.

## Project structure

- `src/simulation.py`: claim and payment simulation
- `src/reinsurance.py`: excess-of-loss and indexation calculations
- `src/triangles.py`: triangle construction
- `src/reserving.py`: classical reserving methods
- `src/ml_models.py`: statistical and machine-learning methods
- `src/evaluation.py`: reserve-error calculations
- `src/figures.py`: dissertation figures
- `config.py`: central assumptions and experiment settings
- `tests/`: automated validation tests
- `data/`: generated pilot and final datasets
- `outputs/`: figures, result tables and logs

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```
Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Run all automated tests using:

```bash
python -m pytest -q
```

Reproducibility

Random-number generation will use fixed and recorded seeds. Final
experiment parameters will be frozen before the final Monte Carlo
simulation is run.
---
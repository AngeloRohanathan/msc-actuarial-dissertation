"""Run Step 27 Expected Loss/BF prior sensitivity."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import pandas as pd

from config import (
    BF_BENCHMARK_INCREMENTAL_PATTERNS,
    BF_STRUCTURAL_BREAK_ACCIDENT_YEAR,
    EXPECTED_LOSS_PRIOR_VERSION,
    VALUATION_YEAR,
)
from src.prior_misspecification_sensitivity import (
    BASELINE_REPRODUCTION_ATOL,
    BASELINE_REPRODUCTION_RTOL,
    PRIOR_MULTIPLIERS,
    build_acceptance_report,
    build_failure_summary,
    build_prior_multiplier_comparison,
    build_sensitivity_summary,
    compare_multiplier_one_to_frozen_baselines,
    load_frozen_independent_prior,
    run_prior_misspecification_sensitivity,
    select_frozen_evaluation_rows,
    sha256_file,
)


DEFAULT_PRIOR = Path(
    "data/calibration/expected_loss_prior/"
    "final_2000/expected_loss_prior.csv"
)
DEFAULT_STEP20_DETAIL = Path(
    "outputs/step20_expected_loss/final_50/"
    "expected_loss_by_accident_year.csv"
)
DEFAULT_STEP20_RESULTS = Path(
    "outputs/step20_expected_loss/final_50/"
    "expected_loss_results.csv"
)
DEFAULT_STEP21_DETAIL = Path(
    "outputs/step21_paid_bf/final_50/"
    "bf_by_accident_year.csv"
)
DEFAULT_STEP21_RESULTS = Path(
    "outputs/step21_paid_bf/final_50/"
    "bf_results.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/step27_prior_misspecification"
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line inputs."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulations",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--prior-multiplier",
        action="append",
        type=float,
        default=None,
        help=(
            "Repeat to select multipliers. Defaults to the frozen "
            "five-value grid. The baseline gate must use only 1.0."
        ),
    )
    parser.add_argument(
        "--run-label",
        required=True,
    )
    parser.add_argument(
        "--prior",
        type=Path,
        default=DEFAULT_PRIOR,
    )
    parser.add_argument(
        "--step20-detail",
        type=Path,
        default=DEFAULT_STEP20_DETAIL,
    )
    parser.add_argument(
        "--step20-results",
        type=Path,
        default=DEFAULT_STEP20_RESULTS,
    )
    parser.add_argument(
        "--step21-detail",
        type=Path,
        default=DEFAULT_STEP21_DETAIL,
    )
    parser.add_argument(
        "--step21-results",
        type=Path,
        default=DEFAULT_STEP21_RESULTS,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    return parser.parse_args()


def _filter_portfolio_source(
    source: pd.DataFrame,
    selected_detail: pd.DataFrame,
) -> pd.DataFrame:
    """Retain frozen rows for exactly the selected portfolios."""

    keys = [
        "scenario_id",
        "simulation_id",
        "basis",
    ]
    selected_keys = selected_detail[keys].drop_duplicates()

    return source.merge(
        selected_keys,
        on=keys,
        how="inner",
        validate="many_to_one",
    )


def _method_note(
    *,
    multipliers: tuple[float, ...],
    simulations: int,
    scenarios: int,
) -> str:
    """Return the Step 27 methodological note."""

    multiplier_text = ", ".join(
        f"{value:.2f}" for value in multipliers
    )
    attempts = (
        scenarios
        * simulations
        * 2
        * 3
        * len(multipliers)
    )

    return rf"""# Step 27 — Prior Misspecification Sensitivity

## Objective

This experiment measures how Expected Loss and paid
Bornhuetter–Ferguson reserves respond when the frozen independent
expected-ultimate prior is systematically too low or too high. Model
accuracy is a research result rather than a technical acceptance
criterion.

## Frozen prior and multiplier design

The baseline prior is the independently calibrated Step 19 pricing
Monte Carlo prior `{EXPECTED_LOSS_PRIOR_VERSION}`. Calibration used a
seed range separate from the evaluation experiment and did not use
evaluation reserve truth.

For each basis and accident year, the sensitivity prior is

\[
U^{{sensitivity}}_i = m U^{{baseline}}_i.
\]

The configured grid is `0.80, 0.90, 1.00, 1.10, 1.20`. This run uses:
`{multiplier_text}`. The same multiplier is applied to gross, ceded and
retained prior ultimates. No multiplier is calibrated from observed
evaluation errors.

## Frozen models

Expected Loss uses

\[
\widehat R_i = \max(U_i - P_i, 0).
\]

Standard and Break-Aware paid BF use

\[
\widehat R_i = U_i(1-p_i).
\]

The existing Step 20 and Step 21 implementations are called directly.
The BF pattern definitions and the structural-break accident year are
unchanged. No regime-mismatch sensitivity is included.

## Reused evaluation data and leakage safeguards

No portfolio is resimulated. The experiment reuses the frozen Step 20
accident-year rows, preserving scenario, simulation ID, basis,
paid-to-date and true reserve. BF assumptions are checked against the
frozen Step 21 detail rows. Estimator inputs are constructed without
truth; truth is merged only after each reserve estimate exists.

The source prior SHA-256 digest is checked before and after the run.
Multiplier 1.00 is compared row-by-row with the frozen Step 20 Expected
Loss and Step 21 BF portfolio results.

## Experimental design

This run contains {scenarios} scenarios × {simulations} simulations ×
2 bases × 3 models × {len(multipliers)} multiplier value(s), giving
{attempts:,} model attempts. Gross and ceded results remain separate.
Model failures, if any, remain explicit and accuracy is summarised only
for successful estimates.
"""


def main() -> None:
    """Run Step 27 from frozen Step 19–21 files."""

    arguments = parse_arguments()
    started = time.perf_counter()

    source_paths = [
        arguments.prior,
        arguments.step20_detail,
        arguments.step20_results,
        arguments.step21_detail,
        arguments.step21_results,
    ]

    for path in source_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    output_directory = (
        arguments.output_root / arguments.run_label
    )

    if output_directory.exists():
        raise FileExistsError(
            "Output folder already exists: "
            f"{output_directory}"
        )

    multipliers = tuple(
        arguments.prior_multiplier
        if arguments.prior_multiplier is not None
        else PRIOR_MULTIPLIERS
    )

    if not any(
        abs(float(value) - 1.0) <= 1e-12
        for value in multipliers
    ):
        raise ValueError(
            "Every Step 27 run must include multiplier 1.00 so the "
            "frozen baseline reproduction check is performed."
        )

    prior_hash_before = sha256_file(arguments.prior)
    prior = load_frozen_independent_prior(
        arguments.prior
    )

    step20_detail = pd.read_csv(
        arguments.step20_detail
    )
    selected_detail = select_frozen_evaluation_rows(
        step20_detail,
        simulations_per_scenario=arguments.simulations,
        scenario_ids=arguments.scenario,
    )
    scenario_ids = sorted(
        selected_detail["scenario_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    frozen_step20_results = _filter_portfolio_source(
        pd.read_csv(arguments.step20_results),
        selected_detail,
    )
    frozen_step21_results = _filter_portfolio_source(
        pd.read_csv(arguments.step21_results),
        selected_detail,
    )
    frozen_step21_detail = _filter_portfolio_source(
        pd.read_csv(arguments.step21_detail),
        selected_detail,
    )

    results, detail_results = (
        run_prior_misspecification_sensitivity(
            expected_loss_prior=prior,
            evaluation_detail=selected_detail,
            prior_multipliers=multipliers,
            valuation_year=VALUATION_YEAR,
            structural_break_year=(
                BF_STRUCTURAL_BREAK_ACCIDENT_YEAR
            ),
            benchmark_patterns=(
                BF_BENCHMARK_INCREMENTAL_PATTERNS
            ),
        )
    )

    baseline_reproduction = (
        compare_multiplier_one_to_frozen_baselines(
            results=results,
            frozen_step20_results=(
                frozen_step20_results
            ),
            frozen_step21_results=(
                frozen_step21_results
            ),
            rtol=BASELINE_REPRODUCTION_RTOL,
            atol=BASELINE_REPRODUCTION_ATOL,
        )
    )

    summary = build_sensitivity_summary(results)
    multiplier_comparison = (
        build_prior_multiplier_comparison(summary)
    )
    failure_summary = build_failure_summary(results)
    prior_hash_after = sha256_file(arguments.prior)

    acceptance_report = build_acceptance_report(
        results=results,
        detail_results=detail_results,
        evaluation_detail=selected_detail,
        frozen_bf_detail=frozen_step21_detail,
        baseline_reproduction=baseline_reproduction,
        expected_scenario_ids=scenario_ids,
        simulations_per_scenario=arguments.simulations,
        requested_multipliers=multipliers,
        prior_hash_before=prior_hash_before,
        prior_hash_after=prior_hash_after,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    results.to_csv(
        output_directory / "results.csv",
        index=False,
    )
    detail_results.to_csv(
        output_directory / "by_accident_year.csv",
        index=False,
    )
    summary.to_csv(
        output_directory / "summary.csv",
        index=False,
    )
    multiplier_comparison.to_csv(
        output_directory
        / "prior_multiplier_comparison.csv",
        index=False,
    )
    acceptance_report.to_csv(
        output_directory / "acceptance_report.csv",
        index=False,
    )
    failure_summary.to_csv(
        output_directory / "failure_summary.csv",
        index=False,
    )
    baseline_reproduction.to_csv(
        output_directory / "baseline_reproduction.csv",
        index=False,
    )

    elapsed_seconds = time.perf_counter() - started
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "step": 27,
        "method": "prior_misspecification_sensitivity",
        "run_label": arguments.run_label,
        "baseline_reproduction_gate": bool(
            len(multipliers) == 1
            and abs(float(multipliers[0]) - 1.0)
            <= 1e-12
        ),
        "prior_multipliers": [
            float(value) for value in multipliers
        ],
        "configured_prior_multiplier_grid": list(
            PRIOR_MULTIPLIERS
        ),
        "models": [
            "expected_loss",
            "bornhuetter_ferguson_standard",
            "bornhuetter_ferguson_break_aware",
        ],
        "bases": ["gross", "ceded"],
        "scenario_ids": scenario_ids,
        "simulations_per_scenario": (
            arguments.simulations
        ),
        "result_rows": int(len(results)),
        "accident_year_detail_rows": int(
            len(detail_results)
        ),
        "successes": int(results["success"].sum()),
        "failures": int((~results["success"]).sum()),
        "valuation_year": int(VALUATION_YEAR),
        "baseline_prior_version": (
            EXPECTED_LOSS_PRIOR_VERSION
        ),
        "source_files": {
            "step19_prior": str(arguments.prior),
            "step20_detail": str(
                arguments.step20_detail
            ),
            "step20_results": str(
                arguments.step20_results
            ),
            "step21_detail": str(
                arguments.step21_detail
            ),
            "step21_results": str(
                arguments.step21_results
            ),
        },
        "step19_prior_sha256_before": prior_hash_before,
        "step19_prior_sha256_after": prior_hash_after,
        "portfolio_resimulation_performed": False,
        "evaluation_truth_used_for_prior_scaling": False,
        "evaluation_truth_used_for": (
            "evaluation_only_after_estimation"
        ),
        "baseline_reproduction_rtol": (
            BASELINE_REPRODUCTION_RTOL
        ),
        "baseline_reproduction_atol": (
            BASELINE_REPRODUCTION_ATOL
        ),
        "elapsed_seconds": float(elapsed_seconds),
    }

    with (
        output_directory / "manifest.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    (
        output_directory / "STEP27_METHOD_NOTE.md"
    ).write_text(
        _method_note(
            multipliers=tuple(
                float(value) for value in multipliers
            ),
            simulations=arguments.simulations,
            scenarios=len(scenario_ids),
        ),
        encoding="utf-8",
    )

    print("\nStep 27 run completed.")
    print(f"Output: {output_directory}")
    print(f"Result rows: {len(results):,}")
    print(
        "Success/failure: "
        f"{int(results['success'].sum()):,}/"
        f"{int((~results['success']).sum()):,}"
    )
    print("\nBaseline reproduction:")
    print(baseline_reproduction.to_string(index=False))
    print("\nAcceptance report:")
    print(acceptance_report.to_string(index=False))
    print(f"\nElapsed seconds: {elapsed_seconds:.3f}")

    applicable = acceptance_report["applicable"]
    if not acceptance_report.loc[
        applicable,
        "passed",
    ].all():
        raise RuntimeError(
            "At least one applicable Step 27 acceptance check failed."
        )


if __name__ == "__main__":
    main()

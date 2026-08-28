"""Run the Step 25 Regularized Poisson versus Tweedie experiment."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    END_TO_END_BASE_SEED,
    END_TO_END_PILOT_SIMULATIONS,
    END_TO_END_SCENARIOS,
    EXPERIMENT_CLAUSE_TYPE,
    EXPERIMENT_STRUCTURAL_BREAK_YEAR,
    INFLATION_SCENARIOS,
    ML_AMOUNT_SCALE,
    ML_MIN_TRAINING_DIAGONALS,
    ML_MIN_VALIDATION_FOLDS,
    ML_POISSON_ALPHA_GRID,
    ML_TWEEDIE_ALPHA_GRID,
    ML_TWEEDIE_POWER_GRID,
    PILOT_XOL_ATTACHMENT,
    PILOT_XOL_LIMIT,
    VALUATION_YEAR,
    ensure_directories,
    validate_config,
)
from src.evaluation import (
    build_failure_summary,
    build_result_row,
    save_summary_plots,
    summarise_experiment_results,
)
from src.ml_models import (
    fit_regularised_poisson_reserving_model,
    fit_regularised_tweedie_reserving_model,
)
from src.reinsurance import apply_xol_to_payments
from src.simulation import (
    build_inflation_index,
    simulate_portfolio,
)
from src.triangles import build_triangle_package


MODEL_NAMES = [
    "regularized_poisson",
    "regularized_tweedie",
]

BASES = [
    "gross",
    "ceded",
]

STEP25_OUTPUT_DIR = Path(
    "outputs/step25_regularized_tweedie"
)


def parse_arguments() -> argparse.Namespace:
    """Read command-line experiment settings."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the Step 25 paired comparison of baseline "
            "Regularized Poisson and Regularized Tweedie."
        )
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=END_TO_END_PILOT_SIMULATIONS,
        help="Number of simulations per scenario.",
    )

    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help=(
            "Run only the named scenario. "
            "May be supplied multiple times."
        ),
    )

    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Optional name for the output folder.",
    )

    return parser.parse_args()


def select_scenarios(
    requested_scenarios: list[str] | None,
) -> list[dict[str, Any]]:
    """Select all configured scenarios or a requested subset."""

    scenarios = [
        dict(scenario)
        for scenario in END_TO_END_SCENARIOS
    ]

    if requested_scenarios is None:
        return scenarios

    requested = set(requested_scenarios)

    selected = [
        scenario
        for scenario in scenarios
        if scenario["scenario_id"] in requested
    ]

    found = {
        scenario["scenario_id"]
        for scenario in selected
    }

    missing = requested - found

    if missing:
        raise ValueError(
            "Unknown requested scenarios: "
            f"{sorted(missing)}"
        )

    return selected


def build_scenario_inflation_index(
    inflation_scenario: str,
) -> dict[int, float]:
    """Build the inflation index required by both models."""

    inflation_table = build_inflation_index(
        INFLATION_SCENARIOS[
            inflation_scenario
        ]
    )

    return (
        inflation_table
        .set_index("calendar_year")[
            "inflation_index"
        ]
        .to_dict()
    )


def verify_pipeline_reconciliations(
    reinsured_payments: pd.DataFrame,
    triangle_outputs: dict[str, pd.DataFrame],
) -> bool:
    """Check payment and reserve-truth accounting identities."""

    gross_total = float(
        reinsured_payments[
            "nominal_gross_payment"
        ].sum()
    )
    ceded_total = float(
        reinsured_payments[
            "nominal_ceded_payment"
        ].sum()
    )
    retained_total = float(
        reinsured_payments[
            "nominal_retained_payment"
        ].sum()
    )

    reinsurance_passed = np.isclose(
        gross_total,
        ceded_total + retained_total,
        rtol=1e-12,
        atol=1e-6,
    )

    totals = triangle_outputs[
        "true_reserve_totals"
    ].iloc[0]

    gross_triangle_passed = np.isclose(
        totals["total_observed_gross_paid"]
        + totals["total_true_gross_reserve"],
        totals["total_gross_ultimate"],
        rtol=1e-12,
        atol=1e-6,
    )

    ceded_triangle_passed = np.isclose(
        totals["total_observed_ceded_paid"]
        + totals["total_true_ceded_reserve"],
        totals["total_ceded_ultimate"],
        rtol=1e-12,
        atol=1e-6,
    )

    return bool(
        reinsurance_passed
        and gross_triangle_passed
        and ceded_triangle_passed
    )


def run_model_attempt(
    *,
    model_name: str,
    basis: str,
    simulation_id: int,
    seed: int,
    scenario: dict[str, Any],
    inflation_index: dict[int, float],
    triangle_outputs: dict[str, pd.DataFrame],
    pipeline_reconciliation_passed: bool,
) -> dict[str, Any]:
    """Run one model-basis attempt and return a standard result row."""

    incremental_triangle = triangle_outputs[
        f"{basis}_incremental"
    ]

    true_totals = triangle_outputs[
        "true_reserve_totals"
    ].iloc[0]

    if basis == "gross":
        true_reserve = float(
            true_totals[
                "total_true_gross_reserve"
            ]
        )
        observed_paid = float(
            true_totals[
                "total_observed_gross_paid"
            ]
        )
        true_ultimate = float(
            true_totals[
                "total_gross_ultimate"
            ]
        )
    else:
        true_reserve = float(
            true_totals[
                "total_true_ceded_reserve"
            ]
        )
        observed_paid = float(
            true_totals[
                "total_observed_ceded_paid"
            ]
        )
        true_ultimate = float(
            true_totals[
                "total_ceded_ultimate"
            ]
        )

    start_time = time.perf_counter()

    estimated_reserve: float | None = None
    selected_alpha: float | None = None
    selected_power: float | None = None
    model_reconciliation_passed = False
    failure_reason = ""
    status = "failed"

    try:
        structural_break_year = (
            EXPERIMENT_STRUCTURAL_BREAK_YEAR
            if scenario[
                "apply_structural_break"
            ]
            else None
        )

        if model_name == "regularized_poisson":
            result = (
                fit_regularised_poisson_reserving_model(
                    incremental_triangle=(
                        incremental_triangle
                    ),
                    inflation_index=inflation_index,
                    valuation_year=VALUATION_YEAR,
                    basis=basis,
                    alpha_grid=(
                        ML_POISSON_ALPHA_GRID
                    ),
                    minimum_training_diagonals=(
                        ML_MIN_TRAINING_DIAGONALS
                    ),
                    minimum_validation_folds=(
                        ML_MIN_VALIDATION_FOLDS
                    ),
                    amount_scale=ML_AMOUNT_SCALE,
                    structural_break_year=(
                        structural_break_year
                    ),
                )
            )

        elif model_name == "regularized_tweedie":
            result = (
                fit_regularised_tweedie_reserving_model(
                    incremental_triangle=(
                        incremental_triangle
                    ),
                    inflation_index=inflation_index,
                    valuation_year=VALUATION_YEAR,
                    basis=basis,
                    alpha_grid=(
                        ML_TWEEDIE_ALPHA_GRID
                    ),
                    power_grid=(
                        ML_TWEEDIE_POWER_GRID
                    ),
                    minimum_training_diagonals=(
                        ML_MIN_TRAINING_DIAGONALS
                    ),
                    minimum_validation_folds=(
                        ML_MIN_VALIDATION_FOLDS
                    ),
                    amount_scale=ML_AMOUNT_SCALE,
                    structural_break_year=(
                        structural_break_year
                    ),
                )
            )

            selected_power = float(
                result["summary"].loc[
                    0,
                    "selected_power",
                ]
            )

        else:
            raise ValueError(
                f"Unknown model: {model_name}"
            )

        selected_alpha = float(
            result["summary"].loc[
                0,
                "selected_alpha",
            ]
        )

        estimated_reserve = float(
            result["summary"].loc[
                0,
                "total_estimated_reserve",
            ]
        )

        reserve_table_total = float(
            result[
                "reserve_by_accident_year"
            ]["estimated_reserve"].sum()
        )

        model_reconciliation_passed = bool(
            np.isclose(
                estimated_reserve,
                reserve_table_total,
                rtol=1e-12,
                atol=1e-6,
            )
        )

        if not np.isfinite(estimated_reserve):
            raise ValueError(
                "Estimated reserve is not finite."
            )

        if estimated_reserve < -1e-6:
            raise ValueError(
                "Estimated reserve is negative."
            )

        if not model_reconciliation_passed:
            raise ValueError(
                "Model summary does not reconcile with "
                "the accident-year reserve table."
            )

        status = "success"

    except Exception as error:
        failure_reason = (
            f"{type(error).__name__}: {error}"
        )

    runtime_seconds = (
        time.perf_counter()
        - start_time
    )

    row = build_result_row(
        simulation_id=simulation_id,
        seed=seed,
        scenario=scenario,
        clause_type=EXPERIMENT_CLAUSE_TYPE,
        model_name=model_name,
        basis=basis,
        true_reserve=true_reserve,
        observed_paid=observed_paid,
        true_ultimate=true_ultimate,
        estimated_reserve=(
            estimated_reserve
            if status == "success"
            else None
        ),
        runtime_seconds=runtime_seconds,
        success_or_failure=status,
        failure_reason=failure_reason,
        pipeline_reconciliation_passed=(
            pipeline_reconciliation_passed
        ),
        model_reconciliation_passed=(
            model_reconciliation_passed
        ),
        selected_alpha=selected_alpha,
    )

    row["selected_power"] = (
        float(selected_power)
        if selected_power is not None
        else np.nan
    )

    return row


def run_one_portfolio(
    *,
    simulation_id: int,
    seed: int,
    scenario: dict[str, Any],
) -> list[dict[str, Any]]:
    """Simulate once and fit both models to the same triangles."""

    _, simulated_payments = simulate_portfolio(
        simulation_id=simulation_id,
        frequency_scenario=scenario[
            "frequency_scenario"
        ],
        tail_type=scenario["tail_type"],
        inflation_scenario=scenario[
            "inflation_scenario"
        ],
        apply_structural_break=scenario[
            "apply_structural_break"
        ],
        seed=seed,
    )

    simulated_payments = simulated_payments.copy()
    simulated_payments[
        "scenario_id"
    ] = scenario["scenario_id"]

    reinsured_payments, _ = apply_xol_to_payments(
        payments=simulated_payments,
        attachment=PILOT_XOL_ATTACHMENT,
        limit=PILOT_XOL_LIMIT,
    )

    triangle_outputs = build_triangle_package(
        payments=reinsured_payments,
        valuation_year=VALUATION_YEAR,
    )

    pipeline_reconciliation_passed = (
        verify_pipeline_reconciliations(
            reinsured_payments=reinsured_payments,
            triangle_outputs=triangle_outputs,
        )
    )

    inflation_index = build_scenario_inflation_index(
        scenario["inflation_scenario"]
    )

    rows: list[dict[str, Any]] = []

    for model_name in MODEL_NAMES:
        for basis in BASES:
            rows.append(
                run_model_attempt(
                    model_name=model_name,
                    basis=basis,
                    simulation_id=simulation_id,
                    seed=seed,
                    scenario=scenario,
                    inflation_index=inflation_index,
                    triangle_outputs=triangle_outputs,
                    pipeline_reconciliation_passed=(
                        pipeline_reconciliation_passed
                    ),
                )
            )

    return rows


def build_step25_acceptance_report(
    *,
    results: pd.DataFrame,
    expected_rows: int,
    expected_scenario_ids: set[str],
) -> pd.DataFrame:
    """Evaluate Step 25 structure, pairing, and model integrity."""

    successful = results.loc[
        results["success_or_failure"].eq(
            "success"
        )
    ]
    failed = results.loc[
        results["success_or_failure"].eq(
            "failed"
        )
    ]
    tweedie_success = successful.loc[
        successful["model"].eq(
            "regularized_tweedie"
        )
    ]
    poisson = results.loc[
        results["model"].eq(
            "regularized_poisson"
        )
    ]

    paired_truth = (
        results.groupby(
            [
                "simulation_id",
                "scenario_id",
                "basis",
            ],
            dropna=False,
        )[
            [
                "true_reserve",
                "true_ultimate",
                "observed_paid",
                "seed",
            ]
        ]
        .nunique(dropna=False)
    )

    failures_complete = bool(
        failed.empty
        or (
            failed["failure_reason"]
            .fillna("")
            .str.len()
            .gt(0)
            .all()
            and failed["failure_category"]
            .fillna("")
            .str.len()
            .gt(0)
            .all()
        )
    )

    checks = [
        {
            "check": "expected_row_count",
            "passed": len(results) == expected_rows,
            "detail": (
                f"actual={len(results)}, expected={expected_rows}"
            ),
        },
        {
            "check": "all_expected_scenarios_present",
            "passed": set(results["scenario_id"]) == (
                expected_scenario_ids
            ),
            "detail": "",
        },
        {
            "check": "both_bases_present",
            "passed": set(results["basis"]) == set(BASES),
            "detail": "",
        },
        {
            "check": "both_models_present",
            "passed": set(results["model"]) == set(MODEL_NAMES),
            "detail": "",
        },
        {
            "check": "unique_model_portfolio_rows",
            "passed": not results.duplicated(
                [
                    "simulation_id",
                    "scenario_id",
                    "model",
                    "basis",
                ]
            ).any(),
            "detail": "",
        },
        {
            "check": "selected_tweedie_power_in_grid",
            "passed": bool(
                tweedie_success[
                    "selected_power"
                ].isin(
                    ML_TWEEDIE_POWER_GRID
                ).all()
            ),
            "detail": "",
        },
        {
            "check": "selected_alpha_in_model_grid",
            "passed": bool(
                successful.loc[
                    successful["model"].eq(
                        "regularized_poisson"
                    ),
                    "selected_alpha",
                ].isin(
                    ML_POISSON_ALPHA_GRID
                ).all()
                and tweedie_success[
                    "selected_alpha"
                ].isin(
                    ML_TWEEDIE_ALPHA_GRID
                ).all()
            ),
            "detail": "",
        },
        {
            "check": "poisson_selected_power_is_missing",
            "passed": bool(
                poisson["selected_power"].isna().all()
            ),
            "detail": "",
        },
        {
            "check": "successful_estimates_nonnegative",
            "passed": bool(
                successful[
                    "estimated_reserve"
                ].ge(0.0).all()
            ),
            "detail": "",
        },
        {
            "check": "pipeline_reconciliations_passed",
            "passed": bool(
                results[
                    "pipeline_reconciliation_passed"
                ].all()
            ),
            "detail": "",
        },
        {
            "check": "successful_reserve_reconciliations_passed",
            "passed": bool(
                successful[
                    "model_reconciliation_passed"
                ].all()
            ),
            "detail": "",
        },
        {
            "check": "paired_models_use_same_truth",
            "passed": bool(
                paired_truth.eq(1).all().all()
                and paired_truth.shape[0]
                == (
                    len(results)
                    // len(MODEL_NAMES)
                )
            ),
            "detail": "",
        },
        {
            "check": "failure_reporting_complete",
            "passed": failures_complete,
            "detail": f"failed_rows={len(failed)}",
        },
    ]

    return pd.DataFrame(checks)


def main() -> None:
    """Run the controlled paired Poisson/Tweedie experiment."""

    arguments = parse_arguments()

    ensure_directories()
    validate_config()

    if arguments.simulations < 1:
        raise ValueError(
            "--simulations must be positive."
        )

    scenarios = select_scenarios(
        arguments.scenario
    )

    run_label = (
        arguments.run_label
        or (
            "step25_regularized_tweedie_"
            f"{arguments.simulations}_simulations"
        )
    )

    output_directory = (
        STEP25_OUTPUT_DIR
        / run_label
    )
    figure_directory = (
        output_directory
        / "figures"
    )

    if output_directory.exists():
        raise FileExistsError(
            "Output directory already exists; choose a new "
            f"--run-label: {output_directory}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    all_rows: list[dict[str, Any]] = []
    experiment_start = time.perf_counter()

    total_portfolios = (
        arguments.simulations
        * len(scenarios)
    )
    completed_portfolios = 0

    for simulation_id in range(
        1,
        arguments.simulations + 1,
    ):
        seed = (
            END_TO_END_BASE_SEED
            + simulation_id
        )

        for scenario in scenarios:
            completed_portfolios += 1

            print(
                f"[{completed_portfolios}/{total_portfolios}] "
                f"simulation={simulation_id}, "
                f"scenario={scenario['scenario_id']}"
            )

            all_rows.extend(
                run_one_portfolio(
                    simulation_id=simulation_id,
                    seed=seed,
                    scenario=scenario,
                )
            )

            pd.DataFrame(all_rows).to_csv(
                output_directory
                / "results_checkpoint.csv",
                index=False,
            )

    results = pd.DataFrame(all_rows)

    expected_rows = (
        arguments.simulations
        * len(scenarios)
        * len(MODEL_NAMES)
        * len(BASES)
    )

    summary = summarise_experiment_results(
        results
    )
    failure_summary = build_failure_summary(
        results
    )
    acceptance_report = (
        build_step25_acceptance_report(
            results=results,
            expected_rows=expected_rows,
            expected_scenario_ids={
                scenario["scenario_id"]
                for scenario in scenarios
            },
        )
    )

    results.to_csv(
        output_directory / "results.csv",
        index=False,
    )
    summary.to_csv(
        output_directory / "summary.csv",
        index=False,
    )
    failure_summary.to_csv(
        output_directory / "failure_summary.csv",
        index=False,
    )
    acceptance_report.to_csv(
        output_directory / "acceptance_report.csv",
        index=False,
    )

    save_summary_plots(
        summary=summary,
        output_directory=figure_directory,
    )

    elapsed_seconds = (
        time.perf_counter()
        - experiment_start
    )

    manifest = {
        "run_label": run_label,
        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "experiment": (
            "regularized_poisson_vs_regularized_tweedie"
        ),
        "selection_criterion": (
            "historical_rolling_diagonal_mean_absolute_error"
        ),
        "evaluation_truth_used_for_hyperparameter_selection": False,
        "simulations_per_scenario": arguments.simulations,
        "scenario_ids": [
            scenario["scenario_id"]
            for scenario in scenarios
        ],
        "number_of_scenarios": len(scenarios),
        "expected_result_rows": expected_rows,
        "actual_result_rows": int(len(results)),
        "base_seed": END_TO_END_BASE_SEED,
        "valuation_year": VALUATION_YEAR,
        "treaty_attachment": PILOT_XOL_ATTACHMENT,
        "treaty_limit": PILOT_XOL_LIMIT,
        "clause_type": EXPERIMENT_CLAUSE_TYPE,
        "models": MODEL_NAMES,
        "bases": BASES,
        "tweedie_power_grid": list(
            ML_TWEEDIE_POWER_GRID
        ),
        "tweedie_alpha_grid": list(
            ML_TWEEDIE_ALPHA_GRID
        ),
        "elapsed_seconds": elapsed_seconds,
    }

    with (
        output_directory / "manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print("\nStep 25 experiment completed.")
    print(f"Result rows: {len(results):,}")
    print("\nAcceptance report:")
    print(
        acceptance_report.to_string(
            index=False
        )
    )
    print("\nOutputs saved to:")
    print(output_directory)


if __name__ == "__main__":
    main()

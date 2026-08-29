"""Run Step 20 expected-loss reserving experiment."""

from __future__ import annotations

import argparse
import inspect
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    END_TO_END_BASE_SEED,
    END_TO_END_SCENARIOS,
    PILOT_XOL_ATTACHMENT,
    PILOT_XOL_LIMIT,
    VALUATION_YEAR,
)

from src.distributional_diagnostics import (
    build_distributional_summary,
)

from src.expected_loss_reserving import (
    aggregate_paid_to_date,
    build_expected_loss_by_accident_year,
)

from src.reinsurance import (
    apply_xol_to_payments,
)

from src.simulation import (
    simulate_portfolio,
)


DEFAULT_PRIOR = Path(
    "data/calibration/expected_loss_prior/"
    "final_2000/expected_loss_prior.csv"
)

DEFAULT_BASELINE = Path(
    "data/final/baseline_step16/results.csv"
)

DEFAULT_OUTPUT_ROOT = Path(
    "outputs/step20_expected_loss"
)


def parse_arguments() -> argparse.Namespace:
    """Read Step 20 experiment options without changing frozen defaults."""

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
        "--run-label",
        required=True,
    )

    parser.add_argument(
        "--prior",
        type=Path,
        default=DEFAULT_PRIOR,
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    return parser.parse_args()


def select_scenarios(
    requested: list[str] | None,
) -> list[dict[str, Any]]:
    """Select requested frozen scenarios while preserving configured order."""

    scenarios = [
        dict(scenario)
        for scenario in END_TO_END_SCENARIOS
    ]

    if requested is None:
        return scenarios

    requested_set = set(
        requested
    )

    selected = [
        scenario
        for scenario in scenarios
        if scenario[
            "scenario_id"
        ]
        in requested_set
    ]

    found = {
        scenario[
            "scenario_id"
        ]
        for scenario in selected
    }

    missing = (
        requested_set
        - found
    )

    if missing:
        raise ValueError(
            "Unknown scenarios: "
            f"{sorted(missing)}"
        )

    return selected


def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str:
    """Return the first available source column from an approved alias list."""

    for column in candidates:
        if column in frame.columns:
            return column

    raise ValueError(
        "None of the required column names "
        f"were found: {candidates}. "
        f"Available columns: {list(frame.columns)}"
    )


def get_seed(
    *,
    baseline: pd.DataFrame,
    scenario_id: str,
    simulation_id: int,
) -> int:
    """Recover the recorded evaluation seed for one frozen portfolio."""

    subset = baseline.loc[
        baseline[
            "scenario_id"
        ].eq(
            scenario_id
        )
        & baseline[
            "simulation_id"
        ].eq(
            simulation_id
        )
    ]

    for seed_column in [
        "random_seed",
        "seed",
    ]:
        if seed_column in subset.columns:
            values = (
                subset[
                    seed_column
                ]
                .dropna()
                .unique()
            )

            if len(values) == 1:
                return int(
                    values[0]
                )

    return int(
        END_TO_END_BASE_SEED
        + simulation_id
    )


def build_baseline_truth(
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Recover one frozen true reserve per simulation and basis."""

    poisson = baseline.loc[
        baseline[
            "model"
        ].eq(
            "regularized_poisson"
        )
        & baseline[
            "success_or_failure"
        ].eq(
            "success"
        )
    ].copy()

    key_columns = [
        "scenario_id",
        "simulation_id",
        "basis",
    ]

    truth_candidates = [
        "true_reserve",
        "true_ibnr",
        "true_future_payments",
    ]

    truth_column = next(
        (
            column
            for column in truth_candidates
            if column in poisson.columns
        ),
        None,
    )

    if truth_column is not None:
        truth = poisson[
            key_columns
            + [
                truth_column,
            ]
        ].copy()

        truth = truth.rename(
            columns={
                truth_column: (
                    "baseline_true_reserve"
                )
            }
        )

    else:
        estimate_candidates = [
            "estimated_reserve",
            "estimated_ibnr",
        ]

        estimate_column = next(
            (
                column
                for column in estimate_candidates
                if column in poisson.columns
            ),
            None,
        )

        if (
            estimate_column is None
            or "signed_error"
            not in poisson.columns
        ):
            raise ValueError(
                "Could not recover baseline true "
                "reserve from frozen results."
            )

        truth = poisson[
            key_columns
            + [
                estimate_column,
                "signed_error",
            ]
        ].copy()

        truth[
            "baseline_true_reserve"
        ] = (
            truth[
                estimate_column
            ]
            - truth[
                "signed_error"
            ]
        )

        truth = truth[
            key_columns
            + [
                "baseline_true_reserve",
            ]
        ]

    return (
        truth.drop_duplicates(
            key_columns
        )
        .reset_index(
            drop=True
        )
    )


def aggregate_future_truth(
    *,
    payments: pd.DataFrame,
    amount_column: str,
    year_column: str,
    accident_years: list[int],
) -> pd.DataFrame:
    """Aggregate future payments for evaluation only."""

    future = payments.loc[
        pd.to_numeric(
            payments[
                year_column
            ],
            errors="raise",
        )
        > VALUATION_YEAR
    ].copy()

    future[
        amount_column
    ] = pd.to_numeric(
        future[
            amount_column
        ],
        errors="raise",
    )

    return (
        future.groupby(
            "accident_year"
        )[
            amount_column
        ]
        .sum()
        .reindex(
            accident_years,
            fill_value=0.0,
        )
        .rename(
            "true_reserve"
        )
        .rename_axis(
            "accident_year"
        )
        .reset_index()
    )


def main() -> None:
    """Run and validate the Step 20 Expected Loss experiment."""

    arguments = parse_arguments()

    if arguments.simulations < 1:
        raise ValueError(
            "--simulations must be positive."
        )

    if not arguments.prior.exists():
        raise FileNotFoundError(
            arguments.prior
        )

    if not arguments.baseline.exists():
        raise FileNotFoundError(
            arguments.baseline
        )

    output_directory = (
        arguments.output_root
        / arguments.run_label
    )

    if output_directory.exists():
        raise FileExistsError(
            "Output folder already exists: "
            f"{output_directory}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    prior = pd.read_csv(
        arguments.prior
    )

    baseline = pd.read_csv(
        arguments.baseline
    )

    scenarios = select_scenarios(
        arguments.scenario
    )

    baseline_truth = (
        build_baseline_truth(
            baseline
        )
    )

    all_detail: list[
        pd.DataFrame
    ] = []

    result_rows: list[
        dict[str, Any]
    ] = []

    for scenario in scenarios:
        scenario_id = (
            scenario[
                "scenario_id"
            ]
        )

        available_ids = sorted(
            baseline.loc[
                baseline[
                    "scenario_id"
                ].eq(
                    scenario_id
                ),
                "simulation_id",
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        simulation_ids = (
            available_ids[
                : arguments.simulations
            ]
        )

        if (
            len(simulation_ids)
            != arguments.simulations
        ):
            raise ValueError(
                f"Scenario {scenario_id} has only "
                f"{len(simulation_ids)} baseline simulations."
            )

        print(
            f"\nScenario: {scenario_id}"
        )

        for position, simulation_id in enumerate(
            simulation_ids,
            start=1,
        ):
            if (
                position == 1
                or position % 10 == 0
                or position
                == len(simulation_ids)
            ):
                print(
                    f"  simulation "
                    f"{position}/"
                    f"{len(simulation_ids)}"
                )

            seed = get_seed(
                baseline=baseline,
                scenario_id=scenario_id,
                simulation_id=simulation_id,
            )

            _, simulated_payments = (
                simulate_portfolio(
                    simulation_id=(
                        simulation_id
                    ),
                    frequency_scenario=(
                        scenario[
                            "frequency_scenario"
                        ]
                    ),
                    tail_type=(
                        scenario[
                            "tail_type"
                        ]
                    ),
                    inflation_scenario=(
                        scenario[
                            "inflation_scenario"
                        ]
                    ),
                    apply_structural_break=(
                        scenario[
                            "apply_structural_break"
                        ]
                    ),
                    seed=seed,
                )
            )

            (
                reinsured_payments,
                _,
            ) = apply_xol_to_payments(
                payments=(
                    simulated_payments
                ),
                attachment=(
                    PILOT_XOL_ATTACHMENT
                ),
                limit=(
                    PILOT_XOL_LIMIT
                ),
            )

            year_column = find_column(
                reinsured_payments,
                [
                    "payment_calendar_year",
                    "calendar_year",
                ],
            )

            amount_columns = {
                "gross": find_column(
                    reinsured_payments,
                    [
                        "nominal_gross_payment",
                        "gross_payment",
                    ],
                ),
                "ceded": find_column(
                    reinsured_payments,
                    [
                        "nominal_ceded_payment",
                        "ceded_payment",
                    ],
                ),
            }

            accident_years = sorted(
                prior.loc[
                    prior[
                        "scenario_id"
                    ].eq(
                        scenario_id
                    ),
                    "accident_year",
                ]
                .astype(int)
                .unique()
                .tolist()
            )

            baseline_scenario_rows = (
                baseline.loc[
                    baseline[
                        "scenario_id"
                    ].eq(
                        scenario_id
                    )
                ]
            )

            if (
                "clause_type"
                in baseline_scenario_rows.columns
            ):
                clause_values = (
                    baseline_scenario_rows[
                        "clause_type"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                )

                clause_type = (
                    clause_values[0]
                    if len(clause_values)
                    > 0
                    else "none"
                )
            else:
                clause_type = "none"

            for basis in [
                "gross",
                "ceded",
            ]:
                amount_column = (
                    amount_columns[
                        basis
                    ]
                )

                observed_paid = (
                    aggregate_paid_to_date(
                        payments=(
                            reinsured_payments
                        ),
                        amount_column=(
                            amount_column
                        ),
                        calendar_year_column=(
                            year_column
                        ),
                        valuation_year=(
                            VALUATION_YEAR
                        ),
                        accident_years=(
                            accident_years
                        ),
                    )
                )

                # -------------------------------------------------
                # ESTIMATOR
                #
                # At this point the model receives ONLY:
                # 1. independent Step 19 prior;
                # 2. paid claims observed by valuation date.
                #
                # No true future payments enter this calculation.
                # -------------------------------------------------

                start = time.perf_counter()

                detail = (
                    build_expected_loss_by_accident_year(
                        expected_loss_prior=(
                            prior
                        ),
                        observed_paid=(
                            observed_paid
                        ),
                        scenario_id=(
                            scenario_id
                        ),
                        basis=(
                            basis
                        ),
                    )
                )

                runtime_seconds = (
                    time.perf_counter()
                    - start
                )

                # -------------------------------------------------
                # EVALUATION ONLY
                #
                # Truth is calculated only AFTER the estimate exists.
                # -------------------------------------------------

                truth = (
                    aggregate_future_truth(
                        payments=(
                            reinsured_payments
                        ),
                        amount_column=(
                            amount_column
                        ),
                        year_column=(
                            year_column
                        ),
                        accident_years=(
                            accident_years
                        ),
                    )
                )

                detail = detail.merge(
                    truth,
                    on="accident_year",
                    how="left",
                    validate="one_to_one",
                )

                detail[
                    "simulation_id"
                ] = simulation_id

                detail[
                    "seed"
                ] = seed

                detail[
                    "tail_type"
                ] = scenario[
                    "tail_type"
                ]

                detail[
                    "inflation_scenario"
                ] = scenario[
                    "inflation_scenario"
                ]

                detail[
                    "structural_break"
                ] = bool(
                    scenario[
                        "apply_structural_break"
                    ]
                )

                detail[
                    "clause_type"
                ] = clause_type

                detail[
                    "signed_error"
                ] = (
                    detail[
                        "estimated_reserve"
                    ]
                    - detail[
                        "true_reserve"
                    ]
                )

                all_detail.append(
                    detail
                )

                estimated_reserve = float(
                    detail[
                        "estimated_reserve"
                    ].sum()
                )

                true_reserve = float(
                    detail[
                        "true_reserve"
                    ].sum()
                )

                signed_error = (
                    estimated_reserve
                    - true_reserve
                )

                absolute_error = abs(
                    signed_error
                )

                if true_reserve > 0:
                    percentage_error = (
                        100.0
                        * signed_error
                        / true_reserve
                    )

                    absolute_percentage_error = abs(
                        percentage_error
                    )
                else:
                    percentage_error = np.nan
                    absolute_percentage_error = np.nan

                result_rows.append(
                    {
                        "simulation_id": (
                            simulation_id
                        ),
                        "scenario_id": (
                            scenario_id
                        ),
                        "tail_type": (
                            scenario[
                                "tail_type"
                            ]
                        ),
                        "inflation_scenario": (
                            scenario[
                                "inflation_scenario"
                            ]
                        ),
                        "structural_break": bool(
                            scenario[
                                "apply_structural_break"
                            ]
                        ),
                        "clause_type": (
                            clause_type
                        ),
                        "model": (
                            "expected_loss"
                        ),
                        "basis": (
                            basis
                        ),
                        "success_or_failure": (
                            "success"
                        ),
                        "estimated_reserve": (
                            estimated_reserve
                        ),
                        "true_reserve": (
                            true_reserve
                        ),
                        "signed_error": (
                            signed_error
                        ),
                        "absolute_error": (
                            absolute_error
                        ),
                        "percentage_error": (
                            percentage_error
                        ),
                        "absolute_percentage_error": (
                            absolute_percentage_error
                        ),
                        "runtime_seconds": (
                            runtime_seconds
                        ),
                        "seed": (
                            seed
                        ),
                        "pricing_assumption_version": (
                            detail[
                                "pricing_assumption_version"
                            ].iloc[0]
                        ),
                    }
                )

    detail_results = pd.concat(
        all_detail,
        ignore_index=True,
    )

    results = pd.DataFrame(
        result_rows
    )

    selected_scenario_ids = {
        scenario[
            "scenario_id"
        ]
        for scenario in scenarios
    }

    baseline_comparison = baseline.loc[
        baseline[
            "scenario_id"
        ].isin(
            selected_scenario_ids
        )
        & baseline[
            "model"
        ].isin(
            [
                "chain_ladder",
                "regularized_poisson",
            ]
        )
        & baseline[
            "simulation_id"
        ].isin(
            results[
                "simulation_id"
            ].unique()
        )
    ].copy()

    comparison_results = pd.concat(
        [
            baseline_comparison,
            results,
        ],
        ignore_index=True,
        sort=False,
    )

    comparison_summary = (
        build_distributional_summary(
            comparison_results
        )
    )

    # -------------------------------------------------
    # Acceptance checks
    # -------------------------------------------------

    expected_detail_rows = (
        len(scenarios)
        * arguments.simulations
        * 2
        * 15
    )

    expected_result_rows = (
        len(scenarios)
        * arguments.simulations
        * 2
    )

    formula_reconciles = bool(
        np.allclose(
            detail_results[
                "estimated_reserve"
            ],
            np.maximum(
                detail_results[
                    "expected_loss_prior_ultimate"
                ]
                - detail_results[
                    "paid_to_date"
                ],
                0.0,
            ),
            rtol=1e-12,
            atol=1e-6,
        )
    )

    total_reconciliation = (
        detail_results.groupby(
            [
                "scenario_id",
                "simulation_id",
                "basis",
            ],
            as_index=False,
        )[
            "estimated_reserve"
        ]
        .sum()
        .merge(
            results[
                [
                    "scenario_id",
                    "simulation_id",
                    "basis",
                    "estimated_reserve",
                ]
            ],
            on=[
                "scenario_id",
                "simulation_id",
                "basis",
            ],
            suffixes=(
                "_detail",
                "_result",
            ),
            validate="one_to_one",
        )
    )

    totals_reconcile = bool(
        np.allclose(
            total_reconciliation[
                "estimated_reserve_detail"
            ],
            total_reconciliation[
                "estimated_reserve_result"
            ],
            rtol=1e-12,
            atol=1e-5,
        )
    )

    generated_truth = results[
        [
            "scenario_id",
            "simulation_id",
            "basis",
            "true_reserve",
        ]
    ].merge(
        baseline_truth,
        on=[
            "scenario_id",
            "simulation_id",
            "basis",
        ],
        how="left",
        validate="one_to_one",
    )

    baseline_truth_match = bool(
        generated_truth[
            "baseline_true_reserve"
        ].notna().all()
        and np.allclose(
            generated_truth[
                "true_reserve"
            ],
            generated_truth[
                "baseline_true_reserve"
            ],
            rtol=1e-10,
            atol=1e-3,
        )
    )

    estimator_parameters = set(
        inspect.signature(
            build_expected_loss_by_accident_year
        ).parameters
    )

    no_oracle_parameters = bool(
        not any(
            any(
                forbidden
                in parameter.lower()
                for forbidden in [
                    "true",
                    "future",
                    "actual",
                ]
            )
            for parameter
            in estimator_parameters
        )
    )

    checks = [
        {
            "check": "expected_detail_row_count",
            "passed": (
                len(detail_results)
                == expected_detail_rows
            ),
            "detail": (
                f"actual={len(detail_results)}, "
                f"expected={expected_detail_rows}"
            ),
        },
        {
            "check": "expected_result_row_count",
            "passed": (
                len(results)
                == expected_result_rows
            ),
            "detail": (
                f"actual={len(results)}, "
                f"expected={expected_result_rows}"
            ),
        },
        {
            "check": "all_expected_loss_fits_successful",
            "passed": bool(
                results[
                    "success_or_failure"
                ]
                .eq(
                    "success"
                )
                .all()
            ),
            "detail": "",
        },
        {
            "check": "reserves_are_nonnegative",
            "passed": bool(
                (
                    results[
                        "estimated_reserve"
                    ]
                    >= 0
                ).all()
            ),
            "detail": "",
        },
        {
            "check": "accident_year_formula_reconciles",
            "passed": (
                formula_reconciles
            ),
            "detail": "",
        },
        {
            "check": "total_reserve_reconciles",
            "passed": (
                totals_reconcile
            ),
            "detail": "",
        },
        {
            "check": "regenerated_truth_matches_frozen_baseline",
            "passed": (
                baseline_truth_match
            ),
            "detail": (
                "Confirms the same evaluation "
                "portfolios were used."
            ),
        },
        {
            "check": "pricing_prior_version_is_correct",
            "passed": bool(
                results[
                    "pricing_assumption_version"
                ]
                .eq(
                    "pricing_mc_v1"
                )
                .all()
            ),
            "detail": "",
        },
        {
            "check": "estimator_has_no_oracle_inputs",
            "passed": (
                no_oracle_parameters
            ),
            "detail": (
                "Truth is calculated only after "
                "the reserve estimate."
            ),
        },
    ]

    acceptance_report = pd.DataFrame(
        checks
    )

    detail_results.to_csv(
        output_directory
        / "expected_loss_by_accident_year.csv",
        index=False,
    )

    results.to_csv(
        output_directory
        / "expected_loss_results.csv",
        index=False,
    )

    comparison_results.to_csv(
        output_directory
        / "comparison_results.csv",
        index=False,
    )

    comparison_summary.to_csv(
        output_directory
        / "comparison_summary.csv",
        index=False,
    )

    acceptance_report.to_csv(
        output_directory
        / "acceptance_report.csv",
        index=False,
    )

    manifest = {
        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "method": (
            "expected_loss"
        ),
        "prior_source": str(
            arguments.prior
        ),
        "baseline_source": str(
            arguments.baseline
        ),
        "valuation_year": int(
            VALUATION_YEAR
        ),
        "simulations_per_scenario": (
            arguments.simulations
        ),
        "number_of_scenarios": (
            len(scenarios)
        ),
        "expected_loss_result_rows": int(
            len(results)
        ),
        "accident_year_detail_rows": int(
            len(detail_results)
        ),
        "estimator_inputs": [
            (
                "independent Step 19 "
                "expected-loss prior"
            ),
            (
                "payments observed up to "
                "the valuation year"
            ),
        ],
        "oracle_inputs_to_estimator": [],
        "true_future_payments_used_for": (
            "evaluation_only_after_estimation"
        ),
    }

    with (
        output_directory
        / "manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print(
        "\nStep 20 completed."
    )

    print(
        f"Expected-loss result rows: "
        f"{len(results):,}"
    )

    print(
        f"Accident-year detail rows: "
        f"{len(detail_results):,}"
    )

    print(
        "\nAcceptance report:"
    )

    print(
        acceptance_report.to_string(
            index=False
        )
    )

    print(
        "\nComparison summary:"
    )

    print(
        comparison_summary[
            [
                "scenario_id",
                "model",
                "basis",
                "successful_fits",
                "success_rate",
                "mean_percentage_error",
                "mean_absolute_percentage_error",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nOutputs saved to:"
    )

    print(
        output_directory
    )

    if not acceptance_report[
        "passed"
    ].all():
        raise RuntimeError(
            "At least one Step 20 "
            "acceptance check failed."
        )


if __name__ == "__main__":
    main()

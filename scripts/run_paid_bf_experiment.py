"""Run Step 21 paid Bornhuetter-Ferguson experiment."""

from __future__ import annotations

import argparse
import inspect
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BF_BENCHMARK_INCREMENTAL_PATTERNS,
    BF_STRUCTURAL_BREAK_ACCIDENT_YEAR,
    VALUATION_YEAR,
)

from src.bornhuetter_ferguson import (
    BF_BREAK_AWARE,
    BF_STANDARD,
    build_paid_bf_by_accident_year,
)

from src.distributional_diagnostics import (
    build_distributional_summary,
)


DEFAULT_DETAIL_INPUT = Path(
    "outputs/step20_expected_loss/"
    "final_50/"
    "expected_loss_by_accident_year.csv"
)

DEFAULT_STEP20_RESULTS = Path(
    "outputs/step20_expected_loss/"
    "final_50/"
    "expected_loss_results.csv"
)

DEFAULT_BASELINE = Path(
    "data/final/baseline_step16/"
    "results.csv"
)

DEFAULT_OUTPUT_ROOT = Path(
    "outputs/step21_paid_bf"
)


def parse_arguments() -> argparse.Namespace:
    """Read Step 21 paid BF experiment options."""

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
        "--detail-input",
        type=Path,
        default=DEFAULT_DETAIL_INPUT,
    )

    parser.add_argument(
        "--step20-results",
        type=Path,
        default=DEFAULT_STEP20_RESULTS,
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


def as_bool_series(
    values: pd.Series,
) -> pd.Series:
    """Convert boolean-like values safely."""

    return (
        values
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )


def build_pattern_table() -> pd.DataFrame:
    """Create a readable table of fixed BF patterns."""

    records = []

    for pattern_name, pattern in (
        BF_BENCHMARK_INCREMENTAL_PATTERNS.items()
    ):
        cumulative = 0.0

        for development_year, proportion in enumerate(
            pattern,
            start=1,
        ):
            cumulative += float(
                proportion
            )

            records.append(
                {
                    "benchmark_pattern": (
                        pattern_name
                    ),
                    "development_year": (
                        development_year
                    ),
                    "incremental_proportion": (
                        float(
                            proportion
                        )
                    ),
                    "cumulative_paid_proportion": (
                        cumulative
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


def main() -> None:
    """Run and validate the Step 21 paid BF experiment."""

    arguments = parse_arguments()

    if arguments.simulations < 1:
        raise ValueError(
            "--simulations must be positive."
        )

    for path in [
        arguments.detail_input,
        arguments.step20_results,
        arguments.baseline,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
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

    source_detail = pd.read_csv(
        arguments.detail_input
    )

    step20_results = pd.read_csv(
        arguments.step20_results
    )

    baseline = pd.read_csv(
        arguments.baseline
    )

    available_scenarios = sorted(
        source_detail[
            "scenario_id"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if arguments.scenario is None:
        scenario_ids = (
            available_scenarios
        )
    else:
        requested = set(
            arguments.scenario
        )

        missing = (
            requested
            - set(
                available_scenarios
            )
        )

        if missing:
            raise ValueError(
                "Unknown scenarios requested: "
                f"{sorted(missing)}"
            )

        scenario_ids = sorted(
            requested
        )

    selected_parts = []

    for scenario_id in scenario_ids:
        scenario_rows = (
            source_detail.loc[
                source_detail[
                    "scenario_id"
                ].eq(
                    scenario_id
                )
            ]
        )

        simulation_ids = sorted(
            scenario_rows[
                "simulation_id"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )[
            :arguments.simulations
        ]

        if len(
            simulation_ids
        ) != arguments.simulations:
            raise ValueError(
                f"Scenario {scenario_id} has only "
                f"{len(simulation_ids)} simulations."
            )

        selected_parts.append(
            scenario_rows.loc[
                scenario_rows[
                    "simulation_id"
                ].isin(
                    simulation_ids
                )
            ]
        )

    selected_detail = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    selected_detail[
        "structural_break"
    ] = as_bool_series(
        selected_detail[
            "structural_break"
        ]
    )

    estimator_required = [
        "scenario_id",
        "simulation_id",
        "accident_year",
        "basis",
        "tail_type",
        "structural_break",
        "expected_loss_prior_ultimate",
        "paid_to_date",
        "pricing_assumption_version",
    ]

    optional_estimator_columns = [
        "inflation_scenario",
        "clause_type",
        "seed",
    ]

    estimator_columns = (
        estimator_required
        + [
            column
            for column
            in optional_estimator_columns
            if column
            in selected_detail.columns
        ]
    )

    variants = [
        BF_STANDARD,
        BF_BREAK_AWARE,
    ]

    all_detail = []
    result_rows = []

    grouped = selected_detail.groupby(
        [
            "scenario_id",
            "simulation_id",
            "basis",
        ],
        sort=True,
    )

    for (
        scenario_id,
        simulation_id,
        basis,
    ), group in grouped:
        group = (
            group.sort_values(
                "accident_year"
            )
            .reset_index(
                drop=True
            )
        )

        truth = group[
            [
                "accident_year",
                "true_reserve",
            ]
        ].copy()

        estimator_input = group[
            estimator_columns
        ].copy()

        for variant in variants:
            start = time.perf_counter()

            estimate = (
                build_paid_bf_by_accident_year(
                    estimator_input=(
                        estimator_input
                    ),
                    variant=variant,
                    valuation_year=(
                        VALUATION_YEAR
                    ),
                    structural_break_year=(
                        BF_STRUCTURAL_BREAK_ACCIDENT_YEAR
                    ),
                    benchmark_patterns=(
                        BF_BENCHMARK_INCREMENTAL_PATTERNS
                    ),
                )
            )

            runtime_seconds = (
                time.perf_counter()
                - start
            )

            # Truth is merged only AFTER estimation.
            estimate = estimate.merge(
                truth,
                on="accident_year",
                how="left",
                validate="one_to_one",
            )

            estimate[
                "signed_error"
            ] = (
                estimate[
                    "estimated_reserve"
                ]
                - estimate[
                    "true_reserve"
                ]
            )

            all_detail.append(
                estimate
            )

            estimated_reserve = float(
                estimate[
                    "estimated_reserve"
                ].sum()
            )

            true_reserve = float(
                estimate[
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

            first_row = (
                estimate.iloc[0]
            )

            result_rows.append(
                {
                    "simulation_id": int(
                        simulation_id
                    ),
                    "scenario_id": (
                        scenario_id
                    ),
                    "tail_type": (
                        first_row[
                            "tail_type"
                        ]
                    ),
                    "inflation_scenario": (
                        first_row.get(
                            "inflation_scenario",
                            "unknown",
                        )
                    ),
                    "structural_break": bool(
                        first_row[
                            "structural_break"
                        ]
                    ),
                    "clause_type": (
                        first_row.get(
                            "clause_type",
                            "none",
                        )
                    ),
                    "model": variant,
                    "basis": basis,
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
                    "pricing_assumption_version": (
                        first_row[
                            "pricing_assumption_version"
                        ]
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

    # ---------------------------------------------------------
    # Explicit misspecified-pattern sensitivity
    #
    # In post-break long-tail AYs:
    # standard BF keeps the old long-tail pattern;
    # break-aware BF uses accelerated settlement.
    # ---------------------------------------------------------

    break_mask = (
        as_bool_series(
            detail_results[
                "structural_break"
            ]
        )
        & detail_results[
            "tail_type"
        ].astype(
            str
        ).str.lower().eq(
            "long"
        )
        & (
            detail_results[
                "accident_year"
            ]
            >= BF_STRUCTURAL_BREAK_ACCIDENT_YEAR
        )
    )

    post_break = (
        detail_results.loc[
            break_mask
        ]
    )

    sensitivity_keys = [
        "scenario_id",
        "simulation_id",
        "accident_year",
        "basis",
    ]

    standard = (
        post_break.loc[
            post_break[
                "model"
            ].eq(
                BF_STANDARD
            ),
            sensitivity_keys
            + [
                "expected_loss_prior_ultimate",
                "benchmark_paid_proportion",
                "estimated_reserve",
            ],
        ]
        .rename(
            columns={
                "benchmark_paid_proportion": (
                    "standard_paid_proportion"
                ),
                "estimated_reserve": (
                    "standard_bf_reserve"
                ),
            }
        )
    )

    break_aware = (
        post_break.loc[
            post_break[
                "model"
            ].eq(
                BF_BREAK_AWARE
            ),
            sensitivity_keys
            + [
                "benchmark_paid_proportion",
                "estimated_reserve",
            ],
        ]
        .rename(
            columns={
                "benchmark_paid_proportion": (
                    "break_aware_paid_proportion"
                ),
                "estimated_reserve": (
                    "break_aware_bf_reserve"
                ),
            }
        )
    )

    if (
        not standard.empty
        and not break_aware.empty
    ):
        sensitivity = standard.merge(
            break_aware,
            on=sensitivity_keys,
            how="inner",
            validate="one_to_one",
        )

        sensitivity[
            "paid_proportion_difference"
        ] = (
            sensitivity[
                "break_aware_paid_proportion"
            ]
            - sensitivity[
                "standard_paid_proportion"
            ]
        )

        sensitivity[
            "reserve_difference_old_minus_break_aware"
        ] = (
            sensitivity[
                "standard_bf_reserve"
            ]
            - sensitivity[
                "break_aware_bf_reserve"
            ]
        )

        sensitivity_summary = (
            sensitivity.groupby(
                [
                    "scenario_id",
                    "basis",
                ],
                as_index=False,
            )
            .agg(
                post_break_rows=(
                    "accident_year",
                    "size",
                ),
                mean_standard_bf_reserve=(
                    "standard_bf_reserve",
                    "mean",
                ),
                mean_break_aware_bf_reserve=(
                    "break_aware_bf_reserve",
                    "mean",
                ),
                mean_reserve_difference=(
                    "reserve_difference_old_minus_break_aware",
                    "mean",
                ),
                total_reserve_difference=(
                    "reserve_difference_old_minus_break_aware",
                    "sum",
                ),
                mean_paid_proportion_difference=(
                    "paid_proportion_difference",
                    "mean",
                ),
            )
        )

    else:
        sensitivity = pd.DataFrame()

        sensitivity_summary = (
            pd.DataFrame()
        )

    # ---------------------------------------------------------
    # Model comparison
    # ---------------------------------------------------------

    selected_pairs = (
        results[
            [
                "scenario_id",
                "simulation_id",
            ]
        ]
        .drop_duplicates()
    )

    baseline_selected = (
        baseline.merge(
            selected_pairs,
            on=[
                "scenario_id",
                "simulation_id",
            ],
            how="inner",
            validate="many_to_one",
        )
    )

    baseline_selected = (
        baseline_selected.loc[
            baseline_selected[
                "model"
            ].isin(
                [
                    "chain_ladder",
                    "regularized_poisson",
                ]
            )
        ]
    )

    step20_selected = (
        step20_results.merge(
            selected_pairs,
            on=[
                "scenario_id",
                "simulation_id",
            ],
            how="inner",
            validate="many_to_one",
        )
    )

    comparison_results = pd.concat(
        [
            baseline_selected,
            step20_selected,
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

    # ---------------------------------------------------------
    # Acceptance checks
    # ---------------------------------------------------------

    number_of_scenarios = len(
        scenario_ids
    )

    number_of_variants = len(
        variants
    )

    number_of_bases = 2

    number_of_accident_years = (
        selected_detail[
            "accident_year"
        ]
        .nunique()
    )

    expected_detail_rows = (
        number_of_scenarios
        * arguments.simulations
        * number_of_bases
        * number_of_variants
        * number_of_accident_years
    )

    expected_result_rows = (
        number_of_scenarios
        * arguments.simulations
        * number_of_bases
        * number_of_variants
    )

    formula_reconciles = bool(
        np.allclose(
            detail_results[
                "estimated_reserve"
            ],
            detail_results[
                "expected_loss_prior_ultimate"
            ]
            * (
                1.0
                - detail_results[
                    "benchmark_paid_proportion"
                ]
            ),
            rtol=1e-12,
            atol=1e-6,
        )
    )

    proportions_valid = bool(
        detail_results[
            "benchmark_paid_proportion"
        ]
        .between(
            0.0,
            1.0,
            inclusive="both",
        )
        .all()
    )

    expected_model_basis_count = (
        number_of_scenarios
        * arguments.simulations
    )

    model_basis_counts = (
        results.groupby(
            [
                "model",
                "basis",
            ]
        )
        .size()
    )

    gross_ceded_complete = bool(
        (
            model_basis_counts
            == expected_model_basis_count
        ).all()
    )

    no_break = results.loc[
        ~as_bool_series(
            results[
                "structural_break"
            ]
        )
    ]

    standard_no_break = (
        no_break.loc[
            no_break[
                "model"
            ].eq(
                BF_STANDARD
            ),
            [
                "scenario_id",
                "simulation_id",
                "basis",
                "estimated_reserve",
            ],
        ]
        .rename(
            columns={
                "estimated_reserve": (
                    "standard_reserve"
                )
            }
        )
    )

    aware_no_break = (
        no_break.loc[
            no_break[
                "model"
            ].eq(
                BF_BREAK_AWARE
            ),
            [
                "scenario_id",
                "simulation_id",
                "basis",
                "estimated_reserve",
            ],
        ]
        .rename(
            columns={
                "estimated_reserve": (
                    "aware_reserve"
                )
            }
        )
    )

    no_break_compare = (
        standard_no_break.merge(
            aware_no_break,
            on=[
                "scenario_id",
                "simulation_id",
                "basis",
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    no_break_variants_equal = bool(
        np.allclose(
            no_break_compare[
                "standard_reserve"
            ],
            no_break_compare[
                "aware_reserve"
            ],
            rtol=1e-12,
            atol=1e-6,
        )
    )

    break_rows = detail_results.loc[
        as_bool_series(
            detail_results[
                "structural_break"
            ]
        )
    ]

    pre_break = break_rows.loc[
        break_rows[
            "accident_year"
        ]
        < BF_STRUCTURAL_BREAK_ACCIDENT_YEAR
    ]

    pre_standard = (
        pre_break.loc[
            pre_break[
                "model"
            ].eq(
                BF_STANDARD
            ),
            sensitivity_keys
            + [
                "estimated_reserve",
            ],
        ]
        .rename(
            columns={
                "estimated_reserve": (
                    "standard_reserve"
                )
            }
        )
    )

    pre_aware = (
        pre_break.loc[
            pre_break[
                "model"
            ].eq(
                BF_BREAK_AWARE
            ),
            sensitivity_keys
            + [
                "estimated_reserve",
            ],
        ]
        .rename(
            columns={
                "estimated_reserve": (
                    "aware_reserve"
                )
            }
        )
    )

    pre_compare = (
        pre_standard.merge(
            pre_aware,
            on=sensitivity_keys,
            how="inner",
            validate="one_to_one",
        )
    )

    pre_break_variants_equal = bool(
        pre_compare.empty
        or np.allclose(
            pre_compare[
                "standard_reserve"
            ],
            pre_compare[
                "aware_reserve"
            ],
            rtol=1e-12,
            atol=1e-6,
        )
    )

    post_break_aware = (
        post_break.loc[
            post_break[
                "model"
            ].eq(
                BF_BREAK_AWARE
            )
        ]
    )

    accelerated_used_correctly = bool(
        post_break_aware.empty
        or post_break_aware[
            "benchmark_pattern"
        ]
        .eq(
            "accelerated_long"
        )
        .all()
    )

    post_break_standard = (
        post_break.loc[
            post_break[
                "model"
            ].eq(
                BF_STANDARD
            )
        ]
    )

    old_pattern_used_for_sensitivity = bool(
        post_break_standard.empty
        or post_break_standard[
            "benchmark_pattern"
        ]
        .eq(
            "long"
        )
        .all()
    )

    estimator_parameters = set(
        inspect.signature(
            build_paid_bf_by_accident_year
        ).parameters
    )

    oracle_free = bool(
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

    step20_truth = (
        step20_selected[
            [
                "scenario_id",
                "simulation_id",
                "basis",
                "true_reserve",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "true_reserve": (
                    "step20_true_reserve"
                )
            }
        )
    )

    bf_truth = results.merge(
        step20_truth,
        on=[
            "scenario_id",
            "simulation_id",
            "basis",
        ],
        how="left",
        validate="many_to_one",
    )

    truth_matches_step20 = bool(
        bf_truth[
            "step20_true_reserve"
        ].notna().all()
        and np.allclose(
            bf_truth[
                "true_reserve"
            ],
            bf_truth[
                "step20_true_reserve"
            ],
            rtol=1e-12,
            atol=1e-6,
        )
    )

    has_break_scenario = bool(
        as_bool_series(
            selected_detail[
                "structural_break"
            ]
        ).any()
    )

    sensitivity_created = bool(
        (
            not has_break_scenario
        )
        or (
            not sensitivity.empty
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
            "check": "all_bf_fits_successful",
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
            "check": "bf_reserves_nonnegative",
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
            "check": "bf_formula_reconciles",
            "passed": (
                formula_reconciles
            ),
            "detail": "",
        },
        {
            "check": "benchmark_proportions_valid",
            "passed": (
                proportions_valid
            ),
            "detail": "",
        },
        {
            "check": "gross_and_ceded_results_complete",
            "passed": (
                gross_ceded_complete
            ),
            "detail": "",
        },
        {
            "check": "all_requested_scenarios_tested",
            "passed": (
                results[
                    "scenario_id"
                ].nunique()
                == number_of_scenarios
            ),
            "detail": (
                f"scenarios={number_of_scenarios}"
            ),
        },
        {
            "check": "standard_equals_break_aware_without_break",
            "passed": (
                no_break_variants_equal
            ),
            "detail": "",
        },
        {
            "check": "pre_break_variants_are_equal",
            "passed": (
                pre_break_variants_equal
            ),
            "detail": "",
        },
        {
            "check": "break_aware_uses_accelerated_post_break",
            "passed": (
                accelerated_used_correctly
            ),
            "detail": "",
        },
        {
            "check": "standard_uses_old_pattern_post_break",
            "passed": (
                old_pattern_used_for_sensitivity
            ),
            "detail": (
                "This is the deliberate "
                "pattern-misspecification case."
            ),
        },
        {
            "check": "misspecification_sensitivity_created",
            "passed": (
                sensitivity_created
            ),
            "detail": "",
        },
        {
            "check": "bf_truth_matches_step20",
            "passed": (
                truth_matches_step20
            ),
            "detail": (
                "Same validated evaluation "
                "portfolios are used."
            ),
        },
        {
            "check": "bf_estimator_has_no_oracle_inputs",
            "passed": (
                oracle_free
            ),
            "detail": (
                "Truth is merged only "
                "after BF estimation."
            ),
        },
    ]

    acceptance_report = pd.DataFrame(
        checks
    )

    pattern_table = (
        build_pattern_table()
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    detail_results.to_csv(
        output_directory
        / "bf_by_accident_year.csv",
        index=False,
    )

    results.to_csv(
        output_directory
        / "bf_results.csv",
        index=False,
    )

    pattern_table.to_csv(
        output_directory
        / "bf_benchmark_patterns.csv",
        index=False,
    )

    sensitivity.to_csv(
        output_directory
        / "pattern_misspecification_sensitivity.csv",
        index=False,
    )

    sensitivity_summary.to_csv(
        output_directory
        / "pattern_misspecification_summary.csv",
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
            "paid_bornhuetter_ferguson"
        ),
        "valuation_year": int(
            VALUATION_YEAR
        ),
        "structural_break_accident_year": int(
            BF_STRUCTURAL_BREAK_ACCIDENT_YEAR
        ),
        "variants": variants,
        "benchmark_pattern_source": (
            "fixed_design_assumptions"
        ),
        "realised_future_used_for_pattern_selection": (
            False
        ),
        "true_reserve_used_for": (
            "evaluation_only_after_estimation"
        ),
        "scenarios": scenario_ids,
        "simulations_per_scenario": int(
            arguments.simulations
        ),
        "detail_rows": int(
            len(detail_results)
        ),
        "result_rows": int(
            len(results)
        ),
        "pattern_sensitivity_rows": int(
            len(sensitivity)
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
        "\nStep 21 paid BF completed."
    )

    print(
        f"BF detail rows: "
        f"{len(detail_results):,}"
    )

    print(
        f"BF result rows: "
        f"{len(results):,}"
    )

    print(
        f"Pattern sensitivity rows: "
        f"{len(sensitivity):,}"
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

    columns = [
        "scenario_id",
        "model",
        "basis",
        "successful_fits",
        "success_rate",
        "mean_percentage_error",
        "mean_absolute_percentage_error",
    ]

    print(
        comparison_summary[
            columns
        ].to_string(
            index=False
        )
    )

    if not sensitivity_summary.empty:
        print(
            "\nPattern misspecification summary:"
        )

        print(
            sensitivity_summary.to_string(
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
            "At least one Step 21 "
            "acceptance check failed."
        )


if __name__ == "__main__":
    main()

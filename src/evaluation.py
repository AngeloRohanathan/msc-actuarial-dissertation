"""Evaluation utilities for the end-to-end reserving experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_KEY_COLUMNS = [
    "simulation_id",
    "scenario_id",
    "clause_type",
    "model",
    "basis",
]


REQUIRED_RESULT_COLUMNS = {
    "simulation_id",
    "seed",
    "scenario_id",
    "tail_type",
    "inflation_scenario",
    "structural_break",
    "clause_type",
    "model",
    "basis",
    "true_reserve",
    "estimated_reserve",
    "signed_error",
    "absolute_error",
    "percentage_error",
    "absolute_percentage_error",
    "normalised_error",
    "runtime_seconds",
    "success_or_failure",
    "failure_category",
    "failure_reason",
    "pipeline_reconciliation_passed",
    "model_reconciliation_passed",
}


def calculate_error_metrics(
    estimated_reserve: float | None,
    true_reserve: float,
) -> dict[str, float]:
    """Calculate reserve-error measures without producing infinities."""

    true_reserve = float(true_reserve)

    empty_metrics = {
        "signed_error": np.nan,
        "absolute_error": np.nan,
        "percentage_error": np.nan,
        "absolute_percentage_error": np.nan,
        "normalised_error": np.nan,
    }

    if estimated_reserve is None:
        return empty_metrics

    estimated_reserve = float(
        estimated_reserve
    )

    if not np.isfinite(
        estimated_reserve
    ):
        return empty_metrics

    signed_error = (
        estimated_reserve
        - true_reserve
    )

    absolute_error = abs(
        signed_error
    )

    if np.isclose(
        true_reserve,
        0.0,
        atol=1e-12,
    ):
        percentage_error = np.nan
        absolute_percentage_error = np.nan
        normalised_error = np.nan
    else:
        normalised_error = (
            signed_error
            / true_reserve
        )

        percentage_error = (
            100.0
            * normalised_error
        )

        absolute_percentage_error = abs(
            percentage_error
        )

    return {
        "signed_error": float(
            signed_error
        ),
        "absolute_error": float(
            absolute_error
        ),
        "percentage_error": float(
            percentage_error
        ),
        "absolute_percentage_error": float(
            absolute_percentage_error
        ),
        "normalised_error": float(
            normalised_error
        ),
    }


def classify_model_failure(
    model_name: str,
    basis: str,
    failure_reason: str,
) -> str:
    """Classify known model limitations separately from new failures."""

    chain_ladder_models = {
        "chain_ladder",
        "inflation_adjusted_chain_ladder",
        "cashflow_uplift",
    }

    if (
        basis == "ceded"
        and model_name in chain_ladder_models
        and "denominator is zero" in failure_reason.lower()
    ):
        return (
            "known_sparse_ceded_"
            "chain_ladder_limitation"
        )

    return "unexplained_failure"


def build_result_row(
    *,
    simulation_id: int,
    seed: int,
    scenario: dict[str, Any],
    clause_type: str,
    model_name: str,
    basis: str,
    true_reserve: float,
    observed_paid: float,
    true_ultimate: float,
    estimated_reserve: float | None,
    runtime_seconds: float,
    success_or_failure: str,
    failure_reason: str,
    pipeline_reconciliation_passed: bool,
    model_reconciliation_passed: bool,
    selected_alpha: float | None = None,
) -> dict[str, Any]:
    """Construct one standardised experiment-result row."""

    metrics = calculate_error_metrics(
        estimated_reserve=estimated_reserve,
        true_reserve=true_reserve,
    )

    if success_or_failure == "success":
        failure_category = ""
    else:
        failure_category = (
            classify_model_failure(
                model_name=model_name,
                basis=basis,
                failure_reason=failure_reason,
            )
        )

    row = {
        "simulation_id": int(
            simulation_id
        ),
        "seed": int(seed),
        "scenario_id": scenario[
            "scenario_id"
        ],
        "frequency_scenario": scenario[
            "frequency_scenario"
        ],
        "tail_type": scenario[
            "tail_type"
        ],
        "inflation_scenario": scenario[
            "inflation_scenario"
        ],
        "structural_break": bool(
            scenario[
                "apply_structural_break"
            ]
        ),
        "clause_type": clause_type,
        "model": model_name,
        "basis": basis,
        "observed_paid": float(
            observed_paid
        ),
        "true_reserve": float(
            true_reserve
        ),
        "true_ultimate": float(
            true_ultimate
        ),
        "estimated_reserve": (
            float(estimated_reserve)
            if estimated_reserve is not None
            else np.nan
        ),
        "selected_alpha": (
            float(selected_alpha)
            if selected_alpha is not None
            else np.nan
        ),
        "runtime_seconds": float(
            runtime_seconds
        ),
        "success_or_failure": (
            success_or_failure
        ),
        "failure_category": (
            failure_category
        ),
        "failure_reason": (
            failure_reason
        ),
        "pipeline_reconciliation_passed": bool(
            pipeline_reconciliation_passed
        ),
        "model_reconciliation_passed": bool(
            model_reconciliation_passed
        ),
    }

    row.update(metrics)

    return row


def summarise_experiment_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise success rates, errors and runtimes by scenario."""

    grouping_columns = [
        "scenario_id",
        "tail_type",
        "inflation_scenario",
        "structural_break",
        "clause_type",
        "model",
        "basis",
    ]

    summary_rows: list[
        dict[str, Any]
    ] = []

    for group_values, group in (
        results.groupby(
            grouping_columns,
            dropna=False,
        )
    ):
        group_information = dict(
            zip(
                grouping_columns,
                group_values,
            )
        )

        successful = group.loc[
            group[
                "success_or_failure"
            ]
            == "success"
        ]

        attempts = int(
            len(group)
        )

        successes = int(
            len(successful)
        )

        row = {
            **group_information,
            "attempts": attempts,
            "successes": successes,
            "failures": (
                attempts - successes
            ),
            "success_rate": (
                successes / attempts
                if attempts > 0
                else np.nan
            ),
            "mean_runtime_seconds": float(
                group[
                    "runtime_seconds"
                ].mean()
            ),
        }

        if successful.empty:
            row.update(
                {
                    "mean_signed_error": np.nan,
                    "mean_absolute_error": np.nan,
                    "root_mean_squared_error": np.nan,
                    "mean_percentage_error": np.nan,
                    "mean_absolute_percentage_error": np.nan,
                    "median_absolute_percentage_error": np.nan,
                    "mean_normalised_error": np.nan,
                }
            )
        else:
            signed_errors = successful[
                "signed_error"
            ]

            row.update(
                {
                    "mean_signed_error": float(
                        signed_errors.mean()
                    ),
                    "mean_absolute_error": float(
                        successful[
                            "absolute_error"
                        ].mean()
                    ),
                    "root_mean_squared_error": float(
                        np.sqrt(
                            np.mean(
                                signed_errors
                                ** 2
                            )
                        )
                    ),
                    "mean_percentage_error": float(
                        successful[
                            "percentage_error"
                        ].mean()
                    ),
                    "mean_absolute_percentage_error": float(
                        successful[
                            "absolute_percentage_error"
                        ].mean()
                    ),
                    "median_absolute_percentage_error": float(
                        successful[
                            "absolute_percentage_error"
                        ].median()
                    ),
                    "mean_normalised_error": float(
                        successful[
                            "normalised_error"
                        ].mean()
                    ),
                }
            )

        summary_rows.append(row)

    return pd.DataFrame(
        summary_rows
    )


def build_failure_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Count model failures by scenario, model and reason."""

    failed = results.loc[
        results["success_or_failure"]
        == "failed"
    ]

    if failed.empty:
        return pd.DataFrame(
            columns=[
                "scenario_id",
                "model",
                "basis",
                "failure_category",
                "failure_reason",
                "failure_count",
            ]
        )

    return (
        failed.groupby(
            [
                "scenario_id",
                "model",
                "basis",
                "failure_category",
                "failure_reason",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="failure_count"
        )
    )


def build_acceptance_report(
    results: pd.DataFrame,
    expected_rows: int,
) -> pd.DataFrame:
    """Evaluate the Step 16 pilot acceptance criteria."""

    missing_columns = (
        REQUIRED_RESULT_COLUMNS
        - set(results.columns)
    )

    unique_keys = not results.duplicated(
        RESULT_KEY_COLUMNS
    ).any()

    numeric_columns = [
        "true_reserve",
        "estimated_reserve",
        "signed_error",
        "absolute_error",
        "percentage_error",
        "absolute_percentage_error",
        "normalised_error",
        "runtime_seconds",
    ]

    numeric_values = (
        results[numeric_columns]
        .to_numpy(dtype=float)
    )

    no_infinities = not np.isinf(
        numeric_values
    ).any()

    successful = results.loc[
        results["success_or_failure"]
        == "success"
    ]

    failed = results.loc[
        results["success_or_failure"]
        == "failed"
    ]

    successful_estimates_are_finite = (
        np.isfinite(
            successful[
                "estimated_reserve"
            ]
        ).all()
    )

    failures_have_reasons = (
        failed["failure_reason"]
        .fillna("")
        .str.len()
        .gt(0)
        .all()
    )

    no_unexplained_failures = not (
        failed["failure_category"]
        == "unexplained_failure"
    ).any()

    all_pipeline_reconciliations_passed = (
        results[
            "pipeline_reconciliation_passed"
        ].all()
    )

    successful_model_reconciliations_passed = (
        successful[
            "model_reconciliation_passed"
        ].all()
    )

    checks = [
        {
            "check": "required_columns_present",
            "passed": (
                len(missing_columns) == 0
            ),
            "detail": (
                ""
                if not missing_columns
                else str(
                    sorted(missing_columns)
                )
            ),
        },
        {
            "check": "expected_row_count",
            "passed": (
                len(results)
                == expected_rows
            ),
            "detail": (
                f"actual={len(results)}, "
                f"expected={expected_rows}"
            ),
        },
        {
            "check": "one_unique_row_per_model_scenario",
            "passed": unique_keys,
            "detail": "",
        },
        {
            "check": "no_numeric_infinities",
            "passed": no_infinities,
            "detail": "",
        },
        {
            "check": "successful_estimates_are_finite",
            "passed": bool(
                successful_estimates_are_finite
            ),
            "detail": "",
        },
        {
            "check": "failures_have_reasons",
            "passed": bool(
                failures_have_reasons
            ),
            "detail": "",
        },
        {
            "check": "no_unexplained_model_failures",
            "passed": bool(
                no_unexplained_failures
            ),
            "detail": "",
        },
        {
            "check": "pipeline_reconciliations_passed",
            "passed": bool(
                all_pipeline_reconciliations_passed
            ),
            "detail": "",
        },
        {
            "check": "successful_model_reconciliations_passed",
            "passed": bool(
                successful_model_reconciliations_passed
            ),
            "detail": "",
        },
    ]

    return pd.DataFrame(checks)


def save_summary_plots(
    summary: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Generate pilot error and success-rate plots automatically."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for basis in [
        "gross",
        "ceded",
    ]:
        basis_summary = summary.loc[
            (
                summary["basis"] == basis
            )
            & (
                summary["successes"] > 0
            )
        ]

        if basis_summary.empty:
            continue

        pivot = basis_summary.pivot(
            index="scenario_id",
            columns="model",
            values=(
                "mean_absolute_percentage_error"
            ),
        )

        figure, axis = plt.subplots(
            figsize=(12, 6)
        )

        pivot.plot(
            kind="bar",
            ax=axis,
        )

        axis.set_title(
            "Mean absolute percentage error "
            f"— {basis}"
        )

        axis.set_xlabel(
            "Scenario"
        )

        axis.set_ylabel(
            "Mean absolute percentage error (%)"
        )

        axis.tick_params(
            axis="x",
            rotation=45,
        )

        figure.tight_layout()

        figure.savefig(
            output_directory
            / (
                "mean_absolute_percentage_error_"
                f"{basis}.png"
            ),
            dpi=200,
        )

        plt.close(figure)

    success_pivot = summary.pivot_table(
        index="model",
        columns="basis",
        values="success_rate",
        aggfunc="mean",
    )

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    success_pivot.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title(
        "Model success rate"
    )

    axis.set_xlabel(
        "Model"
    )

    axis.set_ylabel(
        "Success rate"
    )

    axis.set_ylim(
        0.0,
        1.05,
    )

    axis.tick_params(
        axis="x",
        rotation=30,
    )

    figure.tight_layout()

    figure.savefig(
        output_directory
        / "model_success_rate.png",
        dpi=200,
    )

    plt.close(figure)
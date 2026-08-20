"""Tests for experiment evaluation utilities."""

import numpy as np
import pandas as pd

from src.evaluation import (
    build_acceptance_report,
    calculate_error_metrics,
    classify_model_failure,
    summarise_experiment_results,
)


def test_error_metrics_are_correct() -> None:
    metrics = calculate_error_metrics(
        estimated_reserve=90.0,
        true_reserve=100.0,
    )

    assert np.isclose(
        metrics["signed_error"],
        -10.0,
    )

    assert np.isclose(
        metrics["absolute_error"],
        10.0,
    )

    assert np.isclose(
        metrics["percentage_error"],
        -10.0,
    )

    assert np.isclose(
        metrics["normalised_error"],
        -0.10,
    )


def test_zero_true_reserve_does_not_create_infinity() -> None:
    metrics = calculate_error_metrics(
        estimated_reserve=10.0,
        true_reserve=0.0,
    )

    assert np.isnan(
        metrics["percentage_error"]
    )

    assert np.isnan(
        metrics[
            "absolute_percentage_error"
        ]
    )

    assert np.isnan(
        metrics["normalised_error"]
    )


def test_known_ceded_failure_is_classified() -> None:
    category = classify_model_failure(
        model_name="chain_ladder",
        basis="ceded",
        failure_reason=(
            "The Chain Ladder denominator "
            "is zero for development 1 to 2."
        ),
    )

    assert category == (
        "known_sparse_ceded_"
        "chain_ladder_limitation"
    )


def test_unknown_failure_is_not_silently_accepted() -> None:
    category = classify_model_failure(
        model_name="regularized_poisson",
        basis="gross",
        failure_reason=(
            "Unexpected numerical failure."
        ),
    )

    assert category == (
        "unexplained_failure"
    )


def create_result_data() -> pd.DataFrame:
    """Create minimal valid experiment results."""

    rows = []

    for simulation_id in [
        1,
        2,
    ]:
        rows.append(
            {
                "simulation_id": (
                    simulation_id
                ),
                "seed": (
                    100 + simulation_id
                ),
                "scenario_id": (
                    "test_scenario"
                ),
                "tail_type": "long",
                "inflation_scenario": (
                    "stable"
                ),
                "structural_break": False,
                "clause_type": "none",
                "model": "chain_ladder",
                "basis": "gross",
                "true_reserve": 100.0,
                "estimated_reserve": 90.0,
                "signed_error": -10.0,
                "absolute_error": 10.0,
                "percentage_error": -10.0,
                "absolute_percentage_error": 10.0,
                "normalised_error": -0.1,
                "runtime_seconds": 0.1,
                "success_or_failure": (
                    "success"
                ),
                "failure_category": "",
                "failure_reason": "",
                "pipeline_reconciliation_passed": (
                    True
                ),
                "model_reconciliation_passed": (
                    True
                ),
            }
        )

    return pd.DataFrame(rows)


def test_summary_counts_attempts() -> None:
    results = create_result_data()

    summary = (
        summarise_experiment_results(
            results
        )
    )

    assert summary.loc[
        0,
        "attempts",
    ] == 2

    assert summary.loc[
        0,
        "successes",
    ] == 2

    assert np.isclose(
        summary.loc[
            0,
            "mean_absolute_percentage_error",
        ],
        10.0,
    )


def test_acceptance_report_detects_correct_row_count() -> None:
    results = create_result_data()

    report = build_acceptance_report(
        results=results,
        expected_rows=2,
    )

    row_count_check = report.loc[
        report["check"]
        == "expected_row_count"
    ].iloc[0]

    assert bool(
        row_count_check["passed"]
    )
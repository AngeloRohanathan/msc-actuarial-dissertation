"""Tests for Step 18 distributional diagnostics."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.distributional_diagnostics import (
    build_distributional_summary,
    save_distributional_boxplots,
)


def make_row(
    *,
    simulation_id: int,
    model: str,
    basis: str,
    status: str,
    signed_error: float | None,
    percentage_error: float | None,
) -> dict:
    """Construct one small experiment-results row."""

    if signed_error is None:
        absolute_error = np.nan
    else:
        absolute_error = abs(
            signed_error
        )

    if percentage_error is None:
        absolute_percentage_error = np.nan
    else:
        absolute_percentage_error = abs(
            percentage_error
        )

    return {
        "simulation_id": simulation_id,
        "scenario_id": "test_scenario",
        "tail_type": "long",
        "inflation_scenario": "stable",
        "structural_break": False,
        "clause_type": "none",
        "model": model,
        "basis": basis,
        "success_or_failure": status,
        "signed_error": signed_error,
        "absolute_error": absolute_error,
        "percentage_error": percentage_error,
        "absolute_percentage_error": (
            absolute_percentage_error
        ),
        "runtime_seconds": 0.1,
    }


def test_distributional_metrics_are_correct() -> None:
    """Check central diagnostics using two successful fits."""

    results = pd.DataFrame(
        [
            make_row(
                simulation_id=1,
                model="regularized_poisson",
                basis="gross",
                status="success",
                signed_error=-10.0,
                percentage_error=-10.0,
            ),
            make_row(
                simulation_id=2,
                model="regularized_poisson",
                basis="gross",
                status="success",
                signed_error=10.0,
                percentage_error=10.0,
            ),
        ]
    )

    summary = (
        build_distributional_summary(
            results
        )
    )

    row = summary.iloc[0]

    assert row[
        "attempts"
    ] == 2

    assert row[
        "successful_fits"
    ] == 2

    assert np.isclose(
        row[
            "mean_signed_error"
        ],
        0.0,
    )

    assert np.isclose(
        row[
            "median_signed_error"
        ],
        0.0,
    )

    assert np.isclose(
        row[
            "mean_absolute_percentage_error"
        ],
        10.0,
    )

    assert np.isclose(
        row[
            "median_absolute_percentage_error"
        ],
        10.0,
    )

    assert np.isclose(
        row[
            "root_mean_squared_error"
        ],
        10.0,
    )

    assert np.isclose(
        row[
            "mcse_mean_percentage_error"
        ],
        10.0,
    )

    assert row[
        "accuracy_scope"
    ] == (
        "all_attempts_successful"
    )


def test_partial_success_is_labelled_conditional() -> None:
    """Accuracy must be labelled conditional when some fits fail."""

    results = pd.DataFrame(
        [
            make_row(
                simulation_id=1,
                model="chain_ladder",
                basis="ceded",
                status="success",
                signed_error=20.0,
                percentage_error=20.0,
            ),
            make_row(
                simulation_id=2,
                model="chain_ladder",
                basis="ceded",
                status="failed",
                signed_error=None,
                percentage_error=None,
            ),
        ]
    )

    summary = (
        build_distributional_summary(
            results
        )
    )

    row = summary.iloc[0]

    assert row[
        "successful_fits"
    ] == 1

    assert np.isclose(
        row[
            "success_rate"
        ],
        0.5,
    )

    assert row[
        "accuracy_scope"
    ] == (
        "conditional_on_successful_fits"
    )

    assert bool(
        row[
            "conditional_accuracy_warning"
        ]
    )


def test_no_success_produces_missing_accuracy_metrics() -> None:
    """No-success groups must not receive artificial zero errors."""

    results = pd.DataFrame(
        [
            make_row(
                simulation_id=1,
                model="chain_ladder",
                basis="ceded",
                status="failed",
                signed_error=None,
                percentage_error=None,
            ),
            make_row(
                simulation_id=2,
                model="chain_ladder",
                basis="ceded",
                status="failed",
                signed_error=None,
                percentage_error=None,
            ),
        ]
    )

    summary = (
        build_distributional_summary(
            results
        )
    )

    row = summary.iloc[0]

    assert row[
        "successful_fits"
    ] == 0

    assert row[
        "accuracy_scope"
    ] == (
        "not_available_no_successful_fits"
    )

    assert np.isnan(
        row[
            "mean_signed_error"
        ]
    )

    assert np.isnan(
        row[
            "mean_absolute_percentage_error"
        ]
    )


def test_boxplots_are_created(
    tmp_path: Path,
) -> None:
    """Both signed and absolute percentage-error plots are saved."""

    results = pd.DataFrame(
        [
            make_row(
                simulation_id=1,
                model="regularized_poisson",
                basis="gross",
                status="success",
                signed_error=-10.0,
                percentage_error=-10.0,
            ),
            make_row(
                simulation_id=2,
                model="regularized_poisson",
                basis="gross",
                status="success",
                signed_error=5.0,
                percentage_error=5.0,
            ),
        ]
    )

    manifest = (
        save_distributional_boxplots(
            results=results,
            output_directory=(
                tmp_path
            ),
        )
    )

    assert len(
        manifest
    ) == 2

    for filename in manifest[
        "filename"
    ]:
        assert (
            tmp_path
            / filename
        ).exists()
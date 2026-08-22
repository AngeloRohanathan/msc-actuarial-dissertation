"""Tests for BF diagnostic decomposition."""

import numpy as np
import pandas as pd

from src.bf_diagnostics import (
    add_bf_error_decomposition,
)


def make_example() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario_id": ["test_break"],
            "simulation_id": [1],
            "accident_year": [2024],
            "basis": ["gross"],
            "model": [
                "bornhuetter_ferguson_standard"
            ],
            "tail_type": ["long"],
            "structural_break": [True],
            "development_year_at_valuation": [1],
            "paid_to_date": [40.0],
            "expected_loss_prior_ultimate": [100.0],
            "benchmark_paid_proportion": [0.40],
            "estimated_reserve": [60.0],
            "true_reserve": [60.0],
        }
    )


def test_true_ultimate_reconciles() -> None:
    result = add_bf_error_decomposition(
        make_example(),
        structural_break_year=2018,
    )

    assert np.isclose(
        result.loc[0, "true_ultimate"],
        100.0,
    )


def test_true_paid_proportion() -> None:
    result = add_bf_error_decomposition(
        make_example(),
        structural_break_year=2018,
    )

    assert np.isclose(
        result.loc[0, "true_paid_proportion"],
        0.40,
    )


def test_exact_error_decomposition() -> None:
    data = make_example()

    data["expected_loss_prior_ultimate"] = 90.0
    data["benchmark_paid_proportion"] = 0.50
    data["estimated_reserve"] = 45.0

    result = add_bf_error_decomposition(
        data,
        structural_break_year=2018,
    )

    assert np.isclose(
        result.loc[0, "bf_reserve_error"],
        -15.0,
    )

    assert np.isclose(
        result.loc[0, "prior_error_component"],
        -5.0,
    )

    assert np.isclose(
        result.loc[0, "development_error_component"],
        -10.0,
    )

    assert np.isclose(
        result.loc[0, "decomposition_sum"],
        -15.0,
    )

    assert np.isclose(
        result.loc[0, "decomposition_residual"],
        0.0,
    )


def test_post_break_label() -> None:
    result = add_bf_error_decomposition(
        make_example(),
        structural_break_year=2018,
    )

    assert (
        result.loc[0, "break_period"]
        == "post_break"
    )


def test_prior_percentage_error() -> None:
    data = make_example()
    data["expected_loss_prior_ultimate"] = 110.0

    result = add_bf_error_decomposition(
        data,
        structural_break_year=2018,
    )

    assert np.isclose(
        result.loc[0, "prior_ultimate_error_pct"],
        10.0,
    )
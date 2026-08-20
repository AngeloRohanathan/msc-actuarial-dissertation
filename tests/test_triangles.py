"""Tests for paid triangles and true reserve calculations."""

import numpy as np
import pandas as pd
import pytest

from config import (
    PILOT_XOL_ATTACHMENT,
    PILOT_XOL_LIMIT,
    VALUATION_YEAR,
)

from src.reinsurance import (
    apply_xol_to_payments,
)

from src.simulation import simulate_portfolio

from src.triangles import (
    build_cumulative_paid_triangle,
    build_incremental_paid_triangle,
    build_observation_mask,
    build_triangle_package,
    calculate_true_reserves,
)


@pytest.fixture
def toy_payments() -> pd.DataFrame:
    """Create a small portfolio with known observed and future values."""

    return pd.DataFrame(
        {
            "accident_year": [
                2020,
                2020,
                2020,
                2021,
                2021,
                2021,
                2022,
                2022,
                2022,
            ],
            "development_year": [
                1,
                3,
                4,
                1,
                2,
                3,
                1,
                2,
                3,
            ],
            "payment_calendar_year": [
                2020,
                2022,
                2023,
                2021,
                2022,
                2023,
                2022,
                2023,
                2024,
            ],
            "nominal_gross_payment": [
                100.0,
                25.0,
                10.0,
                200.0,
                80.0,
                40.0,
                300.0,
                90.0,
                30.0,
            ],
            "nominal_ceded_payment": [
                10.0,
                2.5,
                1.0,
                20.0,
                8.0,
                4.0,
                30.0,
                9.0,
                3.0,
            ],
        }
    )


def test_observation_mask() -> None:
    mask = build_observation_mask(
        accident_years=[
            2020,
            2021,
            2022,
        ],
        development_years=[
            1,
            2,
            3,
            4,
        ],
        valuation_year=2022,
    )

    expected = np.asarray(
        [
            [True, True, True, False],
            [True, True, False, False],
            [True, False, False, False],
        ]
    )

    np.testing.assert_array_equal(
        mask.to_numpy(),
        expected,
    )


def test_incremental_triangle_has_correct_values(
    toy_payments: pd.DataFrame,
) -> None:
    triangle = build_incremental_paid_triangle(
        payments=toy_payments,
        amount_column=(
            "nominal_gross_payment"
        ),
        valuation_year=2022,
        accident_years=[
            2020,
            2021,
            2022,
        ],
        max_development_year=4,
    )

    assert np.isclose(
        triangle.loc[2020, 1],
        100.0,
    )

    assert np.isclose(
        triangle.loc[2020, 2],
        0.0,
    )

    assert np.isclose(
        triangle.loc[2020, 3],
        25.0,
    )

    assert np.isclose(
        triangle.loc[2021, 2],
        80.0,
    )

    assert np.isclose(
        triangle.loc[2022, 1],
        300.0,
    )


def test_future_triangle_cells_remain_missing(
    toy_payments: pd.DataFrame,
) -> None:
    triangle = build_incremental_paid_triangle(
        payments=toy_payments,
        amount_column=(
            "nominal_gross_payment"
        ),
        valuation_year=2022,
        accident_years=[
            2020,
            2021,
            2022,
        ],
        max_development_year=4,
    )

    assert np.isnan(
        triangle.loc[2020, 4]
    )

    assert np.isnan(
        triangle.loc[2021, 3]
    )

    assert np.isnan(
        triangle.loc[2022, 2]
    )


def test_observed_zero_is_not_treated_as_missing(
    toy_payments: pd.DataFrame,
) -> None:
    triangle = build_incremental_paid_triangle(
        payments=toy_payments,
        amount_column=(
            "nominal_gross_payment"
        ),
        valuation_year=2022,
        accident_years=[
            2020,
            2021,
            2022,
        ],
        max_development_year=4,
    )

    assert triangle.loc[2020, 2] == 0.0
    assert not np.isnan(
        triangle.loc[2020, 2]
    )


def test_cumulative_triangle_is_correct(
    toy_payments: pd.DataFrame,
) -> None:
    incremental = (
        build_incremental_paid_triangle(
            payments=toy_payments,
            amount_column=(
                "nominal_gross_payment"
            ),
            valuation_year=2022,
            accident_years=[
                2020,
                2021,
                2022,
            ],
            max_development_year=4,
        )
    )

    cumulative = (
        build_cumulative_paid_triangle(
            incremental
        )
    )

    assert np.isclose(
        cumulative.loc[2020, 1],
        100.0,
    )

    assert np.isclose(
        cumulative.loc[2020, 2],
        100.0,
    )

    assert np.isclose(
        cumulative.loc[2020, 3],
        125.0,
    )

    assert np.isclose(
        cumulative.loc[2021, 2],
        280.0,
    )

    assert np.isnan(
        cumulative.loc[2021, 3]
    )


def test_future_payments_do_not_enter_observed_triangle(
    toy_payments: pd.DataFrame,
) -> None:
    original_triangle = (
        build_incremental_paid_triangle(
            payments=toy_payments,
            amount_column=(
                "nominal_gross_payment"
            ),
            valuation_year=2022,
            accident_years=[
                2020,
                2021,
                2022,
            ],
            max_development_year=4,
        )
    )

    altered_payments = toy_payments.copy()

    future_mask = (
        altered_payments[
            "payment_calendar_year"
        ]
        > 2022
    )

    altered_payments.loc[
        future_mask,
        "nominal_gross_payment",
    ] *= 1_000.0

    altered_triangle = (
        build_incremental_paid_triangle(
            payments=altered_payments,
            amount_column=(
                "nominal_gross_payment"
            ),
            valuation_year=2022,
            accident_years=[
                2020,
                2021,
                2022,
            ],
            max_development_year=4,
        )
    )

    np.testing.assert_allclose(
        original_triangle.to_numpy(),
        altered_triangle.to_numpy(),
        rtol=1e-12,
        atol=1e-6,
        equal_nan=True,
    )


def test_true_reserves_are_correct(
    toy_payments: pd.DataFrame,
) -> None:
    reserves, totals = (
        calculate_true_reserves(
            payments=toy_payments,
            valuation_year=2022,
            accident_years=[
                2020,
                2021,
                2022,
            ],
        )
    )

    reserves = reserves.set_index(
        "accident_year"
    )

    assert np.isclose(
        reserves.loc[
            2020,
            "true_gross_reserve",
        ],
        10.0,
    )

    assert np.isclose(
        reserves.loc[
            2021,
            "true_gross_reserve",
        ],
        40.0,
    )

    assert np.isclose(
        reserves.loc[
            2022,
            "true_gross_reserve",
        ],
        120.0,
    )

    assert np.isclose(
        totals.loc[
            0,
            "total_true_gross_reserve",
        ],
        170.0,
    )

    assert np.isclose(
        totals.loc[
            0,
            "total_true_ceded_reserve",
        ],
        17.0,
    )


def test_observed_plus_future_equals_ultimate(
    toy_payments: pd.DataFrame,
) -> None:
    reserves, _ = calculate_true_reserves(
        payments=toy_payments,
        valuation_year=2022,
        accident_years=[
            2020,
            2021,
            2022,
        ],
    )

    np.testing.assert_allclose(
        (
            reserves["observed_gross_paid"]
            + reserves["true_gross_reserve"]
        ),
        reserves["gross_ultimate"],
        rtol=1e-12,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        (
            reserves["observed_ceded_paid"]
            + reserves["true_ceded_reserve"]
        ),
        reserves["ceded_ultimate"],
        rtol=1e-12,
        atol=1e-6,
    )


def test_observed_triangle_reconciles_to_payments(
    toy_payments: pd.DataFrame,
) -> None:
    triangle = build_incremental_paid_triangle(
        payments=toy_payments,
        amount_column=(
            "nominal_gross_payment"
        ),
        valuation_year=2022,
        accident_years=[
            2020,
            2021,
            2022,
        ],
        max_development_year=4,
    )

    expected_observed_total = (
        toy_payments.loc[
            toy_payments[
                "payment_calendar_year"
            ]
            <= 2022,
            "nominal_gross_payment",
        ]
        .sum()
    )

    assert np.isclose(
        np.nansum(
            triangle.to_numpy()
        ),
        expected_observed_total,
    )


def test_gross_and_ceded_triangles_match_dimensions(
    toy_payments: pd.DataFrame,
) -> None:
    outputs = build_triangle_package(
        payments=toy_payments,
        valuation_year=2022,
        accident_years=[
            2020,
            2021,
            2022,
        ],
        max_development_year=4,
    )

    assert (
        outputs[
            "gross_incremental"
        ].shape
        ==
        outputs[
            "ceded_incremental"
        ].shape
    )

    assert (
        outputs[
            "gross_cumulative"
        ].shape
        ==
        outputs[
            "ceded_cumulative"
        ].shape
    )


def test_simulated_reinsured_portfolio_builds_triangles() -> None:
    _, simulated_payments = (
        simulate_portfolio(
            simulation_id=1,
            frequency_scenario="constant",
            tail_type="long",
            inflation_scenario="stable",
            apply_structural_break=False,
            seed=12345,
        )
    )

    (
        reinsured_payments,
        _,
    ) = apply_xol_to_payments(
        payments=simulated_payments,
        attachment=PILOT_XOL_ATTACHMENT,
        limit=PILOT_XOL_LIMIT,
    )

    outputs = build_triangle_package(
        payments=reinsured_payments,
        valuation_year=VALUATION_YEAR,
    )

    assert not outputs[
        "gross_incremental"
    ].empty

    assert not outputs[
        "ceded_incremental"
    ].empty

    assert (
        outputs[
            "gross_incremental"
        ].shape
        ==
        outputs[
            "ceded_incremental"
        ].shape
    )

    totals = outputs[
        "true_reserve_totals"
    ]

    assert (
        totals.loc[
            0,
            "total_true_gross_reserve",
        ]
        >= 0.0
    )

    assert (
        totals.loc[
            0,
            "total_true_ceded_reserve",
        ]
        >= 0.0
    )
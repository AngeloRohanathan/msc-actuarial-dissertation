"""Tests for classical reserving models."""

import numpy as np
import pandas as pd
import pytest

from src.reserving import (
    calculate_development_factors,
    cashflow_uplift,
    chain_ladder,
    inflation_adjusted_chain_ladder,
)


@pytest.fixture
def toy_cumulative_triangle() -> pd.DataFrame:
    """Cumulative triangle with hand-calculable factors."""

    return pd.DataFrame(
        {
            1: [
                100.0,
                120.0,
                140.0,
                160.0,
            ],
            2: [
                150.0,
                180.0,
                210.0,
                np.nan,
            ],
            3: [
                180.0,
                216.0,
                np.nan,
                np.nan,
            ],
            4: [
                200.0,
                np.nan,
                np.nan,
                np.nan,
            ],
        },
        index=[
            2020,
            2021,
            2022,
            2023,
        ],
    )


@pytest.fixture
def toy_incremental_triangle() -> pd.DataFrame:
    """Incremental version of the hand-worked triangle."""

    return pd.DataFrame(
        {
            1: [
                100.0,
                120.0,
                140.0,
                160.0,
            ],
            2: [
                50.0,
                60.0,
                70.0,
                np.nan,
            ],
            3: [
                30.0,
                36.0,
                np.nan,
                np.nan,
            ],
            4: [
                20.0,
                np.nan,
                np.nan,
                np.nan,
            ],
        },
        index=[
            2020,
            2021,
            2022,
            2023,
        ],
    )


def test_hand_worked_development_factors(
    toy_cumulative_triangle: pd.DataFrame,
) -> None:
    factors = calculate_development_factors(
        toy_cumulative_triangle
    )

    expected = np.asarray(
        [
            1.5,
            1.2,
            200.0 / 180.0,
        ]
    )

    np.testing.assert_allclose(
        factors[
            "development_factor"
        ].to_numpy(),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_chain_ladder_cumulative_factors(
    toy_cumulative_triangle: pd.DataFrame,
) -> None:
    result = chain_ladder(
        toy_cumulative_triangle
    )

    actual = result[
        "cumulative_development_factors"
    ]

    expected = pd.Series(
        {
            1: 2.0,
            2: 4.0 / 3.0,
            3: 10.0 / 9.0,
            4: 1.0,
        },
        name="cumulative_development_factor",
    )

    expected.index.name = (
        "development_year"
    )

    np.testing.assert_allclose(
        actual.to_numpy(),
        expected.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_chain_ladder_reserves(
    toy_cumulative_triangle: pd.DataFrame,
) -> None:
    result = chain_ladder(
        toy_cumulative_triangle
    )

    reserves = (
        result[
            "reserve_by_accident_year"
        ]
        .set_index("accident_year")
    )

    expected_reserves = {
        2020: 0.0,
        2021: 24.0,
        2022: 70.0,
        2023: 160.0,
    }

    for accident_year, expected in (
        expected_reserves.items()
    ):
        assert np.isclose(
            reserves.loc[
                accident_year,
                "estimated_reserve",
            ],
            expected,
        )

    assert np.isclose(
        result["summary"].loc[
            0,
            "total_estimated_reserve",
        ],
        254.0,
    )


def test_projected_future_increments(
    toy_cumulative_triangle: pd.DataFrame,
) -> None:
    result = chain_ladder(
        toy_cumulative_triangle
    )

    future = result[
        "projected_future_incremental"
    ]

    assert np.isclose(
        future.loc[2021, 4],
        24.0,
    )

    assert np.isclose(
        future.loc[2022, 3],
        42.0,
    )

    assert np.isclose(
        future.loc[2022, 4],
        28.0,
    )

    assert np.isclose(
        future.loc[2023, 2],
        80.0,
    )

    assert np.isclose(
        future.loc[2023, 3],
        48.0,
    )

    assert np.isclose(
        future.loc[2023, 4],
        32.0,
    )


def test_observed_cells_are_not_returned_as_future(
    toy_cumulative_triangle: pd.DataFrame,
) -> None:
    result = chain_ladder(
        toy_cumulative_triangle
    )

    future = result[
        "projected_future_incremental"
    ]

    assert np.isnan(
        future.loc[2020, 1]
    )

    assert np.isnan(
        future.loc[2021, 3]
    )

    assert np.isnan(
        future.loc[2022, 2]
    )


def test_internal_missing_value_is_rejected() -> None:
    invalid = pd.DataFrame(
        {
            1: [100.0],
            2: [np.nan],
            3: [150.0],
        },
        index=[2020],
    )

    with pytest.raises(
        ValueError,
        match="internal missing",
    ):
        chain_ladder(invalid)


def test_flat_inflation_iacl_equals_chain_ladder(
    toy_cumulative_triangle: pd.DataFrame,
    toy_incremental_triangle: pd.DataFrame,
) -> None:
    flat_index = {
        year: 1.0
        for year in range(
            2020,
            2031,
        )
    }

    chain_ladder_result = chain_ladder(
        toy_cumulative_triangle
    )

    iacl_result = (
        inflation_adjusted_chain_ladder(
            nominal_incremental_triangle=(
                toy_incremental_triangle
            ),
            inflation_index=flat_index,
            valuation_year=2023,
        )
    )

    assert np.isclose(
        chain_ladder_result[
            "summary"
        ].loc[
            0,
            "total_estimated_reserve",
        ],
        iacl_result[
            "summary"
        ].loc[
            0,
            "total_estimated_reserve",
        ],
    )


def _constant_inflation_index(
    start_year: int,
    end_year: int,
    annual_rate: float,
) -> dict[int, float]:
    """Create a cumulative inflation index."""

    output = {
        start_year: 1.0
    }

    for year in range(
        start_year + 1,
        end_year + 1,
    ):
        output[year] = (
            output[year - 1]
            * (1.0 + annual_rate)
        )

    return output


def test_cashflow_uplift_equals_chain_ladder_when_forecast_matches_embedded(
    toy_cumulative_triangle: pd.DataFrame,
) -> None:
    index = _constant_inflation_index(
        start_year=2020,
        end_year=2030,
        annual_rate=0.04,
    )

    chain_ladder_result = chain_ladder(
        toy_cumulative_triangle
    )

    uplift_result = cashflow_uplift(
        nominal_cumulative_triangle=(
            toy_cumulative_triangle
        ),
        forecast_inflation_index=index,
        valuation_year=2023,
        embedded_annual_inflation=0.04,
    )

    assert np.isclose(
        chain_ladder_result[
            "summary"
        ].loc[
            0,
            "total_estimated_reserve",
        ],
        uplift_result[
            "summary"
        ].loc[
            0,
            "total_estimated_reserve",
        ],
    )


def test_higher_forecast_inflation_increases_cashflow_uplift_reserve(
    toy_cumulative_triangle: pd.DataFrame,
) -> None:
    index: dict[int, float] = {
        2020: 1.0
    }

    for year in range(
        2021,
        2031,
    ):
        rate = (
            0.04
            if year <= 2023
            else 0.08
        )

        index[year] = (
            index[year - 1]
            * (1.0 + rate)
        )

    chain_ladder_result = chain_ladder(
        toy_cumulative_triangle
    )

    uplift_result = cashflow_uplift(
        nominal_cumulative_triangle=(
            toy_cumulative_triangle
        ),
        forecast_inflation_index=index,
        valuation_year=2023,
        embedded_annual_inflation=0.04,
    )

    standard_reserve = float(
        chain_ladder_result[
            "summary"
        ].loc[
            0,
            "total_estimated_reserve",
        ]
    )

    uplifted_reserve = float(
        uplift_result[
            "summary"
        ].loc[
            0,
            "total_estimated_reserve",
        ]
    )

    assert uplifted_reserve > standard_reserve
"""Tests for the claim and payment simulation."""

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from config import (
    ACCIDENT_YEARS,
    PARETO_SCALE,
)

from src.simulation import (
    build_inflation_index,
    calculate_frequency_mean,
    simulate_portfolio,
)


@pytest.fixture(scope="module")
def pilot_portfolio() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Generate one portfolio shared by several tests."""

    return simulate_portfolio(
        simulation_id=1,
        frequency_scenario="constant",
        tail_type="long",
        inflation_scenario="stable",
        seed=12345,
    )


def test_same_seed_produces_same_data() -> None:
    claims_1, payments_1 = simulate_portfolio(
        simulation_id=1,
        frequency_scenario="constant",
        tail_type="long",
        inflation_scenario="stable",
        seed=12345,
    )

    claims_2, payments_2 = simulate_portfolio(
        simulation_id=1,
        frequency_scenario="constant",
        tail_type="long",
        inflation_scenario="stable",
        seed=12345,
    )

    assert_frame_equal(claims_1, claims_2)
    assert_frame_equal(payments_1, payments_2)


def test_constant_frequency_mean() -> None:
    mean = calculate_frequency_mean(
        accident_year=min(ACCIDENT_YEARS),
        frequency_scenario="constant",
    )

    assert mean == 50.0


def test_decreasing_frequency_mean() -> None:
    first_year = min(ACCIDENT_YEARS)

    mean = calculate_frequency_mean(
        accident_year=first_year + 1,
        frequency_scenario="decreasing",
    )

    assert np.isclose(mean, 47.5)


def test_increasing_frequency_mean() -> None:
    first_year = min(ACCIDENT_YEARS)

    mean = calculate_frequency_mean(
        accident_year=first_year + 1,
        frequency_scenario="increasing",
    )

    assert np.isclose(mean, 52.5)


def test_claim_counts_are_non_negative_integers(
    pilot_portfolio: tuple[
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    claims, _ = pilot_portfolio

    claim_counts = (
        claims
        .groupby("accident_year")
        .size()
        .reindex(
            ACCIDENT_YEARS,
            fill_value=0,
        )
    )

    assert (claim_counts >= 0).all()

    assert all(
        float(count).is_integer()
        for count in claim_counts
    )


def test_severities_are_at_least_pareto_scale(
    pilot_portfolio: tuple[
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    claims, _ = pilot_portfolio

    assert (
        claims["ultimate_real_severity"]
        >= PARETO_SCALE
    ).all()


def test_report_year_is_correct(
    pilot_portfolio: tuple[
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    claims, _ = pilot_portfolio

    expected_report_years = (
        claims["accident_year"]
        + claims["report_delay"]
    )

    assert (
        claims["report_year"]
        == expected_report_years
    ).all()


def test_no_payment_occurs_before_reporting(
    pilot_portfolio: tuple[
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    _, payments = pilot_portfolio

    assert (
        payments["payment_calendar_year"]
        >= payments["report_year"]
    ).all()


def test_real_payments_sum_to_real_ultimate(
    pilot_portfolio: tuple[
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    claims, payments = pilot_portfolio

    payment_totals = (
        payments
        .groupby("claim_id")["real_payment"]
        .sum()
        .sort_index()
    )

    ultimate_severities = (
        claims
        .set_index("claim_id")[
            "ultimate_real_severity"
        ]
        .sort_index()
    )

    np.testing.assert_allclose(
        payment_totals.to_numpy(),
        ultimate_severities.to_numpy(),
        rtol=1e-12,
        atol=1e-6,
    )


def test_nominal_payment_calculation(
    pilot_portfolio: tuple[
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    _, payments = pilot_portfolio

    expected_nominal_payments = (
        payments["real_payment"]
        * payments["inflation_index"]
    )

    np.testing.assert_allclose(
        payments["nominal_gross_payment"],
        expected_nominal_payments,
        rtol=1e-12,
        atol=1e-6,
    )


def test_inflation_index_compounds_correctly() -> None:
    annual_rates = {
        2010: 0.04,
        2011: 0.05,
        2012: 0.10,
    }

    inflation_table = build_inflation_index(
        annual_rates=annual_rates,
        base_year=2010,
    ).set_index("calendar_year")

    assert np.isclose(
        inflation_table.loc[
            2010,
            "inflation_index",
        ],
        1.0,
    )

    assert np.isclose(
        inflation_table.loc[
            2011,
            "inflation_index",
        ],
        1.05,
    )

    assert np.isclose(
        inflation_table.loc[
            2012,
            "inflation_index",
        ],
        1.155,
    )


def test_generated_data_has_no_missing_values(
    pilot_portfolio: tuple[
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    claims, payments = pilot_portfolio

    assert not claims.isna().any().any()
    assert not payments.isna().any().any()
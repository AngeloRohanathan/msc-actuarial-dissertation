"""Tests for the expected-loss reserving method."""

import inspect

import numpy as np
import pandas as pd

from src.expected_loss_reserving import (
    aggregate_paid_to_date,
    build_expected_loss_by_accident_year,
    expected_loss_reserve,
)


def test_expected_loss_formula() -> None:
    reserve = expected_loss_reserve(
        prior_ultimate=100.0,
        paid_to_date=40.0,
    )

    assert np.isclose(
        reserve,
        60.0,
    )


def test_expected_loss_reserve_is_floored_at_zero() -> None:
    reserve = expected_loss_reserve(
        prior_ultimate=100.0,
        paid_to_date=120.0,
    )

    assert np.isclose(
        reserve,
        0.0,
    )


def test_only_observed_payments_are_used() -> None:
    payments = pd.DataFrame(
        {
            "accident_year": [
                2023,
                2023,
                2024,
            ],
            "payment_calendar_year": [
                2024,
                2025,
                2024,
            ],
            "nominal_gross_payment": [
                30.0,
                70.0,
                20.0,
            ],
        }
    )

    paid = aggregate_paid_to_date(
        payments=payments,
        amount_column=(
            "nominal_gross_payment"
        ),
        calendar_year_column=(
            "payment_calendar_year"
        ),
        valuation_year=2024,
        accident_years=[
            2023,
            2024,
        ],
    )

    year_2023 = paid.loc[
        paid[
            "accident_year"
        ].eq(
            2023
        )
    ].iloc[0]

    assert np.isclose(
        year_2023[
            "paid_to_date"
        ],
        30.0,
    )


def test_gross_and_ceded_priors_are_selected_correctly() -> None:
    prior = pd.DataFrame(
        {
            "scenario_id": [
                "test",
            ],
            "accident_year": [
                2024,
            ],
            "expected_gross_ultimate": [
                100.0,
            ],
            "expected_ceded_ultimate": [
                40.0,
            ],
            "pricing_assumption_version": [
                "pricing_mc_v1",
            ],
        }
    )

    observed = pd.DataFrame(
        {
            "accident_year": [
                2024,
            ],
            "paid_to_date": [
                25.0,
            ],
        }
    )

    gross = (
        build_expected_loss_by_accident_year(
            expected_loss_prior=prior,
            observed_paid=observed,
            scenario_id="test",
            basis="gross",
        )
    )

    ceded = (
        build_expected_loss_by_accident_year(
            expected_loss_prior=prior,
            observed_paid=observed,
            scenario_id="test",
            basis="ceded",
        )
    )

    assert np.isclose(
        gross.iloc[0][
            "estimated_reserve"
        ],
        75.0,
    )

    assert np.isclose(
        ceded.iloc[0][
            "estimated_reserve"
        ],
        15.0,
    )


def test_estimator_has_no_oracle_inputs() -> None:
    parameters = set(
        inspect.signature(
            build_expected_loss_by_accident_year
        ).parameters
    )

    forbidden_fragments = {
        "true",
        "future",
        "actual",
    }

    for parameter in parameters:
        assert not any(
            fragment in parameter.lower()
            for fragment in forbidden_fragments
        )
"""Tests for the independent expected-loss prior."""

import numpy as np
import pandas as pd

from src.expected_loss_prior import (
    build_calibration_rows,
    build_calibration_seed_schedule,
    seed_ranges_are_disjoint,
    summarise_expected_loss_prior,
)


def test_seed_schedule_is_deterministic() -> None:
    first = build_calibration_seed_schedule(
        number_of_simulations=3,
        seed_base=100,
    )

    second = build_calibration_seed_schedule(
        number_of_simulations=3,
        seed_base=100,
    )

    assert first == [
        101,
        102,
        103,
    ]

    assert first == second


def test_calibration_and_evaluation_seeds_can_be_disjoint() -> None:
    assert seed_ranges_are_disjoint(
        first_seed_base=100,
        first_count=10,
        second_seed_base=1_000,
        second_count=10,
    )

    assert not seed_ranges_are_disjoint(
        first_seed_base=100,
        first_count=10,
        second_seed_base=105,
        second_count=10,
    )


def test_build_calibration_rows_fills_missing_years() -> None:
    payments = pd.DataFrame(
        {
            "accident_year": [
                2010,
                2010,
                2012,
            ],
            "claim_id": [
                1,
                2,
                3,
            ],
            "nominal_gross_payment": [
                100.0,
                200.0,
                300.0,
            ],
            "nominal_ceded_payment": [
                10.0,
                20.0,
                30.0,
            ],
            "nominal_retained_payment": [
                90.0,
                180.0,
                270.0,
            ],
        }
    )

    scenario = {
        "scenario_id": "test",
        "frequency_scenario": "constant",
        "tail_type": "long",
        "inflation_scenario": "stable",
        "apply_structural_break": False,
    }

    rows = build_calibration_rows(
        reinsured_payments=payments,
        scenario=scenario,
        calibration_id=1,
        seed=101,
        accident_years=[
            2010,
            2011,
            2012,
        ],
    )

    assert len(rows) == 3

    year_2011 = rows.loc[
        rows[
            "accident_year"
        ]
        == 2011
    ].iloc[0]

    assert np.isclose(
        year_2011[
            "gross_ultimate"
        ],
        0.0,
    )

    assert year_2011[
        "claim_count"
    ] == 0

    year_2010 = rows.loc[
        rows[
            "accident_year"
        ]
        == 2010
    ].iloc[0]

    assert np.isclose(
        year_2010[
            "gross_ultimate"
        ],
        300.0,
    )

    assert np.isclose(
        year_2010[
            "ceded_ultimate"
        ],
        30.0,
    )


def test_prior_summary_uses_independent_sample_mean() -> None:
    records = pd.DataFrame(
        {
            "scenario_id": [
                "test",
                "test",
            ],
            "frequency_scenario": [
                "constant",
                "constant",
            ],
            "tail_type": [
                "long",
                "long",
            ],
            "inflation_scenario": [
                "stable",
                "stable",
            ],
            "structural_break": [
                False,
                False,
            ],
            "calibration_id": [
                1,
                2,
            ],
            "accident_year": [
                2010,
                2010,
            ],
            "gross_ultimate": [
                100.0,
                300.0,
            ],
            "ceded_ultimate": [
                20.0,
                60.0,
            ],
            "retained_ultimate": [
                80.0,
                240.0,
            ],
            "claim_count": [
                10,
                20,
            ],
        }
    )

    prior = summarise_expected_loss_prior(
        calibration_records=records,
        pricing_assumption_version=(
            "test_v1"
        ),
        calibration_simulations=2,
        calibration_seed_base=100,
        attachment=2.0,
        limit=5.0,
    )

    row = prior.iloc[0]

    assert np.isclose(
        row[
            "expected_gross_ultimate"
        ],
        200.0,
    )

    assert np.isclose(
        row[
            "expected_ceded_ultimate"
        ],
        40.0,
    )

    assert np.isclose(
        row[
            "expected_retained_ultimate"
        ],
        160.0,
    )

    assert np.isclose(
        row[
            "exposure_measure"
        ],
        15.0,
    )

    assert row[
        "unique_calibrations"
    ] == 2
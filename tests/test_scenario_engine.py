"""Tests for the complete Step 11 scenario engine."""

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from config import (
    CLAIMS_INFLATION_BASE_YEAR,
    INFLATION_SCENARIOS,
    PAYMENT_PATTERNS,
    PARETO_SCALE,
    PARETO_SHAPE,
    STRUCTURAL_BREAK_ACCIDENT_YEAR,
)

from src.simulation import (
    build_inflation_index,
    build_scenario_metadata,
    simulate_portfolio,
)


@pytest.fixture(scope="module")
def long_shock_break_portfolio() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Generate a long-tail shock-inflation break scenario."""

    return simulate_portfolio(
        simulation_id=1,
        frequency_scenario="constant",
        tail_type="long",
        inflation_scenario="shock",
        apply_structural_break=True,
        seed=12345,
    )


def test_same_seed_produces_identical_data() -> None:
    claims_1, payments_1 = simulate_portfolio(
        simulation_id=1,
        frequency_scenario="constant",
        tail_type="long",
        inflation_scenario="shock",
        apply_structural_break=True,
        seed=12345,
    )

    claims_2, payments_2 = simulate_portfolio(
        simulation_id=1,
        frequency_scenario="constant",
        tail_type="long",
        inflation_scenario="shock",
        apply_structural_break=True,
        seed=12345,
    )

    assert_frame_equal(claims_1, claims_2)
    assert_frame_equal(payments_1, payments_2)


def test_different_seeds_produce_different_data() -> None:
    claims_1, _ = simulate_portfolio(
        simulation_id=1,
        frequency_scenario="constant",
        tail_type="long",
        inflation_scenario="stable",
        apply_structural_break=False,
        seed=12345,
    )

    claims_2, _ = simulate_portfolio(
        simulation_id=1,
        frequency_scenario="constant",
        tail_type="long",
        inflation_scenario="stable",
        apply_structural_break=False,
        seed=54321,
    )

    assert not claims_1.equals(claims_2)


def test_all_payment_patterns_sum_to_one() -> None:
    for pattern_name, pattern in (
        PAYMENT_PATTERNS.items()
    ):
        assert np.isclose(
            sum(pattern),
            1.0,
        ), (
            f"Payment pattern {pattern_name} "
            "does not sum to one."
        )


def test_payments_reconcile_to_claim_ultimates(
    long_shock_break_portfolio: tuple[
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    claims, payments = (
        long_shock_break_portfolio
    )

    payment_totals = (
        payments
        .groupby("claim_id")["real_payment"]
        .sum()
        .sort_index()
    )

    claim_ultimates = (
        claims
        .set_index("claim_id")[
            "ultimate_real_severity"
        ]
        .sort_index()
    )

    np.testing.assert_allclose(
        payment_totals.to_numpy(),
        claim_ultimates.to_numpy(),
        rtol=1e-12,
        atol=1e-6,
    )


def test_every_inflation_index_compounds_correctly() -> None:
    for scenario_name, annual_rates in (
        INFLATION_SCENARIOS.items()
    ):
        inflation_table = build_inflation_index(
            annual_rates=annual_rates,
            base_year=(
                CLAIMS_INFLATION_BASE_YEAR
            ),
        ).set_index("calendar_year")

        assert np.isclose(
            inflation_table.loc[
                CLAIMS_INFLATION_BASE_YEAR,
                "inflation_index",
            ],
            1.0,
        )

        years = list(
            inflation_table.index
        )

        for year in years[1:]:
            previous_year = year - 1

            expected_index = (
                inflation_table.loc[
                    previous_year,
                    "inflation_index",
                ]
                * (
                    1.0
                    + inflation_table.loc[
                        year,
                        "inflation_rate",
                    ]
                )
            )

            actual_index = (
                inflation_table.loc[
                    year,
                    "inflation_index",
                ]
            )

            assert np.isclose(
                actual_index,
                expected_index,
            ), (
                "Incorrect compounding in "
                f"{scenario_name}, year {year}."
            )


def test_no_break_scenario_has_no_affected_claims() -> None:
    claims, _ = simulate_portfolio(
        simulation_id=1,
        frequency_scenario="constant",
        tail_type="long",
        inflation_scenario="stable",
        apply_structural_break=False,
        seed=12345,
    )

    assert (
        claims["structural_break_indicator"]
        == 0
    ).all()


def test_break_changes_only_intended_long_tail_claims(
    long_shock_break_portfolio: tuple[
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    claims, _ = (
        long_shock_break_portfolio
    )

    expected_indicator = (
        claims["accident_year"]
        >= STRUCTURAL_BREAK_ACCIDENT_YEAR
    ).astype(int)

    np.testing.assert_array_equal(
        claims[
            "structural_break_indicator"
        ].to_numpy(),
        expected_indicator.to_numpy(),
    )


def test_short_tail_structural_break_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="long-tail",
    ):
        simulate_portfolio(
            simulation_id=1,
            frequency_scenario="constant",
            tail_type="short",
            inflation_scenario="stable",
            apply_structural_break=True,
            seed=12345,
        )


def test_accelerated_claims_use_accelerated_pattern(
    long_shock_break_portfolio: tuple[
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    claims, payments = (
        long_shock_break_portfolio
    )

    affected_claims = claims.loc[
        claims["structural_break_indicator"]
        == 1
    ]

    assert not affected_claims.empty

    claim = affected_claims.iloc[0]

    claim_payments = (
        payments.loc[
            payments["claim_id"]
            == claim["claim_id"]
        ]
        .sort_values("payment_sequence")
    )

    assert set(
        claim_payments[
            "payment_pattern_name"
        ]
    ) == {"accelerated_long"}

    observed_proportions = (
        claim_payments["real_payment"]
        / claim["ultimate_real_severity"]
    )

    np.testing.assert_allclose(
        observed_proportions.to_numpy(),
        np.asarray(
            PAYMENT_PATTERNS[
                "accelerated_long"
            ],
            dtype=float,
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_pre_break_claims_use_original_long_pattern(
    long_shock_break_portfolio: tuple[
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    claims, payments = (
        long_shock_break_portfolio
    )

    pre_break_claims = claims.loc[
        claims["accident_year"]
        < STRUCTURAL_BREAK_ACCIDENT_YEAR
    ]

    assert not pre_break_claims.empty

    pre_break_ids = set(
        pre_break_claims["claim_id"]
    )

    pre_break_payments = payments.loc[
        payments["claim_id"].isin(
            pre_break_ids
        )
    ]

    assert set(
        pre_break_payments[
            "payment_pattern_name"
        ]
    ) == {"long"}


def test_payment_years_are_covered_by_inflation_table(
    long_shock_break_portfolio: tuple[
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    _, payments = (
        long_shock_break_portfolio
    )

    inflation_table = build_inflation_index(
        INFLATION_SCENARIOS["shock"],
        base_year=CLAIMS_INFLATION_BASE_YEAR,
    )

    covered_years = set(
        inflation_table["calendar_year"]
    )

    payment_years = set(
        payments["payment_calendar_year"]
    )

    assert payment_years.issubset(
        covered_years
    )


def test_no_payment_occurs_before_reporting(
    long_shock_break_portfolio: tuple[
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    _, payments = (
        long_shock_break_portfolio
    )

    assert (
        payments["payment_calendar_year"]
        >= payments["report_year"]
    ).all()


def test_all_scenario_labels_are_saved(
    long_shock_break_portfolio: tuple[
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    claims, payments = (
        long_shock_break_portfolio
    )

    assert set(
        claims["frequency_scenario"]
    ) == {"constant"}

    assert set(
        claims["tail_type"]
    ) == {"long"}

    assert set(
        claims["inflation_scenario"]
    ) == {"shock"}

    assert set(
        claims["apply_structural_break"]
    ) == {True}

    assert set(
        payments["frequency_scenario"]
    ) == {"constant"}

    assert set(
        payments["tail_type"]
    ) == {"long"}

    assert set(
        payments["inflation_scenario"]
    ) == {"shock"}

    assert set(
        payments["apply_structural_break"]
    ) == {True}


def test_scenario_metadata_is_correct(
    long_shock_break_portfolio: tuple[
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    claims, _ = (
        long_shock_break_portfolio
    )

    metadata = build_scenario_metadata(
        claims=claims,
        simulation_id=1,
        seed=12345,
        tail_type="long",
        frequency_scenario="constant",
        inflation_scenario="shock",
        apply_structural_break=True,
    )

    assert len(metadata) == 1

    row = metadata.iloc[0]

    assert row["simulation_id"] == 1
    assert row["seed"] == 12345
    assert row["tail_type"] == "long"

    assert (
        row["frequency_scenario"]
        == "constant"
    )

    assert (
        row["inflation_scenario"]
        == "shock"
    )

    assert bool(
        row["structural_break"]
    )

    assert (
        row["structural_break_year"]
        == STRUCTURAL_BREAK_ACCIDENT_YEAR
    )

    assert np.isclose(
        row["pareto_scale"],
        PARETO_SCALE,
    )

    assert np.isclose(
        row["pareto_shape"],
        PARETO_SHAPE,
    )

    assert row["claim_count"] == len(claims)
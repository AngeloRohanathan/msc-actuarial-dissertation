"""Tests for the basic Excess-of-Loss treaty."""

import numpy as np
import pandas as pd
import pytest

from config import (
    PILOT_XOL_ATTACHMENT,
    PILOT_XOL_LIMIT,
)

from src.reinsurance import (
    apply_xol_to_payments,
    calculate_xol_ceded,
)

from src.simulation import simulate_portfolio


@pytest.mark.parametrize(
    (
        "claim_amount",
        "expected_ceded",
    ),
    [
        (1_000_000.0, 0.0),
        (2_000_000.0, 0.0),
        (3_000_000.0, 1_000_000.0),
        (7_000_000.0, 5_000_000.0),
        (10_000_000.0, 5_000_000.0),
    ],
)
def test_hand_worked_xol_examples(
    claim_amount: float,
    expected_ceded: float,
) -> None:
    actual_ceded = calculate_xol_ceded(
        claim_amount=claim_amount,
        attachment=2_000_000.0,
        limit=5_000_000.0,
    )

    assert np.isclose(
        actual_ceded,
        expected_ceded,
    )


@pytest.mark.parametrize(
    (
        "attachment",
        "limit",
    ),
    [
        (-1.0, 5_000_000.0),
        (2_000_000.0, 0.0),
        (2_000_000.0, -1.0),
    ],
)
def test_invalid_treaty_terms_are_rejected(
    attachment: float,
    limit: float,
) -> None:
    with pytest.raises(ValueError):
        calculate_xol_ceded(
            claim_amount=5_000_000.0,
            attachment=attachment,
            limit=limit,
        )


def test_negative_claim_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="negative",
    ):
        calculate_xol_ceded(
            claim_amount=-1.0,
            attachment=2_000_000.0,
            limit=5_000_000.0,
        )


@pytest.fixture
def cumulative_payment_example() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "simulation_id": [1, 1, 1, 1],
            "claim_id": [
                "TEST_CLAIM",
                "TEST_CLAIM",
                "TEST_CLAIM",
                "TEST_CLAIM",
            ],
            "payment_sequence": [1, 2, 3, 4],
            "payment_calendar_year": [
                2024,
                2025,
                2026,
                2027,
            ],
            "nominal_gross_payment": [
                1_000_000.0,
                1_500_000.0,
                3_000_000.0,
                4_000_000.0,
            ],
        }
    )


def test_cumulative_xol_payment_allocation(
    cumulative_payment_example: pd.DataFrame,
) -> None:
    reinsured, summary = apply_xol_to_payments(
        payments=cumulative_payment_example,
        attachment=2_000_000.0,
        limit=5_000_000.0,
    )

    expected_ceded_payments = np.asarray(
        [
            0.0,
            500_000.0,
            3_000_000.0,
            1_500_000.0,
        ]
    )

    expected_retained_payments = np.asarray(
        [
            1_000_000.0,
            1_000_000.0,
            0.0,
            2_500_000.0,
        ]
    )

    np.testing.assert_allclose(
        reinsured[
            "nominal_ceded_payment"
        ].to_numpy(),
        expected_ceded_payments,
        rtol=1e-12,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        reinsured[
            "nominal_retained_payment"
        ].to_numpy(),
        expected_retained_payments,
        rtol=1e-12,
        atol=1e-6,
    )

    assert np.isclose(
        summary.loc[
            0,
            "nominal_gross_ultimate",
        ],
        9_500_000.0,
    )

    assert np.isclose(
        summary.loc[
            0,
            "nominal_ceded_ultimate",
        ],
        5_000_000.0,
    )

    assert np.isclose(
        summary.loc[
            0,
            "nominal_retained_ultimate",
        ],
        4_500_000.0,
    )


def test_gross_equals_ceded_plus_retained(
    cumulative_payment_example: pd.DataFrame,
) -> None:
    reinsured, _ = apply_xol_to_payments(
        payments=cumulative_payment_example,
        attachment=2_000_000.0,
        limit=5_000_000.0,
    )

    expected_gross = (
        reinsured["nominal_ceded_payment"]
        + reinsured["nominal_retained_payment"]
    )

    np.testing.assert_allclose(
        reinsured["nominal_gross_payment"],
        expected_gross,
        rtol=1e-12,
        atol=1e-6,
    )


def test_treaty_limit_is_not_exceeded(
    cumulative_payment_example: pd.DataFrame,
) -> None:
    reinsured, summary = apply_xol_to_payments(
        payments=cumulative_payment_example,
        attachment=2_000_000.0,
        limit=5_000_000.0,
    )

    assert (
        reinsured[
            "cumulative_nominal_ceded_amount"
        ]
        <= 5_000_000.0 + 1e-6
    ).all()

    assert (
        summary["nominal_ceded_ultimate"]
        <= 5_000_000.0 + 1e-6
    ).all()


def test_simulated_portfolio_reconciles() -> None:
    _, payments = simulate_portfolio(
        simulation_id=1,
        frequency_scenario="constant",
        tail_type="long",
        inflation_scenario="stable",
        apply_structural_break=False,
        seed=12345,
    )

    reinsured, claim_summary = (
        apply_xol_to_payments(
            payments=payments,
            attachment=PILOT_XOL_ATTACHMENT,
            limit=PILOT_XOL_LIMIT,
        )
    )

    np.testing.assert_allclose(
        reinsured["nominal_gross_payment"],
        (
            reinsured["nominal_ceded_payment"]
            + reinsured[
                "nominal_retained_payment"
            ]
        ),
        rtol=1e-12,
        atol=1e-6,
    )

    assert (
        claim_summary[
            "nominal_ceded_ultimate"
        ]
        <= PILOT_XOL_LIMIT + 1e-6
    ).all()

    assert len(claim_summary) == (
        payments["claim_id"].nunique()
    )


def test_claims_below_attachment_have_zero_ceded() -> None:
    payments = pd.DataFrame(
        {
            "simulation_id": [1, 1],
            "claim_id": [
                "SMALL_CLAIM",
                "SMALL_CLAIM",
            ],
            "payment_sequence": [1, 2],
            "payment_calendar_year": [
                2024,
                2025,
            ],
            "nominal_gross_payment": [
                750_000.0,
                750_000.0,
            ],
        }
    )

    reinsured, summary = apply_xol_to_payments(
        payments=payments,
        attachment=2_000_000.0,
        limit=5_000_000.0,
    )

    assert np.isclose(
        reinsured[
            "nominal_ceded_payment"
        ].sum(),
        0.0,
    )

    assert np.isclose(
        summary.loc[
            0,
            "nominal_ceded_ultimate",
        ],
        0.0,
    )
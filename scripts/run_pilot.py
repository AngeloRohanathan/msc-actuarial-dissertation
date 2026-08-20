"""Run and validate the first dissertation pilot experiment.

The first pilot uses:

- one simulation;
- constant claim frequency;
- a long-tail portfolio;
- stable inflation;
- the original long-tail payment pattern;
- no structural break;
- no reinsurance;
- no reserving model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    MASTER_RANDOM_SEED,
    PAYMENT_PATTERNS,
    PILOT_DATA_DIR,
    ensure_directories,
    validate_config,
)

from src.simulation import simulate_portfolio


CLAIMS_OUTPUT_PATH = (
    PILOT_DATA_DIR / "claims_pilot.parquet"
)

PAYMENTS_OUTPUT_PATH = (
    PILOT_DATA_DIR / "payments_pilot.parquet"
)


def validate_pilot_outputs(
    claims: pd.DataFrame,
    payments: pd.DataFrame,
) -> None:
    """Check that the output matches the first-pilot specification."""

    if claims.empty:
        raise ValueError(
            "The pilot claim table is empty."
        )

    if payments.empty:
        raise ValueError(
            "The pilot payment table is empty."
        )

    if claims.isna().any().any():
        raise ValueError(
            "The pilot claim table contains missing values."
        )

    if payments.isna().any().any():
        raise ValueError(
            "The pilot payment table contains missing values."
        )

    if set(claims["simulation_id"]) != {1}:
        raise ValueError(
            "The pilot must contain only simulation_id 1."
        )

    if set(claims["frequency_scenario"]) != {
        "constant"
    }:
        raise ValueError(
            "The pilot must use constant claim frequency."
        )

    if set(claims["tail_type"]) != {"long"}:
        raise ValueError(
            "The pilot must contain only long-tail claims."
        )

    if set(payments["inflation_scenario"]) != {
        "stable"
    }:
        raise ValueError(
            "The pilot must use stable inflation."
        )

    if (
        claims["structural_break_indicator"] != 0
    ).any():
        raise ValueError(
            "The structural break must be turned off "
            "for the first pilot."
        )

    payment_pattern_length = len(
        PAYMENT_PATTERNS["long"]
    )

    payments_per_claim = (
        payments
        .groupby("claim_id")
        .size()
    )

    if not (
        payments_per_claim
        == payment_pattern_length
    ).all():
        raise ValueError(
            "Each claim should have exactly "
            f"{payment_pattern_length} payments under "
            "the original long-tail pattern."
        )

    expected_payment_rows = (
        len(claims)
        * payment_pattern_length
    )

    if len(payments) != expected_payment_rows:
        raise ValueError(
            "The number of payment rows does not "
            "reconcile with the number of claims."
        )

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

    if not payment_totals.index.equals(
        ultimate_severities.index
    ):
        raise ValueError(
            "The claim IDs in the claim and payment "
            "tables do not reconcile."
        )

    if not np.allclose(
        payment_totals.to_numpy(),
        ultimate_severities.to_numpy(),
        rtol=1e-12,
        atol=1e-6,
    ):
        raise ValueError(
            "Real payments do not sum to real "
            "ultimate claim severities."
        )

    expected_nominal_payments = (
        payments["real_payment"]
        * payments["inflation_index"]
    )

    if not np.allclose(
        payments["nominal_gross_payment"],
        expected_nominal_payments,
        rtol=1e-12,
        atol=1e-6,
    ):
        raise ValueError(
            "Nominal payments do not equal real "
            "payments multiplied by the inflation index."
        )

    if not (
        payments["payment_calendar_year"]
        >= payments["report_year"]
    ).all():
        raise ValueError(
            "At least one payment occurs before "
            "the claim is reported."
        )


def save_and_verify_parquet(
    claims: pd.DataFrame,
    payments: pd.DataFrame,
) -> None:
    """Save the pilot data and verify that it can be read back."""

    claims.to_parquet(
        CLAIMS_OUTPUT_PATH,
        index=False,
        engine="pyarrow",
    )

    payments.to_parquet(
        PAYMENTS_OUTPUT_PATH,
        index=False,
        engine="pyarrow",
    )

    reloaded_claims = pd.read_parquet(
        CLAIMS_OUTPUT_PATH,
        engine="pyarrow",
    )

    reloaded_payments = pd.read_parquet(
        PAYMENTS_OUTPUT_PATH,
        engine="pyarrow",
    )

    pd.testing.assert_frame_equal(
        claims.reset_index(drop=True),
        reloaded_claims.reset_index(drop=True),
    )

    pd.testing.assert_frame_equal(
        payments.reset_index(drop=True),
        reloaded_payments.reset_index(drop=True),
    )


def main() -> None:
    """Run, validate and save the first pilot experiment."""

    ensure_directories()
    validate_config()

    claims, payments = simulate_portfolio(
        simulation_id=1,
        frequency_scenario="constant",
        tail_type="long",
        inflation_scenario="stable",
        apply_structural_break=False,
        seed=MASTER_RANDOM_SEED,
    )

    validate_pilot_outputs(
        claims=claims,
        payments=payments,
    )

    save_and_verify_parquet(
        claims=claims,
        payments=payments,
    )

    total_real_ultimate = (
        claims["ultimate_real_severity"].sum()
    )

    total_nominal_payments = (
        payments["nominal_gross_payment"].sum()
    )

    print("Pilot experiment completed successfully.")
    print(f"Claims: {len(claims):,}")
    print(f"Payments: {len(payments):,}")

    print(
        "Total real ultimate: "
        f"£{total_real_ultimate:,.2f}"
    )

    print(
        "Total nominal gross payments: "
        f"£{total_nominal_payments:,.2f}"
    )

    print(
        "Claims saved to: "
        f"{CLAIMS_OUTPUT_PATH}"
    )

    print(
        "Payments saved to: "
        f"{PAYMENTS_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
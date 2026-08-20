"""Excess-of-Loss reinsurance calculations.

This module implements the baseline fixed, unindexed,
per-claim Excess-of-Loss treaty.

Recoveries are calculated using cumulative nominal claim
payments. Indexation and stability clauses are deliberately
excluded at this stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    PILOT_XOL_ATTACHMENT,
    PILOT_XOL_LIMIT,
)


REQUIRED_PAYMENT_COLUMNS = {
    "claim_id",
    "payment_sequence",
    "payment_calendar_year",
    "nominal_gross_payment",
}


REINSURANCE_OUTPUT_COLUMNS = [
    "treaty_attachment",
    "treaty_limit",
    "clause_type",
    "treaty_basis",
    "cumulative_nominal_gross_payment",
    "cumulative_nominal_ceded_amount",
    "nominal_ceded_payment",
    "nominal_retained_payment",
]


def validate_treaty_terms(
    attachment: float,
    limit: float,
) -> None:
    """Validate the attachment and limit of an XoL treaty."""

    if isinstance(attachment, bool):
        raise TypeError(
            "attachment must be a numerical amount."
        )

    if isinstance(limit, bool):
        raise TypeError(
            "limit must be a numerical amount."
        )

    if not isinstance(
        attachment,
        (int, float, np.number),
    ):
        raise TypeError(
            "attachment must be a numerical amount."
        )

    if not isinstance(
        limit,
        (int, float, np.number),
    ):
        raise TypeError(
            "limit must be a numerical amount."
        )

    if not np.isfinite(attachment):
        raise ValueError(
            "attachment must be finite."
        )

    if not np.isfinite(limit):
        raise ValueError(
            "limit must be finite."
        )

    if attachment < 0:
        raise ValueError(
            "attachment cannot be negative."
        )

    if limit <= 0:
        raise ValueError(
            "limit must be positive."
        )


def calculate_xol_ceded(
    claim_amount: float,
    attachment: float = PILOT_XOL_ATTACHMENT,
    limit: float = PILOT_XOL_LIMIT,
) -> float:
    """Calculate the total ceded amount for one claim.

    Parameters
    ----------
    claim_amount:
        Total nominal amount of the claim.

    attachment:
        Amount retained before the treaty begins to respond.

    limit:
        Maximum amount recoverable from the reinsurer.

    Returns
    -------
    float
        Total amount ceded to the reinsurer.
    """

    validate_treaty_terms(
        attachment=attachment,
        limit=limit,
    )

    if isinstance(claim_amount, bool):
        raise TypeError(
            "claim_amount must be numerical."
        )

    if not isinstance(
        claim_amount,
        (int, float, np.number),
    ):
        raise TypeError(
            "claim_amount must be numerical."
        )

    if not np.isfinite(claim_amount):
        raise ValueError(
            "claim_amount must be finite."
        )

    if claim_amount < 0:
        raise ValueError(
            "claim_amount cannot be negative."
        )

    ceded_amount = min(
        max(
            float(claim_amount)
            - float(attachment),
            0.0,
        ),
        float(limit),
    )

    return float(ceded_amount)


def _get_claim_group_columns(
    payments: pd.DataFrame,
) -> list[str]:
    """Determine the columns uniquely identifying each claim.

    scenario_id is included when present because the same claim ID
    may occur under several paired scenarios.
    """

    group_columns: list[str] = []

    if "scenario_id" in payments.columns:
        group_columns.append("scenario_id")

    if "simulation_id" in payments.columns:
        group_columns.append("simulation_id")

    group_columns.append("claim_id")

    return group_columns


def validate_payment_input(
    payments: pd.DataFrame,
) -> None:
    """Validate payment data before applying reinsurance."""

    missing_columns = (
        REQUIRED_PAYMENT_COLUMNS
        - set(payments.columns)
    )

    if missing_columns:
        raise ValueError(
            "Payment data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if payments.empty:
        raise ValueError(
            "Payment data cannot be empty."
        )

    if payments[
        list(REQUIRED_PAYMENT_COLUMNS)
    ].isna().any().any():
        raise ValueError(
            "Required payment columns contain missing values."
        )

    if (
        payments["nominal_gross_payment"] < 0
    ).any():
        raise ValueError(
            "Nominal gross payments cannot be negative."
        )

    group_columns = _get_claim_group_columns(
        payments
    )

    duplicate_columns = (
        group_columns
        + ["payment_sequence"]
    )

    if payments.duplicated(
        duplicate_columns
    ).any():
        raise ValueError(
            "A claim contains duplicate payment sequences."
        )


def apply_xol_to_payments(
    payments: pd.DataFrame,
    attachment: float = PILOT_XOL_ATTACHMENT,
    limit: float = PILOT_XOL_LIMIT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a fixed per-claim XoL treaty to payment data.

    Recoveries are calculated on a cumulative paid basis:

        cumulative ceded
        = min(max(cumulative gross - attachment, 0), limit)

    Incremental ceded payments are calculated as the change in
    cumulative ceded amounts between consecutive payment periods.

    Parameters
    ----------
    payments:
        Payment-level data from the simulation engine.

    attachment:
        Fixed nominal attachment amount.

    limit:
        Fixed nominal treaty limit.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Reinsured payment-level data followed by claim-level
        treaty summaries.
    """

    validate_treaty_terms(
        attachment=attachment,
        limit=limit,
    )

    validate_payment_input(payments)

    output = payments.copy()

    group_columns = _get_claim_group_columns(
        output
    )

    sort_columns = (
        group_columns
        + [
            "payment_calendar_year",
            "payment_sequence",
        ]
    )

    output = (
        output
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )

    output["treaty_attachment"] = float(
        attachment
    )

    output["treaty_limit"] = float(limit)

    output["clause_type"] = "none"

    output["treaty_basis"] = (
        "cumulative_nominal_paid"
    )

    output[
        "cumulative_nominal_gross_payment"
    ] = (
        output
        .groupby(
            group_columns,
            sort=False,
        )["nominal_gross_payment"]
        .cumsum()
    )

    previous_cumulative_gross = (
        output[
            "cumulative_nominal_gross_payment"
        ]
        - output["nominal_gross_payment"]
    )

    output[
        "cumulative_nominal_ceded_amount"
    ] = (
        output[
            "cumulative_nominal_gross_payment"
        ]
        .sub(float(attachment))
        .clip(
            lower=0.0,
            upper=float(limit),
        )
    )

    previous_cumulative_ceded = (
        previous_cumulative_gross
        .sub(float(attachment))
        .clip(
            lower=0.0,
            upper=float(limit),
        )
    )

    output["nominal_ceded_payment"] = (
        output[
            "cumulative_nominal_ceded_amount"
        ]
        - previous_cumulative_ceded
    )

    # Remove any tiny negative values caused by floating-point
    # arithmetic.
    output["nominal_ceded_payment"] = (
        output["nominal_ceded_payment"]
        .clip(lower=0.0)
    )

    output["nominal_retained_payment"] = (
        output["nominal_gross_payment"]
        - output["nominal_ceded_payment"]
    )

    label_columns = [
        column
        for column in [
            "accident_year",
            "tail_type",
            "frequency_scenario",
            "inflation_scenario",
            "apply_structural_break",
            "structural_break_indicator",
        ]
        if (
            column in output.columns
            and column not in group_columns
        )
    ]

    aggregations: dict[str, str] = {
        column: "first"
        for column in label_columns
    }

    aggregations.update(
        {
            "nominal_gross_payment": "sum",
            "nominal_ceded_payment": "sum",
            "nominal_retained_payment": "sum",
        }
    )

    claim_summary = (
        output
        .groupby(
            group_columns,
            as_index=False,
            sort=False,
        )
        .agg(aggregations)
        .rename(
            columns={
                "nominal_gross_payment": (
                    "nominal_gross_ultimate"
                ),
                "nominal_ceded_payment": (
                    "nominal_ceded_ultimate"
                ),
                "nominal_retained_payment": (
                    "nominal_retained_ultimate"
                ),
            }
        )
    )

    claim_summary["treaty_attachment"] = float(
        attachment
    )

    claim_summary["treaty_limit"] = float(
        limit
    )

    claim_summary["clause_type"] = "none"

    claim_summary["treaty_basis"] = (
        "cumulative_nominal_paid"
    )

    claim_summary["attachment_breached"] = (
        claim_summary["nominal_ceded_ultimate"]
        > 0.0
    )

    claim_summary["limit_exhausted"] = np.isclose(
        claim_summary["nominal_ceded_ultimate"],
        float(limit),
        rtol=1e-10,
        atol=1e-6,
    )

    claim_summary["ceded_share"] = np.where(
        claim_summary["nominal_gross_ultimate"]
        > 0.0,
        (
            claim_summary[
                "nominal_ceded_ultimate"
            ]
            / claim_summary[
                "nominal_gross_ultimate"
            ]
        ),
        0.0,
    )

    validate_xol_outputs(
        reinsured_payments=output,
        claim_summary=claim_summary,
        attachment=attachment,
        limit=limit,
    )

    return output, claim_summary


def validate_xol_outputs(
    reinsured_payments: pd.DataFrame,
    claim_summary: pd.DataFrame,
    attachment: float,
    limit: float,
) -> None:
    """Validate payment-level and claim-level XoL outputs."""

    validate_treaty_terms(
        attachment=attachment,
        limit=limit,
    )

    required_output_columns = {
        "nominal_gross_payment",
        "nominal_ceded_payment",
        "nominal_retained_payment",
        "cumulative_nominal_ceded_amount",
    }

    missing_columns = (
        required_output_columns
        - set(reinsured_payments.columns)
    )

    if missing_columns:
        raise ValueError(
            "Reinsurance output is missing columns: "
            f"{sorted(missing_columns)}"
        )

    expected_gross = (
        reinsured_payments[
            "nominal_ceded_payment"
        ]
        + reinsured_payments[
            "nominal_retained_payment"
        ]
    )

    if not np.allclose(
        reinsured_payments[
            "nominal_gross_payment"
        ],
        expected_gross,
        rtol=1e-12,
        atol=1e-6,
    ):
        raise ValueError(
            "Gross payments do not reconcile with "
            "ceded and retained payments."
        )

    if (
        reinsured_payments[
            "nominal_ceded_payment"
        ]
        < -1e-6
    ).any():
        raise ValueError(
            "A ceded payment is negative."
        )

    if (
        reinsured_payments[
            "nominal_retained_payment"
        ]
        < -1e-6
    ).any():
        raise ValueError(
            "A retained payment is negative."
        )

    if (
        reinsured_payments[
            "cumulative_nominal_ceded_amount"
        ]
        > float(limit) + 1e-6
    ).any():
        raise ValueError(
            "A claim exceeds the treaty limit."
        )

    expected_claim_ceded = (
        claim_summary["nominal_gross_ultimate"]
        .apply(
            lambda gross_amount: (
                calculate_xol_ceded(
                    claim_amount=float(
                        gross_amount
                    ),
                    attachment=attachment,
                    limit=limit,
                )
            )
        )
    )

    if not np.allclose(
        claim_summary[
            "nominal_ceded_ultimate"
        ],
        expected_claim_ceded,
        rtol=1e-12,
        atol=1e-6,
    ):
        raise ValueError(
            "Claim-level ceded amounts do not match "
            "the XoL treaty formula."
        )

    expected_retained = (
        claim_summary["nominal_gross_ultimate"]
        - claim_summary["nominal_ceded_ultimate"]
    )

    if not np.allclose(
        claim_summary[
            "nominal_retained_ultimate"
        ],
        expected_retained,
        rtol=1e-12,
        atol=1e-6,
    ):
        raise ValueError(
            "Claim-level retained amounts do not reconcile."
        )
"""Expected-loss reserving using an independent pricing prior."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


BASIS_TO_PRIOR_COLUMN = {
    "gross": "expected_gross_ultimate",
    "ceded": "expected_ceded_ultimate",
}


def expected_loss_reserve(
    *,
    prior_ultimate: float,
    paid_to_date: float,
) -> float:
    """Calculate a non-negative expected-loss reserve."""

    prior_ultimate = float(
        prior_ultimate
    )

    paid_to_date = float(
        paid_to_date
    )

    if not np.isfinite(
        prior_ultimate
    ):
        raise ValueError(
            "prior_ultimate must be finite."
        )

    if not np.isfinite(
        paid_to_date
    ):
        raise ValueError(
            "paid_to_date must be finite."
        )

    if prior_ultimate < 0:
        raise ValueError(
            "prior_ultimate cannot be negative."
        )

    if paid_to_date < 0:
        raise ValueError(
            "paid_to_date cannot be negative."
        )

    return float(
        max(
            prior_ultimate
            - paid_to_date,
            0.0,
        )
    )


def aggregate_paid_to_date(
    *,
    payments: pd.DataFrame,
    amount_column: str,
    calendar_year_column: str,
    valuation_year: int,
    accident_years: Iterable[int],
) -> pd.DataFrame:
    """Aggregate only payments observable by the valuation date."""

    required = {
        "accident_year",
        amount_column,
        calendar_year_column,
    }

    missing = (
        required
        - set(
            payments.columns
        )
    )

    if missing:
        raise ValueError(
            "Payment data are missing columns: "
            f"{sorted(missing)}"
        )

    accident_years = [
        int(year)
        for year in accident_years
    ]

    observed = payments.loc[
        pd.to_numeric(
            payments[
                calendar_year_column
            ],
            errors="raise",
        )
        <= int(
            valuation_year
        )
    ].copy()

    observed[
        amount_column
    ] = pd.to_numeric(
        observed[
            amount_column
        ],
        errors="raise",
    )

    paid = (
        observed.groupby(
            "accident_year"
        )[
            amount_column
        ]
        .sum()
        .reindex(
            accident_years,
            fill_value=0.0,
        )
        .rename(
            "paid_to_date"
        )
        .rename_axis(
            "accident_year"
        )
        .reset_index()
    )

    return paid


def build_expected_loss_by_accident_year(
    *,
    expected_loss_prior: pd.DataFrame,
    observed_paid: pd.DataFrame,
    scenario_id: str,
    basis: str,
) -> pd.DataFrame:
    """Apply the expected-loss method by accident year."""

    if basis not in BASIS_TO_PRIOR_COLUMN:
        raise ValueError(
            "basis must be either "
            "'gross' or 'ceded'."
        )

    prior_column = (
        BASIS_TO_PRIOR_COLUMN[
            basis
        ]
    )

    required_prior_columns = {
        "scenario_id",
        "accident_year",
        prior_column,
        "pricing_assumption_version",
    }

    missing_prior = (
        required_prior_columns
        - set(
            expected_loss_prior.columns
        )
    )

    if missing_prior:
        raise ValueError(
            "Expected-loss prior is missing "
            f"columns: {sorted(missing_prior)}"
        )

    required_paid_columns = {
        "accident_year",
        "paid_to_date",
    }

    missing_paid = (
        required_paid_columns
        - set(
            observed_paid.columns
        )
    )

    if missing_paid:
        raise ValueError(
            "Observed paid data are missing "
            f"columns: {sorted(missing_paid)}"
        )

    scenario_prior = (
        expected_loss_prior.loc[
            expected_loss_prior[
                "scenario_id"
            ].eq(
                scenario_id
            ),
            [
                "scenario_id",
                "accident_year",
                prior_column,
                "pricing_assumption_version",
            ],
        ]
        .copy()
    )

    if scenario_prior.empty:
        raise ValueError(
            "No prior rows found for scenario: "
            f"{scenario_id}"
        )

    if scenario_prior[
        "accident_year"
    ].duplicated().any():
        raise ValueError(
            "The prior contains duplicate "
            "accident-year rows."
        )

    if observed_paid[
        "accident_year"
    ].duplicated().any():
        raise ValueError(
            "Observed-paid data contain duplicate "
            "accident-year rows."
        )

    scenario_prior = (
        scenario_prior.rename(
            columns={
                prior_column: (
                    "expected_loss_prior_ultimate"
                )
            }
        )
    )

    result = scenario_prior.merge(
        observed_paid,
        on="accident_year",
        how="left",
        validate="one_to_one",
    )

    result[
        "paid_to_date"
    ] = (
        result[
            "paid_to_date"
        ]
        .fillna(
            0.0
        )
        .astype(float)
    )

    result[
        "expected_loss_prior_ultimate"
    ] = pd.to_numeric(
        result[
            "expected_loss_prior_ultimate"
        ],
        errors="raise",
    ).astype(float)

    if (
        result[
            "expected_loss_prior_ultimate"
        ]
        < 0
    ).any():
        raise ValueError(
            "Prior ultimates cannot be negative."
        )

    if (
        result[
            "paid_to_date"
        ]
        < 0
    ).any():
        raise ValueError(
            "Paid-to-date amounts cannot be negative."
        )

    result[
        "estimated_reserve"
    ] = np.maximum(
        result[
            "expected_loss_prior_ultimate"
        ]
        - result[
            "paid_to_date"
        ],
        0.0,
    )

    result[
        "estimated_ultimate"
    ] = (
        result[
            "paid_to_date"
        ]
        + result[
            "estimated_reserve"
        ]
    )

    result[
        "prior_exhausted"
    ] = (
        result[
            "paid_to_date"
        ]
        >= result[
            "expected_loss_prior_ultimate"
        ]
    )

    result[
        "basis"
    ] = basis

    result[
        "model"
    ] = "expected_loss"

    return (
        result.sort_values(
            "accident_year"
        )
        .reset_index(
            drop=True
        )
    )
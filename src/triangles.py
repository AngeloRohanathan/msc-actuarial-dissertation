"""Paid-triangle construction and true-reserve calculations.

This module separates the complete simulated payment history into:

1. Payments observed by the valuation date.
2. Future payments that are unknown at the valuation date.

Observed payments are aggregated into incremental and cumulative
paid triangles. Future payments are retained only for evaluating
the true outstanding reserve.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from config import (
    ACCIDENT_YEARS,
    VALUATION_YEAR,
)


REQUIRED_TRIANGLE_COLUMNS = {
    "accident_year",
    "development_year",
    "payment_calendar_year",
    "nominal_gross_payment",
    "nominal_ceded_payment",
}


TRIANGLE_OUTPUT_KEYS = {
    "gross_incremental",
    "gross_cumulative",
    "ceded_incremental",
    "ceded_cumulative",
    "true_reserves_by_accident_year",
    "true_reserve_totals",
}


def validate_triangle_input(
    payments: pd.DataFrame,
) -> None:
    """Validate payment data before triangle construction."""

    if payments.empty:
        raise ValueError(
            "Payment data cannot be empty."
        )

    missing_columns = (
        REQUIRED_TRIANGLE_COLUMNS
        - set(payments.columns)
    )

    if missing_columns:
        raise ValueError(
            "Payment data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    required_columns = sorted(
        REQUIRED_TRIANGLE_COLUMNS
    )

    if payments[
        required_columns
    ].isna().any().any():
        raise ValueError(
            "Required triangle columns contain missing values."
        )

    if (
        payments["development_year"] < 1
    ).any():
        raise ValueError(
            "Development years must begin at 1."
        )

    if (
        payments["nominal_gross_payment"] < 0
    ).any():
        raise ValueError(
            "Nominal gross payments cannot be negative."
        )

    if (
        payments["nominal_ceded_payment"] < 0
    ).any():
        raise ValueError(
            "Nominal ceded payments cannot be negative."
        )

    if (
        payments["nominal_ceded_payment"]
        >
        payments["nominal_gross_payment"] + 1e-6
    ).any():
        raise ValueError(
            "A ceded payment exceeds its gross payment."
        )

    expected_development_year = (
        payments["payment_calendar_year"]
        - payments["accident_year"]
        + 1
    )

    if not (
        payments["development_year"]
        == expected_development_year
    ).all():
        raise ValueError(
            "Development years are inconsistent with "
            "accident and payment calendar years."
        )

    configured_accident_years = set(
        ACCIDENT_YEARS
    )

    payment_accident_years = set(
        payments["accident_year"]
    )

    unexpected_accident_years = (
        payment_accident_years
        - configured_accident_years
    )

    if unexpected_accident_years:
        raise ValueError(
            "Payment data contains unconfigured accident years: "
            f"{sorted(unexpected_accident_years)}"
        )

    # A combined scenario or Monte Carlo dataset should be split
    # before constructing an individual reserving triangle.
    for column in [
        "scenario_id",
        "simulation_id",
        "clause_type",
    ]:
        if (
            column in payments.columns
            and payments[column].nunique(
                dropna=False
            ) > 1
        ):
            raise ValueError(
                "Triangle construction requires one portfolio "
                f"at a time, but {column} contains multiple values."
            )


def build_observation_mask(
    accident_years: Sequence[int],
    development_years: Sequence[int],
    valuation_year: int,
) -> pd.DataFrame:
    """Identify triangle cells observed by the valuation year.

    A cell is observable where:

        accident year + development year - 1
        <= valuation year
    """

    mask = pd.DataFrame(
        [
            [
                (
                    int(accident_year)
                    + int(development_year)
                    - 1
                    <= int(valuation_year)
                )
                for development_year
                in development_years
            ]
            for accident_year in accident_years
        ],
        index=list(accident_years),
        columns=list(development_years),
        dtype=bool,
    )

    mask.index.name = "accident_year"
    mask.columns.name = "development_year"

    return mask


def build_incremental_paid_triangle(
    payments: pd.DataFrame,
    amount_column: str,
    valuation_year: int = VALUATION_YEAR,
    accident_years: Sequence[int] = ACCIDENT_YEARS,
    max_development_year: int | None = None,
) -> pd.DataFrame:
    """Construct an observed incremental paid triangle.

    Observed cells with no payment are recorded as zero.

    Unobserved future cells are recorded as NaN, because a future
    cell is unknown rather than an observed zero.
    """

    validate_triangle_input(payments)

    if amount_column not in payments.columns:
        raise ValueError(
            f"Unknown amount column: {amount_column}"
        )

    accident_years = sorted(
        int(year)
        for year in accident_years
    )

    if not accident_years:
        raise ValueError(
            "accident_years cannot be empty."
        )

    maximum_actual_development = int(
        payments["development_year"].max()
    )

    if max_development_year is None:
        max_development_year = (
            maximum_actual_development
        )

    if max_development_year < 1:
        raise ValueError(
            "max_development_year must be positive."
        )

    if (
        max_development_year
        < maximum_actual_development
    ):
        raise ValueError(
            "max_development_year cannot be smaller "
            "than the maximum development year in the data."
        )

    development_years = list(
        range(
            1,
            int(max_development_year) + 1,
        )
    )

    observed_payments = payments.loc[
        payments["payment_calendar_year"]
        <= int(valuation_year)
    ]

    if observed_payments.empty:
        grouped = pd.DataFrame(
            index=accident_years,
            columns=development_years,
            dtype=float,
        )
    else:
        grouped = (
            observed_payments
            .groupby(
                [
                    "accident_year",
                    "development_year",
                ]
            )[amount_column]
            .sum()
            .unstack("development_year")
        )

    # Reindex creates the full rectangular grid.
    # Missing observed combinations are then interpreted as
    # genuine observed zero payments.
    triangle = (
        grouped
        .reindex(
            index=accident_years,
            columns=development_years,
        )
        .fillna(0.0)
        .astype(float)
    )

    observation_mask = build_observation_mask(
        accident_years=accident_years,
        development_years=development_years,
        valuation_year=valuation_year,
    )

    # This replaces future cells with NaN while retaining observed
    # zero-payment cells as 0.0.
    triangle = triangle.where(
        observation_mask,
        np.nan,
    )

    triangle.index.name = "accident_year"
    triangle.columns.name = "development_year"

    return triangle


def build_cumulative_paid_triangle(
    incremental_triangle: pd.DataFrame,
) -> pd.DataFrame:
    """Convert an incremental paid triangle to cumulative form."""

    if incremental_triangle.empty:
        raise ValueError(
            "Incremental triangle cannot be empty."
        )

    cumulative_triangle = (
        incremental_triangle
        .cumsum(
            axis=1,
            skipna=True,
        )
    )

    # Preserve the missing future cells. Without this mask, later
    # transformations could accidentally treat missing cells as
    # observed values.
    cumulative_triangle = (
        cumulative_triangle.where(
            incremental_triangle.notna()
        )
    )

    cumulative_triangle.index.name = (
        incremental_triangle.index.name
    )

    cumulative_triangle.columns.name = (
        incremental_triangle.columns.name
    )

    return cumulative_triangle


def calculate_true_reserves(
    payments: pd.DataFrame,
    valuation_year: int = VALUATION_YEAR,
    accident_years: Sequence[int] = ACCIDENT_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate true gross and ceded outstanding reserves.

    The simulated future is used only to calculate the truth against
    which reserve estimates will later be assessed.
    """

    validate_triangle_input(payments)

    accident_years = sorted(
        int(year)
        for year in accident_years
    )

    observed_mask = (
        payments["payment_calendar_year"]
        <= int(valuation_year)
    )

    future_mask = ~observed_mask

    result = pd.DataFrame(
        index=accident_years
    )

    result.index.name = "accident_year"

    amount_columns = {
        "gross": "nominal_gross_payment",
        "ceded": "nominal_ceded_payment",
    }

    for basis, amount_column in (
        amount_columns.items()
    ):
        observed_by_year = (
            payments.loc[observed_mask]
            .groupby("accident_year")[
                amount_column
            ]
            .sum()
            .reindex(
                accident_years,
                fill_value=0.0,
            )
        )

        future_by_year = (
            payments.loc[future_mask]
            .groupby("accident_year")[
                amount_column
            ]
            .sum()
            .reindex(
                accident_years,
                fill_value=0.0,
            )
        )

        ultimate_by_year = (
            payments
            .groupby("accident_year")[
                amount_column
            ]
            .sum()
            .reindex(
                accident_years,
                fill_value=0.0,
            )
        )

        result[
            f"observed_{basis}_paid"
        ] = observed_by_year.astype(float)

        result[
            f"true_{basis}_reserve"
        ] = future_by_year.astype(float)

        result[
            f"{basis}_ultimate"
        ] = ultimate_by_year.astype(float)

        if not np.allclose(
            observed_by_year + future_by_year,
            ultimate_by_year,
            rtol=1e-12,
            atol=1e-6,
        ):
            raise ValueError(
                f"Observed and future {basis} payments "
                "do not reconcile with ultimate payments."
            )

    reserves_by_accident_year = (
        result.reset_index()
    )

    true_reserve_totals = pd.DataFrame(
        [
            {
                "valuation_year": int(
                    valuation_year
                ),
                "total_observed_gross_paid": float(
                    result[
                        "observed_gross_paid"
                    ].sum()
                ),
                "total_true_gross_reserve": float(
                    result[
                        "true_gross_reserve"
                    ].sum()
                ),
                "total_gross_ultimate": float(
                    result[
                        "gross_ultimate"
                    ].sum()
                ),
                "total_observed_ceded_paid": float(
                    result[
                        "observed_ceded_paid"
                    ].sum()
                ),
                "total_true_ceded_reserve": float(
                    result[
                        "true_ceded_reserve"
                    ].sum()
                ),
                "total_ceded_ultimate": float(
                    result[
                        "ceded_ultimate"
                    ].sum()
                ),
            }
        ]
    )

    return (
        reserves_by_accident_year,
        true_reserve_totals,
    )


def build_triangle_package(
    payments: pd.DataFrame,
    valuation_year: int = VALUATION_YEAR,
    accident_years: Sequence[int] = ACCIDENT_YEARS,
    max_development_year: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Build all Step 13 triangle and reserve outputs."""

    validate_triangle_input(payments)

    if max_development_year is None:
        max_development_year = int(
            payments["development_year"].max()
        )

    gross_incremental = (
        build_incremental_paid_triangle(
            payments=payments,
            amount_column=(
                "nominal_gross_payment"
            ),
            valuation_year=valuation_year,
            accident_years=accident_years,
            max_development_year=(
                max_development_year
            ),
        )
    )

    ceded_incremental = (
        build_incremental_paid_triangle(
            payments=payments,
            amount_column=(
                "nominal_ceded_payment"
            ),
            valuation_year=valuation_year,
            accident_years=accident_years,
            max_development_year=(
                max_development_year
            ),
        )
    )

    gross_cumulative = (
        build_cumulative_paid_triangle(
            gross_incremental
        )
    )

    ceded_cumulative = (
        build_cumulative_paid_triangle(
            ceded_incremental
        )
    )

    (
        true_reserves_by_accident_year,
        true_reserve_totals,
    ) = calculate_true_reserves(
        payments=payments,
        valuation_year=valuation_year,
        accident_years=accident_years,
    )

    outputs = {
        "gross_incremental": (
            gross_incremental
        ),
        "gross_cumulative": (
            gross_cumulative
        ),
        "ceded_incremental": (
            ceded_incremental
        ),
        "ceded_cumulative": (
            ceded_cumulative
        ),
        "true_reserves_by_accident_year": (
            true_reserves_by_accident_year
        ),
        "true_reserve_totals": (
            true_reserve_totals
        ),
    }

    validate_triangle_package(
        payments=payments,
        outputs=outputs,
        valuation_year=valuation_year,
    )

    return outputs


def validate_triangle_package(
    payments: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    valuation_year: int = VALUATION_YEAR,
) -> None:
    """Perform internal reconciliation checks on triangle outputs."""

    missing_outputs = (
        TRIANGLE_OUTPUT_KEYS
        - set(outputs)
    )

    if missing_outputs:
        raise ValueError(
            "Triangle package is missing outputs: "
            f"{sorted(missing_outputs)}"
        )

    gross_incremental = outputs[
        "gross_incremental"
    ]

    gross_cumulative = outputs[
        "gross_cumulative"
    ]

    ceded_incremental = outputs[
        "ceded_incremental"
    ]

    ceded_cumulative = outputs[
        "ceded_cumulative"
    ]

    reserves = outputs[
        "true_reserves_by_accident_year"
    ]

    totals = outputs[
        "true_reserve_totals"
    ]

    if gross_incremental.shape != (
        ceded_incremental.shape
    ):
        raise ValueError(
            "Gross and ceded incremental triangles "
            "have different dimensions."
        )

    if gross_cumulative.shape != (
        ceded_cumulative.shape
    ):
        raise ValueError(
            "Gross and ceded cumulative triangles "
            "have different dimensions."
        )

    expected_gross_cumulative = (
        build_cumulative_paid_triangle(
            gross_incremental
        )
    )

    expected_ceded_cumulative = (
        build_cumulative_paid_triangle(
            ceded_incremental
        )
    )

    if not np.allclose(
        gross_cumulative.to_numpy(),
        expected_gross_cumulative.to_numpy(),
        rtol=1e-12,
        atol=1e-6,
        equal_nan=True,
    ):
        raise ValueError(
            "Gross cumulative triangle does not equal "
            "the cumulative sum of the incremental triangle."
        )

    if not np.allclose(
        ceded_cumulative.to_numpy(),
        expected_ceded_cumulative.to_numpy(),
        rtol=1e-12,
        atol=1e-6,
        equal_nan=True,
    ):
        raise ValueError(
            "Ceded cumulative triangle does not equal "
            "the cumulative sum of the incremental triangle."
        )

    observed_payments = payments.loc[
        payments["payment_calendar_year"]
        <= int(valuation_year)
    ]

    expected_observed_gross = float(
        observed_payments[
            "nominal_gross_payment"
        ].sum()
    )

    expected_observed_ceded = float(
        observed_payments[
            "nominal_ceded_payment"
        ].sum()
    )

    if not np.isclose(
        np.nansum(
            gross_incremental.to_numpy()
        ),
        expected_observed_gross,
        rtol=1e-12,
        atol=1e-6,
    ):
        raise ValueError(
            "Gross incremental triangle does not "
            "reconcile with observed gross payments."
        )

    if not np.isclose(
        np.nansum(
            ceded_incremental.to_numpy()
        ),
        expected_observed_ceded,
        rtol=1e-12,
        atol=1e-6,
    ):
        raise ValueError(
            "Ceded incremental triangle does not "
            "reconcile with observed ceded payments."
        )

    observation_mask = build_observation_mask(
        accident_years=(
            gross_incremental.index.tolist()
        ),
        development_years=(
            gross_incremental.columns.tolist()
        ),
        valuation_year=valuation_year,
    )

    future_positions = (
        ~observation_mask.to_numpy()
    )

    if not np.isnan(
        gross_incremental.to_numpy()[
            future_positions
        ]
    ).all():
        raise ValueError(
            "A future gross triangle cell is not missing."
        )

    if not np.isnan(
        ceded_incremental.to_numpy()[
            future_positions
        ]
    ).all():
        raise ValueError(
            "A future ceded triangle cell is not missing."
        )

    if not np.allclose(
        (
            reserves["observed_gross_paid"]
            + reserves["true_gross_reserve"]
        ),
        reserves["gross_ultimate"],
        rtol=1e-12,
        atol=1e-6,
    ):
        raise ValueError(
            "Gross reserve components do not reconcile."
        )

    if not np.allclose(
        (
            reserves["observed_ceded_paid"]
            + reserves["true_ceded_reserve"]
        ),
        reserves["ceded_ultimate"],
        rtol=1e-12,
        atol=1e-6,
    ):
        raise ValueError(
            "Ceded reserve components do not reconcile."
        )

    if not np.isclose(
        totals.loc[
            0,
            "total_true_gross_reserve",
        ],
        reserves[
            "true_gross_reserve"
        ].sum(),
        rtol=1e-12,
        atol=1e-6,
    ):
        raise ValueError(
            "Total gross reserve does not reconcile."
        )

    if not np.isclose(
        totals.loc[
            0,
            "total_true_ceded_reserve",
        ],
        reserves[
            "true_ceded_reserve"
        ].sum(),
        rtol=1e-12,
        atol=1e-6,
    ):
        raise ValueError(
            "Total ceded reserve does not reconcile."
        )
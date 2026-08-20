"""Classical loss-reserving models.

This module implements:

1. Deterministic Chain Ladder.
2. Inflation-Adjusted Chain Ladder.
3. A stylised Cashflow Uplift benchmark.

The models operate only on the observed upper triangle. Simulated
future payments are not supplied to any reserving model.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


CHAIN_LADDER_OUTPUT_KEYS = {
    "development_factors",
    "cumulative_development_factors",
    "projected_cumulative_triangle",
    "projected_future_incremental",
    "reserve_by_accident_year",
    "summary",
}


def _prepare_triangle(
    triangle: pd.DataFrame,
    *,
    cumulative: bool,
) -> pd.DataFrame:
    """Validate and standardise a reserving triangle."""

    if not isinstance(triangle, pd.DataFrame):
        raise TypeError(
            "triangle must be a pandas DataFrame."
        )

    if triangle.empty:
        raise ValueError(
            "triangle cannot be empty."
        )

    output = triangle.copy()

    try:
        output.index = [
            int(value)
            for value in output.index
        ]

        output.columns = [
            int(value)
            for value in output.columns
        ]

    except (TypeError, ValueError) as error:
        raise ValueError(
            "Triangle index and columns must be "
            "convertible to integers."
        ) from error

    if output.index.duplicated().any():
        raise ValueError(
            "Triangle contains duplicate accident years."
        )

    if output.columns.duplicated().any():
        raise ValueError(
            "Triangle contains duplicate development years."
        )

    output = output.sort_index()
    output = output.sort_index(axis=1)
    output = output.astype(float)

    development_years = list(output.columns)

    expected_development_years = list(
        range(
            1,
            max(development_years) + 1,
        )
    )

    if development_years != expected_development_years:
        raise ValueError(
            "Development years must be consecutive "
            "and begin at 1."
        )

    if output.iloc[:, 0].isna().any():
        raise ValueError(
            "Development year 1 must be observed "
            "for every accident year."
        )

    if (
        output.stack().to_numpy()
        < -1e-6
    ).any():
        raise ValueError(
            "Observed triangle values cannot be negative."
        )

    # Each row must contain observed values followed by missing
    # future values. An observed value cannot appear after a gap.
    for accident_year, row in output.iterrows():
        observed = row.notna().to_numpy()

        first_missing = np.flatnonzero(
            ~observed
        )

        if len(first_missing) > 0:
            first_missing_position = int(
                first_missing[0]
            )

            if observed[
                first_missing_position:
            ].any():
                raise ValueError(
                    "Triangle contains an internal missing "
                    f"value for accident year {accident_year}."
                )

    if cumulative:
        differences = output.diff(axis=1)

        adjacent_observed = (
            output.notna()
            & output.shift(axis=1).notna()
        )

        negative_differences = (
            differences.where(adjacent_observed)
            < -1e-6
        )

        if negative_differences.any().any():
            raise ValueError(
                "A cumulative triangle decreases between "
                "observed development periods."
            )

    output.index.name = "accident_year"
    output.columns.name = "development_year"

    return output


def prepare_cumulative_triangle(
    triangle: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and standardise a cumulative triangle."""

    return _prepare_triangle(
        triangle,
        cumulative=True,
    )


def prepare_incremental_triangle(
    triangle: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and standardise an incremental triangle."""

    return _prepare_triangle(
        triangle,
        cumulative=False,
    )


def incremental_to_cumulative(
    incremental_triangle: pd.DataFrame,
) -> pd.DataFrame:
    """Convert an observed incremental triangle to cumulative form."""

    incremental = prepare_incremental_triangle(
        incremental_triangle
    )

    cumulative = incremental.cumsum(
        axis=1,
        skipna=True,
    )

    cumulative = cumulative.where(
        incremental.notna()
    )

    cumulative.index.name = "accident_year"
    cumulative.columns.name = "development_year"

    return cumulative


def calculate_development_factors(
    cumulative_triangle: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate volume-weighted age-to-age factors.

    For development period j:

        f_j = sum(C_(i,j+1)) / sum(C_(i,j))

    Only rows in which both development periods are observed
    are used.
    """

    triangle = prepare_cumulative_triangle(
        cumulative_triangle
    )

    development_years = list(
        triangle.columns
    )

    rows: list[dict[str, float | int]] = []

    for position in range(
        len(development_years) - 1
    ):
        current_development = (
            development_years[position]
        )

        next_development = (
            development_years[position + 1]
        )

        current_values = triangle[
            current_development
        ]

        next_values = triangle[
            next_development
        ]

        paired_observations = (
            current_values.notna()
            & next_values.notna()
        )

        rows_used = int(
            paired_observations.sum()
        )

        if rows_used == 0:
            raise ValueError(
                "No paired observations are available "
                f"for development {current_development} "
                f"to {next_development}."
            )

        denominator = float(
            current_values.loc[
                paired_observations
            ].sum()
        )

        numerator = float(
            next_values.loc[
                paired_observations
            ].sum()
        )

        if denominator <= 0.0:
            raise ValueError(
                "The Chain Ladder denominator is zero "
                f"for development {current_development} "
                f"to {next_development}."
            )

        factor = numerator / denominator

        if factor < 1.0 - 1e-10:
            raise ValueError(
                "A paid cumulative development factor "
                "is below 1. Check that the cumulative "
                "triangle is non-decreasing."
            )

        rows.append(
            {
                "from_development_year": int(
                    current_development
                ),
                "to_development_year": int(
                    next_development
                ),
                "numerator": numerator,
                "denominator": denominator,
                "development_factor": float(
                    factor
                ),
                "rows_used": rows_used,
            }
        )

    return pd.DataFrame(rows)


def calculate_cumulative_development_factors(
    development_factors: pd.DataFrame,
    development_years: list[int],
) -> pd.Series:
    """Calculate factors from each development age to ultimate."""

    required_columns = {
        "from_development_year",
        "development_factor",
    }

    missing_columns = (
        required_columns
        - set(development_factors.columns)
    )

    if missing_columns:
        raise ValueError(
            "Development-factor table is missing: "
            f"{sorted(missing_columns)}"
        )

    factor_lookup = (
        development_factors
        .set_index("from_development_year")[
            "development_factor"
        ]
        .to_dict()
    )

    maximum_development = max(
        development_years
    )

    cumulative_factors: dict[int, float] = {}

    for development_year in development_years:
        factor = 1.0

        for current_year in range(
            int(development_year),
            int(maximum_development),
        ):
            if current_year not in factor_lookup:
                raise ValueError(
                    "Missing development factor from "
                    f"development year {current_year}."
                )

            factor *= float(
                factor_lookup[current_year]
            )

        cumulative_factors[
            int(development_year)
        ] = float(factor)

    output = pd.Series(
        cumulative_factors,
        name="cumulative_development_factor",
        dtype=float,
    )

    output.index.name = "development_year"

    return output


def project_cumulative_triangle(
    cumulative_triangle: pd.DataFrame,
    development_factors: pd.DataFrame,
) -> pd.DataFrame:
    """Fill future cumulative cells using Chain Ladder factors."""

    triangle = prepare_cumulative_triangle(
        cumulative_triangle
    )

    projected = triangle.copy()

    factor_lookup = (
        development_factors
        .set_index("from_development_year")[
            "development_factor"
        ]
        .to_dict()
    )

    development_years = list(
        projected.columns
    )

    for row_position in range(
        len(projected.index)
    ):
        for column_position in range(
            1,
            len(development_years),
        ):
            if pd.isna(
                projected.iat[
                    row_position,
                    column_position,
                ]
            ):
                previous_development = (
                    development_years[
                        column_position - 1
                    ]
                )

                previous_value = float(
                    projected.iat[
                        row_position,
                        column_position - 1,
                    ]
                )

                factor = float(
                    factor_lookup[
                        previous_development
                    ]
                )

                projected.iat[
                    row_position,
                    column_position,
                ] = previous_value * factor

    return projected


def cumulative_to_incremental(
    cumulative_triangle: pd.DataFrame,
) -> pd.DataFrame:
    """Convert a complete cumulative triangle to incremental form."""

    cumulative = cumulative_triangle.copy()

    incremental = cumulative.diff(axis=1)

    incremental.iloc[:, 0] = (
        cumulative.iloc[:, 0]
    )

    incremental.index.name = "accident_year"
    incremental.columns.name = "development_year"

    return incremental


def _build_reserve_table(
    observed_cumulative: pd.DataFrame,
    projected_future_incremental: pd.DataFrame,
    cumulative_development_factors: (
        pd.Series | None
    ) = None,
) -> pd.DataFrame:
    """Build reserve estimates by accident year."""

    rows: list[dict[str, float | int]] = []

    for accident_year, row in (
        observed_cumulative.iterrows()
    ):
        observed_values = row.dropna()

        latest_development_year = int(
            observed_values.index[-1]
        )

        latest_cumulative_paid = float(
            observed_values.iloc[-1]
        )

        estimated_reserve = float(
            projected_future_incremental.loc[
                accident_year
            ]
            .fillna(0.0)
            .sum()
        )

        estimated_ultimate = (
            latest_cumulative_paid
            + estimated_reserve
        )

        result: dict[str, float | int] = {
            "accident_year": int(
                accident_year
            ),
            "latest_development_year": (
                latest_development_year
            ),
            "latest_cumulative_paid": (
                latest_cumulative_paid
            ),
            "estimated_ultimate": float(
                estimated_ultimate
            ),
            "estimated_reserve": float(
                estimated_reserve
            ),
        }

        if (
            cumulative_development_factors
            is not None
        ):
            result[
                "cumulative_development_factor"
            ] = float(
                cumulative_development_factors.loc[
                    latest_development_year
                ]
            )

        rows.append(result)

    return pd.DataFrame(rows)


def _build_summary(
    reserve_by_accident_year: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Create a one-row model summary."""

    return pd.DataFrame(
        [
            {
                "model": model_name,
                "total_observed_paid": float(
                    reserve_by_accident_year[
                        "latest_cumulative_paid"
                    ].sum()
                ),
                "total_estimated_ultimate": float(
                    reserve_by_accident_year[
                        "estimated_ultimate"
                    ].sum()
                ),
                "total_estimated_reserve": float(
                    reserve_by_accident_year[
                        "estimated_reserve"
                    ].sum()
                ),
            }
        ]
    )


def chain_ladder(
    cumulative_triangle: pd.DataFrame,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Run deterministic Chain Ladder."""

    triangle = prepare_cumulative_triangle(
        cumulative_triangle
    )

    development_factors = (
        calculate_development_factors(
            triangle
        )
    )

    cumulative_development_factors = (
        calculate_cumulative_development_factors(
            development_factors=(
                development_factors
            ),
            development_years=list(
                triangle.columns
            ),
        )
    )

    projected_cumulative = (
        project_cumulative_triangle(
            cumulative_triangle=triangle,
            development_factors=(
                development_factors
            ),
        )
    )

    projected_incremental = (
        cumulative_to_incremental(
            projected_cumulative
        )
    )

    # Keep only future projected payments. Observed cells are
    # deliberately returned as missing in this table.
    projected_future_incremental = (
        projected_incremental.where(
            triangle.isna()
        )
    )

    reserve_by_accident_year = (
        _build_reserve_table(
            observed_cumulative=triangle,
            projected_future_incremental=(
                projected_future_incremental
            ),
            cumulative_development_factors=(
                cumulative_development_factors
            ),
        )
    )

    summary = _build_summary(
        reserve_by_accident_year=(
            reserve_by_accident_year
        ),
        model_name="chain_ladder",
    )

    return {
        "development_factors": (
            development_factors
        ),
        "cumulative_development_factors": (
            cumulative_development_factors
        ),
        "projected_cumulative_triangle": (
            projected_cumulative
        ),
        "projected_future_incremental": (
            projected_future_incremental
        ),
        "reserve_by_accident_year": (
            reserve_by_accident_year
        ),
        "summary": summary,
    }


def _validate_inflation_index(
    inflation_index: Mapping[int, float],
    required_years: set[int],
) -> dict[int, float]:
    """Validate and standardise a calendar-year inflation index."""

    standardised = {
        int(year): float(value)
        for year, value
        in inflation_index.items()
    }

    missing_years = (
        required_years
        - set(standardised)
    )

    if missing_years:
        raise ValueError(
            "Inflation index is missing calendar years: "
            f"{sorted(missing_years)}"
        )

    invalid_values = [
        year
        for year in required_years
        if (
            not np.isfinite(
                standardised[year]
            )
            or standardised[year] <= 0.0
        )
    ]

    if invalid_values:
        raise ValueError(
            "Inflation-index values must be finite "
            "and positive. Invalid years: "
            f"{sorted(invalid_values)}"
        )

    return standardised


def _required_calendar_years(
    triangle: pd.DataFrame,
) -> set[int]:
    """Return calendar years represented by triangle cells."""

    return {
        int(accident_year)
        + int(development_year)
        - 1
        for accident_year in triangle.index
        for development_year in triangle.columns
    }


def deflate_incremental_triangle(
    nominal_incremental_triangle: pd.DataFrame,
    inflation_index: Mapping[int, float],
    valuation_year: int,
) -> pd.DataFrame:
    """Restate nominal payments in valuation-year prices.

    For calendar year t:

        real_at_valuation = nominal_t * I_v / I_t
    """

    nominal = prepare_incremental_triangle(
        nominal_incremental_triangle
    )

    required_years = (
        _required_calendar_years(nominal)
        | {int(valuation_year)}
    )

    index = _validate_inflation_index(
        inflation_index=inflation_index,
        required_years=required_years,
    )

    valuation_index = float(
        index[int(valuation_year)]
    )

    real_triangle = nominal.copy()

    for accident_year in nominal.index:
        for development_year in (
            nominal.columns
        ):
            nominal_amount = nominal.loc[
                accident_year,
                development_year,
            ]

            if pd.isna(nominal_amount):
                continue

            calendar_year = (
                int(accident_year)
                + int(development_year)
                - 1
            )

            real_triangle.loc[
                accident_year,
                development_year,
            ] = (
                float(nominal_amount)
                * valuation_index
                / float(index[calendar_year])
            )

    return real_triangle


def inflation_adjusted_chain_ladder(
    nominal_incremental_triangle: pd.DataFrame,
    inflation_index: Mapping[int, float],
    valuation_year: int,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Run an Inflation-Adjusted Chain Ladder model.

    The implementation:

    1. Deflates observed nominal increments to valuation-year prices.
    2. Applies Chain Ladder to the resulting real-terms triangle.
    3. Projects future real cashflows.
    4. Re-inflates projected future cashflows using the supplied
       inflation path.

    Future inflation indices are used only after model fitting.
    """

    nominal_incremental = (
        prepare_incremental_triangle(
            nominal_incremental_triangle
        )
    )

    required_years = (
        _required_calendar_years(
            nominal_incremental
        )
        | {int(valuation_year)}
    )

    index = _validate_inflation_index(
        inflation_index=inflation_index,
        required_years=required_years,
    )

    real_incremental = (
        deflate_incremental_triangle(
            nominal_incremental_triangle=(
                nominal_incremental
            ),
            inflation_index=index,
            valuation_year=valuation_year,
        )
    )

    real_cumulative = (
        incremental_to_cumulative(
            real_incremental
        )
    )

    real_chain_ladder = chain_ladder(
        real_cumulative
    )

    future_real_incremental = (
        real_chain_ladder[
            "projected_future_incremental"
        ].copy()
    )

    future_nominal_incremental = (
        future_real_incremental.copy()
    )

    valuation_index = float(
        index[int(valuation_year)]
    )

    for accident_year in (
        future_nominal_incremental.index
    ):
        for development_year in (
            future_nominal_incremental.columns
        ):
            real_amount = (
                future_nominal_incremental.loc[
                    accident_year,
                    development_year,
                ]
            )

            if pd.isna(real_amount):
                continue

            calendar_year = (
                int(accident_year)
                + int(development_year)
                - 1
            )

            future_nominal_incremental.loc[
                accident_year,
                development_year,
            ] = (
                float(real_amount)
                * float(index[calendar_year])
                / valuation_index
            )

    nominal_cumulative = (
        incremental_to_cumulative(
            nominal_incremental
        )
    )

    reserve_by_accident_year = (
        _build_reserve_table(
            observed_cumulative=(
                nominal_cumulative
            ),
            projected_future_incremental=(
                future_nominal_incremental
            ),
        )
    )

    summary = _build_summary(
        reserve_by_accident_year=(
            reserve_by_accident_year
        ),
        model_name=(
            "inflation_adjusted_chain_ladder"
        ),
    )

    return {
        "real_incremental_triangle": (
            real_incremental
        ),
        "real_cumulative_triangle": (
            real_cumulative
        ),
        "development_factors": (
            real_chain_ladder[
                "development_factors"
            ]
        ),
        "cumulative_development_factors": (
            real_chain_ladder[
                "cumulative_development_factors"
            ]
        ),
        "projected_future_real_incremental": (
            future_real_incremental
        ),
        "projected_future_nominal_incremental": (
            future_nominal_incremental
        ),
        "reserve_by_accident_year": (
            reserve_by_accident_year
        ),
        "summary": summary,
    }


def cashflow_uplift(
    nominal_cumulative_triangle: pd.DataFrame,
    forecast_inflation_index: Mapping[int, float],
    valuation_year: int,
    embedded_annual_inflation: float,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Apply a stylised Cashflow Uplift adjustment.

    Standard Chain Ladder first projects baseline nominal future
    cashflows. Each projected future payment is then multiplied by:

        forecast cumulative inflation
        --------------------------------
        embedded cumulative inflation

    The embedded rate represents the inflation assumption already
    reflected in the historical Chain Ladder factors.

    This is a transparent dissertation benchmark. Its precise
    contractual and methodological interpretation should be checked
    against the Cashflow Uplift source used in the literature review.
    """

    if not np.isfinite(
        embedded_annual_inflation
    ):
        raise ValueError(
            "embedded_annual_inflation must be finite."
        )

    if embedded_annual_inflation <= -1.0:
        raise ValueError(
            "embedded_annual_inflation must exceed -100%."
        )

    cumulative = prepare_cumulative_triangle(
        nominal_cumulative_triangle
    )

    required_years = (
        _required_calendar_years(cumulative)
        | {int(valuation_year)}
    )

    index = _validate_inflation_index(
        inflation_index=(
            forecast_inflation_index
        ),
        required_years=required_years,
    )

    standard_result = chain_ladder(
        cumulative
    )

    baseline_future = standard_result[
        "projected_future_incremental"
    ].copy()

    uplift_factors = pd.DataFrame(
        np.nan,
        index=baseline_future.index,
        columns=baseline_future.columns,
    )

    adjusted_future = baseline_future.copy()

    valuation_index = float(
        index[int(valuation_year)]
    )

    for accident_year in (
        baseline_future.index
    ):
        for development_year in (
            baseline_future.columns
        ):
            baseline_amount = (
                baseline_future.loc[
                    accident_year,
                    development_year,
                ]
            )

            if pd.isna(baseline_amount):
                continue

            calendar_year = (
                int(accident_year)
                + int(development_year)
                - 1
            )

            years_after_valuation = (
                calendar_year
                - int(valuation_year)
            )

            forecast_growth = (
                float(index[calendar_year])
                / valuation_index
            )

            embedded_growth = (
                1.0
                + float(
                    embedded_annual_inflation
                )
            ) ** years_after_valuation

            uplift_factor = (
                forecast_growth
                / embedded_growth
            )

            uplift_factors.loc[
                accident_year,
                development_year,
            ] = uplift_factor

            adjusted_future.loc[
                accident_year,
                development_year,
            ] = (
                float(baseline_amount)
                * uplift_factor
            )

    reserve_by_accident_year = (
        _build_reserve_table(
            observed_cumulative=cumulative,
            projected_future_incremental=(
                adjusted_future
            ),
        )
    )

    summary = _build_summary(
        reserve_by_accident_year=(
            reserve_by_accident_year
        ),
        model_name="cashflow_uplift",
    )

    return {
        "standard_chain_ladder_result": (
            standard_result
        ),
        "baseline_future_incremental": (
            baseline_future
        ),
        "uplift_factors": uplift_factors,
        "projected_future_incremental": (
            adjusted_future
        ),
        "reserve_by_accident_year": (
            reserve_by_accident_year
        ),
        "summary": summary,
    }
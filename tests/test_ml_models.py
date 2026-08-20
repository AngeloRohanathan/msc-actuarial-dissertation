"""Tests for the regularised Poisson reserving model."""

import numpy as np
import pandas as pd
import pytest

from src.ml_models import (
    generate_rolling_diagonal_splits,
    fit_regularised_poisson_reserving_model,
    triangle_to_cell_dataset,
)


VALUATION_YEAR = 2022

ACCIDENT_YEARS = list(
    range(2015, 2023)
)

DEVELOPMENT_YEARS = list(
    range(1, 7)
)


def create_test_inflation_index() -> dict[int, float]:
    """Create a simple 2% annual inflation index."""

    index = {
        2015: 1.0
    }

    for year in range(
        2016,
        2030,
    ):
        index[year] = (
            index[year - 1]
            * 1.02
        )

    return index


def create_test_triangle(
    zero_first_development: bool = False,
) -> pd.DataFrame:
    """Create an upper triangle with known observation timing."""

    triangle = pd.DataFrame(
        np.nan,
        index=ACCIDENT_YEARS,
        columns=DEVELOPMENT_YEARS,
        dtype=float,
    )

    for accident_year in ACCIDENT_YEARS:
        for development_year in (
            DEVELOPMENT_YEARS
        ):
            calendar_year = (
                accident_year
                + development_year
                - 1
            )

            if calendar_year > VALUATION_YEAR:
                continue

            if (
                zero_first_development
                and development_year == 1
            ):
                amount = 0.0
            else:
                accident_year_level = (
                    accident_year
                    - 2014
                )

                amount = (
                    100_000.0
                    * accident_year_level
                    * np.exp(
                        -0.30
                        * (
                            development_year
                            - 1
                        )
                    )
                )

            triangle.loc[
                accident_year,
                development_year,
            ] = amount

    triangle.index.name = "accident_year"
    triangle.columns.name = (
        "development_year"
    )

    return triangle


def test_triangle_is_converted_to_cell_data() -> None:
    triangle = create_test_triangle()

    cells = triangle_to_cell_dataset(
        incremental_triangle=triangle,
        inflation_index=(
            create_test_inflation_index()
        ),
        valuation_year=VALUATION_YEAR,
        basis="gross",
        amount_scale=1_000_000.0,
    )

    observed_cell = cells.loc[
        (
            cells["accident_year"] == 2020
        )
        & (
            cells["development_year"] == 3
        )
    ].iloc[0]

    future_cell = cells.loc[
        (
            cells["accident_year"] == 2021
        )
        & (
            cells["development_year"] == 3
        )
    ].iloc[0]

    assert observed_cell["calendar_year"] == 2022
    assert observed_cell["is_observed"]
    assert not np.isnan(
        observed_cell[
            "incremental_paid_scaled"
        ]
    )

    assert future_cell["calendar_year"] == 2023
    assert not future_cell["is_observed"]
    assert np.isnan(
        future_cell[
            "incremental_paid_scaled"
        ]
    )


def test_rolling_splits_respect_calendar_time() -> None:
    triangle = create_test_triangle()

    cells = triangle_to_cell_dataset(
        incremental_triangle=triangle,
        inflation_index=(
            create_test_inflation_index()
        ),
        valuation_year=VALUATION_YEAR,
        basis="gross",
        amount_scale=1_000_000.0,
    )

    observed = (
        cells.loc[
            cells["is_observed"]
        ]
        .reset_index(drop=True)
    )

    splits = generate_rolling_diagonal_splits(
        observed_cells=observed,
        minimum_training_diagonals=3,
    )

    assert len(splits) > 0

    for (
        validation_year,
        training_indices,
        validation_indices,
    ) in splits:

        training_years = observed.iloc[
            training_indices
        ]["calendar_year"]

        validation_years = observed.iloc[
            validation_indices
        ]["calendar_year"]

        assert (
            training_years.max()
            < validation_year
        )

        assert (
            validation_years
            == validation_year
        ).all()


def test_selected_alpha_comes_from_grid() -> None:
    triangle = create_test_triangle()

    alpha_grid = [
        0.01,
        0.1,
    ]

    result = (
        fit_regularised_poisson_reserving_model(
            incremental_triangle=triangle,
            inflation_index=(
                create_test_inflation_index()
            ),
            valuation_year=VALUATION_YEAR,
            basis="gross",
            alpha_grid=alpha_grid,
            minimum_training_diagonals=3,
            minimum_validation_folds=3,
            amount_scale=1_000_000.0,
        )
    )

    selected_alpha = float(
        result["summary"].loc[
            0,
            "selected_alpha",
        ]
    )

    assert selected_alpha in alpha_grid


def test_future_predictions_are_non_negative() -> None:
    triangle = create_test_triangle()

    result = (
        fit_regularised_poisson_reserving_model(
            incremental_triangle=triangle,
            inflation_index=(
                create_test_inflation_index()
            ),
            valuation_year=VALUATION_YEAR,
            basis="gross",
            alpha_grid=[
                0.01,
                0.1,
            ],
            minimum_training_diagonals=3,
            minimum_validation_folds=3,
            amount_scale=1_000_000.0,
        )
    )

    predictions = result[
        "future_predictions"
    ][
        "predicted_incremental_paid"
    ]

    assert predictions.notna().all()

    assert (
        predictions >= 0.0
    ).all()


def test_poisson_model_handles_zero_first_development_column() -> None:
    triangle = create_test_triangle(
        zero_first_development=True
    )

    result = (
        fit_regularised_poisson_reserving_model(
            incremental_triangle=triangle,
            inflation_index=(
                create_test_inflation_index()
            ),
            valuation_year=VALUATION_YEAR,
            basis="ceded",
            alpha_grid=[
                0.01,
                0.1,
            ],
            minimum_training_diagonals=3,
            minimum_validation_folds=3,
            amount_scale=1_000_000.0,
        )
    )

    estimated_reserve = float(
        result["summary"].loc[
            0,
            "total_estimated_reserve",
        ]
    )

    assert estimated_reserve >= 0.0

    assert not result[
        "future_predictions"
    ].empty


def test_reserve_table_reconciles_with_summary() -> None:
    triangle = create_test_triangle()

    result = (
        fit_regularised_poisson_reserving_model(
            incremental_triangle=triangle,
            inflation_index=(
                create_test_inflation_index()
            ),
            valuation_year=VALUATION_YEAR,
            basis="gross",
            alpha_grid=[
                0.01,
                0.1,
            ],
            minimum_training_diagonals=3,
            minimum_validation_folds=3,
            amount_scale=1_000_000.0,
        )
    )

    reserve_table_total = float(
        result[
            "reserve_by_accident_year"
        ][
            "estimated_reserve"
        ].sum()
    )

    summary_total = float(
        result["summary"].loc[
            0,
            "total_estimated_reserve",
        ]
    )

    assert np.isclose(
        reserve_table_total,
        summary_total,
    )


def test_observed_rows_only_are_used_for_final_fit() -> None:
    triangle = create_test_triangle()

    result = (
        fit_regularised_poisson_reserving_model(
            incremental_triangle=triangle,
            inflation_index=(
                create_test_inflation_index()
            ),
            valuation_year=VALUATION_YEAR,
            basis="gross",
            alpha_grid=[
                0.01,
                0.1,
            ],
            minimum_training_diagonals=3,
            minimum_validation_folds=3,
            amount_scale=1_000_000.0,
        )
    )

    observed_count = int(
        result["cell_dataset"][
            "is_observed"
        ].sum()
    )

    reported_training_rows = int(
        result["summary"].loc[
            0,
            "training_rows",
        ]
    )

    assert (
        observed_count
        == reported_training_rows
    )
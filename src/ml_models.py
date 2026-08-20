"""Regularised statistical-learning reserving models.

This module implements a regularised Poisson regression model for
incremental paid triangles.

The regularisation parameter is selected using rolling
calendar-year diagonal validation. Random train-test splitting is
deliberately avoided because reserving is a forecasting problem.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


NUMERIC_FEATURE_COLUMNS = [
    "accident_year_centered",
    "relative_log_inflation",
    "structural_break_indicator",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "development_year",
]

MODEL_FEATURE_COLUMNS = (
    NUMERIC_FEATURE_COLUMNS
    + CATEGORICAL_FEATURE_COLUMNS
)


def prepare_incremental_triangle(
    triangle: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and standardise an incremental triangle."""

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
            "Triangle labels must be convertible "
            "to integers."
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

    expected_development_years = list(
        range(
            1,
            max(output.columns) + 1,
        )
    )

    if list(output.columns) != (
        expected_development_years
    ):
        raise ValueError(
            "Development years must begin at 1 "
            "and be consecutive."
        )

    observed_values = (
        output.stack().to_numpy()
    )

    if (
        observed_values < -1e-8
    ).any():
        raise ValueError(
            "Observed incremental payments "
            "cannot be negative."
        )

    for accident_year, row in output.iterrows():
        observed_mask = (
            row.notna().to_numpy()
        )

        missing_positions = np.flatnonzero(
            ~observed_mask
        )

        if len(missing_positions) == 0:
            continue

        first_missing = int(
            missing_positions[0]
        )

        if observed_mask[first_missing:].any():
            raise ValueError(
                "Triangle contains an internal missing "
                f"value for accident year {accident_year}."
            )

    output.index.name = "accident_year"
    output.columns.name = "development_year"

    return output


def validate_inflation_index(
    inflation_index: Mapping[int, float],
    required_years: set[int],
) -> dict[int, float]:
    """Validate and standardise an inflation index."""

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
            "Inflation index is missing years: "
            f"{sorted(missing_years)}"
        )

    invalid_years = [
        year
        for year in required_years
        if (
            not np.isfinite(
                standardised[year]
            )
            or standardised[year] <= 0.0
        )
    ]

    if invalid_years:
        raise ValueError(
            "Inflation-index values must be "
            "finite and positive. Invalid years: "
            f"{sorted(invalid_years)}"
        )

    return standardised


def triangle_to_cell_dataset(
    incremental_triangle: pd.DataFrame,
    inflation_index: Mapping[int, float],
    valuation_year: int,
    basis: str,
    amount_scale: float,
    structural_break_year: int | None = None,
) -> pd.DataFrame:
    """Convert a triangle into portfolio-cell modelling data.

    Observed upper-triangle cells contain a response value.
    Future lower-triangle cells retain a missing response and
    are used only after the model has been fitted.
    """

    if basis not in {
        "gross",
        "ceded",
    }:
        raise ValueError(
            "basis must be either 'gross' or 'ceded'."
        )

    if amount_scale <= 0.0:
        raise ValueError(
            "amount_scale must be positive."
        )

    triangle = prepare_incremental_triangle(
        incremental_triangle
    )

    accident_year_origin = int(
        min(triangle.index)
    )

    required_years = {
        int(accident_year)
        + int(development_year)
        - 1
        for accident_year in triangle.index
        for development_year in triangle.columns
    }

    required_years.add(
        int(valuation_year)
    )

    index = validate_inflation_index(
        inflation_index=inflation_index,
        required_years=required_years,
    )

    valuation_index = float(
        index[int(valuation_year)]
    )

    rows: list[
        dict[str, Any]
    ] = []

    for accident_year in triangle.index:
        for development_year in (
            triangle.columns
        ):
            calendar_year = (
                int(accident_year)
                + int(development_year)
                - 1
            )

            amount = triangle.loc[
                accident_year,
                development_year,
            ]

            is_observed = (
                calendar_year
                <= int(valuation_year)
            )

            if (
                is_observed
                and pd.isna(amount)
            ):
                raise ValueError(
                    "An observable triangle cell is missing: "
                    f"accident year {accident_year}, "
                    f"development year {development_year}."
                )

            if (
                not is_observed
                and not pd.isna(amount)
            ):
                raise ValueError(
                    "A future triangle cell contains an "
                    "observed value."
                )

            if pd.isna(amount):
                scaled_amount = np.nan
                original_amount = np.nan
            else:
                original_amount = float(amount)
                scaled_amount = (
                    original_amount
                    / float(amount_scale)
                )

            relative_log_inflation = float(
                np.log(
                    float(index[calendar_year])
                    / valuation_index
                )
            )

            if structural_break_year is None:
                structural_break_indicator = 0
            else:
                structural_break_indicator = int(
                    calendar_year
                    >= int(structural_break_year)
                )

            rows.append(
                {
                    "basis": basis,
                    "accident_year": int(
                        accident_year
                    ),
                    "development_year": int(
                        development_year
                    ),
                    "calendar_year": int(
                        calendar_year
                    ),
                    "accident_year_centered": (
                        int(accident_year)
                        - accident_year_origin
                    ),
                    "relative_log_inflation": (
                        relative_log_inflation
                    ),
                    "structural_break_indicator": (
                        structural_break_indicator
                    ),
                    "is_observed": bool(
                        is_observed
                    ),
                    "incremental_paid": (
                        original_amount
                    ),
                    "incremental_paid_scaled": (
                        scaled_amount
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_poisson_pipeline(
    alpha: float,
) -> Pipeline:
    """Create a preprocessing and Poisson-regression pipeline."""

    if not np.isfinite(alpha):
        raise ValueError(
            "alpha must be finite."
        )

    if alpha <= 0.0:
        raise ValueError(
            "alpha must be positive."
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURE_COLUMNS,
            ),
            (
                "development",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURE_COLUMNS,
            ),
        ],
        remainder="drop",
    )

    model = PoissonRegressor(
        alpha=float(alpha),
        max_iter=2_000,
        tol=1e-8,
    )

    return Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


def generate_rolling_diagonal_splits(
    observed_cells: pd.DataFrame,
    minimum_training_diagonals: int,
) -> list[
    tuple[int, np.ndarray, np.ndarray]
]:
    """Create expanding-window calendar-year validation splits.

    For validation year v:

        training cells: calendar year < v
        validation cells: calendar year = v
    """

    if minimum_training_diagonals < 1:
        raise ValueError(
            "minimum_training_diagonals "
            "must be positive."
        )

    if observed_cells.empty:
        raise ValueError(
            "Observed cell data cannot be empty."
        )

    if not observed_cells[
        "is_observed"
    ].all():
        raise ValueError(
            "generate_rolling_diagonal_splits "
            "requires observed cells only."
        )

    data = observed_cells.reset_index(
        drop=True
    )

    calendar_years = sorted(
        int(year)
        for year in data[
            "calendar_year"
        ].unique()
    )

    if (
        len(calendar_years)
        <= minimum_training_diagonals
    ):
        raise ValueError(
            "Insufficient calendar-year diagonals "
            "for rolling validation."
        )

    splits: list[
        tuple[int, np.ndarray, np.ndarray]
    ] = []

    for position in range(
        minimum_training_diagonals,
        len(calendar_years),
    ):
        validation_year = (
            calendar_years[position]
        )

        training_indices = data.index[
            data["calendar_year"]
            < validation_year
        ].to_numpy()

        validation_indices = data.index[
            data["calendar_year"]
            == validation_year
        ].to_numpy()

        if (
            len(training_indices) == 0
            or len(validation_indices) == 0
        ):
            continue

        splits.append(
            (
                validation_year,
                training_indices,
                validation_indices,
            )
        )

    return splits


def select_regularisation_alpha(
    cell_dataset: pd.DataFrame,
    alpha_grid: Sequence[float],
    minimum_training_diagonals: int,
    minimum_validation_folds: int,
    amount_scale: float,
) -> tuple[
    float,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Select alpha using rolling calendar-diagonal validation."""

    if not alpha_grid:
        raise ValueError(
            "alpha_grid cannot be empty."
        )

    if minimum_validation_folds < 1:
        raise ValueError(
            "minimum_validation_folds "
            "must be positive."
        )

    observed_cells = (
        cell_dataset.loc[
            cell_dataset["is_observed"]
        ]
        .reset_index(drop=True)
    )

    splits = generate_rolling_diagonal_splits(
        observed_cells=observed_cells,
        minimum_training_diagonals=(
            minimum_training_diagonals
        ),
    )

    fold_rows: list[
        dict[str, Any]
    ] = []

    alpha_rows: list[
        dict[str, Any]
    ] = []

    for alpha in alpha_grid:
        alpha = float(alpha)

        if alpha <= 0.0:
            raise ValueError(
                "All alpha values must be positive."
            )

        successful_scores: list[float] = []
        successful_maes: list[float] = []

        for (
            validation_year,
            training_indices,
            validation_indices,
        ) in splits:

            training_data = (
                observed_cells.iloc[
                    training_indices
                ]
            )

            validation_data = (
                observed_cells.iloc[
                    validation_indices
                ]
            )

            y_train = training_data[
                "incremental_paid_scaled"
            ].to_numpy(dtype=float)

            y_validation = validation_data[
                "incremental_paid_scaled"
            ].to_numpy(dtype=float)

            # A Poisson model cannot be estimated when every
            # training response is zero. This can occur in very
            # early ceded diagonals, so those folds are skipped.
            if np.isclose(
                y_train.sum(),
                0.0,
            ):
                fold_rows.append(
                    {
                        "alpha": alpha,
                        "validation_year": (
                            validation_year
                        ),
                        "fold_status": (
                            "skipped_zero_training_total"
                        ),
                        "training_rows": int(
                            len(training_data)
                        ),
                        "validation_rows": int(
                            len(validation_data)
                        ),
                        "mean_poisson_deviance": (
                            np.nan
                        ),
                        "mean_absolute_error": (
                            np.nan
                        ),
                    }
                )

                continue

            pipeline = build_poisson_pipeline(
                alpha=alpha
            )

            pipeline.fit(
                training_data[
                    MODEL_FEATURE_COLUMNS
                ],
                y_train,
            )

            prediction = pipeline.predict(
                validation_data[
                    MODEL_FEATURE_COLUMNS
                ]
            )

            prediction = np.clip(
                prediction,
                1e-12,
                None,
            )

            poisson_deviance = float(
                mean_poisson_deviance(
                    y_validation,
                    prediction,
                )
            )

            mae_scaled = float(
                mean_absolute_error(
                    y_validation,
                    prediction,
                )
            )

            mae_original = (
                mae_scaled
                * float(amount_scale)
            )

            successful_scores.append(
                poisson_deviance
            )

            successful_maes.append(
                mae_original
            )

            fold_rows.append(
                {
                    "alpha": alpha,
                    "validation_year": (
                        validation_year
                    ),
                    "fold_status": "success",
                    "training_rows": int(
                        len(training_data)
                    ),
                    "validation_rows": int(
                        len(validation_data)
                    ),
                    "mean_poisson_deviance": (
                        poisson_deviance
                    ),
                    "mean_absolute_error": (
                        mae_original
                    ),
                }
            )

        successful_fold_count = len(
            successful_scores
        )

        if successful_fold_count == 0:
            mean_deviance = np.nan
            mean_mae = np.nan
        else:
            mean_deviance = float(
                np.mean(successful_scores)
            )

            mean_mae = float(
                np.mean(successful_maes)
            )

        alpha_rows.append(
            {
                "alpha": alpha,
                "successful_folds": (
                    successful_fold_count
                ),
                "mean_poisson_deviance": (
                    mean_deviance
                ),
                "mean_absolute_error": (
                    mean_mae
                ),
            }
        )

    alpha_summary = pd.DataFrame(
        alpha_rows
    )

    eligible = alpha_summary.loc[
        (
            alpha_summary[
                "successful_folds"
            ]
            >= int(
                minimum_validation_folds
            )
        )
        & (
            alpha_summary[
                "mean_poisson_deviance"
            ].notna()
        )
    ]

    if eligible.empty:
        raise ValueError(
            "No alpha value produced enough successful "
            "rolling validation folds."
        )

    # Lowest validation deviance is preferred. When two values
    # tie, prefer the larger alpha and therefore the more
    # regularised model.
    best_row = (
        eligible
        .sort_values(
            [
                "mean_poisson_deviance",
                "alpha",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .iloc[0]
    )

    best_alpha = float(
        best_row["alpha"]
    )

    fold_results = pd.DataFrame(
        fold_rows
    )

    return (
        best_alpha,
        alpha_summary,
        fold_results,
    )


def fit_regularised_poisson_reserving_model(
    incremental_triangle: pd.DataFrame,
    inflation_index: Mapping[int, float],
    valuation_year: int,
    basis: str,
    alpha_grid: Sequence[float],
    minimum_training_diagonals: int,
    minimum_validation_folds: int,
    amount_scale: float,
    structural_break_year: int | None = None,
) -> dict[str, Any]:
    """Fit a regularised Poisson reserving model."""

    cell_dataset = triangle_to_cell_dataset(
        incremental_triangle=(
            incremental_triangle
        ),
        inflation_index=inflation_index,
        valuation_year=valuation_year,
        basis=basis,
        amount_scale=amount_scale,
        structural_break_year=(
            structural_break_year
        ),
    )

    observed_cells = (
        cell_dataset.loc[
            cell_dataset["is_observed"]
        ]
        .copy()
    )

    future_cells = (
        cell_dataset.loc[
            ~cell_dataset["is_observed"]
        ]
        .copy()
    )

    y_observed = observed_cells[
        "incremental_paid_scaled"
    ].to_numpy(dtype=float)

    if np.isclose(
        y_observed.sum(),
        0.0,
    ):
        raise ValueError(
            "The complete observed response is zero."
        )

    (
        best_alpha,
        alpha_validation,
        fold_validation,
    ) = select_regularisation_alpha(
        cell_dataset=cell_dataset,
        alpha_grid=alpha_grid,
        minimum_training_diagonals=(
            minimum_training_diagonals
        ),
        minimum_validation_folds=(
            minimum_validation_folds
        ),
        amount_scale=amount_scale,
    )

    pipeline = build_poisson_pipeline(
        alpha=best_alpha
    )

    pipeline.fit(
        observed_cells[
            MODEL_FEATURE_COLUMNS
        ],
        y_observed,
    )

    future_prediction_scaled = (
        pipeline.predict(
            future_cells[
                MODEL_FEATURE_COLUMNS
            ]
        )
    )

    future_prediction_scaled = np.clip(
        future_prediction_scaled,
        0.0,
        None,
    )

    future_cells[
        "predicted_incremental_paid_scaled"
    ] = future_prediction_scaled

    future_cells[
        "predicted_incremental_paid"
    ] = (
        future_prediction_scaled
        * float(amount_scale)
    )

    reserve_rows: list[
        dict[str, Any]
    ] = []

    for accident_year in sorted(
        cell_dataset[
            "accident_year"
        ].unique()
    ):
        accident_observed = (
            observed_cells.loc[
                observed_cells[
                    "accident_year"
                ]
                == accident_year
            ]
        )

        accident_future = (
            future_cells.loc[
                future_cells[
                    "accident_year"
                ]
                == accident_year
            ]
        )

        observed_paid = float(
            accident_observed[
                "incremental_paid"
            ].sum()
        )

        estimated_reserve = float(
            accident_future[
                "predicted_incremental_paid"
            ].sum()
        )

        if accident_observed.empty:
            latest_development_year = 0
        else:
            latest_development_year = int(
                accident_observed[
                    "development_year"
                ].max()
            )

        reserve_rows.append(
            {
                "accident_year": int(
                    accident_year
                ),
                "latest_development_year": (
                    latest_development_year
                ),
                "observed_paid": observed_paid,
                "estimated_reserve": (
                    estimated_reserve
                ),
                "estimated_ultimate": (
                    observed_paid
                    + estimated_reserve
                ),
            }
        )

    reserve_by_accident_year = (
        pd.DataFrame(reserve_rows)
    )

    best_validation_row = (
        alpha_validation.loc[
            alpha_validation["alpha"]
            == best_alpha
        ]
        .iloc[0]
    )

    summary = pd.DataFrame(
        [
            {
                "model": (
                    "regularized_poisson"
                ),
                "basis": basis,
                "valuation_year": int(
                    valuation_year
                ),
                "selected_alpha": (
                    best_alpha
                ),
                "successful_validation_folds": int(
                    best_validation_row[
                        "successful_folds"
                    ]
                ),
                "validation_mean_poisson_deviance": float(
                    best_validation_row[
                        "mean_poisson_deviance"
                    ]
                ),
                "validation_mean_absolute_error": float(
                    best_validation_row[
                        "mean_absolute_error"
                    ]
                ),
                "training_rows": int(
                    len(observed_cells)
                ),
                "future_rows": int(
                    len(future_cells)
                ),
                "total_observed_paid": float(
                    reserve_by_accident_year[
                        "observed_paid"
                    ].sum()
                ),
                "total_estimated_reserve": float(
                    reserve_by_accident_year[
                        "estimated_reserve"
                    ].sum()
                ),
                "total_estimated_ultimate": float(
                    reserve_by_accident_year[
                        "estimated_ultimate"
                    ].sum()
                ),
            }
        ]
    )

    return {
        "pipeline": pipeline,
        "cell_dataset": cell_dataset,
        "future_predictions": future_cells,
        "alpha_validation": alpha_validation,
        "fold_validation": fold_validation,
        "reserve_by_accident_year": (
            reserve_by_accident_year
        ),
        "summary": summary,
    }
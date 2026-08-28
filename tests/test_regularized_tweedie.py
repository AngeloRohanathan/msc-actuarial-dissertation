"""Focused tests for the Regularized Tweedie reserving model."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import TweedieRegressor

from config import (
    ML_TWEEDIE_ALPHA_GRID,
    ML_TWEEDIE_POWER_GRID,
)
from src.ml_models import (
    REGULARIZED_TWEEDIE_MODEL_NAME,
    build_tweedie_pipeline,
    fit_regularised_tweedie_reserving_model,
)


VALUATION_YEAR = 2022
ACCIDENT_YEARS = list(range(2015, 2023))
DEVELOPMENT_YEARS = list(range(1, 7))


def create_test_inflation_index() -> dict[int, float]:
    """Create a deterministic 2% annual inflation index."""

    inflation_index = {2015: 1.0}

    for year in range(2016, 2030):
        inflation_index[year] = (
            inflation_index[year - 1]
            * 1.02
        )

    return inflation_index


def create_test_triangle() -> pd.DataFrame:
    """Create an upper triangle with positive and zero payments."""

    triangle = pd.DataFrame(
        np.nan,
        index=ACCIDENT_YEARS,
        columns=DEVELOPMENT_YEARS,
        dtype=float,
    )

    for accident_year in ACCIDENT_YEARS:
        for development_year in DEVELOPMENT_YEARS:
            calendar_year = (
                accident_year
                + development_year
                - 1
            )

            if calendar_year > VALUATION_YEAR:
                continue

            amount = (
                90_000.0
                * (accident_year - 2014)
                * np.exp(
                    -0.35
                    * (development_year - 1)
                )
            )

            if (
                development_year == 1
                and accident_year % 2 == 0
            ):
                amount = 0.0

            triangle.loc[
                accident_year,
                development_year,
            ] = amount

    triangle.index.name = "accident_year"
    triangle.columns.name = "development_year"

    return triangle


@pytest.fixture(scope="module")
def fitted_tweedie() -> dict[str, object]:
    """Fit one compact model shared by the focused assertions."""

    return fit_regularised_tweedie_reserving_model(
        incremental_triangle=create_test_triangle(),
        inflation_index=create_test_inflation_index(),
        valuation_year=VALUATION_YEAR,
        basis="ceded",
        alpha_grid=ML_TWEEDIE_ALPHA_GRID[:2],
        power_grid=ML_TWEEDIE_POWER_GRID[:2],
        minimum_training_diagonals=3,
        minimum_validation_folds=3,
        amount_scale=1_000_000.0,
        structural_break_year=2018,
    )


def test_pipeline_uses_requested_tweedie_parameters() -> None:
    pipeline = build_tweedie_pipeline(
        alpha=0.1,
        power=1.7,
    )

    estimator = pipeline.named_steps["model"]

    assert isinstance(estimator, TweedieRegressor)
    assert estimator.link == "log"
    assert estimator.power == 1.7
    assert estimator.alpha == 0.1


def test_selected_power_belongs_to_configured_grid(
    fitted_tweedie: dict[str, object],
) -> None:
    summary = fitted_tweedie["summary"]
    selected_power = float(
        summary.loc[0, "selected_power"]
    )

    assert selected_power in ML_TWEEDIE_POWER_GRID


def test_selected_alpha_belongs_to_configured_grid(
    fitted_tweedie: dict[str, object],
) -> None:
    summary = fitted_tweedie["summary"]
    selected_alpha = float(
        summary.loc[0, "selected_alpha"]
    )

    assert selected_alpha in ML_TWEEDIE_ALPHA_GRID


def test_predictions_are_finite_and_nonnegative(
    fitted_tweedie: dict[str, object],
) -> None:
    predictions = fitted_tweedie[
        "future_predictions"
    ]["predicted_incremental_paid"]

    assert np.isfinite(predictions).all()
    assert (predictions >= 0.0).all()


def test_accident_year_reserves_reconcile(
    fitted_tweedie: dict[str, object],
) -> None:
    accident_year_total = float(
        fitted_tweedie[
            "reserve_by_accident_year"
        ]["estimated_reserve"].sum()
    )

    summary_total = float(
        fitted_tweedie[
            "summary"
        ].loc[0, "total_estimated_reserve"]
    )

    future_cell_total = float(
        fitted_tweedie[
            "future_predictions"
        ]["predicted_incremental_paid"].sum()
    )

    assert np.isclose(
        accident_year_total,
        summary_total,
    )
    assert np.isclose(
        accident_year_total,
        future_cell_total,
    )


def test_summary_uses_correct_model_name(
    fitted_tweedie: dict[str, object],
) -> None:
    assert (
        fitted_tweedie["summary"].loc[0, "model"]
        == REGULARIZED_TWEEDIE_MODEL_NAME
    )


def test_fitting_api_does_not_accept_evaluation_truth() -> None:
    parameters = inspect.signature(
        fit_regularised_tweedie_reserving_model
    ).parameters

    forbidden_parameters = {
        "true_reserve",
        "true_ultimate",
        "future_truth",
        "evaluation_truth",
    }

    assert forbidden_parameters.isdisjoint(parameters)


def test_validation_tables_record_power_and_alpha(
    fitted_tweedie: dict[str, object],
) -> None:
    hyperparameters = fitted_tweedie[
        "hyperparameter_validation"
    ]
    folds = fitted_tweedie["fold_validation"]

    assert {"power", "alpha"}.issubset(
        hyperparameters.columns
    )
    assert {"power", "alpha"}.issubset(
        folds.columns
    )

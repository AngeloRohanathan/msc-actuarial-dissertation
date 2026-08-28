"""Utilities for Step 26 historical data-window sensitivity.

These helpers change only the accident years supplied to an existing
reserving model.  They do not change model definitions, validation
rules, or the fixed accident-year range used for evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


HISTORY_WINDOWS: dict[int, int] = {
    15: 2010,
    10: 2015,
    7: 2018,
    5: 2020,
}

EVALUATION_START_AY = 2020
EVALUATION_END_AY = 2024
EVALUATION_ACCIDENT_YEARS = tuple(
    range(EVALUATION_START_AY, EVALUATION_END_AY + 1)
)

CLASSICAL_HISTORY_WINDOW_MODELS = frozenset(
    {
        "chain_ladder",
        "inflation_adjusted_chain_ladder",
        "cashflow_uplift",
    }
)
ML_HISTORY_WINDOW_MODELS = frozenset(
    {
        "regularized_poisson",
        "regularized_poisson_break_interaction",
        "regularized_tweedie",
    }
)
APPLICABLE_BY_DESIGN = "applicable_by_design"
NOT_APPLICABLE_BY_DESIGN = "not_applicable_by_design"


@dataclass(frozen=True)
class ModelApplicability:
    """Geometry-only Step 26 applicability assessment."""

    status: str
    reason: str
    available_calendar_diagonals: int
    potential_validation_folds: int | None
    maximum_development_year: int

    @property
    def fit_should_be_attempted(self) -> bool:
        return self.status == APPLICABLE_BY_DESIGN


def assess_model_applicability(
    *,
    model_name: str,
    triangle: pd.DataFrame,
    minimum_training_diagonals: int,
    minimum_validation_folds: int,
) -> ModelApplicability:
    """Assess structural applicability without inspecting payment amounts.

    Only impossibility that follows from the frozen model-window combination
    for every simulated portfolio is pre-classified. ML applicability depends
    on the number of calendar diagonals available under the frozen rolling-
    validation requirements. Classical development horizons and denominators
    depend on the realised triangle, so every classical row is attempted and
    any missing-pair or zero-denominator error is retained as data-dependent.
    """

    if not isinstance(triangle, pd.DataFrame):
        raise TypeError("triangle must be a pandas DataFrame.")
    if triangle.empty:
        raise ValueError("triangle cannot be empty.")

    known_models = CLASSICAL_HISTORY_WINDOW_MODELS | ML_HISTORY_WINDOW_MODELS
    if model_name not in known_models:
        raise ValueError(f"Unknown Step 26 model: {model_name}")
    if minimum_training_diagonals < 1:
        raise ValueError("minimum_training_diagonals must be positive.")
    if minimum_validation_folds < 1:
        raise ValueError("minimum_validation_folds must be positive.")

    data = triangle.copy(deep=False)
    accident_years = [int(value) for value in data.index]
    development_years = sorted(int(value) for value in data.columns)
    maximum_development_year = max(development_years)
    available_calendar_years = {
        accident_year + development_year - 1
        for accident_year in accident_years
        for development_year in development_years
        if not pd.isna(data.loc[accident_year, development_year])
    }
    available_diagonals = len(available_calendar_years)

    if model_name in CLASSICAL_HISTORY_WINDOW_MODELS:
        return ModelApplicability(
            status=APPLICABLE_BY_DESIGN,
            reason=(
                "Classical paired-observation and denominator availability "
                "depends on the realised triangle; the frozen estimator must "
                "be attempted."
            ),
            available_calendar_diagonals=available_diagonals,
            potential_validation_folds=None,
            maximum_development_year=maximum_development_year,
        )

    potential_validation_folds = max(
        0,
        available_diagonals - int(minimum_training_diagonals),
    )
    if potential_validation_folds < int(minimum_validation_folds):
        return ModelApplicability(
            status=NOT_APPLICABLE_BY_DESIGN,
            reason=(
                f"Only {available_diagonals} calendar diagonals are available, "
                f"giving {potential_validation_folds} potential rolling folds "
                f"after {minimum_training_diagonals} training diagonals; the "
                f"frozen minimum is {minimum_validation_folds} validation folds."
            ),
            available_calendar_diagonals=available_diagonals,
            potential_validation_folds=potential_validation_folds,
            maximum_development_year=maximum_development_year,
        )

    return ModelApplicability(
        status=APPLICABLE_BY_DESIGN,
        reason=(
            f"Geometry supplies {potential_validation_folds} potential rolling "
            "validation folds under the frozen requirements."
        ),
        available_calendar_diagonals=available_diagonals,
        potential_validation_folds=potential_validation_folds,
        maximum_development_year=maximum_development_year,
    )


def filter_triangle_to_history_window(
    triangle: pd.DataFrame,
    history_start_ay: int,
    valuation_year: int,
) -> pd.DataFrame:
    """Return a copied triangle restricted to the requested AY window.

    Development-year columns and observed/future cell values are left
    unchanged.  In particular, future cells remain missing and the
    input object is never modified in place.
    """

    if not isinstance(triangle, pd.DataFrame):
        raise TypeError("triangle must be a pandas DataFrame.")

    if triangle.empty:
        raise ValueError("triangle cannot be empty.")

    history_start_ay = int(history_start_ay)
    valuation_year = int(valuation_year)

    if history_start_ay > valuation_year:
        raise ValueError(
            "history_start_ay cannot be later than valuation_year."
        )

    output = triangle.copy(deep=True)

    try:
        output.index = [int(value) for value in output.index]
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Triangle accident years must be convertible to integers."
        ) from error

    if output.index.duplicated().any():
        raise ValueError("Triangle contains duplicate accident years.")

    output = output.sort_index()
    expected_accident_years = list(
        range(history_start_ay, valuation_year + 1)
    )

    missing_accident_years = (
        set(expected_accident_years) - set(output.index)
    )
    if missing_accident_years:
        raise ValueError(
            "Triangle is missing requested history accident years: "
            f"{sorted(missing_accident_years)}"
        )

    filtered = output.loc[expected_accident_years].copy(deep=True)
    filtered.index.name = triangle.index.name or "accident_year"
    filtered.columns.name = triangle.columns.name

    return filtered


def future_cells_are_hidden(
    triangle: pd.DataFrame,
    valuation_year: int,
) -> bool:
    """Return whether future cells are missing and observed cells exist."""

    for accident_year in triangle.index:
        for development_year in triangle.columns:
            calendar_year = (
                int(accident_year) + int(development_year) - 1
            )
            value = triangle.loc[accident_year, development_year]

            if calendar_year <= int(valuation_year):
                if pd.isna(value):
                    return False
            elif not pd.isna(value):
                return False

    return True


def aggregate_accident_year_amount(
    accident_year_table: pd.DataFrame,
    value_column: str,
    evaluation_start_ay: int = EVALUATION_START_AY,
    evaluation_end_ay: int = EVALUATION_END_AY,
) -> float:
    """Sum one accident-year amount over the fixed evaluation target."""

    required_columns = {"accident_year", value_column}
    missing_columns = required_columns - set(accident_year_table.columns)
    if missing_columns:
        raise ValueError(
            "Accident-year table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    data = accident_year_table.copy(deep=True)
    data["accident_year"] = data["accident_year"].astype(int)

    if data["accident_year"].duplicated().any():
        raise ValueError("Accident-year table contains duplicate years.")

    expected_years = set(
        range(int(evaluation_start_ay), int(evaluation_end_ay) + 1)
    )
    available_years = set(data["accident_year"])
    missing_years = expected_years - available_years

    if missing_years:
        raise ValueError(
            "Accident-year table is missing evaluation years: "
            f"{sorted(missing_years)}"
        )

    target = data.loc[data["accident_year"].isin(expected_years)]
    values = target[value_column].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"Evaluation values in {value_column} must be finite."
        )

    return float(values.sum())


def calculate_true_target_values(
    true_reserves_by_accident_year: pd.DataFrame,
    basis: str,
    evaluation_start_ay: int = EVALUATION_START_AY,
    evaluation_end_ay: int = EVALUATION_END_AY,
) -> dict[str, float]:
    """Return observed paid, reserve, and ultimate for the fixed target."""

    if basis not in {"gross", "ceded"}:
        raise ValueError("basis must be either 'gross' or 'ceded'.")

    observed_paid = aggregate_accident_year_amount(
        true_reserves_by_accident_year,
        f"observed_{basis}_paid",
        evaluation_start_ay,
        evaluation_end_ay,
    )
    true_reserve = aggregate_accident_year_amount(
        true_reserves_by_accident_year,
        f"true_{basis}_reserve",
        evaluation_start_ay,
        evaluation_end_ay,
    )
    true_ultimate = aggregate_accident_year_amount(
        true_reserves_by_accident_year,
        f"{basis}_ultimate",
        evaluation_start_ay,
        evaluation_end_ay,
    )

    if not np.isclose(
        observed_paid + true_reserve,
        true_ultimate,
        rtol=1e-12,
        atol=1e-6,
    ):
        raise ValueError(
            "Fixed-target observed paid and true reserve do not "
            "reconcile with true ultimate."
        )

    return {
        "observed_target_paid": observed_paid,
        "true_target_reserve": true_reserve,
        "true_target_ultimate": true_ultimate,
    }


def validate_history_windows(
    history_windows: Mapping[int, int] = HISTORY_WINDOWS,
) -> None:
    """Validate the frozen Step 26 history-window definition."""

    expected = {
        15: 2010,
        10: 2015,
        7: 2018,
        5: 2020,
    }

    if dict(history_windows) != expected:
        raise ValueError(
            "Step 26 history windows must remain equal to "
            f"{expected}."
        )

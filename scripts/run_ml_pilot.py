"""Run the regularised Poisson reserving pilot."""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import (
    INFLATION_SCENARIOS,
    ML_AMOUNT_SCALE,
    ML_MIN_TRAINING_DIAGONALS,
    ML_MIN_VALIDATION_FOLDS,
    ML_POISSON_ALPHA_GRID,
    PILOT_DATA_DIR,
    VALUATION_YEAR,
    ensure_directories,
    validate_config,
)

from src.ml_models import (
    fit_regularised_poisson_reserving_model,
)

from src.simulation import (
    build_inflation_index,
)


TRIANGLE_DIR = (
    PILOT_DATA_DIR / "triangles"
)

OUTPUT_DIR = (
    PILOT_DATA_DIR / "ml"
)


def read_triangle(
    filename: str,
) -> pd.DataFrame:
    """Read a triangle and restore integer labels."""

    path = TRIANGLE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Triangle file not found: {path}"
        )

    triangle = pd.read_csv(
        path,
        index_col=0,
    )

    triangle.index = (
        triangle.index.astype(int)
    )

    triangle.columns = [
        int(column)
        for column in triangle.columns
    ]

    triangle.index.name = "accident_year"
    triangle.columns.name = (
        "development_year"
    )

    return triangle


def run_model_safely(
    basis: str,
    triangle: pd.DataFrame,
    inflation_index: dict[int, float],
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any],
]:
    """Run one Poisson model and record failure information."""

    try:
        result = (
            fit_regularised_poisson_reserving_model(
                incremental_triangle=triangle,
                inflation_index=inflation_index,
                valuation_year=VALUATION_YEAR,
                basis=basis,
                alpha_grid=(
                    ML_POISSON_ALPHA_GRID
                ),
                minimum_training_diagonals=(
                    ML_MIN_TRAINING_DIAGONALS
                ),
                minimum_validation_folds=(
                    ML_MIN_VALIDATION_FOLDS
                ),
                amount_scale=ML_AMOUNT_SCALE,
                structural_break_year=None,
            )
        )

        status = {
            "model": "regularized_poisson",
            "basis": basis,
            "model_status": "success",
            "failure_reason": "",
        }

        return result, status

    except ValueError as error:
        status = {
            "model": "regularized_poisson",
            "basis": basis,
            "model_status": "failed",
            "failure_reason": str(error),
        }

        print(
            f"Regularised Poisson failed "
            f"for {basis}: {error}"
        )

        return None, status


def main() -> None:
    """Run and save the gross and ceded ML pilot."""

    ensure_directories()
    validate_config()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    gross_triangle = read_triangle(
        "gross_incremental_paid_triangle.csv"
    )

    ceded_triangle = read_triangle(
        "ceded_incremental_paid_triangle.csv"
    )

    true_totals_path = (
        TRIANGLE_DIR
        / "true_reserve_totals.csv"
    )

    if not true_totals_path.exists():
        raise FileNotFoundError(
            f"True-reserve file not found: "
            f"{true_totals_path}"
        )

    true_totals = pd.read_csv(
        true_totals_path
    ).iloc[0]

    inflation_table = (
        build_inflation_index(
            INFLATION_SCENARIOS["stable"]
        )
    )

    inflation_index = (
        inflation_table
        .set_index("calendar_year")[
            "inflation_index"
        ]
        .to_dict()
    )

    model_results: dict[
        str,
        dict[str, Any],
    ] = {}

    status_rows: list[
        dict[str, Any]
    ] = []

    for basis, triangle in {
        "gross": gross_triangle,
        "ceded": ceded_triangle,
    }.items():

        result, status = run_model_safely(
            basis=basis,
            triangle=triangle,
            inflation_index=inflation_index,
        )

        status_rows.append(status)

        if result is not None:
            model_results[basis] = result

    summary_rows: list[
        dict[str, Any]
    ] = []

    for basis, result in (
        model_results.items()
    ):
        reserve_table = result[
            "reserve_by_accident_year"
        ]

        future_predictions = result[
            "future_predictions"
        ]

        alpha_validation = result[
            "alpha_validation"
        ]

        fold_validation = result[
            "fold_validation"
        ]

        reserve_table.to_csv(
            OUTPUT_DIR
            / (
                "regularized_poisson_"
                f"{basis}_by_accident_year.csv"
            ),
            index=False,
        )

        future_predictions.to_csv(
            OUTPUT_DIR
            / (
                "regularized_poisson_"
                f"{basis}_future_cells.csv"
            ),
            index=False,
        )

        alpha_validation.to_csv(
            OUTPUT_DIR
            / (
                "regularized_poisson_"
                f"{basis}_alpha_validation.csv"
            ),
            index=False,
        )

        fold_validation.to_csv(
            OUTPUT_DIR
            / (
                "regularized_poisson_"
                f"{basis}_fold_validation.csv"
            ),
            index=False,
        )

        summary_row = (
            result["summary"]
            .iloc[0]
            .to_dict()
        )

        estimated_reserve = float(
            summary_row[
                "total_estimated_reserve"
            ]
        )

        if basis == "gross":
            true_reserve = float(
                true_totals[
                    "total_true_gross_reserve"
                ]
            )
        else:
            true_reserve = float(
                true_totals[
                    "total_true_ceded_reserve"
                ]
            )

        signed_error = (
            estimated_reserve
            - true_reserve
        )

        summary_row.update(
            {
                "inflation_scenario": (
                    "stable"
                ),
                "true_reserve": (
                    true_reserve
                ),
                "signed_error": (
                    signed_error
                ),
                "absolute_error": abs(
                    signed_error
                ),
                "signed_percentage_error": (
                    100.0
                    * signed_error
                    / true_reserve
                    if true_reserve != 0.0
                    else float("nan")
                ),
                "absolute_percentage_error": (
                    100.0
                    * abs(signed_error)
                    / true_reserve
                    if true_reserve != 0.0
                    else float("nan")
                ),
            }
        )

        summary_rows.append(
            summary_row
        )

    model_summary = pd.DataFrame(
        summary_rows
    )

    model_status = pd.DataFrame(
        status_rows
    )

    model_summary.to_csv(
        OUTPUT_DIR / "model_summary.csv",
        index=False,
    )

    model_status.to_csv(
        OUTPUT_DIR / "model_status.csv",
        index=False,
    )

    print(
        "Regularised Poisson reserving "
        "pilot completed."
    )

    print(
        f"Valuation year: {VALUATION_YEAR}"
    )

    print(
        "Inflation scenario: stable"
    )

    print("\nModel summary:")

    if model_summary.empty:
        print(
            "No models completed successfully."
        )
    else:
        print(
            model_summary.to_string(
                index=False
            )
        )

    print("\nModel execution status:")

    print(
        model_status.to_string(
            index=False
        )
    )

    print(
        "\nOutputs saved to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
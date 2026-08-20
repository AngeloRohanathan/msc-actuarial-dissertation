"""Run the classical reserving models on the pilot triangles."""

from __future__ import annotations

import pandas as pd

from config import (
    CASHFLOW_UPLIFT_EMBEDDED_INFLATION,
    INFLATION_SCENARIOS,
    PILOT_DATA_DIR,
    VALUATION_YEAR,
    ensure_directories,
    validate_config,
)

from src.reserving import (
    cashflow_uplift,
    chain_ladder,
    inflation_adjusted_chain_ladder,
)

from src.simulation import (
    build_inflation_index,
)


TRIANGLE_DIR = (
    PILOT_DATA_DIR / "triangles"
)

OUTPUT_DIR = (
    PILOT_DATA_DIR / "reserving"
)


def read_triangle(
    filename: str,
) -> pd.DataFrame:
    """Read a saved triangle and restore integer labels."""

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
    triangle.columns.name = "development_year"

    return triangle

def run_model_safely(
    model_name: str,
    basis: str,
    model_function,
    **model_arguments,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    """Run one reserving model and record any model failure."""

    try:
        result = model_function(
            **model_arguments
        )

        estimated_reserve = float(
            result["summary"].loc[
                0,
                "total_estimated_reserve",
            ]
        )

        status = {
            "model": model_name,
            "basis": basis,
            "model_status": "success",
            "failure_reason": "",
            "estimated_reserve": (
                estimated_reserve
            ),
        }

        return result, status

    except ValueError as error:
        status = {
            "model": model_name,
            "basis": basis,
            "model_status": "failed",
            "failure_reason": str(error),
            "estimated_reserve": float("nan"),
        }

        print(
            f"\n{model_name} failed for "
            f"{basis}: {error}"
        )

        return None, status


def main() -> None:
    """Run and save the classical pilot results."""

    ensure_directories()
    validate_config()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    gross_incremental = read_triangle(
        "gross_incremental_paid_triangle.csv"
    )

    gross_cumulative = read_triangle(
        "gross_cumulative_paid_triangle.csv"
    )

    ceded_incremental = read_triangle(
        "ceded_incremental_paid_triangle.csv"
    )

    ceded_cumulative = read_triangle(
        "ceded_cumulative_paid_triangle.csv"
    )

    true_totals_path = (
        TRIANGLE_DIR
        / "true_reserve_totals.csv"
    )

    if not true_totals_path.exists():
        raise FileNotFoundError(
            f"True reserve file not found: "
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

    results: dict[
        tuple[str, str],
        dict[str, object],
    ] = {}
    
    
    results: dict[
    tuple[str, str],
    dict[str, object],
    ] = {}

    model_status_rows: list[
        dict[str, object]
    ] = []
    
    
    # ---------------------------------------------------------
    # Standard Chain Ladder
    # ---------------------------------------------------------

    gross_cl, gross_cl_status = (
        run_model_safely(
            model_name="chain_ladder",
            basis="gross",
            model_function=chain_ladder,
            cumulative_triangle=gross_cumulative,
        )
    )

    model_status_rows.append(
        gross_cl_status
    )

    if gross_cl is not None:
        results[
            ("chain_ladder", "gross")
        ] = gross_cl


    ceded_cl, ceded_cl_status = (
        run_model_safely(
            model_name="chain_ladder",
            basis="ceded",
            model_function=chain_ladder,
            cumulative_triangle=ceded_cumulative,
        )
    )

    model_status_rows.append(
        ceded_cl_status
    )

    if ceded_cl is not None:
        results[
            ("chain_ladder", "ceded")
        ] = ceded_cl


    # ---------------------------------------------------------
    # Inflation-Adjusted Chain Ladder
    # ---------------------------------------------------------

    gross_iacl, gross_iacl_status = (
        run_model_safely(
            model_name=(
                "inflation_adjusted_chain_ladder"
            ),
            basis="gross",
            model_function=(
                inflation_adjusted_chain_ladder
            ),
            nominal_incremental_triangle=(
                gross_incremental
            ),
            inflation_index=inflation_index,
            valuation_year=VALUATION_YEAR,
        )
    )

    model_status_rows.append(
        gross_iacl_status
    )

    if gross_iacl is not None:
        results[
            (
                "inflation_adjusted_chain_ladder",
                "gross",
            )
        ] = gross_iacl


    ceded_iacl, ceded_iacl_status = (
        run_model_safely(
            model_name=(
                "inflation_adjusted_chain_ladder"
            ),
            basis="ceded",
            model_function=(
                inflation_adjusted_chain_ladder
            ),
            nominal_incremental_triangle=(
                ceded_incremental
            ),
            inflation_index=inflation_index,
            valuation_year=VALUATION_YEAR,
        )
    )

    model_status_rows.append(
        ceded_iacl_status
    )

    if ceded_iacl is not None:
        results[
            (
                "inflation_adjusted_chain_ladder",
                "ceded",
            )
        ] = ceded_iacl

    # ---------------------------------------------------------
    # Cashflow Uplift
    # ---------------------------------------------------------

    gross_uplift, gross_uplift_status = (
        run_model_safely(
            model_name="cashflow_uplift",
            basis="gross",
            model_function=cashflow_uplift,
            nominal_cumulative_triangle=(
                gross_cumulative
            ),
            forecast_inflation_index=(
                inflation_index
            ),
            valuation_year=VALUATION_YEAR,
            embedded_annual_inflation=(
                CASHFLOW_UPLIFT_EMBEDDED_INFLATION
            ),
        )
    )

    model_status_rows.append(
        gross_uplift_status
    )

    if gross_uplift is not None:
        results[
            ("cashflow_uplift", "gross")
        ] = gross_uplift


    ceded_uplift, ceded_uplift_status = (
        run_model_safely(
            model_name="cashflow_uplift",
            basis="ceded",
            model_function=cashflow_uplift,
            nominal_cumulative_triangle=(
                ceded_cumulative
            ),
            forecast_inflation_index=(
                inflation_index
            ),
            valuation_year=VALUATION_YEAR,
            embedded_annual_inflation=(
                CASHFLOW_UPLIFT_EMBEDDED_INFLATION
            ),
        )
    )

    model_status_rows.append(
        ceded_uplift_status
    )

    if ceded_uplift is not None:
        results[
            ("cashflow_uplift", "ceded")
        ] = ceded_uplift

    summary_rows: list[
        dict[str, object]
    ] = []

    output_names = {
        (
            "chain_ladder",
            "gross",
        ): "chain_ladder_gross_by_accident_year.csv",

        (
            "chain_ladder",
            "ceded",
        ): "chain_ladder_ceded_by_accident_year.csv",

        (
            "inflation_adjusted_chain_ladder",
            "gross",
        ): "iacl_gross_by_accident_year.csv",

        (
            "inflation_adjusted_chain_ladder",
            "ceded",
        ): "iacl_ceded_by_accident_year.csv",

        (
            "cashflow_uplift",
            "gross",
        ): "cashflow_uplift_gross_by_accident_year.csv",

        (
            "cashflow_uplift",
            "ceded",
        ): "cashflow_uplift_ceded_by_accident_year.csv",
    }

    for (
        model_name,
        basis,
    ), result in results.items():

        reserve_table = result[
            "reserve_by_accident_year"
        ]

        reserve_table.to_csv(
            OUTPUT_DIR
            / output_names[
                (model_name, basis)
            ],
            index=False,
        )

        estimated_reserve = float(
            result["summary"].loc[
                0,
                "total_estimated_reserve",
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

        summary_rows.append(
            {
                "model": model_name,
                "basis": basis,
                "valuation_year": (
                    VALUATION_YEAR
                ),
                "inflation_scenario": (
                    "stable"
                ),
                "estimated_reserve": (
                    estimated_reserve
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
            }
        )

    model_summary = pd.DataFrame(
        summary_rows
    )

    model_summary.to_csv(
        OUTPUT_DIR / "model_summary.csv",
        index=False,
    )

    print(
        "Classical reserving pilot completed."
    )

    print(
        f"Valuation year: {VALUATION_YEAR}"
    )

    print(
        "Inflation scenario: stable"
    )

    print("\nModel summary:")
    print(
        model_summary.to_string(
            index=False
        )
    )
    
    model_status = pd.DataFrame(
    model_status_rows
    )

    model_status.to_csv(
        OUTPUT_DIR / "model_status.csv",
        index=False,
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
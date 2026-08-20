"""Construct and save the baseline observed paid triangles."""

from __future__ import annotations

import pandas as pd

from config import (
    PILOT_DATA_DIR,
    VALUATION_YEAR,
    ensure_directories,
    validate_config,
)

from src.triangles import (
    build_triangle_package,
)


INPUT_PATH = (
    PILOT_DATA_DIR
    / "reinsurance"
    / "reinsured_payments_pilot.parquet"
)

OUTPUT_DIR = (
    PILOT_DATA_DIR
    / "triangles"
)


def main() -> None:
    """Build and save gross and ceded pilot triangles."""

    ensure_directories()
    validate_config()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Reinsured payment file was not found: "
            f"{INPUT_PATH}. Run Step 12 first."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payments = pd.read_parquet(
        INPUT_PATH,
        engine="pyarrow",
    )

    outputs = build_triangle_package(
        payments=payments,
        valuation_year=VALUATION_YEAR,
    )

    triangle_files = {
        "gross_incremental": (
            "gross_incremental_paid_triangle.csv"
        ),
        "gross_cumulative": (
            "gross_cumulative_paid_triangle.csv"
        ),
        "ceded_incremental": (
            "ceded_incremental_paid_triangle.csv"
        ),
        "ceded_cumulative": (
            "ceded_cumulative_paid_triangle.csv"
        ),
    }

    for output_name, filename in (
        triangle_files.items()
    ):
        outputs[output_name].to_csv(
            OUTPUT_DIR / filename,
            index=True,
        )

    outputs[
        "true_reserves_by_accident_year"
    ].to_csv(
        OUTPUT_DIR
        / "true_reserves_by_accident_year.csv",
        index=False,
    )

    outputs[
        "true_reserve_totals"
    ].to_csv(
        OUTPUT_DIR
        / "true_reserve_totals.csv",
        index=False,
    )

    gross_incremental = outputs[
        "gross_incremental"
    ]

    ceded_incremental = outputs[
        "ceded_incremental"
    ]

    totals = outputs[
        "true_reserve_totals"
    ].iloc[0]

    observed_payment_rows = int(
        (
            payments["payment_calendar_year"]
            <= VALUATION_YEAR
        ).sum()
    )

    future_payment_rows = int(
        (
            payments["payment_calendar_year"]
            > VALUATION_YEAR
        ).sum()
    )

    print(
        "Triangle pilot completed successfully."
    )

    print(
        f"Valuation year: {VALUATION_YEAR}"
    )

    print(
        "Gross triangle dimensions: "
        f"{gross_incremental.shape}"
    )

    print(
        "Ceded triangle dimensions: "
        f"{ceded_incremental.shape}"
    )

    print(
        "Observed payment rows: "
        f"{observed_payment_rows:,}"
    )

    print(
        "Future payment rows: "
        f"{future_payment_rows:,}"
    )

    print(
        "Observed gross paid: "
        f"£{totals['total_observed_gross_paid']:,.2f}"
    )

    print(
        "True gross reserve: "
        f"£{totals['total_true_gross_reserve']:,.2f}"
    )

    print(
        "Gross ultimate: "
        f"£{totals['total_gross_ultimate']:,.2f}"
    )

    print(
        "Observed ceded paid: "
        f"£{totals['total_observed_ceded_paid']:,.2f}"
    )

    print(
        "True ceded reserve: "
        f"£{totals['total_true_ceded_reserve']:,.2f}"
    )

    print(
        "Ceded ultimate: "
        f"£{totals['total_ceded_ultimate']:,.2f}"
    )

    print(
        "Outputs saved to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
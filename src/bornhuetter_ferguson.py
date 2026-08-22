"""Paid Bornhuetter-Ferguson reserving."""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
import inspect

import numpy as np
import pandas as pd


BF_STANDARD = (
    "bornhuetter_ferguson_standard"
)

BF_BREAK_AWARE = (
    "bornhuetter_ferguson_break_aware"
)

VALID_BF_VARIANTS = {
    BF_STANDARD,
    BF_BREAK_AWARE,
}


def _coerce_bool(
    value: object,
) -> bool:
    """Convert common boolean representations."""

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    value_as_text = (
        str(value)
        .strip()
        .lower()
    )

    if value_as_text in {
        "true",
        "1",
        "yes",
    }:
        return True

    if value_as_text in {
        "false",
        "0",
        "no",
    }:
        return False

    raise ValueError(
        f"Cannot interpret as boolean: {value}"
    )


def cumulative_paid_proportion(
    *,
    incremental_pattern: Sequence[float],
    development_year: int,
) -> float:
    """Return expected cumulative paid proportion."""

    if development_year < 1:
        raise ValueError(
            "development_year must be at least 1."
        )

    pattern = [
        float(value)
        for value in incremental_pattern
    ]

    if not pattern:
        raise ValueError(
            "incremental_pattern cannot be empty."
        )

    if any(
        value < 0
        for value in pattern
    ):
        raise ValueError(
            "Payment proportions cannot be negative."
        )

    if not np.isclose(
        sum(pattern),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Payment pattern must sum to 1.0."
        )

    if development_year >= len(
        pattern
    ):
        return 1.0

    return float(
        sum(
            pattern[
                :development_year
            ]
        )
    )


def select_bf_pattern_name(
    *,
    tail_type: str,
    structural_break: bool,
    accident_year: int,
    variant: str,
    structural_break_year: int,
) -> str:
    """Select the benchmark pattern for one AY."""

    if variant not in VALID_BF_VARIANTS:
        raise ValueError(
            "Unknown BF variant: "
            f"{variant}"
        )

    tail_type = str(
        tail_type
    ).strip().lower()

    structural_break = (
        _coerce_bool(
            structural_break
        )
    )

    if tail_type == "short":
        return "short"

    if tail_type != "long":
        raise ValueError(
            "tail_type must be 'short' or 'long'."
        )

    if variant == BF_STANDARD:
        return "long"

    if (
        structural_break
        and int(
            accident_year
        )
        >= int(
            structural_break_year
        )
    ):
        return "accelerated_long"

    return "long"


def paid_bf_reserve(
    *,
    prior_ultimate: float,
    cumulative_paid_proportion: float,
) -> float:
    """Calculate one paid BF reserve."""

    prior_ultimate = float(
        prior_ultimate
    )

    cumulative_paid_proportion = float(
        cumulative_paid_proportion
    )

    if not np.isfinite(
        prior_ultimate
    ):
        raise ValueError(
            "prior_ultimate must be finite."
        )

    if prior_ultimate < 0:
        raise ValueError(
            "prior_ultimate cannot be negative."
        )

    if not (
        0.0
        <= cumulative_paid_proportion
        <= 1.0
    ):
        raise ValueError(
            "cumulative_paid_proportion must "
            "lie between 0 and 1."
        )

    return float(
        prior_ultimate
        * (
            1.0
            - cumulative_paid_proportion
        )
    )


def build_paid_bf_by_accident_year(
    *,
    estimator_input: pd.DataFrame,
    variant: str,
    valuation_year: int,
    structural_break_year: int,
    benchmark_patterns: Mapping[
        str,
        Sequence[float],
    ],
) -> pd.DataFrame:
    """Calculate BF reserves using no future truth."""

    required_columns = {
        "scenario_id",
        "simulation_id",
        "accident_year",
        "basis",
        "tail_type",
        "structural_break",
        "expected_loss_prior_ultimate",
        "paid_to_date",
        "pricing_assumption_version",
    }

    missing = (
        required_columns
        - set(
            estimator_input.columns
        )
    )

    if missing:
        raise ValueError(
            "BF input is missing columns: "
            f"{sorted(missing)}"
        )

    for required_pattern in [
        "short",
        "long",
        "accelerated_long",
    ]:
        if required_pattern not in benchmark_patterns:
            raise ValueError(
                "Missing BF benchmark pattern: "
                f"{required_pattern}"
            )

    records: list[
        dict[str, object]
    ] = []

    for row in estimator_input.to_dict(
        orient="records"
    ):
        accident_year = int(
            row[
                "accident_year"
            ]
        )

        development_year = (
            int(
                valuation_year
            )
            - accident_year
            + 1
        )

        if development_year < 1:
            raise ValueError(
                "Accident year occurs after "
                "the valuation year."
            )

        structural_break = (
            _coerce_bool(
                row[
                    "structural_break"
                ]
            )
        )

        pattern_name = (
            select_bf_pattern_name(
                tail_type=(
                    row[
                        "tail_type"
                    ]
                ),
                structural_break=(
                    structural_break
                ),
                accident_year=(
                    accident_year
                ),
                variant=variant,
                structural_break_year=(
                    structural_break_year
                ),
            )
        )

        cumulative_proportion = (
            cumulative_paid_proportion(
                incremental_pattern=(
                    benchmark_patterns[
                        pattern_name
                    ]
                ),
                development_year=(
                    development_year
                ),
            )
        )

        prior_ultimate = float(
            row[
                "expected_loss_prior_ultimate"
            ]
        )

        paid_to_date = float(
            row[
                "paid_to_date"
            ]
        )

        reserve = paid_bf_reserve(
            prior_ultimate=(
                prior_ultimate
            ),
            cumulative_paid_proportion=(
                cumulative_proportion
            ),
        )

        output_row = dict(
            row
        )

        output_row[
            "structural_break"
        ] = structural_break

        output_row[
            "development_year_at_valuation"
        ] = development_year

        output_row[
            "benchmark_pattern"
        ] = pattern_name

        output_row[
            "benchmark_paid_proportion"
        ] = cumulative_proportion

        output_row[
            "benchmark_unpaid_proportion"
        ] = (
            1.0
            - cumulative_proportion
        )

        output_row[
            "estimated_reserve"
        ] = reserve

        output_row[
            "estimated_ultimate"
        ] = (
            paid_to_date
            + reserve
        )

        output_row[
            "model"
        ] = variant

        output_row[
            "pattern_source"
        ] = (
            "fixed_independent_benchmark"
        )

        records.append(
            output_row
        )

    result = pd.DataFrame(
        records
    )

    return (
        result.sort_values(
            [
                "scenario_id",
                "simulation_id",
                "basis",
                "accident_year",
            ]
        )
        .reset_index(
            drop=True
        )
    )
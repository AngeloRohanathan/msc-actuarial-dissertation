"""Diagnostics for Bornhuetter-Ferguson reserve errors."""

from __future__ import annotations

import numpy as np
import pandas as pd


BF_STANDARD = "bornhuetter_ferguson_standard"
BF_BREAK_AWARE = "bornhuetter_ferguson_break_aware"


def add_bf_error_decomposition(
    data: pd.DataFrame,
    *,
    structural_break_year: int,
) -> pd.DataFrame:
    """
    Add prior adequacy, development-pattern adequacy,
    and exact BF reserve-error decomposition.

    Truth is used for diagnostic evaluation only.
    """

    required_columns = {
        "scenario_id",
        "simulation_id",
        "accident_year",
        "basis",
        "model",
        "tail_type",
        "structural_break",
        "paid_to_date",
        "expected_loss_prior_ultimate",
        "benchmark_paid_proportion",
        "estimated_reserve",
        "true_reserve",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required BF diagnostic columns: "
            f"{sorted(missing)}"
        )

    result = data.copy()

    numeric_columns = [
        "paid_to_date",
        "expected_loss_prior_ultimate",
        "benchmark_paid_proportion",
        "estimated_reserve",
        "true_reserve",
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        )

    # ---------------------------------------------------------
    # True ultimate and realised cumulative paid proportion
    # ---------------------------------------------------------

    result["true_ultimate"] = (
        result["paid_to_date"]
        + result["true_reserve"]
    )

    result["true_paid_proportion"] = np.where(
        result["true_ultimate"] > 0,
        result["paid_to_date"]
        / result["true_ultimate"],
        np.nan,
    )

    result["true_unpaid_proportion"] = np.where(
        result["true_ultimate"] > 0,
        result["true_reserve"]
        / result["true_ultimate"],
        np.nan,
    )

    # ---------------------------------------------------------
    # Prior adequacy
    # ---------------------------------------------------------

    result["prior_ultimate_error"] = (
        result["expected_loss_prior_ultimate"]
        - result["true_ultimate"]
    )

    result["prior_ultimate_error_pct"] = np.where(
        result["true_ultimate"] > 0,
        100.0
        * result["prior_ultimate_error"]
        / result["true_ultimate"],
        np.nan,
    )

    # ---------------------------------------------------------
    # Pattern adequacy
    # Positive value:
    # actual cumulative paid proportion is greater than assumed.
    #
    # Negative value:
    # assumed pattern is more developed than realised experience.
    # ---------------------------------------------------------

    result["paid_proportion_error"] = (
        result["true_paid_proportion"]
        - result["benchmark_paid_proportion"]
    )

    # ---------------------------------------------------------
    # Exact BF reserve-error decomposition
    # ---------------------------------------------------------

    result["prior_error_component"] = (
        result["prior_ultimate_error"]
        * (
            1.0
            - result["benchmark_paid_proportion"]
        )
    )

    result["development_error_component"] = (
        result["true_ultimate"]
        * result["paid_proportion_error"]
    )

    result["bf_reserve_error"] = (
        result["estimated_reserve"]
        - result["true_reserve"]
    )

    result["decomposition_sum"] = (
        result["prior_error_component"]
        + result["development_error_component"]
    )

    result["decomposition_residual"] = (
        result["bf_reserve_error"]
        - result["decomposition_sum"]
    )

    # ---------------------------------------------------------
    # Regime labels
    # ---------------------------------------------------------

    result["structural_break"] = (
        result["structural_break"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    result["break_period"] = np.where(
        (
            result["structural_break"]
            & (
                result["accident_year"]
                >= structural_break_year
            )
        ),
        "post_break",
        "pre_break_or_no_break",
    )

    result["development_age_group"] = pd.cut(
        result["development_year_at_valuation"],
        bins=[0, 2, 4, 7, np.inf],
        labels=[
            "dev_1_2",
            "dev_3_4",
            "dev_5_7",
            "dev_8_plus",
        ],
        right=True,
    )

    return result


def build_prior_adequacy_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise prior ultimate adequacy."""

    return (
        data.groupby(
            [
                "scenario_id",
                "basis",
                "model",
                "break_period",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            rows=("accident_year", "size"),
            mean_prior_error_pct=(
                "prior_ultimate_error_pct",
                "mean",
            ),
            median_prior_error_pct=(
                "prior_ultimate_error_pct",
                "median",
            ),
            mean_prior_ultimate=(
                "expected_loss_prior_ultimate",
                "mean",
            ),
            mean_true_ultimate=(
                "true_ultimate",
                "mean",
            ),
        )
    )


def build_pattern_adequacy_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise assumed versus realised development."""

    return (
        data.groupby(
            [
                "scenario_id",
                "basis",
                "model",
                "break_period",
                "development_year_at_valuation",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            rows=("accident_year", "size"),
            mean_assumed_paid_proportion=(
                "benchmark_paid_proportion",
                "mean",
            ),
            mean_true_paid_proportion=(
                "true_paid_proportion",
                "mean",
            ),
            mean_paid_proportion_error=(
                "paid_proportion_error",
                "mean",
            ),
        )
    )


def build_decomposition_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise BF error components."""

    return (
        data.groupby(
            [
                "scenario_id",
                "basis",
                "model",
                "break_period",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            rows=("accident_year", "size"),
            mean_bf_reserve_error=(
                "bf_reserve_error",
                "mean",
            ),
            mean_prior_error_component=(
                "prior_error_component",
                "mean",
            ),
            mean_development_error_component=(
                "development_error_component",
                "mean",
            ),
            median_bf_reserve_error=(
                "bf_reserve_error",
                "median",
            ),
            mean_absolute_bf_reserve_error=(
                "bf_reserve_error",
                lambda x: np.mean(np.abs(x)),
            ),
            max_absolute_decomposition_residual=(
                "decomposition_residual",
                lambda x: np.max(np.abs(x)),
            ),
        )
    )
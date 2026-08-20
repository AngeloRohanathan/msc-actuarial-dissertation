"""Distributional diagnostics for reserving simulation experiments."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP_COLUMNS = [
    "scenario_id",
    "tail_type",
    "inflation_scenario",
    "structural_break",
    "clause_type",
    "model",
    "basis",
]


REQUIRED_RESULT_COLUMNS = {
    *GROUP_COLUMNS,
    "simulation_id",
    "success_or_failure",
    "signed_error",
    "absolute_error",
    "percentage_error",
    "absolute_percentage_error",
    "runtime_seconds",
}


CLASSICAL_CHAIN_LADDER_MODELS = {
    "chain_ladder",
    "inflation_adjusted_chain_ladder",
    "cashflow_uplift",
}


MODEL_DISPLAY_NAMES = {
    "chain_ladder": "Chain Ladder",
    "inflation_adjusted_chain_ladder": "Inflation-Adjusted CL",
    "cashflow_uplift": "Cashflow Uplift",
    "regularized_poisson": "Regularised Poisson",
    "expected_loss": "Expected Loss",
    "bornhuetter_ferguson": "Bornhuetter–Ferguson",
    "regularized_tweedie": "Regularised Tweedie",
}


PREFERRED_MODEL_ORDER = [
    "chain_ladder",
    "inflation_adjusted_chain_ladder",
    "cashflow_uplift",
    "expected_loss",
    "bornhuetter_ferguson",
    "regularized_poisson",
    "regularized_tweedie",
]


def validate_diagnostics_input(
    results: pd.DataFrame,
) -> None:
    """Validate the experiment-results table."""

    missing_columns = (
        REQUIRED_RESULT_COLUMNS
        - set(results.columns)
    )

    if missing_columns:
        raise ValueError(
            "The experiment results are missing "
            f"required columns: {sorted(missing_columns)}"
        )

    if results.empty:
        raise ValueError(
            "The experiment-results table is empty."
        )

    valid_statuses = {
        "success",
        "failed",
    }

    observed_statuses = set(
        results["success_or_failure"]
        .dropna()
        .astype(str)
    )

    invalid_statuses = (
        observed_statuses
        - valid_statuses
    )

    if invalid_statuses:
        raise ValueError(
            "Unexpected success_or_failure values: "
            f"{sorted(invalid_statuses)}"
        )


def _finite_numeric_series(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return finite numeric values from one column."""

    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    )

    values = values.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return (
        values
        .dropna()
        .astype(float)
    )


def _safe_mean(
    values: pd.Series,
) -> float:
    """Return a mean or NaN when no observations exist."""

    if values.empty:
        return np.nan

    return float(
        values.mean()
    )


def _safe_median(
    values: pd.Series,
) -> float:
    """Return a median or NaN when no observations exist."""

    if values.empty:
        return np.nan

    return float(
        values.median()
    )


def _safe_standard_deviation(
    values: pd.Series,
) -> float:
    """Return the sample standard deviation."""

    if len(values) < 2:
        return np.nan

    return float(
        values.std(
            ddof=1
        )
    )


def _safe_quantile(
    values: pd.Series,
    probability: float,
) -> float:
    """Return a quantile or NaN when no observations exist."""

    if values.empty:
        return np.nan

    return float(
        values.quantile(
            probability
        )
    )


def _safe_rmse(
    values: pd.Series,
) -> float:
    """Calculate root mean squared error."""

    if values.empty:
        return np.nan

    return float(
        np.sqrt(
            np.mean(
                np.square(values)
            )
        )
    )


def _safe_mcse(
    values: pd.Series,
) -> float:
    """Calculate the Monte Carlo standard error of a sample mean."""

    if len(values) < 2:
        return np.nan

    return float(
        values.std(
            ddof=1
        )
        / np.sqrt(
            len(values)
        )
    )


def _accuracy_scope(
    attempts: int,
    successful_fits: int,
) -> str:
    """Describe whether accuracy metrics are conditional on success."""

    if successful_fits == 0:
        return (
            "not_available_no_successful_fits"
        )

    if successful_fits < attempts:
        return (
            "conditional_on_successful_fits"
        )

    return (
        "all_attempts_successful"
    )


def build_distributional_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Build detailed error diagnostics by scenario, model and basis."""

    validate_diagnostics_input(
        results
    )

    summary_rows: list[
        dict[str, Any]
    ] = []

    grouped = results.groupby(
        GROUP_COLUMNS,
        dropna=False,
        sort=True,
    )

    for group_values, group in grouped:
        group_information = dict(
            zip(
                GROUP_COLUMNS,
                group_values,
            )
        )

        successful = group.loc[
            group[
                "success_or_failure"
            ].eq("success")
        ].copy()

        attempts = int(
            len(group)
        )

        successful_fits = int(
            len(successful)
        )

        failed_fits = (
            attempts
            - successful_fits
        )

        success_rate = (
            successful_fits
            / attempts
            if attempts > 0
            else np.nan
        )

        signed_error = (
            _finite_numeric_series(
                successful,
                "signed_error",
            )
        )

        absolute_error = (
            _finite_numeric_series(
                successful,
                "absolute_error",
            )
        )

        percentage_error = (
            _finite_numeric_series(
                successful,
                "percentage_error",
            )
        )

        absolute_percentage_error = (
            _finite_numeric_series(
                successful,
                "absolute_percentage_error",
            )
        )

        runtime = (
            _finite_numeric_series(
                group,
                "runtime_seconds",
            )
        )

        scope = _accuracy_scope(
            attempts=attempts,
            successful_fits=successful_fits,
        )

        row = {
            **group_information,
            "attempts": attempts,
            "successful_fits": successful_fits,
            "failed_fits": failed_fits,
            "success_rate": float(
                success_rate
            ),
            "accuracy_scope": scope,
            "conditional_accuracy_warning": bool(
                scope
                == "conditional_on_successful_fits"
            ),
            "mean_runtime_seconds": _safe_mean(
                runtime
            ),
            "mean_signed_error": _safe_mean(
                signed_error
            ),
            "median_signed_error": _safe_median(
                signed_error
            ),
            "mean_absolute_error": _safe_mean(
                absolute_error
            ),
            "root_mean_squared_error": _safe_rmse(
                signed_error
            ),
            "mean_percentage_error": _safe_mean(
                percentage_error
            ),
            "median_percentage_error": _safe_median(
                percentage_error
            ),
            "mean_absolute_percentage_error": _safe_mean(
                absolute_percentage_error
            ),
            "median_absolute_percentage_error": _safe_median(
                absolute_percentage_error
            ),
            "standard_deviation_percentage_error": (
                _safe_standard_deviation(
                    percentage_error
                )
            ),
            "standard_deviation_absolute_percentage_error": (
                _safe_standard_deviation(
                    absolute_percentage_error
                )
            ),
            "percentage_error_minimum": _safe_quantile(
                percentage_error,
                0.00,
            ),
            "percentage_error_5th_percentile": _safe_quantile(
                percentage_error,
                0.05,
            ),
            "percentage_error_25th_percentile": _safe_quantile(
                percentage_error,
                0.25,
            ),
            "percentage_error_75th_percentile": _safe_quantile(
                percentage_error,
                0.75,
            ),
            "percentage_error_95th_percentile": _safe_quantile(
                percentage_error,
                0.95,
            ),
            "percentage_error_maximum": _safe_quantile(
                percentage_error,
                1.00,
            ),
            "absolute_percentage_error_minimum": _safe_quantile(
                absolute_percentage_error,
                0.00,
            ),
            "absolute_percentage_error_5th_percentile": _safe_quantile(
                absolute_percentage_error,
                0.05,
            ),
            "absolute_percentage_error_25th_percentile": _safe_quantile(
                absolute_percentage_error,
                0.25,
            ),
            "absolute_percentage_error_75th_percentile": _safe_quantile(
                absolute_percentage_error,
                0.75,
            ),
            "absolute_percentage_error_95th_percentile": _safe_quantile(
                absolute_percentage_error,
                0.95,
            ),
            "absolute_percentage_error_maximum": _safe_quantile(
                absolute_percentage_error,
                1.00,
            ),
            "mcse_mean_signed_error": _safe_mcse(
                signed_error
            ),
            "mcse_mean_percentage_error": _safe_mcse(
                percentage_error
            ),
            "mcse_mean_absolute_percentage_error": _safe_mcse(
                absolute_percentage_error
            ),
        }

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    return (
        summary.sort_values(
            [
                "basis",
                "scenario_id",
                "model",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def build_applicability_summary(
    distributional_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create a focused model-success and applicability table."""

    columns = [
        "scenario_id",
        "tail_type",
        "inflation_scenario",
        "structural_break",
        "clause_type",
        "model",
        "basis",
        "attempts",
        "successful_fits",
        "failed_fits",
        "success_rate",
        "accuracy_scope",
        "conditional_accuracy_warning",
    ]

    return (
        distributional_summary[
            columns
        ]
        .copy()
        .sort_values(
            [
                "basis",
                "scenario_id",
                "model",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def _safe_filename(
    value: str,
) -> str:
    """Convert text into a safe filename component."""

    cleaned = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(value),
    )

    return cleaned.strip(
        "_"
    ).lower()


def _ordered_models(
    models: list[str],
) -> list[str]:
    """Place established model names first, followed by new models."""

    ordered = [
        model
        for model in PREFERRED_MODEL_ORDER
        if model in models
    ]

    remaining = sorted(
        set(models)
        - set(ordered)
    )

    return (
        ordered
        + remaining
    )


def save_distributional_boxplots(
    results: pd.DataFrame,
    output_directory: Path,
) -> pd.DataFrame:
    """Save boxplots of signed and absolute percentage errors."""

    validate_diagnostics_input(
        results
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    successful = results.loc[
        results[
            "success_or_failure"
        ].eq("success")
    ].copy()

    metric_definitions = [
        {
            "column": "percentage_error",
            "filename_label": "percentage_error",
            "title": "Signed percentage error",
            "ylabel": "Percentage error (%)",
        },
        {
            "column": "absolute_percentage_error",
            "filename_label": "absolute_percentage_error",
            "title": "Absolute percentage error",
            "ylabel": "Absolute percentage error (%)",
        },
    ]

    manifest_rows: list[
        dict[str, Any]
    ] = []

    scenario_basis_pairs = (
        results[
            [
                "scenario_id",
                "basis",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "scenario_id",
                "basis",
            ]
        )
    )

    for pair in scenario_basis_pairs.itertuples(
        index=False
    ):
        scenario_id = pair.scenario_id
        basis = pair.basis

        attempted_group = results.loc[
            (
                results[
                    "scenario_id"
                ].eq(
                    scenario_id
                )
            )
            & (
                results[
                    "basis"
                ].eq(
                    basis
                )
            )
        ]

        successful_group = successful.loc[
            (
                successful[
                    "scenario_id"
                ].eq(
                    scenario_id
                )
            )
            & (
                successful[
                    "basis"
                ].eq(
                    basis
                )
            )
        ]

        models = _ordered_models(
            attempted_group[
                "model"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        conditional_accuracy = False

        for model_name in models:
            model_attempts = attempted_group.loc[
                attempted_group[
                    "model"
                ].eq(
                    model_name
                )
            ]

            model_successes = successful_group.loc[
                successful_group[
                    "model"
                ].eq(
                    model_name
                )
            ]

            if (
                0
                < len(model_successes)
                < len(model_attempts)
            ):
                conditional_accuracy = True

        for metric in metric_definitions:
            plot_values: list[
                np.ndarray
            ] = []

            plot_labels: list[
                str
            ] = []

            plotted_models: list[
                str
            ] = []

            for model_name in models:
                model_rows = successful_group.loc[
                    successful_group[
                        "model"
                    ].eq(
                        model_name
                    )
                ]

                values = (
                    _finite_numeric_series(
                        model_rows,
                        metric["column"],
                    )
                )

                if values.empty:
                    continue

                plot_values.append(
                    values.to_numpy(
                        dtype=float
                    )
                )

                plot_labels.append(
                    MODEL_DISPLAY_NAMES.get(
                        model_name,
                        model_name.replace(
                            "_",
                            " ",
                        ).title(),
                    )
                )

                plotted_models.append(
                    model_name
                )

            if not plot_values:
                continue

            figure, axis = plt.subplots(
                figsize=(
                    max(
                        9,
                        2.2
                        * len(plot_values),
                    ),
                    6,
                )
            )

            try:
                axis.boxplot(
                    plot_values,
                    tick_labels=plot_labels,
                    showmeans=True,
                )
            except TypeError:
                # Compatibility with older Matplotlib versions.
                axis.boxplot(
                    plot_values,
                    labels=plot_labels,
                    showmeans=True,
                )

            title = (
                f"{metric['title']} — "
                f"{scenario_id} — {basis}"
            )

            if conditional_accuracy:
                title += (
                    "\nAccuracy is conditional on "
                    "successful fits where applicable"
                )

            axis.set_title(
                title
            )

            axis.set_xlabel(
                "Reserving model"
            )

            axis.set_ylabel(
                metric["ylabel"]
            )

            axis.tick_params(
                axis="x",
                rotation=25,
            )

            axis.grid(
                axis="y",
                alpha=0.3,
            )

            figure.tight_layout()

            filename = (
                f"{_safe_filename(scenario_id)}"
                f"__{_safe_filename(basis)}"
                f"__{metric['filename_label']}.png"
            )

            output_path = (
                output_directory
                / filename
            )

            figure.savefig(
                output_path,
                dpi=200,
                bbox_inches="tight",
            )

            plt.close(
                figure
            )

            manifest_rows.append(
                {
                    "scenario_id": scenario_id,
                    "basis": basis,
                    "metric": metric["column"],
                    "filename": filename,
                    "models_plotted": "|".join(
                        plotted_models
                    ),
                    "number_of_models_plotted": len(
                        plotted_models
                    ),
                    "conditional_accuracy_warning": (
                        conditional_accuracy
                    ),
                }
            )

    return pd.DataFrame(
        manifest_rows
    )


def build_diagnostics_acceptance_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    boxplot_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Check that the Step 18 outputs are complete and coherent."""

    expected_summary_rows = int(
        results[
            GROUP_COLUMNS
        ]
        .drop_duplicates()
        .shape[0]
    )

    successful_results = results.loc[
        results[
            "success_or_failure"
        ].eq("success")
    ]

    eligible_scenario_basis_groups = int(
        successful_results[
            [
                "scenario_id",
                "basis",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    expected_boxplots = (
        eligible_scenario_basis_groups
        * 2
    )

    success_counts_reconcile = bool(
        (
            summary[
                "successful_fits"
            ]
            + summary[
                "failed_fits"
            ]
            == summary[
                "attempts"
            ]
        ).all()
    )

    success_rates_valid = bool(
        summary[
            "success_rate"
        ]
        .between(
            0.0,
            1.0,
            inclusive="both",
        )
        .all()
    )

    summary_numeric = (
        summary.select_dtypes(
            include=[
                np.number,
            ]
        )
    )

    no_summary_infinities = bool(
        not np.isinf(
            summary_numeric.to_numpy(
                dtype=float
            )
        ).any()
    )

    no_success_rows = summary.loc[
        summary[
            "successful_fits"
        ].eq(0)
    ]

    no_success_metrics_are_missing = bool(
        no_success_rows[
            [
                "mean_signed_error",
                "mean_absolute_percentage_error",
                "root_mean_squared_error",
            ]
        ]
        .isna()
        .all()
        .all()
    )

    successful_summary_rows = summary.loc[
        summary[
            "successful_fits"
        ].gt(0)
    ]

    successful_metrics_are_finite = bool(
        np.isfinite(
            successful_summary_rows[
                [
                    "mean_signed_error",
                    "mean_absolute_percentage_error",
                    "root_mean_squared_error",
                ]
            ]
            .to_numpy(
                dtype=float
            )
        ).all()
    )

    classical_ceded = summary.loc[
        (
            summary[
                "basis"
            ].eq(
                "ceded"
            )
        )
        & (
            summary[
                "model"
            ].isin(
                CLASSICAL_CHAIN_LADDER_MODELS
            )
        )
        & (
            summary[
                "success_rate"
            ].lt(
                1.0
            )
        )
    ]

    classical_labels_valid = bool(
        classical_ceded.apply(
            lambda row: (
                (
                    row[
                        "successful_fits"
                    ]
                    == 0
                    and row[
                        "accuracy_scope"
                    ]
                    == (
                        "not_available_"
                        "no_successful_fits"
                    )
                )
                or
                (
                    0
                    < row[
                        "successful_fits"
                    ]
                    < row[
                        "attempts"
                    ]
                    and row[
                        "accuracy_scope"
                    ]
                    == (
                        "conditional_on_"
                        "successful_fits"
                    )
                )
            ),
            axis=1,
        ).all()
    )

    checks = [
        {
            "check": "expected_summary_row_count",
            "passed": (
                len(summary)
                == expected_summary_rows
            ),
            "detail": (
                f"actual={len(summary)}, "
                f"expected={expected_summary_rows}"
            ),
        },
        {
            "check": "success_counts_reconcile",
            "passed": (
                success_counts_reconcile
            ),
            "detail": "",
        },
        {
            "check": "success_rates_between_zero_and_one",
            "passed": (
                success_rates_valid
            ),
            "detail": "",
        },
        {
            "check": "no_summary_numeric_infinities",
            "passed": (
                no_summary_infinities
            ),
            "detail": "",
        },
        {
            "check": "no_success_metrics_are_missing",
            "passed": (
                no_success_metrics_are_missing
            ),
            "detail": "",
        },
        {
            "check": "successful_metrics_are_finite",
            "passed": (
                successful_metrics_are_finite
            ),
            "detail": "",
        },
        {
            "check": (
                "conditional_classical_ceded_"
                "results_are_labelled"
            ),
            "passed": (
                classical_labels_valid
            ),
            "detail": (
                f"rows_checked="
                f"{len(classical_ceded)}"
            ),
        },
        {
            "check": "expected_boxplot_count",
            "passed": (
                len(boxplot_manifest)
                == expected_boxplots
            ),
            "detail": (
                f"actual={len(boxplot_manifest)}, "
                f"expected={expected_boxplots}"
            ),
        },
    ]

    return pd.DataFrame(
        checks
    )
"""Step 27 prior-misspecification sensitivity utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import inspect
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from src.bornhuetter_ferguson import (
    BF_BREAK_AWARE,
    BF_STANDARD,
    build_paid_bf_by_accident_year,
)
from src.expected_loss_reserving import (
    build_expected_loss_by_accident_year,
)


PRIOR_MULTIPLIERS = (
    0.80,
    0.90,
    1.00,
    1.10,
    1.20,
)

PRIOR_ULTIMATE_COLUMNS = (
    "expected_gross_ultimate",
    "expected_ceded_ultimate",
    "expected_retained_ultimate",
)

SENSITIVITY_MODELS = (
    "expected_loss",
    BF_STANDARD,
    BF_BREAK_AWARE,
)

RESULT_KEY_COLUMNS = [
    "scenario_id",
    "simulation_id",
    "basis",
    "model",
    "prior_multiplier",
]

DETAIL_KEY_COLUMNS = (
    RESULT_KEY_COLUMNS
    + [
        "accident_year",
    ]
)

BASELINE_REPRODUCTION_RTOL = 1e-12
BASELINE_REPRODUCTION_ATOL = 1e-5


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without changing it."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_frozen_independent_prior(
    path: Path,
) -> pd.DataFrame:
    """Load and validate the frozen Step 19 independent prior."""

    prior = pd.read_csv(path)

    required_columns = {
        "scenario_id",
        "accident_year",
        "pricing_assumption_version",
        *PRIOR_ULTIMATE_COLUMNS,
    }

    missing = required_columns - set(prior.columns)

    if missing:
        raise ValueError(
            "Frozen prior is missing columns: "
            f"{sorted(missing)}"
        )

    if prior.duplicated(
        [
            "scenario_id",
            "accident_year",
        ]
    ).any():
        raise ValueError(
            "Frozen prior contains duplicate scenario/AY rows."
        )

    values = prior[
        list(PRIOR_ULTIMATE_COLUMNS)
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Frozen prior contains non-finite ultimates."
        )

    if (values < 0.0).any():
        raise ValueError(
            "Frozen prior contains negative ultimates."
        )

    return prior


def _canonical_multiplier(
    prior_multiplier: float,
) -> float:
    """Return the configured representation of one multiplier."""

    value = float(prior_multiplier)

    for candidate in PRIOR_MULTIPLIERS:
        if np.isclose(
            value,
            candidate,
            rtol=0.0,
            atol=1e-12,
        ):
            return float(candidate)

    raise ValueError(
        "prior_multiplier must belong to the frozen grid: "
        f"{PRIOR_MULTIPLIERS}"
    )


def apply_prior_multiplier(
    expected_loss_prior: pd.DataFrame,
    *,
    prior_multiplier: float,
) -> pd.DataFrame:
    """Return a scaled copy of the prior without mutating its source."""

    multiplier = _canonical_multiplier(
        prior_multiplier
    )

    missing = set(PRIOR_ULTIMATE_COLUMNS) - set(
        expected_loss_prior.columns
    )

    if missing:
        raise ValueError(
            "Prior is missing scalable ultimate columns: "
            f"{sorted(missing)}"
        )

    scaled = expected_loss_prior.copy(deep=True)

    for column in PRIOR_ULTIMATE_COLUMNS:
        values = pd.to_numeric(
            scaled[column],
            errors="raise",
        ).astype(float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Prior column {column} is not finite."
            )

        if (values < 0.0).any():
            raise ValueError(
                f"Prior column {column} is negative."
            )

        scaled[column] = values * multiplier

    scaled["prior_multiplier"] = multiplier

    return scaled


def select_frozen_evaluation_rows(
    evaluation_detail: pd.DataFrame,
    *,
    simulations_per_scenario: int,
    scenario_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Select deterministic Step 20 evaluation rows without resimulation."""

    required = {
        "scenario_id",
        "simulation_id",
        "accident_year",
        "basis",
        "tail_type",
        "structural_break",
        "expected_loss_prior_ultimate",
        "paid_to_date",
        "pricing_assumption_version",
        "true_reserve",
    }

    missing = required - set(evaluation_detail.columns)

    if missing:
        raise ValueError(
            "Frozen evaluation detail is missing columns: "
            f"{sorted(missing)}"
        )

    if simulations_per_scenario < 1:
        raise ValueError(
            "simulations_per_scenario must be positive."
        )

    available_scenarios = sorted(
        evaluation_detail["scenario_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if scenario_ids is None:
        selected_scenarios = available_scenarios
    else:
        selected_scenarios = sorted(
            {str(value) for value in scenario_ids}
        )

        unknown = set(selected_scenarios) - set(
            available_scenarios
        )

        if unknown:
            raise ValueError(
                "Unknown scenarios requested: "
                f"{sorted(unknown)}"
            )

    selected_parts: list[pd.DataFrame] = []

    for scenario_id in selected_scenarios:
        scenario_rows = evaluation_detail.loc[
            evaluation_detail["scenario_id"].eq(
                scenario_id
            )
        ]

        simulation_ids = sorted(
            scenario_rows["simulation_id"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )[:simulations_per_scenario]

        if len(simulation_ids) != simulations_per_scenario:
            raise ValueError(
                f"Scenario {scenario_id} has only "
                f"{len(simulation_ids)} simulations."
            )

        selected_parts.append(
            scenario_rows.loc[
                scenario_rows["simulation_id"].isin(
                    simulation_ids
                )
            ].copy()
        )

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    selected["simulation_id"] = selected[
        "simulation_id"
    ].astype(int)

    selected["accident_year"] = selected[
        "accident_year"
    ].astype(int)

    selected["structural_break"] = (
        selected["structural_break"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )

    source_keys = [
        "scenario_id",
        "simulation_id",
        "basis",
        "accident_year",
    ]

    if selected.duplicated(source_keys).any():
        raise ValueError(
            "Frozen evaluation detail contains duplicate keys."
        )

    bases = set(
        selected["basis"].astype(str).unique()
    )

    if bases != {"gross", "ceded"}:
        raise ValueError(
            "Frozen evaluation detail must contain gross and ceded rows."
        )

    return (
        selected.sort_values(source_keys)
        .reset_index(drop=True)
    )


def _basis_prior_column(basis: str) -> str:
    """Return the frozen prior column for one evaluated basis."""

    if basis == "gross":
        return "expected_gross_ultimate"

    if basis == "ceded":
        return "expected_ceded_ultimate"

    raise ValueError(
        f"Unsupported reserving basis: {basis}"
    )


def _build_estimator_input(
    *,
    baseline_prior: pd.DataFrame,
    scaled_prior: pd.DataFrame,
    evaluation_group: pd.DataFrame,
    scenario_id: str,
    simulation_id: int,
    basis: str,
) -> pd.DataFrame:
    """Build oracle-free estimator input from frozen observed fields."""

    prior_column = _basis_prior_column(basis)

    prior_columns = [
        "accident_year",
        prior_column,
        "pricing_assumption_version",
    ]

    baseline_lookup = baseline_prior.loc[
        baseline_prior["scenario_id"].eq(
            scenario_id
        ),
        prior_columns,
    ].rename(
        columns={
            prior_column: "baseline_prior_ultimate",
        }
    )

    scaled_lookup = scaled_prior.loc[
        scaled_prior["scenario_id"].eq(
            scenario_id
        ),
        [
            "accident_year",
            prior_column,
        ],
    ].rename(
        columns={
            prior_column: "expected_loss_prior_ultimate",
        }
    )

    metadata_columns = [
        "accident_year",
        "paid_to_date",
        "tail_type",
        "structural_break",
    ]

    for optional in [
        "inflation_scenario",
        "clause_type",
        "seed",
    ]:
        if optional in evaluation_group.columns:
            metadata_columns.append(optional)

    estimator_input = (
        evaluation_group[metadata_columns]
        .copy()
        .merge(
            baseline_lookup,
            on="accident_year",
            how="left",
            validate="one_to_one",
        )
        .merge(
            scaled_lookup,
            on="accident_year",
            how="left",
            validate="one_to_one",
        )
    )

    if estimator_input[
        [
            "baseline_prior_ultimate",
            "expected_loss_prior_ultimate",
        ]
    ].isna().any().any():
        raise ValueError(
            "Prior lookup did not cover every evaluation accident year."
        )

    estimator_input["scenario_id"] = scenario_id
    estimator_input["simulation_id"] = int(
        simulation_id
    )
    estimator_input["basis"] = basis

    return estimator_input


def _attach_evaluation_fields(
    *,
    estimate: pd.DataFrame,
    estimator_input: pd.DataFrame,
    evaluation_group: pd.DataFrame,
    prior_multiplier: float,
) -> pd.DataFrame:
    """Attach frozen truth only after a reserve has been estimated."""

    output = estimate.copy()

    estimator_metadata = estimator_input[
        [
            "accident_year",
            "baseline_prior_ultimate",
        ]
    ]

    if "baseline_prior_ultimate" not in output.columns:
        output = output.merge(
            estimator_metadata,
            on="accident_year",
            how="left",
            validate="one_to_one",
        )

    truth = evaluation_group[
        [
            "accident_year",
            "true_reserve",
        ]
    ].copy()

    output = output.merge(
        truth,
        on="accident_year",
        how="left",
        validate="one_to_one",
    )

    output["prior_multiplier"] = float(
        prior_multiplier
    )
    output["baseline_prior_version"] = output[
        "pricing_assumption_version"
    ]
    output["signed_error"] = (
        output["estimated_reserve"]
        - output["true_reserve"]
    )

    return output


def _successful_portfolio_row(
    *,
    detail: pd.DataFrame,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Aggregate one successful model attempt to portfolio level."""

    first = detail.iloc[0]
    estimated_reserve = float(
        detail["estimated_reserve"].sum()
    )
    true_reserve = float(
        detail["true_reserve"].sum()
    )
    signed_error = estimated_reserve - true_reserve

    if true_reserve > 0.0:
        percentage_error = (
            100.0 * signed_error / true_reserve
        )
        absolute_percentage_error = abs(
            percentage_error
        )
    else:
        percentage_error = np.nan
        absolute_percentage_error = np.nan

    return {
        "simulation_id": int(first["simulation_id"]),
        "scenario_id": str(first["scenario_id"]),
        "tail_type": str(first["tail_type"]),
        "inflation_scenario": first.get(
            "inflation_scenario",
            "unknown",
        ),
        "structural_break": bool(
            first["structural_break"]
        ),
        "clause_type": first.get(
            "clause_type",
            "none",
        ),
        "model": str(first["model"]),
        "basis": str(first["basis"]),
        "prior_multiplier": float(
            first["prior_multiplier"]
        ),
        "baseline_prior_version": str(
            first["baseline_prior_version"]
        ),
        "pricing_assumption_version": str(
            first["pricing_assumption_version"]
        ),
        "success": True,
        "success_or_failure": "success",
        "failure_type": "",
        "failure_message": "",
        "estimated_reserve": estimated_reserve,
        "true_reserve": true_reserve,
        "signed_error": signed_error,
        "absolute_error": abs(signed_error),
        "percentage_error": percentage_error,
        "absolute_percentage_error": (
            absolute_percentage_error
        ),
        "runtime_seconds": float(runtime_seconds),
        "seed": first.get("seed", np.nan),
    }


def _failed_portfolio_row(
    *,
    evaluation_group: pd.DataFrame,
    model: str,
    prior_multiplier: float,
    baseline_prior_version: str,
    error: Exception,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Represent one failed attempt without inventing an estimate."""

    first = evaluation_group.iloc[0]

    return {
        "simulation_id": int(first["simulation_id"]),
        "scenario_id": str(first["scenario_id"]),
        "tail_type": str(first["tail_type"]),
        "inflation_scenario": first.get(
            "inflation_scenario",
            "unknown",
        ),
        "structural_break": bool(
            first["structural_break"]
        ),
        "clause_type": first.get(
            "clause_type",
            "none",
        ),
        "model": model,
        "basis": str(first["basis"]),
        "prior_multiplier": float(prior_multiplier),
        "baseline_prior_version": (
            baseline_prior_version
        ),
        "pricing_assumption_version": (
            baseline_prior_version
        ),
        "success": False,
        "success_or_failure": "failure",
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "estimated_reserve": np.nan,
        "true_reserve": float(
            evaluation_group["true_reserve"].sum()
        ),
        "signed_error": np.nan,
        "absolute_error": np.nan,
        "percentage_error": np.nan,
        "absolute_percentage_error": np.nan,
        "runtime_seconds": float(runtime_seconds),
        "seed": first.get("seed", np.nan),
    }


def run_prior_misspecification_sensitivity(
    *,
    expected_loss_prior: pd.DataFrame,
    evaluation_detail: pd.DataFrame,
    prior_multipliers: Sequence[float],
    valuation_year: int,
    structural_break_year: int,
    benchmark_patterns: Mapping[
        str,
        Sequence[float],
    ],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run frozen Expected Loss and BF models on reused evaluation data."""

    multipliers = tuple(
        _canonical_multiplier(value)
        for value in prior_multipliers
    )

    if not multipliers:
        raise ValueError(
            "At least one prior multiplier is required."
        )

    if len(set(multipliers)) != len(multipliers):
        raise ValueError(
            "Prior multipliers must be unique."
        )

    versions = (
        expected_loss_prior[
            "pricing_assumption_version"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(versions) != 1:
        raise ValueError(
            "Frozen prior must contain exactly one version."
        )

    baseline_prior_version = str(versions[0])
    result_rows: list[dict[str, Any]] = []
    all_detail: list[pd.DataFrame] = []

    grouped = evaluation_detail.groupby(
        [
            "scenario_id",
            "simulation_id",
            "basis",
        ],
        sort=True,
    )

    for multiplier in multipliers:
        scaled_prior = apply_prior_multiplier(
            expected_loss_prior,
            prior_multiplier=multiplier,
        )

        for (
            scenario_id,
            simulation_id,
            basis,
        ), group in grouped:
            group = group.sort_values(
                "accident_year"
            ).reset_index(drop=True)

            estimator_input = _build_estimator_input(
                baseline_prior=expected_loss_prior,
                scaled_prior=scaled_prior,
                evaluation_group=group,
                scenario_id=str(scenario_id),
                simulation_id=int(simulation_id),
                basis=str(basis),
            )

            observed_paid = estimator_input[
                [
                    "accident_year",
                    "paid_to_date",
                ]
            ].copy()

            model_builders = {
                "expected_loss": lambda: (
                    build_expected_loss_by_accident_year(
                        expected_loss_prior=scaled_prior,
                        observed_paid=observed_paid,
                        scenario_id=str(scenario_id),
                        basis=str(basis),
                    ).merge(
                        estimator_input.drop(
                            columns=[
                                "paid_to_date",
                                "expected_loss_prior_ultimate",
                                "pricing_assumption_version",
                                "scenario_id",
                                "basis",
                            ]
                        ),
                        on="accident_year",
                        how="left",
                        validate="one_to_one",
                    )
                ),
                BF_STANDARD: lambda: (
                    build_paid_bf_by_accident_year(
                        estimator_input=estimator_input,
                        variant=BF_STANDARD,
                        valuation_year=valuation_year,
                        structural_break_year=(
                            structural_break_year
                        ),
                        benchmark_patterns=(
                            benchmark_patterns
                        ),
                    )
                ),
                BF_BREAK_AWARE: lambda: (
                    build_paid_bf_by_accident_year(
                        estimator_input=estimator_input,
                        variant=BF_BREAK_AWARE,
                        valuation_year=valuation_year,
                        structural_break_year=(
                            structural_break_year
                        ),
                        benchmark_patterns=(
                            benchmark_patterns
                        ),
                    )
                ),
            }

            for model in SENSITIVITY_MODELS:
                started = time.perf_counter()

                try:
                    estimate = model_builders[model]()
                    runtime_seconds = (
                        time.perf_counter() - started
                    )

                    detail = _attach_evaluation_fields(
                        estimate=estimate,
                        estimator_input=estimator_input,
                        evaluation_group=group,
                        prior_multiplier=multiplier,
                    )

                    all_detail.append(detail)
                    result_rows.append(
                        _successful_portfolio_row(
                            detail=detail,
                            runtime_seconds=runtime_seconds,
                        )
                    )
                except Exception as error:
                    runtime_seconds = (
                        time.perf_counter() - started
                    )
                    result_rows.append(
                        _failed_portfolio_row(
                            evaluation_group=group,
                            model=model,
                            prior_multiplier=multiplier,
                            baseline_prior_version=(
                                baseline_prior_version
                            ),
                            error=error,
                            runtime_seconds=runtime_seconds,
                        )
                    )

    results = pd.DataFrame(result_rows)

    if all_detail:
        detail_results = pd.concat(
            all_detail,
            ignore_index=True,
        )
        detail_results = detail_results.sort_values(
            DETAIL_KEY_COLUMNS
        ).reset_index(drop=True)
    else:
        detail_results = pd.DataFrame()

    return (
        results.sort_values(RESULT_KEY_COLUMNS)
        .reset_index(drop=True),
        detail_results,
    )


def build_sensitivity_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise success and conditional accuracy by design cell."""

    grouping = [
        "scenario_id",
        "basis",
        "model",
        "prior_multiplier",
    ]
    rows: list[dict[str, Any]] = []

    for values, group in results.groupby(
        grouping,
        sort=True,
        dropna=False,
    ):
        successful = group.loc[group["success"]].copy()

        rows.append(
            {
                **dict(zip(grouping, values)),
                "attempted_fits": int(len(group)),
                "successful_fits": int(len(successful)),
                "success_rate": float(group["success"].mean()),
                "mean_percentage_error": float(
                    successful["percentage_error"].mean()
                ),
                "median_percentage_error": float(
                    successful["percentage_error"].median()
                ),
                "mean_absolute_percentage_error": float(
                    successful[
                        "absolute_percentage_error"
                    ].mean()
                ),
                "RMSE": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                successful["signed_error"]
                            )
                        )
                    )
                ) if not successful.empty else np.nan,
                "percentage_error_std": float(
                    successful["percentage_error"].std(
                        ddof=1
                    )
                ),
                "mean_runtime_seconds": float(
                    group["runtime_seconds"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_prior_multiplier_comparison(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Add descriptive changes relative to multiplier 1.00."""

    baseline = summary.loc[
        np.isclose(
            summary["prior_multiplier"],
            1.0,
            rtol=0.0,
            atol=1e-12,
        ),
        [
            "scenario_id",
            "basis",
            "model",
            "mean_percentage_error",
            "mean_absolute_percentage_error",
            "RMSE",
        ],
    ].rename(
        columns={
            "mean_percentage_error": (
                "baseline_mean_percentage_error"
            ),
            "mean_absolute_percentage_error": (
                "baseline_mean_absolute_percentage_error"
            ),
            "RMSE": "baseline_RMSE",
        }
    )

    comparison = summary.merge(
        baseline,
        on=[
            "scenario_id",
            "basis",
            "model",
        ],
        how="left",
        validate="many_to_one",
    )

    comparison["difference_in_MPE_vs_1_00"] = (
        comparison["mean_percentage_error"]
        - comparison["baseline_mean_percentage_error"]
    )
    comparison["difference_in_MAPE_vs_1_00"] = (
        comparison["mean_absolute_percentage_error"]
        - comparison[
            "baseline_mean_absolute_percentage_error"
        ]
    )
    comparison["difference_in_RMSE_vs_1_00"] = (
        comparison["RMSE"]
        - comparison["baseline_RMSE"]
    )

    return comparison


def build_failure_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise failures while retaining an empty stable schema."""

    columns = [
        "scenario_id",
        "basis",
        "model",
        "prior_multiplier",
        "failure_type",
        "failure_message",
        "failures",
    ]
    failures = results.loc[~results["success"]]

    if failures.empty:
        return pd.DataFrame(columns=columns)

    return (
        failures.groupby(
            columns[:-1],
            dropna=False,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "failures"})
    )


def compare_multiplier_one_to_frozen_baselines(
    *,
    results: pd.DataFrame,
    frozen_step20_results: pd.DataFrame,
    frozen_step21_results: pd.DataFrame,
    rtol: float = BASELINE_REPRODUCTION_RTOL,
    atol: float = BASELINE_REPRODUCTION_ATOL,
) -> pd.DataFrame:
    """Compare multiplier 1.00 estimates with Steps 20 and 21."""

    key_columns = [
        "scenario_id",
        "simulation_id",
        "basis",
    ]
    rows: list[dict[str, Any]] = []

    for model in SENSITIVITY_MODELS:
        current = results.loc[
            results["model"].eq(model)
            & np.isclose(
                results["prior_multiplier"],
                1.0,
                rtol=0.0,
                atol=1e-12,
            ),
            key_columns
            + [
                "estimated_reserve",
                "true_reserve",
                "success",
            ],
        ].copy()

        source = (
            frozen_step20_results
            if model == "expected_loss"
            else frozen_step21_results
        )

        frozen = source.loc[
            source["model"].eq(model),
            key_columns
            + [
                "estimated_reserve",
                "true_reserve",
                "success_or_failure",
            ],
        ].copy()

        comparison = current.merge(
            frozen,
            on=key_columns,
            how="outer",
            suffixes=("_step27", "_frozen"),
            indicator=True,
            validate="one_to_one",
        )

        matched = comparison.loc[
            comparison["_merge"].eq("both")
        ].copy()

        differences = np.abs(
            matched["estimated_reserve_step27"]
            - matched["estimated_reserve_frozen"]
        )

        relative = differences / np.maximum(
            np.abs(
                matched["estimated_reserve_frozen"]
            ),
            atol,
        )

        matching = np.isclose(
            matched["estimated_reserve_step27"],
            matched["estimated_reserve_frozen"],
            rtol=rtol,
            atol=atol,
        )

        truth_matching = np.isclose(
            matched["true_reserve_step27"],
            matched["true_reserve_frozen"],
            rtol=rtol,
            atol=atol,
        )

        status_matching = (
            matched["success"].astype(bool)
            == matched["success_or_failure"].eq(
                "success"
            )
        )

        missing_step27 = int(
            comparison["_merge"].eq("right_only").sum()
        )
        missing_frozen = int(
            comparison["_merge"].eq("left_only").sum()
        )
        nonmatching = int((~matching).sum())

        rows.append(
            {
                "model": model,
                "frozen_source_step": (
                    "Step 20"
                    if model == "expected_loss"
                    else "Step 21"
                ),
                "compared_rows": int(len(matched)),
                "missing_step27_rows": missing_step27,
                "missing_frozen_rows": missing_frozen,
                "max_absolute_reserve_difference": (
                    float(differences.max())
                    if not differences.empty
                    else np.nan
                ),
                "max_relative_reserve_difference": (
                    float(relative.max())
                    if not relative.empty
                    else np.nan
                ),
                "nonmatching_reserve_rows": nonmatching,
                "nonmatching_truth_rows": int(
                    (~truth_matching).sum()
                ),
                "nonmatching_status_rows": int(
                    (~status_matching).sum()
                ),
                "rtol": float(rtol),
                "atol": float(atol),
                "passed": bool(
                    missing_step27 == 0
                    and missing_frozen == 0
                    and nonmatching == 0
                    and truth_matching.all()
                    and status_matching.all()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_acceptance_report(
    *,
    results: pd.DataFrame,
    detail_results: pd.DataFrame,
    evaluation_detail: pd.DataFrame,
    frozen_bf_detail: pd.DataFrame,
    baseline_reproduction: pd.DataFrame,
    expected_scenario_ids: Sequence[str],
    simulations_per_scenario: int,
    requested_multipliers: Sequence[float],
    prior_hash_before: str,
    prior_hash_after: str,
) -> pd.DataFrame:
    """Build explicit Step 27 structural and leakage checks."""

    multipliers = tuple(
        _canonical_multiplier(value)
        for value in requested_multipliers
    )
    full_run = set(multipliers) == set(PRIOR_MULTIPLIERS)
    expected_rows = (
        len(expected_scenario_ids)
        * simulations_per_scenario
        * 2
        * len(SENSITIVITY_MODELS)
        * len(multipliers)
    )

    present_multipliers = set(
        results["prior_multiplier"].astype(float).unique()
    )
    present_scenarios = set(
        results["scenario_id"].astype(str).unique()
    )
    present_bases = set(
        results["basis"].astype(str).unique()
    )
    present_models = set(
        results["model"].astype(str).unique()
    )

    successful_detail = detail_results.copy()
    multiplier_exact = bool(
        np.array_equal(
            successful_detail[
                "expected_loss_prior_ultimate"
            ].to_numpy(dtype=float),
            (
                successful_detail[
                    "baseline_prior_ultimate"
                ].to_numpy(dtype=float)
                * successful_detail[
                    "prior_multiplier"
                ].to_numpy(dtype=float)
            ),
        )
    )

    truth_invariant = bool(
        results.groupby(
            [
                "scenario_id",
                "simulation_id",
                "basis",
            ]
        )["true_reserve"]
        .nunique(dropna=False)
        .le(1)
        .all()
    )

    paid_invariant = bool(
        detail_results.groupby(
            [
                "scenario_id",
                "simulation_id",
                "basis",
                "accident_year",
            ]
        )["paid_to_date"]
        .nunique(dropna=False)
        .le(1)
        .all()
    )

    bf_detail = detail_results.loc[
        detail_results["model"].isin(
            [
                BF_STANDARD,
                BF_BREAK_AWARE,
            ]
        )
    ].copy()

    bf_proportions_invariant = bool(
        bf_detail.groupby(
            [
                "scenario_id",
                "simulation_id",
                "basis",
                "model",
                "accident_year",
            ]
        )[
            [
                "benchmark_paid_proportion",
                "benchmark_unpaid_proportion",
            ]
        ]
        .nunique(dropna=False)
        .le(1)
        .all()
        .all()
    )

    bf_one = bf_detail.loc[
        np.isclose(
            bf_detail["prior_multiplier"],
            1.0,
            rtol=0.0,
            atol=1e-12,
        )
    ]

    bf_assumption_keys = [
        "scenario_id",
        "simulation_id",
        "basis",
        "model",
        "accident_year",
    ]

    frozen_assumptions = frozen_bf_detail[
        bf_assumption_keys
        + [
            "benchmark_pattern",
            "benchmark_paid_proportion",
            "benchmark_unpaid_proportion",
        ]
    ].copy()

    frozen_assumption_comparison = bf_one.merge(
        frozen_assumptions,
        on=bf_assumption_keys,
        how="outer",
        suffixes=("_step27", "_frozen"),
        indicator=True,
        validate="one_to_one",
    )

    frozen_bf_assumptions_match = bool(
        frozen_assumption_comparison["_merge"]
        .eq("both")
        .all()
        and frozen_assumption_comparison[
            "benchmark_pattern_step27"
        ].eq(
            frozen_assumption_comparison[
                "benchmark_pattern_frozen"
            ]
        ).all()
        and np.allclose(
            frozen_assumption_comparison[
                "benchmark_paid_proportion_step27"
            ],
            frozen_assumption_comparison[
                "benchmark_paid_proportion_frozen"
            ],
            rtol=0.0,
            atol=1e-15,
        )
        and np.allclose(
            frozen_assumption_comparison[
                "benchmark_unpaid_proportion_step27"
            ],
            frozen_assumption_comparison[
                "benchmark_unpaid_proportion_frozen"
            ],
            rtol=0.0,
            atol=1e-15,
        )
    )

    success_rows = results.loc[results["success"]]
    estimates = success_rows["estimated_reserve"].to_numpy(
        dtype=float
    )
    estimates_valid = bool(
        np.isfinite(estimates).all()
        and (estimates >= 0.0).all()
    )

    statuses_complete = bool(
        len(results) == expected_rows
        and results["success"].notna().all()
        and results["success_or_failure"].isin(
            [
                "success",
                "failure",
            ]
        ).all()
        and results.loc[
            ~results["success"],
            [
                "failure_type",
                "failure_message",
            ],
        ].notna().all().all()
    )

    reconciled = (
        detail_results.groupby(
            RESULT_KEY_COLUMNS,
            as_index=False,
        )["estimated_reserve"]
        .sum()
        .merge(
            success_rows[
                RESULT_KEY_COLUMNS
                + [
                    "estimated_reserve",
                ]
            ],
            on=RESULT_KEY_COLUMNS,
            suffixes=("_detail", "_result"),
            validate="one_to_one",
        )
    )

    totals_reconcile = bool(
        np.allclose(
            reconciled["estimated_reserve_detail"],
            reconciled["estimated_reserve_result"],
            rtol=1e-12,
            atol=1e-5,
        )
    )

    evaluation_paid = evaluation_detail[
        [
            "scenario_id",
            "simulation_id",
            "basis",
            "accident_year",
            "paid_to_date",
        ]
    ].rename(
        columns={
            "paid_to_date": "frozen_paid_to_date",
        }
    )

    paid_source_comparison = detail_results.merge(
        evaluation_paid,
        on=[
            "scenario_id",
            "simulation_id",
            "basis",
            "accident_year",
        ],
        how="left",
        validate="many_to_one",
    )

    paid_matches_source = bool(
        np.allclose(
            paid_source_comparison["paid_to_date"],
            paid_source_comparison["frozen_paid_to_date"],
            rtol=0.0,
            atol=1e-12,
        )
    )

    scaling_parameters = set(
        inspect.signature(
            apply_prior_multiplier
        ).parameters
    )
    scaling_is_oracle_free = bool(
        not any(
            forbidden in parameter.lower()
            for parameter in scaling_parameters
            for forbidden in [
                "true",
                "future",
                "actual",
                "reserve_error",
            ]
        )
    )

    version_traceable = bool(
        results["baseline_prior_version"].notna().all()
        and results["baseline_prior_version"]
        .eq(results["pricing_assumption_version"])
        .all()
    )

    reproduction_lookup = baseline_reproduction.set_index(
        "model"
    )

    checks: list[dict[str, Any]] = []

    def add_check(
        check: str,
        passed: bool,
        detail: str = "",
        *,
        applicable: bool = True,
    ) -> None:
        checks.append(
            {
                "check": check,
                "applicable": applicable,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    add_check(
        "all_five_prior_multipliers_present",
        present_multipliers == set(PRIOR_MULTIPLIERS),
        f"present={sorted(present_multipliers)}",
        applicable=full_run,
    )
    add_check(
        "all_requested_prior_multipliers_present",
        present_multipliers == set(multipliers),
        f"requested={list(multipliers)}",
    )
    add_check(
        "all_expected_scenarios_present",
        present_scenarios == set(expected_scenario_ids),
        f"scenarios={len(present_scenarios)}",
    )
    add_check(
        "both_bases_present",
        present_bases == {"gross", "ceded"},
        f"bases={sorted(present_bases)}",
    )
    add_check(
        "all_three_models_present",
        present_models == set(SENSITIVITY_MODELS),
        f"models={sorted(present_models)}",
    )
    add_check(
        "correct_expected_result_row_count",
        len(results) == expected_rows,
        f"actual={len(results)}, expected={expected_rows}",
    )
    add_check(
        "full_sensitivity_row_count_is_13500",
        len(results) == 13_500,
        f"actual={len(results)}, expected=13500",
        applicable=full_run,
    )
    add_check(
        "result_keys_are_unique",
        not results.duplicated(RESULT_KEY_COLUMNS).any(),
    )
    add_check(
        "detail_keys_are_unique",
        not detail_results.duplicated(DETAIL_KEY_COLUMNS).any(),
    )
    add_check(
        "prior_multiplier_transformation_is_exact",
        multiplier_exact,
    )
    add_check(
        "original_prior_file_is_unchanged",
        prior_hash_before == prior_hash_after,
        f"sha256={prior_hash_after}",
    )

    for model, check_name in [
        (
            "expected_loss",
            "multiplier_1_reproduces_step20_expected_loss",
        ),
        (
            BF_STANDARD,
            "multiplier_1_reproduces_step21_standard_bf",
        ),
        (
            BF_BREAK_AWARE,
            "multiplier_1_reproduces_step21_break_aware_bf",
        ),
    ]:
        row = reproduction_lookup.loc[model]
        add_check(
            check_name,
            bool(row["passed"]),
            (
                "max_abs="
                f"{row['max_absolute_reserve_difference']:.12g}, "
                "max_rel="
                f"{row['max_relative_reserve_difference']:.12g}, "
                "nonmatching="
                f"{int(row['nonmatching_reserve_rows'])}"
            ),
        )

    add_check(
        "truth_is_identical_across_multipliers_and_models",
        truth_invariant,
    )
    add_check(
        "paid_to_date_is_identical_across_multipliers_and_models",
        paid_invariant,
    )
    add_check(
        "paid_to_date_matches_frozen_step20_source",
        paid_matches_source,
    )
    add_check(
        "bf_development_proportions_are_unchanged_across_multipliers",
        bf_proportions_invariant,
    )
    add_check(
        "bf_development_assumptions_match_frozen_step21",
        frozen_bf_assumptions_match,
    )
    add_check(
        "prior_scaling_api_has_no_evaluation_truth_input",
        scaling_is_oracle_free,
    )
    add_check(
        "successful_reserves_are_finite_and_nonnegative",
        estimates_valid,
    )
    add_check(
        "all_failures_are_represented",
        statuses_complete,
    )
    add_check(
        "accident_year_reserves_reconcile_to_portfolio",
        totals_reconcile,
    )
    add_check(
        "source_prior_version_is_traceable",
        version_traceable,
    )

    return pd.DataFrame(checks)

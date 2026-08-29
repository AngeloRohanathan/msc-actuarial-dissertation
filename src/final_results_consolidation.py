"""Traceable consolidation helpers for frozen dissertation results.

Step 32 is deliberately analysis-only: these functions read and harmonise
already-created result tables.  They do not import or call simulation,
calibration, reserving or model-fitting code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


NOT_APPLICABLE_BY_DESIGN = "not_applicable_by_design"
VALID_BASES = frozenset({"gross", "ceded"})
VALID_SCENARIOS = frozenset(
    {
        "short_stable_no_break",
        "short_emerging_no_break",
        "short_shock_no_break",
        "long_stable_no_break",
        "long_emerging_no_break",
        "long_shock_no_break",
        "long_stable_break",
        "long_emerging_break",
        "long_shock_break",
    }
)

CANONICAL_MODEL_MAPPING = {
    "chain_ladder": "chain_ladder",
    "inflation_adjusted_chain_ladder": (
        "inflation_adjusted_chain_ladder"
    ),
    "cashflow_uplift": "cashflow_uplift",
    "expected_loss": "expected_loss",
    "bornhuetter_ferguson_standard": (
        "bornhuetter_ferguson_standard"
    ),
    "bornhuetter_ferguson_break_aware": (
        "bornhuetter_ferguson_break_aware"
    ),
    "bornhuetter_ferguson_ceded_specific": (
        "bornhuetter_ferguson_ceded_specific"
    ),
    "bornhuetter_ferguson_break_aware_ceded_specific": (
        "bornhuetter_ferguson_break_aware_ceded_specific"
    ),
    "regularized_poisson": "regularized_poisson",
    "regularized_poisson_break_interaction": (
        "regularized_poisson_break_interaction"
    ),
    "regularized_tweedie": "regularized_tweedie",
    # Explicit historical aliases.  None of these aliases collapse distinct
    # model specifications; they only standardise spelling.
    "regularised_poisson": "regularized_poisson",
    "regularised_tweedie": "regularized_tweedie",
}
CANONICAL_MODELS = frozenset(CANONICAL_MODEL_MAPPING.values())

BASELINE_MODELS = (
    "chain_ladder",
    "inflation_adjusted_chain_ladder",
    "cashflow_uplift",
    "expected_loss",
    "bornhuetter_ferguson_standard",
    "bornhuetter_ferguson_break_aware",
    "regularized_poisson",
    "regularized_poisson_break_interaction",
    "regularized_tweedie",
)

SENSITIVITY_COLUMNS = (
    "history_window_years",
    "history_start_ay",
    "prior_multiplier",
    "pattern_variant",
    "treaty_variant",
    "calibration_version",
    "prior_version",
)

MASTER_RESULT_COLUMNS = (
    "analysis_type",
    "experiment_step",
    "experiment_name",
    "source_file",
    "scenario_id",
    "simulation_id",
    "seed",
    "model",
    "basis",
    "success",
    "applicable",
    "fit_attempted",
    "status",
    "failure_type",
    "failure_message",
    "applicability_reason",
    "estimated_reserve",
    "true_reserve",
    "reserve_error",
    "percentage_error",
    "absolute_percentage_error",
    "runtime_seconds",
    "tail_type",
    "inflation_scenario",
    "structural_break",
    "history_window_years",
    "history_start_ay",
    "prior_multiplier",
    "pattern_variant",
    "treaty_variant",
    "calibration_version",
    "prior_version",
    "selected_alpha",
    "selected_power",
    "selected_hyperparameter",
    "validation_metric",
    "observed_paid",
    "true_ultimate",
    "pipeline_reconciliation_passed",
    "model_reconciliation_passed",
)

MASTER_AY_COLUMNS = (
    "analysis_type",
    "experiment_step",
    "experiment_name",
    "source_file",
    "scenario_id",
    "simulation_id",
    "seed",
    "accident_year",
    "model",
    "basis",
    "success",
    "applicable",
    "status",
    "failure_type",
    "failure_message",
    "history_window_years",
    "prior_multiplier",
    "pattern_variant",
    "calibration_version",
    "prior_version",
    "paid_to_date",
    "prior_ultimate",
    "assumed_paid_proportion",
    "estimated_reserve",
    "true_reserve",
    "reserve_error",
    "estimated_ultimate",
    "pattern_source",
    "tail_type",
    "inflation_scenario",
    "structural_break",
)


class ConsolidationValidationError(ValueError):
    """Raised when frozen results cannot be harmonised unambiguously."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without changing it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_model_name(value: object) -> str:
    """Map one source model label to its documented canonical name."""

    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in CANONICAL_MODEL_MAPPING:
        raise ConsolidationValidationError(
            f"Unknown model label cannot be canonicalised: {value!r}."
        )
    return CANONICAL_MODEL_MAPPING[key]


def validate_unique_keys(
    frame: pd.DataFrame,
    key_columns: Sequence[str],
    *,
    source_name: str,
) -> None:
    """Require a unique declared key, treating nulls as key values."""

    missing = [column for column in key_columns if column not in frame]
    if missing:
        raise ConsolidationValidationError(
            f"{source_name} is missing key columns: {missing}."
        )
    duplicate = frame.duplicated(list(key_columns), keep=False)
    if duplicate.any():
        example = frame.loc[duplicate, list(key_columns)].iloc[0].to_dict()
        raise ConsolidationValidationError(
            f"{source_name} contains duplicate keys; example={example}."
        )


def _column_or_na(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame:
        return frame[name].copy()
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(_column_or_na(frame, name), errors="coerce")


def _normalise_boolean(series: pd.Series) -> pd.Series:
    """Convert the project's CSV boolean conventions to nullable booleans."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    normalised = series.astype("string").str.strip().str.lower()
    mapped = normalised.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
        }
    )
    return mapped.astype("boolean")


def _success_series(frame: pd.DataFrame) -> pd.Series:
    if "success" in frame:
        return _normalise_boolean(frame["success"]).fillna(False)
    if "success_or_failure" in frame:
        return (
            frame["success_or_failure"]
            .astype("string")
            .str.strip()
            .str.lower()
            .eq("success")
            .astype("boolean")
        )
    raise ConsolidationValidationError("Source has no success-status field.")


def _applicable_series(frame: pd.DataFrame) -> pd.Series:
    if "applicability_status" not in frame:
        return pd.Series(True, index=frame.index, dtype="boolean")
    return (
        ~frame["applicability_status"]
        .astype("string")
        .eq(NOT_APPLICABLE_BY_DESIGN)
    ).astype("boolean")


def _fit_attempted_series(
    frame: pd.DataFrame,
    applicable: pd.Series,
) -> pd.Series:
    if "fit_attempted" not in frame:
        return applicable.copy()
    return _normalise_boolean(frame["fit_attempted"]).fillna(False)


def _first_nonempty(
    frame: pd.DataFrame,
    candidates: Iterable[str],
) -> pd.Series:
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    for candidate in candidates:
        if candidate not in frame:
            continue
        values = frame[candidate].astype("string").str.strip()
        values = values.mask(values.eq(""), pd.NA)
        result = result.fillna(values)
    return result


def harmonise_portfolio_results(
    source: pd.DataFrame,
    *,
    experiment_step: int,
    experiment_name: str,
    source_file: str,
    analysis_type: str,
    include_models: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Harmonise one frozen portfolio result table without imputation."""

    required = {"scenario_id", "simulation_id", "model", "basis"}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ConsolidationValidationError(
            f"{source_file} is missing portfolio fields: {missing}."
        )

    frame = source.copy()
    frame["model"] = frame["model"].map(canonical_model_name)
    if include_models is not None:
        selected = {canonical_model_name(model) for model in include_models}
        frame = frame.loc[frame["model"].isin(selected)].copy()

    success = _success_series(frame)
    applicable = _applicable_series(frame)
    fit_attempted = _fit_attempted_series(frame, applicable)
    status = pd.Series("failed", index=frame.index, dtype="string")
    status.loc[~applicable] = NOT_APPLICABLE_BY_DESIGN
    status.loc[success] = "success"

    out = pd.DataFrame(index=frame.index)
    out["analysis_type"] = analysis_type
    out["experiment_step"] = int(experiment_step)
    out["experiment_name"] = experiment_name
    out["source_file"] = source_file
    for column in ["scenario_id", "simulation_id", "seed", "model", "basis"]:
        out[column] = _column_or_na(frame, column)
    out["success"] = success
    out["applicable"] = applicable
    out["fit_attempted"] = fit_attempted
    out["status"] = status
    out["failure_type"] = _first_nonempty(
        frame,
        ["failure_type", "failure_category"],
    )
    out["failure_message"] = _first_nonempty(
        frame,
        ["failure_message", "failure_reason"],
    )
    out["applicability_reason"] = _first_nonempty(
        frame,
        ["applicability_reason"],
    )
    out["estimated_reserve"] = _numeric_column(frame, "estimated_reserve")
    out["true_reserve"] = _numeric_column(frame, "true_reserve")
    out["reserve_error"] = _numeric_column(frame, "signed_error")
    out["percentage_error"] = _numeric_column(frame, "percentage_error")
    out["absolute_percentage_error"] = _numeric_column(
        frame,
        "absolute_percentage_error",
    )
    out["runtime_seconds"] = _numeric_column(frame, "runtime_seconds")
    for column in ["tail_type", "inflation_scenario"]:
        out[column] = _column_or_na(frame, column)
    out["structural_break"] = _normalise_boolean(
        _column_or_na(frame, "structural_break")
    )

    out["history_window_years"] = (
        _numeric_column(frame, "history_window_years")
        if analysis_type == "history_sensitivity"
        else np.nan
    )
    out["history_start_ay"] = (
        _numeric_column(frame, "history_start_ay")
        if analysis_type == "history_sensitivity"
        else np.nan
    )
    out["prior_multiplier"] = (
        _numeric_column(frame, "prior_multiplier")
        if analysis_type == "prior_sensitivity"
        else np.nan
    )
    out["pattern_variant"] = (
        _column_or_na(frame, "pattern_variant")
        if analysis_type == "development_sensitivity"
        else pd.NA
    )
    out["treaty_variant"] = pd.NA
    out["calibration_version"] = (
        _column_or_na(frame, "calibration_version")
        if analysis_type == "development_sensitivity"
        else pd.NA
    )
    out["prior_version"] = _first_nonempty(
        frame,
        [
            "prior_version",
            "baseline_prior_version",
            "pricing_assumption_version",
        ],
    )

    ml_model = out["model"].isin(
        {
            "regularized_poisson",
            "regularized_poisson_break_interaction",
            "regularized_tweedie",
        }
    )
    out["selected_alpha"] = _numeric_column(frame, "selected_alpha").where(
        ml_model
    )
    out["selected_power"] = _numeric_column(frame, "selected_power").where(
        out["model"].eq("regularized_tweedie")
    )
    out["selected_hyperparameter"] = _column_or_na(
        frame,
        "selected_hyperparameter",
    )
    out["validation_metric"] = _column_or_na(frame, "validation_metric")
    out["observed_paid"] = _numeric_column(frame, "observed_paid")
    out["true_ultimate"] = _numeric_column(frame, "true_ultimate")
    out["pipeline_reconciliation_passed"] = _normalise_boolean(
        _column_or_na(frame, "pipeline_reconciliation_passed")
    )
    out["model_reconciliation_passed"] = _normalise_boolean(
        _column_or_na(frame, "model_reconciliation_passed")
    )
    return out.loc[:, MASTER_RESULT_COLUMNS].reset_index(drop=True)


def harmonise_accident_year_results(
    source: pd.DataFrame,
    *,
    experiment_step: int,
    experiment_name: str,
    source_file: str,
    analysis_type: str,
) -> pd.DataFrame:
    """Harmonise compatible AY detail, leaving status for source-backed join."""

    required = {
        "scenario_id",
        "simulation_id",
        "accident_year",
        "model",
        "basis",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ConsolidationValidationError(
            f"{source_file} is missing AY fields: {missing}."
        )

    frame = source.copy()
    out = pd.DataFrame(index=frame.index)
    out["analysis_type"] = analysis_type
    out["experiment_step"] = int(experiment_step)
    out["experiment_name"] = experiment_name
    out["source_file"] = source_file
    for column in [
        "scenario_id",
        "simulation_id",
        "seed",
        "accident_year",
        "basis",
    ]:
        out[column] = _column_or_na(frame, column)
    out["model"] = frame["model"].map(canonical_model_name)
    out["success"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    out["applicable"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    out["status"] = pd.NA
    out["failure_type"] = pd.NA
    out["failure_message"] = pd.NA
    out["history_window_years"] = (
        _numeric_column(frame, "history_window_years")
        if analysis_type == "history_sensitivity"
        else np.nan
    )
    out["prior_multiplier"] = (
        _numeric_column(frame, "prior_multiplier")
        if analysis_type == "prior_sensitivity"
        else np.nan
    )
    out["pattern_variant"] = (
        _column_or_na(frame, "pattern_variant")
        if analysis_type == "development_sensitivity"
        else pd.NA
    )
    out["calibration_version"] = _column_or_na(
        frame,
        "calibration_version",
    )
    out["prior_version"] = _first_nonempty(
        frame,
        [
            "prior_version",
            "baseline_prior_version",
            "pricing_assumption_version",
        ],
    )
    out["paid_to_date"] = _numeric_column(frame, "paid_to_date")
    out["prior_ultimate"] = _numeric_column(
        frame,
        "expected_loss_prior_ultimate",
    )
    out["assumed_paid_proportion"] = _numeric_column(
        frame,
        "benchmark_paid_proportion",
    )
    out["estimated_reserve"] = _numeric_column(frame, "estimated_reserve")
    out["true_reserve"] = _numeric_column(frame, "true_reserve")
    out["reserve_error"] = _numeric_column(frame, "signed_error")
    out["estimated_ultimate"] = _numeric_column(frame, "estimated_ultimate")
    out["pattern_source"] = _column_or_na(frame, "pattern_source")
    out["tail_type"] = _column_or_na(frame, "tail_type")
    out["inflation_scenario"] = _column_or_na(
        frame,
        "inflation_scenario",
    )
    out["structural_break"] = _normalise_boolean(
        _column_or_na(frame, "structural_break")
    )
    return out.loc[:, MASTER_AY_COLUMNS].reset_index(drop=True)


def attach_portfolio_status_to_accident_year(
    accident_year: pd.DataFrame,
    portfolio: pd.DataFrame,
    *,
    sensitivity_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Attach source-backed portfolio status to compatible AY detail rows."""

    keys = ["scenario_id", "simulation_id", "basis", "model"] + list(
        sensitivity_columns
    )
    status_columns = [
        "success",
        "applicable",
        "status",
        "failure_type",
        "failure_message",
    ]
    lookup = portfolio[keys + status_columns].copy()
    validate_unique_keys(lookup, keys, source_name="portfolio status lookup")
    detail = accident_year.drop(columns=status_columns)
    joined = detail.merge(lookup, on=keys, how="left", validate="many_to_one")
    if joined["status"].isna().any():
        raise ConsolidationValidationError(
            "AY detail could not be matched to portfolio status."
        )
    return joined.loc[:, MASTER_AY_COLUMNS]


def classify_ci_direction(lower: object, upper: object) -> str:
    """Classify a frozen paired bootstrap interval deterministically."""

    try:
        lower_value = float(lower)
        upper_value = float(upper)
    except (TypeError, ValueError):
        return "not_available"
    if not np.isfinite(lower_value) or not np.isfinite(upper_value):
        return "not_available"
    if upper_value < 0.0:
        return "favors_A"
    if lower_value > 0.0:
        return "favors_B"
    return "includes_zero"


def harmonise_paired_comparisons(
    source: pd.DataFrame,
    *,
    source_file: str,
) -> pd.DataFrame:
    """Normalise one frozen Step 31 summary without rerunning inference."""

    required = {
        "comparison_id",
        "source_step",
        "scenario_id",
        "basis",
        "model_a",
        "model_b",
        "required_paired_rows",
        "valid_paired_rows",
        "excluded_pairs",
        "mean_ape_a",
        "mean_ape_b",
        "mean_paired_ape_difference",
        "median_paired_ape_difference",
        "std_paired_ape_difference",
        "win_rate_a",
        "win_rate_b",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ConsolidationValidationError(
            f"{source_file} is missing paired fields: {missing}."
        )

    frame = source.copy()
    actual_model = _column_or_na(frame, "model")
    has_actual_model = actual_model.notna() & actual_model.astype(str).ne("")
    model_a = frame["model_a"].astype("string")
    model_b = frame["model_b"].astype("string")
    if has_actual_model.any():
        canonical_actual = actual_model.loc[has_actual_model].map(
            canonical_model_name
        )
        model_a.loc[has_actual_model] = canonical_actual
        model_b.loc[has_actual_model] = canonical_actual
    core_mask = ~has_actual_model
    model_a.loc[core_mask] = model_a.loc[core_mask].map(canonical_model_name)
    model_b.loc[core_mask] = model_b.loc[core_mask].map(canonical_model_name)

    sensitivity_a = pd.Series(pd.NA, index=frame.index, dtype="string")
    sensitivity_b = pd.Series(pd.NA, index=frame.index, dtype="string")
    if "shorter_window_years" in frame:
        mask = frame["shorter_window_years"].notna()
        sensitivity_a.loc[mask] = (
            frame.loc[mask, "shorter_window_years"].astype("Int64").astype(str)
            + "_year_history"
        )
        sensitivity_b.loc[mask] = (
            frame.loc[mask, "longer_window_years"].astype("Int64").astype(str)
            + "_year_history"
        )
    if "prior_multiplier_a" in frame:
        mask = frame["prior_multiplier_a"].notna()
        sensitivity_a.loc[mask] = frame.loc[
            mask, "prior_multiplier_a"
        ].map(lambda value: f"prior_multiplier_{float(value):.2f}")
        sensitivity_b.loc[mask] = frame.loc[
            mask, "prior_multiplier_b"
        ].map(lambda value: f"prior_multiplier_{float(value):.2f}")

    valid = pd.to_numeric(frame["valid_paired_rows"], errors="coerce")
    ties = pd.to_numeric(_column_or_na(frame, "tie_count"), errors="coerce")
    tie_rate = ties.div(valid.where(valid.ne(0)))
    out = pd.DataFrame(
        {
            "analysis_type": "paired_comparison",
            "comparison_id": frame["comparison_id"],
            "experiment_step": pd.to_numeric(
                frame["source_step"], errors="coerce"
            ),
            "source_file": source_file,
            "analysis_family": _column_or_na(frame, "analysis_family"),
            "scenario_id": frame["scenario_id"],
            "basis": frame["basis"],
            "model_a": model_a,
            "model_b": model_b,
            "sensitivity_a": sensitivity_a,
            "sensitivity_b": sensitivity_b,
            "required_pairs": pd.to_numeric(
                frame["required_paired_rows"], errors="coerce"
            ),
            "valid_pairs": valid,
            "excluded_pairs": pd.to_numeric(
                frame["excluded_pairs"], errors="coerce"
            ),
            "mean_ape_a": pd.to_numeric(frame["mean_ape_a"], errors="coerce"),
            "mean_ape_b": pd.to_numeric(frame["mean_ape_b"], errors="coerce"),
            "mean_paired_difference": pd.to_numeric(
                frame["mean_paired_ape_difference"], errors="coerce"
            ),
            "median_paired_difference": pd.to_numeric(
                frame["median_paired_ape_difference"], errors="coerce"
            ),
            "std_paired_difference": pd.to_numeric(
                frame["std_paired_ape_difference"], errors="coerce"
            ),
            "win_rate_a": pd.to_numeric(frame["win_rate_a"], errors="coerce"),
            "win_rate_b": pd.to_numeric(frame["win_rate_b"], errors="coerce"),
            "tie_rate": tie_rate,
            "bootstrap_ci_lower": pd.to_numeric(
                frame["bootstrap_ci_lower"], errors="coerce"
            ),
            "bootstrap_ci_upper": pd.to_numeric(
                frame["bootstrap_ci_upper"], errors="coerce"
            ),
            "inference_direction": [
                classify_ci_direction(lower, upper)
                for lower, upper in zip(
                    frame["bootstrap_ci_lower"],
                    frame["bootstrap_ci_upper"],
                )
            ],
            "group_type": _column_or_na(frame, "group_type"),
            "pool_label": _column_or_na(frame, "pool_label"),
        }
    )
    return out.reset_index(drop=True)


def harmonise_treaty_sensitivity(
    portfolio: pd.DataFrame,
    accident_year: pd.DataFrame,
    *,
    source_file: str,
) -> pd.DataFrame:
    """Build the Step 29 treaty-mechanics companion without estimator APE."""

    required = {
        "scenario_id",
        "simulation_id",
        "treaty_variant",
        "gross_ultimate",
        "ceded_ultimate",
        "retained_ultimate",
        "ceded_share",
        "attachment_frequency",
        "exhaustion_frequency",
        "gross_true_reserve",
        "ceded_true_reserve",
        "retained_true_reserve",
    }
    missing = sorted(required.difference(portfolio.columns))
    if missing:
        raise ConsolidationValidationError(
            f"{source_file} is missing treaty fields: {missing}."
        )

    keys = ["scenario_id", "simulation_id", "treaty_variant"]
    factor_required = set(keys + ["applied_index_factor"])
    if not factor_required.issubset(accident_year.columns):
        raise ConsolidationValidationError(
            "Step 29 AY source lacks applied index-factor fields."
        )
    factors = (
        accident_year.groupby(keys, dropna=False)["applied_index_factor"]
        .agg(
            minimum_applied_index_factor="min",
            maximum_applied_index_factor="max",
            mean_applied_index_factor="mean",
        )
        .reset_index()
    )
    out = portfolio.copy().merge(
        factors,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    out.insert(0, "analysis_type", "treaty_sensitivity")
    out.insert(1, "experiment_step", 29)
    out.insert(2, "experiment_name", "treaty_indexation_sensitivity")
    out.insert(3, "source_file", source_file)
    out["success"] = _success_series(out)
    out["status"] = np.where(out["success"], "success", "failed")
    columns = [
        "analysis_type",
        "experiment_step",
        "experiment_name",
        "source_file",
        "scenario_id",
        "simulation_id",
        "seed",
        "treaty_variant",
        "tail_type",
        "inflation_scenario",
        "structural_break",
        "gross_ultimate",
        "ceded_ultimate",
        "retained_ultimate",
        "ceded_share",
        "attachment_frequency",
        "exhaustion_frequency",
        "gross_true_reserve",
        "ceded_true_reserve",
        "retained_true_reserve",
        "base_attachment",
        "base_limit",
        "index_reference_year",
        "minimum_applied_index_factor",
        "maximum_applied_index_factor",
        "mean_applied_index_factor",
        "success",
        "status",
        "failure_message",
        "runtime_seconds",
    ]
    return out.loc[:, columns].reset_index(drop=True)


def build_applicability_summary(master: pd.DataFrame) -> pd.DataFrame:
    """Separate design applicability, attempted fits and fit failures."""

    group_columns = [
        "experiment_step",
        "experiment_name",
        "analysis_type",
        "model",
        "basis",
        "scenario_id",
        "history_window_years",
        "history_start_ay",
        "prior_multiplier",
        "pattern_variant",
        "calibration_version",
        "prior_version",
    ]
    rows: list[dict[str, object]] = []
    for key, group in master.groupby(group_columns, dropna=False, sort=True):
        applicable = group["applicable"].fillna(False).astype(bool)
        attempted = group["fit_attempted"].fillna(False).astype(bool)
        successful = group["success"].fillna(False).astype(bool)
        scheduled_count = int(len(group))
        applicable_count = int(applicable.sum())
        attempted_count = int(attempted.sum())
        successful_count = int(successful.sum())
        failed_count = int((applicable & attempted & ~successful).sum())
        row = dict(zip(group_columns, key))
        row.update(
            {
                "scheduled_count": scheduled_count,
                "attempted_count": attempted_count,
                "applicable_count": applicable_count,
                "structurally_not_applicable_count": int(
                    (~applicable).sum()
                ),
                "successful_count": successful_count,
                "failed_count": failed_count,
                "applicability_rate": (
                    applicable_count / scheduled_count
                    if scheduled_count
                    else np.nan
                ),
                "conditional_success_rate": (
                    successful_count / attempted_count
                    if attempted_count
                    else np.nan
                ),
                "unconditional_success_rate": (
                    successful_count / scheduled_count
                    if scheduled_count
                    else np.nan
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def reserve_metric_mismatch_counts(master: pd.DataFrame) -> Mapping[str, int]:
    """Count successful-row reconciliation problems without filling nulls."""

    successful = master.loc[master["success"].fillna(False)].copy()
    finite = np.isfinite(successful["estimated_reserve"]) & np.isfinite(
        successful["true_reserve"]
    )
    expected_error = successful["estimated_reserve"] - successful["true_reserve"]
    error_match = np.isclose(
        successful["reserve_error"],
        expected_error,
        rtol=1e-12,
        atol=1e-6,
        equal_nan=False,
    )
    nonzero_truth = successful["true_reserve"].ne(0.0)
    expected_pe = 100.0 * expected_error / successful["true_reserve"]
    pe_match = np.isclose(
        successful.loc[nonzero_truth, "percentage_error"],
        expected_pe.loc[nonzero_truth],
        rtol=1e-12,
        atol=1e-9,
        equal_nan=False,
    )
    ape_match = np.isclose(
        successful["absolute_percentage_error"],
        successful["percentage_error"].abs(),
        rtol=1e-12,
        atol=1e-9,
        equal_nan=False,
    )
    return {
        "successful_nonfinite_estimate_or_truth": int((~finite).sum()),
        "reserve_error_mismatches": int((~error_match).sum()),
        "percentage_error_mismatches": int((~pe_match).sum()),
        "absolute_percentage_error_mismatches": int((~ape_match).sum()),
        "successful_negative_estimates": int(
            successful["estimated_reserve"].lt(0.0).sum()
        ),
    }


def validate_sensitivity_scope(master: pd.DataFrame) -> Mapping[str, int]:
    """Count sensitivity values populated outside their intended analysis."""

    checks = {
        "history_fields_outside_history": int(
            (
                master["analysis_type"].ne("history_sensitivity")
                & (
                    master["history_window_years"].notna()
                    | master["history_start_ay"].notna()
                )
            ).sum()
        ),
        "prior_multiplier_outside_prior": int(
            (
                master["analysis_type"].ne("prior_sensitivity")
                & master["prior_multiplier"].notna()
            ).sum()
        ),
        "pattern_variant_outside_development": int(
            (
                master["analysis_type"].ne("development_sensitivity")
                & master["pattern_variant"].notna()
            ).sum()
        ),
        "treaty_variant_in_estimator_master": int(
            master["treaty_variant"].notna().sum()
        ),
        "selected_power_for_non_tweedie": int(
            (
                master["model"].ne("regularized_tweedie")
                & master["selected_power"].notna()
            ).sum()
        ),
    }
    return checks


def truth_mismatch_group_count(master: pd.DataFrame) -> int:
    """Count comparable groups containing more than one finite truth value."""

    group_columns = [
        "analysis_type",
        "scenario_id",
        "simulation_id",
        "basis",
    ]
    mismatch = 0
    for _, group in master.groupby(group_columns, dropna=False):
        truth = pd.to_numeric(group["true_reserve"], errors="coerce").dropna()
        if truth.empty:
            continue
        reference = float(truth.iloc[0])
        if not np.allclose(
            truth.to_numpy(dtype=float),
            reference,
            rtol=1e-12,
            atol=1e-6,
        ):
            mismatch += 1
    return mismatch


def ensure_data_dictionary_complete(
    dictionary: pd.DataFrame,
    datasets: Mapping[str, pd.DataFrame],
) -> None:
    """Require one dictionary entry for every emitted CSV field."""

    declared = set(zip(dictionary["dataset"], dictionary["field_name"]))
    missing = sorted(
        (dataset, column)
        for dataset, frame in datasets.items()
        for column in frame.columns
        if (dataset, column) not in declared
    )
    if missing:
        raise ConsolidationValidationError(
            f"Data dictionary is missing output fields: {missing[:5]}."
        )

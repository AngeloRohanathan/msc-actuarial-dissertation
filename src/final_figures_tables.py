"""Deterministic presentation helpers for Step 33.

This module summarises only the frozen Step 32 consolidated datasets.  It
contains no simulation, reserving, calibration, bootstrap or model-fitting
code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


FINAL_ANALYSIS_DIRECTORY = Path("outputs/final_analysis")
STEP32_SOURCE_FILES = {
    "baseline": FINAL_ANALYSIS_DIRECTORY / "baseline_model_results.csv",
    "master": FINAL_ANALYSIS_DIRECTORY / "master_results.csv",
    "accident_year": (
        FINAL_ANALYSIS_DIRECTORY / "master_by_accident_year.csv"
    ),
    "paired": FINAL_ANALYSIS_DIRECTORY / "master_paired_comparisons.csv",
    "applicability": FINAL_ANALYSIS_DIRECTORY / "master_applicability.csv",
    "treaty": FINAL_ANALYSIS_DIRECTORY / "master_treaty_sensitivity.csv",
    "model_dictionary": FINAL_ANALYSIS_DIRECTORY / "model_dictionary.csv",
    "data_dictionary": FINAL_ANALYSIS_DIRECTORY / "data_dictionary.csv",
    "source_inventory": FINAL_ANALYSIS_DIRECTORY / "source_inventory.csv",
    "validation": FINAL_ANALYSIS_DIRECTORY / "validation_report.csv",
    "method_note": (
        FINAL_ANALYSIS_DIRECTORY / "FINAL_ANALYSIS_METHOD_NOTE.md"
    ),
}

SCENARIO_ORDER = (
    "short_stable_no_break",
    "short_emerging_no_break",
    "short_shock_no_break",
    "long_stable_no_break",
    "long_emerging_no_break",
    "long_shock_no_break",
    "long_stable_break",
    "long_emerging_break",
    "long_shock_break",
)
SCENARIO_LABELS = {
    "short_stable_no_break": "Short / Stable",
    "short_emerging_no_break": "Short / Emerging",
    "short_shock_no_break": "Short / Shock",
    "long_stable_no_break": "Long / Stable",
    "long_emerging_no_break": "Long / Emerging",
    "long_shock_no_break": "Long / Shock",
    "long_stable_break": "Long / Stable + Break",
    "long_emerging_break": "Long / Emerging + Break",
    "long_shock_break": "Long / Shock + Break",
}
BREAK_SCENARIOS = (
    "long_stable_break",
    "long_emerging_break",
    "long_shock_break",
)

MODEL_ORDER = (
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
MODEL_LABELS = {
    "chain_ladder": "Chain Ladder",
    "inflation_adjusted_chain_ladder": "Inflation-Adjusted CL",
    "cashflow_uplift": "Cashflow Uplift",
    "expected_loss": "Expected Loss",
    "bornhuetter_ferguson_standard": "Standard BF",
    "bornhuetter_ferguson_break_aware": "Break-Aware BF",
    "bornhuetter_ferguson_ceded_specific": "Ceded-Specific BF",
    "bornhuetter_ferguson_break_aware_ceded_specific": (
        "BA Ceded-Specific BF"
    ),
    "regularized_poisson": "Regularized Poisson",
    "regularized_poisson_break_interaction": (
        "Poisson + Break Interaction"
    ),
    "regularized_tweedie": "Tweedie",
}

HISTORY_MODEL_ORDER = (
    "chain_ladder",
    "inflation_adjusted_chain_ladder",
    "cashflow_uplift",
    "regularized_poisson",
    "regularized_poisson_break_interaction",
    "regularized_tweedie",
)
ML_MODEL_ORDER = (
    "regularized_poisson",
    "regularized_poisson_break_interaction",
    "regularized_tweedie",
)
BF_SENSITIVITY_MODEL_ORDER = (
    "bornhuetter_ferguson_standard",
    "bornhuetter_ferguson_ceded_specific",
    "bornhuetter_ferguson_break_aware",
    "bornhuetter_ferguson_break_aware_ceded_specific",
)
PRIOR_MODEL_ORDER = (
    "expected_loss",
    "bornhuetter_ferguson_standard",
    "bornhuetter_ferguson_break_aware",
)
HISTORY_WINDOW_ORDER = (15, 10, 7, 5)
PRIOR_MULTIPLIER_ORDER = (0.8, 0.9, 1.0, 1.1, 1.2)

KEY_PAIRED_COMPARISON_IDS = (
    "step24_poisson_interaction_vs_poisson",
    "step25_tweedie_vs_poisson",
    "step21_break_aware_bf_vs_standard_bf",
    "step28_ceded_specific_bf_vs_standard_bf",
    "step28_break_aware_ceded_specific_vs_break_aware_bf",
)
PAIRED_COMPARISON_LABELS = {
    "step24_poisson_interaction_vs_poisson": (
        "Poisson + Interaction vs Poisson"
    ),
    "step25_tweedie_vs_poisson": "Tweedie vs Poisson",
    "step21_break_aware_bf_vs_standard_bf": (
        "Break-Aware BF vs Standard BF"
    ),
    "step28_ceded_specific_bf_vs_standard_bf": (
        "Ceded-Specific BF vs Standard BF"
    ),
    "step28_break_aware_ceded_specific_vs_break_aware_bf": (
        "BA Ceded-Specific BF vs Break-Aware BF"
    ),
}

FIGURE_OUTPUTS = {
    "figure01": "figure01_baseline_mape",
    "figure02": "figure02_model_success",
    "figure03": "figure03_ml_break_comparison",
    "figure04": "figure04_key_paired_comparisons",
    "figure05": "figure05_ceded_bf_sensitivity",
    "figure06": "figure06_history_window_sensitivity",
    "figure07": "figure07_prior_sensitivity",
    "figure08": "figure08_treaty_indexation",
}
MAIN_TABLE_FILES = (
    "table01_baseline_model_summary.csv",
    "table02_structural_break_models.csv",
    "table03_key_paired_comparisons.csv",
    "table04_ceded_bf.csv",
    "table05_history_window.csv",
    "table06_prior_sensitivity.csv",
    "table07_treaty_indexation.csv",
)


class PresentationValidationError(ValueError):
    """Raised when Step 33 presentation inputs or outputs are invalid."""


def sha256_file(path: Path) -> str:
    """Return a read-only SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_step32_source_scope(
    source_files: Mapping[str, Path] = STEP32_SOURCE_FILES,
) -> None:
    """Require every analytical input to reside in outputs/final_analysis."""

    for name, path in source_files.items():
        normalised = Path(path)
        if normalised.parent != FINAL_ANALYSIS_DIRECTORY:
            raise PresentationValidationError(
                f"Step 33 source {name} is outside final_analysis: {path}."
            )
        if any(
            part.startswith("step") and part[4:6].isdigit()
            for part in normalised.parts[:-1]
        ):
            raise PresentationValidationError(
                f"Step 33 may not read an experiment directory: {path}."
            )


def load_step32_sources() -> tuple[dict[str, object], dict[str, str]]:
    """Load the declared Step 32 sources and record their hashes."""

    validate_step32_source_scope()
    loaded: dict[str, object] = {}
    hashes: dict[str, str] = {}
    for name, path in STEP32_SOURCE_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[name] = sha256_file(path)
        if path.suffix.lower() == ".csv":
            loaded[name] = pd.read_csv(path, low_memory=False)
        else:
            loaded[name] = path.read_text(encoding="utf-8")
    return loaded, hashes


def _as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def _ordered(
    frame: pd.DataFrame,
    column: str,
    order: Sequence[object],
) -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.Categorical(
        result[column],
        categories=list(order),
        ordered=True,
    )
    return result.sort_values(column, kind="stable").reset_index(drop=True)


def _conditional_accuracy(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Summarise accuracy only over successful estimator rows."""

    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(list(group_columns), dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        success = _as_boolean(group["success"])
        successful = group.loc[success]
        row = dict(zip(group_columns, key))
        row.update(
            {
                "attempted_fits": int(len(group)),
                "successful_fits": int(success.sum()),
                "success_rate_pct": 100.0 * float(success.mean()),
                "mean_percentage_error_pct": pd.to_numeric(
                    successful["percentage_error"], errors="coerce"
                ).mean(),
                "mean_absolute_percentage_error_pct": pd.to_numeric(
                    successful["absolute_percentage_error"], errors="coerce"
                ).mean(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_scenario_summary(baseline: pd.DataFrame) -> pd.DataFrame:
    """Conditional baseline accuracy by scenario, model and basis."""

    summary = _conditional_accuracy(
        baseline,
        ["scenario_id", "model", "basis"],
    )
    summary["scenario_label"] = summary["scenario_id"].map(SCENARIO_LABELS)
    summary["model_label"] = summary["model"].map(MODEL_LABELS)
    summary = _ordered(summary, "scenario_id", SCENARIO_ORDER)
    summary = _ordered(summary, "model", MODEL_ORDER)
    return summary.sort_values(
        ["basis", "scenario_id", "model"], kind="stable"
    ).reset_index(drop=True)


def baseline_model_table(baseline: pd.DataFrame) -> pd.DataFrame:
    """Portfolio-wide descriptive baseline performance by model and basis."""

    long = _conditional_accuracy(baseline, ["model", "basis"])
    metrics = [
        "success_rate_pct",
        "mean_percentage_error_pct",
        "mean_absolute_percentage_error_pct",
    ]
    parts = []
    for basis in ["gross", "ceded"]:
        part = long.loc[long["basis"].eq(basis), ["model", *metrics]].copy()
        part = part.rename(
            columns={
                "success_rate_pct": f"{basis}_success_rate_pct",
                "mean_percentage_error_pct": f"{basis}_mpe_pct",
                "mean_absolute_percentage_error_pct": f"{basis}_mape_pct",
            }
        )
        parts.append(part)
    result = parts[0].merge(parts[1], on="model", validate="one_to_one")
    result.insert(1, "model_label", result["model"].map(MODEL_LABELS))
    return _ordered(result, "model", MODEL_ORDER)


def baseline_applicability_summary(applicability: pd.DataFrame) -> pd.DataFrame:
    """Return baseline applicability and unconditional success by scenario."""

    selected = applicability.loc[
        applicability["analysis_type"].eq("baseline_model")
    ].copy()
    selected["applicability_rate_pct"] = (
        100.0 * pd.to_numeric(selected["applicability_rate"], errors="coerce")
    )
    selected["conditional_success_rate_pct"] = (
        100.0
        * pd.to_numeric(selected["conditional_success_rate"], errors="coerce")
    )
    selected["unconditional_success_rate_pct"] = (
        100.0
        * pd.to_numeric(
            selected["unconditional_success_rate"], errors="coerce"
        )
    )
    selected["scenario_label"] = selected["scenario_id"].map(SCENARIO_LABELS)
    selected["model_label"] = selected["model"].map(MODEL_LABELS)
    columns = [
        "scenario_id",
        "scenario_label",
        "model",
        "model_label",
        "basis",
        "scheduled_count",
        "attempted_count",
        "applicable_count",
        "successful_count",
        "failed_count",
        "applicability_rate_pct",
        "conditional_success_rate_pct",
        "unconditional_success_rate_pct",
    ]
    selected = selected[columns]
    selected = _ordered(selected, "scenario_id", SCENARIO_ORDER)
    selected = _ordered(selected, "model", MODEL_ORDER)
    return selected.sort_values(
        ["basis", "scenario_id", "model"], kind="stable"
    ).reset_index(drop=True)


def structural_break_ml_summary(baseline: pd.DataFrame) -> pd.DataFrame:
    """Summarise the three ML models in the structural-break scenarios."""

    selected = baseline.loc[
        baseline["scenario_id"].isin(BREAK_SCENARIOS)
        & baseline["model"].isin(ML_MODEL_ORDER)
    ].copy()
    summary = _conditional_accuracy(
        selected,
        ["scenario_id", "model", "basis"],
    )
    summary["scenario_label"] = summary["scenario_id"].map(SCENARIO_LABELS)
    summary["model_label"] = summary["model"].map(MODEL_LABELS)
    summary = _ordered(summary, "scenario_id", BREAK_SCENARIOS)
    summary = _ordered(summary, "model", ML_MODEL_ORDER)
    return summary.sort_values(
        ["basis", "scenario_id", "model"], kind="stable"
    ).reset_index(drop=True)


def paired_interpretation(direction: object) -> str:
    """Map frozen Step 31 CI labels to approved dissertation wording."""

    mapping = {
        "favors_A": "Favours A",
        "favors_B": "Favours B",
        "includes_zero": "Includes zero",
        "not_available": "Not available",
    }
    key = str(direction)
    if key not in mapping:
        raise PresentationValidationError(
            f"Unknown paired inference direction: {direction!r}."
        )
    return mapping[key]


def key_paired_comparisons(paired: pd.DataFrame) -> pd.DataFrame:
    """Select pre-existing, decision-relevant ceded break comparisons."""

    selected = paired.loc[
        paired["comparison_id"].isin(KEY_PAIRED_COMPARISON_IDS)
        & paired["scenario_id"].isin(BREAK_SCENARIOS)
        & paired["basis"].eq("ceded")
    ].copy()
    selected["comparison_label"] = selected["comparison_id"].map(
        PAIRED_COMPARISON_LABELS
    )
    selected["scenario_label"] = selected["scenario_id"].map(SCENARIO_LABELS)
    selected["model_a_label"] = selected["model_a"].map(MODEL_LABELS)
    selected["model_b_label"] = selected["model_b"].map(MODEL_LABELS)
    selected["interpretation"] = selected["inference_direction"].map(
        paired_interpretation
    )
    comparison_order = list(KEY_PAIRED_COMPARISON_IDS)
    selected["comparison_id"] = pd.Categorical(
        selected["comparison_id"],
        categories=comparison_order,
        ordered=True,
    )
    selected["scenario_id"] = pd.Categorical(
        selected["scenario_id"],
        categories=list(BREAK_SCENARIOS),
        ordered=True,
    )
    return selected.sort_values(
        ["comparison_id", "scenario_id"], kind="stable"
    ).reset_index(drop=True)


def ceded_bf_sensitivity_summary(master: pd.DataFrame) -> pd.DataFrame:
    """Summarise Step 28 ceded development-pattern sensitivity."""

    selected = master.loc[
        master["analysis_type"].eq("development_sensitivity")
        & master["basis"].eq("ceded")
    ].copy()
    summary = _conditional_accuracy(selected, ["scenario_id", "model", "basis"])
    summary["scenario_label"] = summary["scenario_id"].map(SCENARIO_LABELS)
    summary["model_label"] = summary["model"].map(MODEL_LABELS)
    summary = _ordered(summary, "scenario_id", SCENARIO_ORDER)
    summary = _ordered(summary, "model", BF_SENSITIVITY_MODEL_ORDER)
    return summary.sort_values(["scenario_id", "model"], kind="stable").reset_index(
        drop=True
    )


def history_window_summaries(
    master: pd.DataFrame,
    applicability: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return conditional accuracy and separate applicability by history window."""

    selected = master.loc[
        master["analysis_type"].eq("history_sensitivity")
    ].copy()
    accuracy = _conditional_accuracy(
        selected,
        ["model", "basis", "history_window_years"],
    )
    accuracy["model_label"] = accuracy["model"].map(MODEL_LABELS)

    app = applicability.loc[
        applicability["analysis_type"].eq("history_sensitivity")
    ].copy()
    count_columns = [
        "scheduled_count",
        "attempted_count",
        "applicable_count",
        "structurally_not_applicable_count",
        "successful_count",
        "failed_count",
    ]
    app[count_columns] = app[count_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    app = (
        app.groupby(
            ["model", "basis", "history_window_years"],
            dropna=False,
            as_index=False,
        )[count_columns]
        .sum()
    )
    app["applicability_rate_pct"] = (
        100.0 * app["applicable_count"] / app["scheduled_count"]
    )
    app["conditional_success_rate_pct"] = (
        100.0
        * app["successful_count"]
        / app["attempted_count"].replace(0, np.nan)
    )
    app["unconditional_success_rate_pct"] = (
        100.0 * app["successful_count"] / app["scheduled_count"]
    )
    app["model_label"] = app["model"].map(MODEL_LABELS)
    accuracy["history_window_years"] = pd.to_numeric(
        accuracy["history_window_years"], errors="coerce"
    ).astype("Int64")
    app["history_window_years"] = pd.to_numeric(
        app["history_window_years"], errors="coerce"
    ).astype("Int64")
    return accuracy, app


def history_window_table(
    accuracy: pd.DataFrame,
    applicability: pd.DataFrame,
) -> pd.DataFrame:
    """Create the concise model/basis history-window table."""

    accuracy_pivot = accuracy.pivot(
        index=["model", "basis"],
        columns="history_window_years",
        values="mean_absolute_percentage_error_pct",
    ).reset_index()
    accuracy_pivot = accuracy_pivot.rename(
        columns={15: "mape_15_year_pct", 10: "mape_10_year_pct"}
    )
    for column in ["mape_15_year_pct", "mape_10_year_pct"]:
        if column not in accuracy_pivot:
            accuracy_pivot[column] = np.nan
    accuracy_pivot["mape_change_10_minus_15_pp"] = (
        accuracy_pivot["mape_10_year_pct"]
        - accuracy_pivot["mape_15_year_pct"]
    )

    app_pivot = applicability.pivot(
        index=["model", "basis"],
        columns="history_window_years",
        values="unconditional_success_rate_pct",
    ).reset_index()
    app_pivot = app_pivot.rename(
        columns={
            15: "success_15_year_pct",
            10: "success_10_year_pct",
            7: "success_7_year_pct",
            5: "success_5_year_pct",
        }
    )
    result = accuracy_pivot[
        [
            "model",
            "basis",
            "mape_15_year_pct",
            "mape_10_year_pct",
            "mape_change_10_minus_15_pp",
        ]
    ].merge(app_pivot, on=["model", "basis"], validate="one_to_one")
    result.insert(1, "model_label", result["model"].map(MODEL_LABELS))
    return result.sort_values(
        ["model", "basis"],
        key=lambda series: series.map(
            {model: index for index, model in enumerate(HISTORY_MODEL_ORDER)}
        )
        if series.name == "model"
        else series,
        kind="stable",
    ).reset_index(drop=True)


def prior_sensitivity_summary(master: pd.DataFrame) -> pd.DataFrame:
    """Summarise Step 27 bias and accuracy by model, basis and multiplier."""

    selected = master.loc[master["analysis_type"].eq("prior_sensitivity")].copy()
    summary = _conditional_accuracy(
        selected,
        ["model", "basis", "prior_multiplier"],
    )
    summary["model_label"] = summary["model"].map(MODEL_LABELS)
    summary["prior_multiplier"] = pd.to_numeric(
        summary["prior_multiplier"], errors="coerce"
    )
    return summary.sort_values(
        ["basis", "model", "prior_multiplier"], kind="stable"
    ).reset_index(drop=True)


def treaty_scenario_summary(treaty: pd.DataFrame) -> pd.DataFrame:
    """Summarise Step 29 portfolio mechanics by scenario and treaty variant."""

    if any("percentage_error" in column for column in treaty.columns):
        raise PresentationValidationError(
            "Treaty mechanics must not contain estimator APE fields."
        )
    metrics = [
        "ceded_share",
        "ceded_true_reserve",
        "attachment_frequency",
        "exhaustion_frequency",
        "gross_ultimate",
        "ceded_ultimate",
    ]
    summary = (
        treaty.groupby(["scenario_id", "treaty_variant"], as_index=False)[metrics]
        .mean()
        .rename(
            columns={
                "ceded_share": "mean_ceded_share",
                "ceded_true_reserve": "mean_ceded_true_reserve_gbp",
                "attachment_frequency": "mean_attachment_frequency",
                "exhaustion_frequency": "mean_exhaustion_frequency",
                "gross_ultimate": "mean_gross_ultimate_gbp",
                "ceded_ultimate": "mean_ceded_ultimate_gbp",
            }
        )
    )
    summary["mean_ceded_share_pct"] = 100.0 * summary["mean_ceded_share"]
    summary["mean_attachment_frequency_pct"] = (
        100.0 * summary["mean_attachment_frequency"]
    )
    summary["mean_exhaustion_frequency_pct"] = (
        100.0 * summary["mean_exhaustion_frequency"]
    )
    summary["scenario_label"] = summary["scenario_id"].map(SCENARIO_LABELS)
    summary = _ordered(summary, "scenario_id", SCENARIO_ORDER)
    return summary.sort_values(
        ["scenario_id", "treaty_variant"], kind="stable"
    ).reset_index(drop=True)


def treaty_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Create the concise fixed-versus-indexed Step 29 table."""

    fixed = summary.loc[summary["treaty_variant"].eq("fixed_nominal")].copy()
    indexed = summary.loc[summary["treaty_variant"].eq("fully_indexed")].copy()
    columns = [
        "scenario_id",
        "scenario_label",
        "mean_ceded_share_pct",
        "mean_ceded_true_reserve_gbp",
        "mean_attachment_frequency_pct",
        "mean_exhaustion_frequency_pct",
    ]
    fixed = fixed[columns].add_prefix("fixed_").rename(
        columns={
            "fixed_scenario_id": "scenario_id",
            "fixed_scenario_label": "scenario_label",
        }
    )
    indexed = indexed[columns].add_prefix("indexed_").rename(
        columns={"indexed_scenario_id": "scenario_id"}
    )
    indexed = indexed.drop(columns="indexed_scenario_label")
    result = fixed.merge(indexed, on="scenario_id", validate="one_to_one")
    result["ceded_share_change_indexed_minus_fixed_pp"] = (
        result["indexed_mean_ceded_share_pct"]
        - result["fixed_mean_ceded_share_pct"]
    )
    result["ceded_reserve_change_indexed_minus_fixed_pct"] = 100.0 * (
        result["indexed_mean_ceded_true_reserve_gbp"]
        - result["fixed_mean_ceded_true_reserve_gbp"]
    ) / result["fixed_mean_ceded_true_reserve_gbp"]
    result["attachment_frequency_change_pp"] = (
        result["indexed_mean_attachment_frequency_pct"]
        - result["fixed_mean_attachment_frequency_pct"]
    )
    result["exhaustion_frequency_change_pp"] = (
        result["indexed_mean_exhaustion_frequency_pct"]
        - result["fixed_mean_exhaustion_frequency_pct"]
    )
    return _ordered(result, "scenario_id", SCENARIO_ORDER)


def validate_nonapplicable_accuracy(master: pd.DataFrame) -> None:
    """Require structurally skipped estimators to have no accuracy values."""

    applicable = _as_boolean(master["applicable"])
    skipped = master.loc[~applicable]
    accuracy_columns = [
        "percentage_error",
        "absolute_percentage_error",
        "estimated_reserve",
    ]
    if skipped[accuracy_columns].notna().any().any():
        raise PresentationValidationError(
            "Structurally non-applicable rows contain accuracy values."
        )


def validate_output_index(index: pd.DataFrame, repository_root: Path) -> None:
    """Require unique, non-empty metadata rows that reference real files."""

    required = {
        "output_id",
        "output_type",
        "title",
        "research_question",
        "key_message",
        "main_caveat",
        "backing_csv",
        "image_or_table_path",
        "recommended_location",
    }
    missing = sorted(required.difference(index.columns))
    if missing:
        raise PresentationValidationError(
            f"Output index is missing columns: {missing}."
        )
    required_columns = sorted(required)
    if index.empty or index[required_columns].isna().any().any():
        raise PresentationValidationError("Output index contains empty metadata.")
    if index["output_id"].duplicated().any():
        raise PresentationValidationError("Output IDs must be unique.")
    for column in ["backing_csv", "image_or_table_path"]:
        missing_paths = [
            value
            for value in index[column]
            if not (repository_root / str(value)).is_file()
            or (repository_root / str(value)).stat().st_size == 0
        ]
        if missing_paths:
            raise PresentationValidationError(
                f"Output index references missing {column}: {missing_paths}."
            )


def percentage_unit_for_field(field: str) -> str:
    """Return the approved presentation unit for percentage-like fields."""

    percentage_point_tokens = (
        "difference",
        "change",
        "ci_lower",
        "ci_upper",
    )
    if any(token in field for token in percentage_point_tokens):
        return "percentage points"
    if any(token in field for token in ("mpe", "mape", "rate", "share")):
        return "%"
    raise PresentationValidationError(
        f"No percentage unit rule is defined for {field!r}."
    )

"""Build the Step 32 analysis-ready dataset from frozen result files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Iterable

import pandas as pd

from src.final_results_consolidation import (
    BASELINE_MODELS,
    CANONICAL_MODELS,
    MASTER_AY_COLUMNS,
    MASTER_RESULT_COLUMNS,
    VALID_BASES,
    VALID_SCENARIOS,
    attach_portfolio_status_to_accident_year,
    build_applicability_summary,
    ensure_data_dictionary_complete,
    harmonise_accident_year_results,
    harmonise_paired_comparisons,
    harmonise_portfolio_results,
    harmonise_treaty_sensitivity,
    reserve_metric_mismatch_counts,
    sha256_file,
    truth_mismatch_group_count,
    validate_sensitivity_scope,
    validate_unique_keys,
)


OUTPUT_DIRECTORY = Path("outputs/final_analysis")

EXPECTED_FROZEN_HASHES = {
    "step16_baseline_results": (
        "bf1069b9d85516611851d36f09fa2b3dece7d1f887c0280f8afcef84f1e9fb42"
    ),
    "step19_expected_loss_prior": (
        "06c23adae7ab9d96e0fd657d3af173a0bf05f91bd00e55d911c08d8459023912"
    ),
    "step28_ceded_development_pattern": (
        "a953f05567a417fc287c0cc8263b7668876e1490e42737b8409deb0e9c19468e"
    ),
}


@dataclass(frozen=True)
class SourceSpec:
    """One audited input in the internal Step 32 source map."""

    source_id: str
    experiment_step: int
    experiment_name: str
    path: Path
    purpose: str
    data_level: str
    key_columns: tuple[str, ...]
    sensitivity_dimensions: str
    authoritative_for: str
    notes: str = ""


SOURCE_SPECS = (
    SourceSpec(
        "step16_baseline_results",
        16,
        "frozen_baseline_experiment",
        Path("data/final/baseline_step16/results.csv"),
        "Canonical portfolio results for the four original estimators.",
        "portfolio",
        ("scenario_id", "simulation_id", "basis", "model"),
        "none",
        (
            "chain_ladder;inflation_adjusted_chain_ladder;cashflow_uplift;"
            "regularized_poisson"
        ),
        "Later Poisson reproductions are validation evidence only.",
    ),
    SourceSpec(
        "step19_expected_loss_prior",
        19,
        "independent_expected_loss_prior",
        Path(
            "data/calibration/expected_loss_prior/final_2000/"
            "expected_loss_prior.csv"
        ),
        "Frozen independent-prior provenance reference.",
        "calibration_reference",
        ("scenario_id", "accident_year"),
        "none",
        "independent_prior_hash",
    ),
    SourceSpec(
        "step20_expected_loss_results",
        20,
        "expected_loss_reserving",
        Path(
            "outputs/step20_expected_loss/final_50/"
            "expected_loss_results.csv"
        ),
        "Canonical Expected Loss portfolio results.",
        "portfolio",
        ("scenario_id", "simulation_id", "basis", "model"),
        "none",
        "expected_loss",
    ),
    SourceSpec(
        "step20_expected_loss_ay",
        20,
        "expected_loss_reserving",
        Path(
            "outputs/step20_expected_loss/final_50/"
            "expected_loss_by_accident_year.csv"
        ),
        "Expected Loss accident-year detail.",
        "accident_year",
        (
            "scenario_id",
            "simulation_id",
            "basis",
            "model",
            "accident_year",
        ),
        "none",
        "expected_loss_accident_year",
    ),
    SourceSpec(
        "step21_paid_bf_results",
        21,
        "paid_bornhuetter_ferguson",
        Path("outputs/step21_paid_bf/final_50/bf_results.csv"),
        "Canonical Standard and Break-Aware Paid BF portfolio results.",
        "portfolio",
        ("scenario_id", "simulation_id", "basis", "model"),
        "none",
        (
            "bornhuetter_ferguson_standard;"
            "bornhuetter_ferguson_break_aware"
        ),
    ),
    SourceSpec(
        "step21_paid_bf_ay",
        21,
        "paid_bornhuetter_ferguson",
        Path("outputs/step21_paid_bf/final_50/bf_by_accident_year.csv"),
        "Paid BF accident-year detail.",
        "accident_year",
        (
            "scenario_id",
            "simulation_id",
            "basis",
            "model",
            "accident_year",
        ),
        "none",
        "paid_bf_accident_year",
    ),
    SourceSpec(
        "step24_poisson_interaction_results",
        24,
        "poisson_break_interaction",
        Path(
            "outputs/step24_poisson_break_interaction/final_50/results.csv"
        ),
        "Canonical interaction-model rows; reproduced Poisson rows excluded.",
        "portfolio",
        ("scenario_id", "simulation_id", "basis", "model"),
        "none",
        "regularized_poisson_break_interaction",
    ),
    SourceSpec(
        "step25_tweedie_results",
        25,
        "regularized_tweedie",
        Path("outputs/step25_regularized_tweedie/final_50/results.csv"),
        "Canonical Tweedie rows; reproduced Poisson rows excluded.",
        "portfolio",
        ("scenario_id", "simulation_id", "basis", "model"),
        "none",
        "regularized_tweedie",
    ),
    SourceSpec(
        "step26_history_results",
        26,
        "history_window_sensitivity",
        Path(
            "outputs/step26_history_window_sensitivity/final_50/results.csv"
        ),
        "All frozen history-window sensitivity configurations.",
        "portfolio",
        (
            "scenario_id",
            "simulation_id",
            "basis",
            "model",
            "history_window_years",
        ),
        "history_window_years;history_start_ay",
        "history_window_sensitivity",
    ),
    SourceSpec(
        "step27_prior_results",
        27,
        "prior_misspecification_sensitivity",
        Path(
            "outputs/step27_prior_misspecification/final_50/results.csv"
        ),
        "All frozen prior-multiplier portfolio results.",
        "portfolio",
        (
            "scenario_id",
            "simulation_id",
            "basis",
            "model",
            "prior_multiplier",
        ),
        "prior_multiplier",
        "prior_misspecification_sensitivity",
    ),
    SourceSpec(
        "step27_prior_ay",
        27,
        "prior_misspecification_sensitivity",
        Path(
            "outputs/step27_prior_misspecification/final_50/"
            "by_accident_year.csv"
        ),
        "Prior-multiplier accident-year detail.",
        "accident_year",
        (
            "scenario_id",
            "simulation_id",
            "basis",
            "model",
            "prior_multiplier",
            "accident_year",
        ),
        "prior_multiplier",
        "prior_sensitivity_accident_year",
    ),
    SourceSpec(
        "step28_ceded_pattern_reference",
        28,
        "ceded_bf_development_calibration",
        Path(
            "outputs/step28_ceded_bf_development_sensitivity/calibration/"
            "final_2000/ceded_development_pattern.csv"
        ),
        "Frozen independently calibrated ceded-pattern provenance reference.",
        "calibration_reference",
        (
            "calibration_version",
            "scenario_id",
            "structural_break_regime",
            "development_year",
        ),
        "inflation_scenario;structural_break_regime",
        "ceded_development_pattern_hash",
    ),
    SourceSpec(
        "step28_ceded_bf_results",
        28,
        "ceded_bf_development_sensitivity",
        Path(
            "outputs/step28_ceded_bf_development_sensitivity/final_50/"
            "results.csv"
        ),
        "All frozen ceded development-pattern variants.",
        "portfolio",
        (
            "scenario_id",
            "simulation_id",
            "basis",
            "model",
            "pattern_variant",
        ),
        "pattern_variant;calibration_version",
        "ceded_bf_development_sensitivity",
    ),
    SourceSpec(
        "step28_ceded_bf_ay",
        28,
        "ceded_bf_development_sensitivity",
        Path(
            "outputs/step28_ceded_bf_development_sensitivity/final_50/"
            "by_accident_year.csv"
        ),
        "Ceded development-pattern accident-year detail.",
        "accident_year",
        (
            "scenario_id",
            "simulation_id",
            "basis",
            "model",
            "pattern_variant",
            "accident_year",
        ),
        "pattern_variant;calibration_version",
        "ceded_bf_development_sensitivity_accident_year",
    ),
    SourceSpec(
        "step29_treaty_results",
        29,
        "treaty_indexation_sensitivity",
        Path(
            "outputs/step29_treaty_indexation_sensitivity/final_50/"
            "results.csv"
        ),
        "Portfolio-level treaty mechanics; not estimator accuracy.",
        "treaty_portfolio",
        ("scenario_id", "simulation_id", "treaty_variant"),
        "treaty_variant",
        "treaty_mechanics",
    ),
    SourceSpec(
        "step29_treaty_ay",
        29,
        "treaty_indexation_sensitivity",
        Path(
            "outputs/step29_treaty_indexation_sensitivity/final_50/"
            "by_accident_year.csv"
        ),
        "AY treaty index factors used only to describe portfolio variants.",
        "treaty_accident_year",
        (
            "scenario_id",
            "simulation_id",
            "treaty_variant",
            "accident_year",
        ),
        "treaty_variant;accident_year",
        "treaty_index_factor_provenance",
    ),
    SourceSpec(
        "step31_core_paired_summary",
        31,
        "paired_model_comparisons",
        Path(
            "outputs/step31_paired_comparisons/"
            "paired_model_comparison_summary.csv"
        ),
        "Frozen core-model paired comparison summaries.",
        "paired_summary",
        (
            "comparison_id",
            "scenario_id",
            "basis",
            "group_type",
            "pool_label",
        ),
        "comparison_id",
        "paired_core_comparisons",
    ),
    SourceSpec(
        "step31_history_paired_summary",
        31,
        "paired_model_comparisons",
        Path(
            "outputs/step31_paired_comparisons/"
            "history_window_paired_comparisons.csv"
        ),
        "Frozen history-window paired comparison summaries.",
        "paired_summary",
        (
            "comparison_id",
            "scenario_id",
            "basis",
            "model",
            "group_type",
            "pool_label",
        ),
        "shorter_window_years;longer_window_years",
        "paired_history_comparisons",
    ),
    SourceSpec(
        "step31_prior_paired_summary",
        31,
        "paired_model_comparisons",
        Path(
            "outputs/step31_paired_comparisons/"
            "prior_sensitivity_paired_comparisons.csv"
        ),
        "Frozen prior-multiplier paired comparison summaries.",
        "paired_summary",
        (
            "comparison_id",
            "scenario_id",
            "basis",
            "model",
            "group_type",
            "pool_label",
        ),
        "prior_multiplier_a;prior_multiplier_b",
        "paired_prior_comparisons",
    ),
)

SOURCE_BY_ID = {spec.source_id: spec for spec in SOURCE_SPECS}


def _read_and_audit_sources() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Load each frozen source once, validate its key, and record its hash."""

    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    for spec in SOURCE_SPECS:
        if not spec.path.is_file():
            raise FileNotFoundError(spec.path)
        frame = pd.read_csv(spec.path)
        validate_unique_keys(
            frame,
            spec.key_columns,
            source_name=spec.source_id,
        )
        frames[spec.source_id] = frame
        hashes[spec.source_id] = sha256_file(spec.path)
    return frames, hashes


def _portfolio_tables(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the authoritative baseline table and sensitivity-aware master."""

    baseline_specs = [
        (
            "step16_baseline_results",
            16,
            "frozen_baseline_experiment",
            None,
        ),
        (
            "step20_expected_loss_results",
            20,
            "expected_loss_reserving",
            ["expected_loss"],
        ),
        (
            "step21_paid_bf_results",
            21,
            "paid_bornhuetter_ferguson",
            [
                "bornhuetter_ferguson_standard",
                "bornhuetter_ferguson_break_aware",
            ],
        ),
        (
            "step24_poisson_interaction_results",
            24,
            "poisson_break_interaction",
            ["regularized_poisson_break_interaction"],
        ),
        (
            "step25_tweedie_results",
            25,
            "regularized_tweedie",
            ["regularized_tweedie"],
        ),
    ]
    baseline_parts = []
    for source_id, step, name, models in baseline_specs:
        spec = SOURCE_BY_ID[source_id]
        baseline_parts.append(
            harmonise_portfolio_results(
                frames[source_id],
                experiment_step=step,
                experiment_name=name,
                source_file=spec.path.as_posix(),
                analysis_type="baseline_model",
                include_models=models,
            )
        )
    baseline = pd.concat(baseline_parts, ignore_index=True)

    sensitivity_specs = [
        (
            "step26_history_results",
            26,
            "history_window_sensitivity",
            "history_sensitivity",
        ),
        (
            "step27_prior_results",
            27,
            "prior_misspecification_sensitivity",
            "prior_sensitivity",
        ),
        (
            "step28_ceded_bf_results",
            28,
            "ceded_bf_development_sensitivity",
            "development_sensitivity",
        ),
    ]
    sensitivity_parts = []
    for source_id, step, name, analysis_type in sensitivity_specs:
        spec = SOURCE_BY_ID[source_id]
        sensitivity_parts.append(
            harmonise_portfolio_results(
                frames[source_id],
                experiment_step=step,
                experiment_name=name,
                source_file=spec.path.as_posix(),
                analysis_type=analysis_type,
            )
        )
    master = pd.concat([baseline, *sensitivity_parts], ignore_index=True)
    return master.loc[:, MASTER_RESULT_COLUMNS], baseline.loc[:, MASTER_RESULT_COLUMNS]


def _accident_year_table(
    frames: dict[str, pd.DataFrame],
    master: pd.DataFrame,
) -> pd.DataFrame:
    """Harmonise compatible AY detail and attach frozen portfolio statuses."""

    specifications = [
        (
            "step20_expected_loss_ay",
            20,
            "expected_loss_reserving",
            "baseline_model",
            (),
        ),
        (
            "step21_paid_bf_ay",
            21,
            "paid_bornhuetter_ferguson",
            "baseline_model",
            (),
        ),
        (
            "step27_prior_ay",
            27,
            "prior_misspecification_sensitivity",
            "prior_sensitivity",
            ("prior_multiplier",),
        ),
        (
            "step28_ceded_bf_ay",
            28,
            "ceded_bf_development_sensitivity",
            "development_sensitivity",
            ("pattern_variant",),
        ),
    ]
    parts = []
    for source_id, step, name, analysis_type, sensitivities in specifications:
        spec = SOURCE_BY_ID[source_id]
        detail = harmonise_accident_year_results(
            frames[source_id],
            experiment_step=step,
            experiment_name=name,
            source_file=spec.path.as_posix(),
            analysis_type=analysis_type,
        )
        portfolio = master.loc[master["experiment_step"].eq(step)].copy()
        detail = attach_portfolio_status_to_accident_year(
            detail,
            portfolio,
            sensitivity_columns=sensitivities,
        )
        parts.append(detail)
    return pd.concat(parts, ignore_index=True).loc[:, MASTER_AY_COLUMNS]


def _paired_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine the frozen Step 31 paired-summary families without recomputing."""

    parts = []
    for source_id in [
        "step31_core_paired_summary",
        "step31_history_paired_summary",
        "step31_prior_paired_summary",
    ]:
        spec = SOURCE_BY_ID[source_id]
        parts.append(
            harmonise_paired_comparisons(
                frames[source_id],
                source_file=spec.path.as_posix(),
            )
        )
    return pd.concat(parts, ignore_index=True, sort=False)


def _source_inventory(
    frames: dict[str, pd.DataFrame],
    hashes: dict[str, str],
) -> pd.DataFrame:
    """Describe every frozen Step 32 source and its analytical role."""

    rows = []
    for spec in SOURCE_SPECS:
        frame = frames[spec.source_id]
        raw_models: Iterable[object] = (
            frame["model"].dropna().unique() if "model" in frame else []
        )
        bases: Iterable[object] = (
            frame["basis"].dropna().unique() if "basis" in frame else []
        )
        rows.append(
            {
                "experiment_step": spec.experiment_step,
                "experiment_name": spec.experiment_name,
                "source_path": spec.path.as_posix(),
                "source_hash": hashes[spec.source_id],
                "row_count": len(frame),
                "key_columns": ";".join(spec.key_columns),
                "models": ";".join(sorted(map(str, raw_models))),
                "basis": ";".join(sorted(map(str, bases))),
                "sensitivity_dimensions": spec.sensitivity_dimensions,
                "authoritative_for": spec.authoritative_for,
                "purpose": spec.purpose,
                "data_level": spec.data_level,
                "notes": spec.notes,
            }
        )
    return pd.DataFrame(rows)


def _model_dictionary() -> pd.DataFrame:
    """Return the documented canonical model catalogue used by Step 32."""

    rows = [
        (
            "chain_ladder",
            "classical development",
            "baseline",
            "Paid Chain Ladder reserve estimate.",
            16,
            "Stable historical development factors.",
        ),
        (
            "inflation_adjusted_chain_ladder",
            "classical development",
            "baseline",
            "Chain Ladder applied after calendar inflation adjustment.",
            16,
            "Specified inflation adjustment and stable development.",
        ),
        (
            "cashflow_uplift",
            "deterministic cash flow",
            "baseline",
            "Observed cash flows projected with the frozen uplift method.",
            16,
            "Frozen cash-flow uplift specification.",
        ),
        (
            "expected_loss",
            "expected loss",
            "baseline",
            "Reserve from the independent Step 19 expected-loss prior.",
            20,
            "Pricing prior independent of evaluation simulations.",
        ),
        (
            "bornhuetter_ferguson_standard",
            "Bornhuetter-Ferguson",
            "baseline",
            "Paid BF using the standard frozen development pattern.",
            21,
            "Independent prior and standard paid proportion.",
        ),
        (
            "bornhuetter_ferguson_break_aware",
            "Bornhuetter-Ferguson",
            "baseline",
            "Paid BF using the frozen break-aware development pattern.",
            21,
            "Independent prior and break-aware paid proportion.",
        ),
        (
            "regularized_poisson",
            "regularized GLM",
            "baseline",
            "Baseline regularized Poisson reserving model.",
            16,
            "Log link and frozen rolling-diagonal validation.",
        ),
        (
            "regularized_poisson_break_interaction",
            "regularized GLM",
            "baseline",
            "Poisson model with calendar-regime by development interactions.",
            24,
            "Frozen Step 24 interaction design.",
        ),
        (
            "regularized_tweedie",
            "regularized GLM",
            "baseline",
            "Tweedie reserving model with selected variance power.",
            25,
            "Log link and frozen power/alpha grids.",
        ),
        (
            "bornhuetter_ferguson_ceded_specific",
            "Bornhuetter-Ferguson",
            "sensitivity",
            "Standard BF using the independently calibrated ceded pattern.",
            28,
            "Ceded-only independently calibrated development pattern.",
        ),
        (
            "bornhuetter_ferguson_break_aware_ceded_specific",
            "Bornhuetter-Ferguson",
            "sensitivity",
            "Break-aware BF using the calibrated ceded pattern.",
            28,
            "Ceded-only post-break calibrated development pattern.",
        ),
    ]
    columns = [
        "model",
        "model_family",
        "baseline_or_sensitivity",
        "description",
        "source_step",
        "key_assumptions",
    ]
    return pd.DataFrame(rows, columns=columns)


FIELD_DESCRIPTIONS = {
    "analysis_type": "Role of the row in the final analysis hierarchy.",
    "experiment_step": "Dissertation step that produced the frozen source.",
    "experiment_name": "Human-readable frozen experiment identifier.",
    "source_file": "Repository-relative CSV from which the row was derived.",
    "scenario_id": "Frozen simulation scenario identifier.",
    "simulation_id": "Simulation replicate identifier within scenario.",
    "seed": "Recorded random seed, when retained by the frozen source.",
    "model": "Canonical reserving model identifier.",
    "basis": "Gross or ceded reserving basis.",
    "success": "Whether the estimator or mechanics row succeeded.",
    "applicable": "Whether the estimator was structurally applicable.",
    "fit_attempted": "Whether model fitting or estimation was attempted.",
    "status": "Canonical success, failure, or structural-skip status.",
    "failure_type": "Frozen failure classification, if any.",
    "failure_message": "Frozen failure detail, if any.",
    "estimated_reserve": "Portfolio or AY reserve estimate.",
    "true_reserve": "Matching simulated evaluation reserve truth.",
    "reserve_error": "Estimated reserve minus true reserve.",
    "percentage_error": "Reserve error divided by truth, times 100.",
    "absolute_percentage_error": "Absolute value of percentage error.",
    "runtime_seconds": "Recorded source-row runtime.",
    "history_window_years": "Historical fitting-window length.",
    "history_start_ay": "First accident year admitted to the history window.",
    "prior_multiplier": "Factor applied to the frozen independent prior.",
    "pattern_variant": "Frozen BF development-pattern configuration.",
    "treaty_variant": "Fixed-nominal or fully-indexed treaty configuration.",
    "selected_alpha": "Regularisation strength selected from the frozen grid.",
    "selected_power": "Selected Tweedie variance power.",
    "accident_year": "Accident year represented by the detail row.",
    "paid_to_date": "Paid amount observed by the valuation date.",
    "prior_ultimate": "Expected-loss prior ultimate for the accident year.",
    "assumed_paid_proportion": "BF assumed cumulative paid proportion.",
    "inference_direction": "Label determined solely from the frozen CI.",
    "source_hash": "SHA-256 digest of the frozen source file.",
}


def _field_unit(field: str) -> str:
    """Map a consolidated field name to its documented reporting unit."""

    lower = field.lower()
    if lower in {
        "percentage_error",
        "absolute_percentage_error",
        "mean_ape_a",
        "mean_ape_b",
        "mean_paired_difference",
        "median_paired_difference",
        "std_paired_difference",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
    }:
        return "percentage points"
    if any(
        token in lower
        for token in [
            "reserve",
            "ultimate",
            "paid_to_date",
            "base_attachment",
            "base_limit",
        ]
    ):
        return "GBP"
    if lower.endswith("_rate") or lower in {
        "ceded_share",
        "attachment_frequency",
        "exhaustion_frequency",
        "assumed_paid_proportion",
    }:
        return "proportion (0-1)"
    if "runtime" in lower:
        return "seconds"
    if "history_window" in lower:
        return "years"
    if lower in {
        "accident_year",
        "history_start_ay",
        "index_reference_year",
    }:
        return "calendar year"
    if lower == "source_hash":
        return "SHA-256 hex digest"
    if any(
        token in lower
        for token in ["count", "rows", "pairs", "resamples"]
    ):
        return "count"
    if lower in {
        "prior_multiplier",
        "selected_alpha",
        "selected_power",
        "minimum_applied_index_factor",
        "maximum_applied_index_factor",
        "mean_applied_index_factor",
    }:
        return "dimensionless"
    return "not applicable"


def _field_type(series: pd.Series) -> str:
    """Map a pandas dtype to the compact data-dictionary type vocabulary."""

    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    return "string"


def _data_dictionary(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build field-level metadata for every Step 32 machine-readable output."""

    rows = []
    for dataset, frame in datasets.items():
        for field in frame.columns:
            rows.append(
                {
                    "field_name": field,
                    "dataset": dataset,
                    "description": FIELD_DESCRIPTIONS.get(
                        field,
                        field.replace("_", " ").capitalize() + ".",
                    ),
                    "unit": _field_unit(field),
                    "type": _field_type(frame[field]),
                    "nullable": bool(frame[field].isna().any()),
                    "source": (
                        "frozen source or documented Step 32 derivation"
                    ),
                    "notes": (
                        "Null means unavailable or not meaningful; no imputation."
                    ),
                }
            )
    return pd.DataFrame(rows)


def _check(
    rows: list[dict[str, object]],
    name: str,
    passed: object,
    detail: object = "",
) -> None:
    """Append one normalized validation result row."""

    rows.append(
        {"check": name, "passed": bool(passed), "detail": str(detail)}
    )


def _validation_report(
    *,
    master: pd.DataFrame,
    baseline: pd.DataFrame,
    accident_year: pd.DataFrame,
    treaty: pd.DataFrame,
    paired: pd.DataFrame,
    applicability: pd.DataFrame,
    inventory: pd.DataFrame,
    model_dictionary: pd.DataFrame,
    data_dictionary: pd.DataFrame,
    dictionary_datasets: dict[str, pd.DataFrame],
    hashes_before: dict[str, str],
    hashes_after: dict[str, str],
) -> pd.DataFrame:
    """Run the complete Step 32 provenance and numerical acceptance suite."""

    rows: list[dict[str, object]] = []
    metric_counts = reserve_metric_mismatch_counts(master)
    sensitivity_counts = validate_sensitivity_scope(master)
    baseline_key = ["scenario_id", "simulation_id", "basis", "model"]
    master_key = [
        "experiment_step",
        "scenario_id",
        "simulation_id",
        "basis",
        "model",
        "history_window_years",
        "prior_multiplier",
        "pattern_variant",
    ]
    ay_key = [
        "experiment_step",
        "scenario_id",
        "simulation_id",
        "accident_year",
        "basis",
        "model",
        "prior_multiplier",
        "pattern_variant",
    ]
    expected_baseline_counts = {model: 900 for model in BASELINE_MODELS}
    actual_baseline_counts = baseline["model"].value_counts().to_dict()
    failed = master.loc[
        master["applicable"].fillna(False)
        & master["fit_attempted"].fillna(False)
        & ~master["success"].fillna(False)
    ]
    not_applicable = master.loc[~master["applicable"].fillna(False)]

    _check(rows, "all_declared_source_files_exist", True, len(SOURCE_SPECS))
    _check(
        rows,
        "all_source_hashes_recorded",
        len(hashes_before) == len(SOURCE_SPECS)
        and all(len(value) == 64 for value in hashes_before.values()),
        len(hashes_before),
    )
    _check(
        rows,
        "source_files_unchanged_after_consolidation",
        hashes_before == hashes_after,
        f"changed={sum(hashes_before[k] != hashes_after[k] for k in hashes_before)}",
    )
    _check(
        rows,
        "canonical_model_names_valid",
        set(master["model"]).issubset(CANONICAL_MODELS),
        f"models={master['model'].nunique()}",
    )
    _check(
        rows,
        "scenario_names_valid",
        set(master["scenario_id"]).issubset(VALID_SCENARIOS)
        and set(treaty["scenario_id"]).issubset(VALID_SCENARIOS),
    )
    _check(
        rows,
        "basis_values_valid",
        set(master["basis"]).issubset(VALID_BASES),
    )
    _check(
        rows,
        "baseline_keys_unique",
        not baseline.duplicated(baseline_key).any(),
    )
    _check(
        rows,
        "master_result_keys_unique",
        not master.duplicated(master_key).any(),
    )
    _check(
        rows,
        "accident_year_keys_unique",
        not accident_year.duplicated(ay_key).any(),
    )
    truth_mismatches = truth_mismatch_group_count(master)
    _check(
        rows,
        "truth_matches_across_comparable_models",
        truth_mismatches == 0,
        f"mismatch_groups={truth_mismatches}",
    )
    _check(
        rows,
        "no_invented_truth_values",
        master["true_reserve"].notna().all()
        and accident_year["true_reserve"].notna().all(),
        "truth copied only from row-level frozen sources",
    )
    for name, count in metric_counts.items():
        _check(rows, name, count == 0, count)
    _check(
        rows,
        "failed_rows_retain_failure_metadata",
        (
            failed["failure_type"].notna()
            & failed["failure_message"].notna()
        ).all(),
        f"failed_rows={len(failed)}",
    )
    _check(
        rows,
        "structural_nonapplicability_is_distinct",
        len(not_applicable) == 5400
        and not_applicable["status"].eq("not_applicable_by_design").all()
        and not not_applicable["fit_attempted"].fillna(True).any(),
        f"rows={len(not_applicable)}",
    )
    for name, count in sensitivity_counts.items():
        _check(rows, name, count == 0, count)
    _check(
        rows,
        "treaty_master_contains_no_estimator_ape",
        not any("percentage_error" in column for column in treaty.columns),
    )
    _check(
        rows,
        "paired_row_counts_reconcile",
        len(paired) == 616
        and (
            paired["required_pairs"]
            == paired["valid_pairs"] + paired["excluded_pairs"]
        ).all(),
        f"rows={len(paired)}",
    )
    _check(
        rows,
        "baseline_authoritative_counts_reproduced",
        len(baseline) == 8100
        and actual_baseline_counts == expected_baseline_counts,
        json.dumps(actual_baseline_counts, sort_keys=True),
    )
    _check(
        rows,
        "no_duplicate_reproduction_baselines",
        len(baseline) == 9 * 50 * 2 * len(BASELINE_MODELS),
    )
    _check(
        rows,
        "step19_prior_hash_unchanged",
        hashes_before["step19_expected_loss_prior"]
        == EXPECTED_FROZEN_HASHES["step19_expected_loss_prior"],
        hashes_before["step19_expected_loss_prior"],
    )
    _check(
        rows,
        "step28_pattern_hash_unchanged",
        hashes_before["step28_ceded_pattern_reference"]
        == EXPECTED_FROZEN_HASHES["step28_ceded_development_pattern"],
        hashes_before["step28_ceded_pattern_reference"],
    )
    _check(
        rows,
        "step16_baseline_hash_unchanged",
        hashes_before["step16_baseline_results"]
        == EXPECTED_FROZEN_HASHES["step16_baseline_results"],
        hashes_before["step16_baseline_results"],
    )
    _check(
        rows,
        "master_results_expected_rows",
        len(master) == 45000,
        len(master),
    )
    _check(
        rows,
        "accident_year_expected_rows",
        len(accident_year) == 270000,
        len(accident_year),
    )
    _check(rows, "treaty_expected_rows", len(treaty) == 900, len(treaty))
    _check(
        rows,
        "applicability_counts_reconcile",
        (
            applicability["applicable_count"]
            + applicability["structurally_not_applicable_count"]
            == applicability["scheduled_count"]
        ).all()
        and (
            applicability["successful_count"]
            + applicability["failed_count"]
            == applicability["attempted_count"]
        ).all(),
    )
    _check(
        rows,
        "paired_inference_labels_valid",
        set(paired["inference_direction"]).issubset(
            {"favors_A", "favors_B", "includes_zero", "not_available"}
        ),
    )
    _check(
        rows,
        "source_inventory_complete_and_unique",
        len(inventory) == len(SOURCE_SPECS)
        and not inventory["source_path"].duplicated().any(),
        len(inventory),
    )
    _check(
        rows,
        "model_dictionary_complete",
        set(model_dictionary["model"]) == CANONICAL_MODELS,
        len(model_dictionary),
    )
    try:
        ensure_data_dictionary_complete(data_dictionary, dictionary_datasets)
        dictionary_complete = True
        dictionary_detail = len(data_dictionary)
    except ValueError as error:
        dictionary_complete = False
        dictionary_detail = str(error)
    _check(
        rows,
        "data_dictionary_contains_every_output_field",
        dictionary_complete,
        dictionary_detail,
    )
    _check(
        rows,
        "failed_rows_have_no_accuracy_imputation",
        failed["absolute_percentage_error"].isna().all(),
        f"failed_rows={len(failed)}",
    )
    _check(
        rows,
        "not_applicable_rows_have_no_accuracy_imputation",
        not_applicable["absolute_percentage_error"].isna().all(),
        f"rows={len(not_applicable)}",
    )
    _check(
        rows,
        "successful_estimates_nonnegative",
        master.loc[master["success"], "estimated_reserve"].ge(0.0).all(),
    )
    return pd.DataFrame(rows)


def _method_note() -> str:
    """Return the frozen-source and harmonisation note written with Step 32."""

    return """# Step 32 — Consolidated Final Analysis Dataset

## Purpose

Step 32 is an analysis-only provenance and harmonisation layer over frozen
Step 16–31 outputs. It does not run simulations, refit reserving models,
recalibrate priors, rerun bootstrap resampling, or change any source result.
CSV is the authoritative output format for Step 33 and dissertation writing.

## Source hierarchy and authoritative baseline decisions

The original Step 16 archive is authoritative for Chain Ladder,
Inflation-Adjusted Chain Ladder, Cashflow Uplift and Regularized Poisson.
Step 20 is authoritative for Expected Loss and Step 21 for Standard and
Break-Aware Paid BF. Step 24 contributes only the Poisson break-interaction
model; its reproduced baseline Poisson rows are excluded. Step 25 contributes
only Regularized Tweedie; its Poisson comparator rows are excluded. This gives
one baseline row per scenario, simulation, basis and canonical model.

Step 26, Step 27 and Step 28 remain explicitly separate history, prior and
development sensitivities. Their baseline-like settings are retained because
they are necessary members of the frozen sensitivity designs, not substituted
for the authoritative baseline results. Step 29 is kept as treaty mechanics
and has no estimator APE. Step 31 frozen paired summaries are copied and
labelled; confidence intervals are not recomputed.

## Naming, nulls, failures and applicability

Model labels use one documented canonical mapping in `model_dictionary.csv`.
Unavailable or inapplicable fields are null: there is no zero filling,
hyperparameter invention, prior-multiplier inference, runtime imputation or
truth imputation. Successful, failed and structurally not-applicable rows are
distinct. Conditional success is calculated over attempted applicable fits;
unconditional success retains the full scheduled denominator.

AY detail is included only for compatible Expected Loss/BF result families.
Its status is joined from the matching frozen portfolio row rather than
inferred from the estimate. Step 29 index factors are described by the
minimum, maximum and mean of its own frozen AY schedule for each portfolio
variant.

## Paired comparisons and treaty mechanics

`master_paired_comparisons.csv` harmonises the three frozen Step 31 accuracy
summary tables. `inference_direction` is a deterministic label: an interval
below zero favours A, an interval above zero favours B, an interval containing
zero is `includes_zero`, and a missing interval is `not_available`. Treaty
mechanics remain in `master_treaty_sensitivity.csv` because indexation changes
the ceded outcome rather than estimates a common reserve truth.

## Provenance and validation

Every input has a repository-relative path, declared primary key, row count,
purpose, analytical level and SHA-256 digest in `source_inventory.csv`.
Digests are recorded before consolidation and checked again afterwards. The
known frozen hashes of the Step 16 results, Step 19 prior and Step 28 calibrated
ceded pattern are checked explicitly.

Validation covers source existence and stability; canonical scenarios, bases
and models; unique master and baseline keys; authoritative baseline counts;
truth consistency; reserve-error, PE and APE reconciliation; finite and
nonnegative successful estimates; failure metadata; structural applicability;
sensitivity-field scope; treaty/paired separation; and full data-dictionary
coverage. `validation_report.csv` is the machine-readable acceptance record.
"""


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write one authoritative Step 32 table without an index column."""

    frame.to_csv(path, index=False)


def main() -> None:
    """Build and validate all Step 32 outputs from frozen source CSVs."""

    if OUTPUT_DIRECTORY.exists():
        raise FileExistsError(
            f"Step 32 output already exists and will not be overwritten: "
            f"{OUTPUT_DIRECTORY}"
        )

    started = time.perf_counter()
    frames, hashes_before = _read_and_audit_sources()
    master, baseline = _portfolio_tables(frames)
    accident_year = _accident_year_table(frames, master)
    treaty = harmonise_treaty_sensitivity(
        frames["step29_treaty_results"],
        frames["step29_treaty_ay"],
        source_file=SOURCE_BY_ID["step29_treaty_results"].path.as_posix(),
    )
    paired = _paired_table(frames)
    applicability = build_applicability_summary(master)
    inventory = _source_inventory(frames, hashes_before)
    model_dictionary = _model_dictionary()

    validation_stub = pd.DataFrame(columns=["check", "passed", "detail"])
    dictionary_stub = pd.DataFrame(
        columns=[
            "field_name",
            "dataset",
            "description",
            "unit",
            "type",
            "nullable",
            "source",
            "notes",
        ]
    )
    dictionary_datasets = {
        "master_results.csv": master,
        "baseline_model_results.csv": baseline,
        "master_by_accident_year.csv": accident_year,
        "master_treaty_sensitivity.csv": treaty,
        "master_paired_comparisons.csv": paired,
        "master_applicability.csv": applicability,
        "source_inventory.csv": inventory,
        "model_dictionary.csv": model_dictionary,
        "validation_report.csv": validation_stub,
        "data_dictionary.csv": dictionary_stub,
    }
    data_dictionary = _data_dictionary(dictionary_datasets)
    hashes_after = {
        spec.source_id: sha256_file(spec.path) for spec in SOURCE_SPECS
    }
    validation = _validation_report(
        master=master,
        baseline=baseline,
        accident_year=accident_year,
        treaty=treaty,
        paired=paired,
        applicability=applicability,
        inventory=inventory,
        model_dictionary=model_dictionary,
        data_dictionary=data_dictionary,
        dictionary_datasets=dictionary_datasets,
        hashes_before=hashes_before,
        hashes_after=hashes_after,
    )
    elapsed_seconds = time.perf_counter() - started

    if not validation["passed"].all():
        failed = validation.loc[~validation["passed"]]
        raise RuntimeError(
            "Step 32 acceptance checks failed before output creation:\n"
            + failed.to_string(index=False)
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    outputs = {
        "master_results.csv": master,
        "baseline_model_results.csv": baseline,
        "master_by_accident_year.csv": accident_year,
        "master_treaty_sensitivity.csv": treaty,
        "master_paired_comparisons.csv": paired,
        "master_applicability.csv": applicability,
        "source_inventory.csv": inventory,
        "model_dictionary.csv": model_dictionary,
        "data_dictionary.csv": data_dictionary,
        "validation_report.csv": validation,
    }
    for name, frame in outputs.items():
        _write_csv(frame, OUTPUT_DIRECTORY / name)

    (OUTPUT_DIRECTORY / "FINAL_ANALYSIS_METHOD_NOTE.md").write_text(
        _method_note(),
        encoding="utf-8",
    )
    sensitivity_configurations = (
        master.loc[master["analysis_type"].ne("baseline_model")]
        .loc[
            :,
            [
                "analysis_type",
                "history_window_years",
                "prior_multiplier",
                "pattern_variant",
            ],
        ]
        .drop_duplicates()
    )
    manifest: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "step": 32,
        "analysis_only": True,
        "experiments_rerun": False,
        "models_refitted": False,
        "calibrations_rerun": False,
        "bootstrap_rerun": False,
        "output_directory": OUTPUT_DIRECTORY.as_posix(),
        "source_file_count": len(SOURCE_SPECS),
        "source_hashes_before": hashes_before,
        "source_hashes_after": hashes_after,
        "row_counts": {name: int(len(frame)) for name, frame in outputs.items()},
        "canonical_model_count": int(model_dictionary["model"].nunique()),
        "baseline_model_count": len(BASELINE_MODELS),
        "sensitivity_configuration_count": int(
            len(sensitivity_configurations)
        ),
        "validation_check_count": int(len(validation)),
        "validation_checks_passed": int(validation["passed"].sum()),
        "elapsed_seconds": elapsed_seconds,
    }
    (OUTPUT_DIRECTORY / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Step 32 final-analysis consolidation completed.")
    print(f"Output: {OUTPUT_DIRECTORY}")
    print(f"Frozen source files: {len(SOURCE_SPECS)}")
    print(f"Master result rows: {len(master)}")
    print(f"Baseline rows: {len(baseline)}")
    print(f"AY detail rows: {len(accident_year)}")
    print(f"Treaty rows: {len(treaty)}")
    print(f"Paired comparison rows: {len(paired)}")
    print(f"Applicability rows: {len(applicability)}")
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()

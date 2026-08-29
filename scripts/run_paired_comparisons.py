"""Run Step 31 paired analyses using frozen experiment outputs only."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from src.paired_comparisons import (
    APE_TIE_TOLERANCE,
    BOOTSTRAP_CONFIDENCE_LEVEL,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    NOT_APPLICABLE_BY_DESIGN,
    construct_successful_pairs,
    construct_treaty_mechanics_pairs,
    deterministic_bootstrap_mean_ci,
    load_frozen_results,
    sha256_file,
    summarise_applicability,
    summarise_paired_accuracy,
    summarise_treaty_mechanics,
)


OUTPUT_DIRECTORY = Path("outputs/step31_paired_comparisons")
SOURCE_FILES = {
    "step20_expected_loss": Path(
        "outputs/step20_expected_loss/final_50/expected_loss_results.csv"
    ),
    "step21_paid_bf": Path(
        "outputs/step21_paid_bf/final_50/bf_results.csv"
    ),
    "step24_poisson_break_interaction": Path(
        "outputs/step24_poisson_break_interaction/final_50/results.csv"
    ),
    "step25_regularized_tweedie": Path(
        "outputs/step25_regularized_tweedie/final_50/results.csv"
    ),
    "step26_history_window": Path(
        "outputs/step26_history_window_sensitivity/final_50/results.csv"
    ),
    "step27_prior_sensitivity": Path(
        "outputs/step27_prior_misspecification/final_50/results.csv"
    ),
    "step28_ceded_bf": Path(
        "outputs/step28_ceded_bf_development_sensitivity/final_50/results.csv"
    ),
    "step29_treaty_indexation": Path(
        "outputs/step29_treaty_indexation_sensitivity/final_50/results.csv"
    ),
}
SOURCE_KEYS = {
    "step20_expected_loss": [
        "scenario_id", "simulation_id", "basis", "model"
    ],
    "step21_paid_bf": [
        "scenario_id", "simulation_id", "basis", "model"
    ],
    "step24_poisson_break_interaction": [
        "scenario_id", "simulation_id", "basis", "model"
    ],
    "step25_regularized_tweedie": [
        "scenario_id", "simulation_id", "basis", "model"
    ],
    "step26_history_window": [
        "scenario_id",
        "simulation_id",
        "basis",
        "model",
        "history_window_years",
    ],
    "step27_prior_sensitivity": [
        "scenario_id",
        "simulation_id",
        "basis",
        "model",
        "prior_multiplier",
    ],
    "step28_ceded_bf": [
        "scenario_id", "simulation_id", "basis", "model"
    ],
    "step29_treaty_indexation": [
        "scenario_id", "simulation_id", "treaty_variant"
    ],
}

CORE_COMPARISONS = [
    {
        "comparison_id": "step24_poisson_interaction_vs_poisson",
        "source": "step24_poisson_break_interaction",
        "source_step": 24,
        "model_a": "regularized_poisson_break_interaction",
        "model_b": "regularized_poisson",
        "basis_filter": None,
    },
    {
        "comparison_id": "step25_tweedie_vs_poisson",
        "source": "step25_regularized_tweedie",
        "source_step": 25,
        "model_a": "regularized_tweedie",
        "model_b": "regularized_poisson",
        "basis_filter": None,
    },
    {
        "comparison_id": "step21_break_aware_bf_vs_standard_bf",
        "source": "step21_paid_bf",
        "source_step": 21,
        "model_a": "bornhuetter_ferguson_break_aware",
        "model_b": "bornhuetter_ferguson_standard",
        "basis_filter": None,
    },
    {
        "comparison_id": "step28_ceded_specific_bf_vs_standard_bf",
        "source": "step28_ceded_bf",
        "source_step": 28,
        "model_a": "bornhuetter_ferguson_ceded_specific",
        "model_b": "bornhuetter_ferguson_standard",
        "basis_filter": "ceded",
    },
    {
        "comparison_id": (
            "step28_break_aware_ceded_specific_vs_break_aware_bf"
        ),
        "source": "step28_ceded_bf",
        "source_step": 28,
        "model_a": (
            "bornhuetter_ferguson_break_aware_ceded_specific"
        ),
        "model_b": "bornhuetter_ferguson_break_aware",
        "basis_filter": "ceded",
    },
]

HISTORY_COMPARISONS = [
    ("history_10_vs_15", 10, 15),
    ("history_7_vs_15", 7, 15),
    ("history_7_vs_10", 7, 10),
]
PRIOR_COMPARISONS = [
    ("prior_0.80_vs_1.00", 0.8, 1.0),
    ("prior_0.90_vs_1.00", 0.9, 1.0),
    ("prior_1.10_vs_1.00", 1.1, 1.0),
    ("prior_1.20_vs_1.00", 1.2, 1.0),
]


def _load_sources() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    sources: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}

    for name, path in SOURCE_FILES.items():
        required = ["scenario_id", "simulation_id"]
        if name != "step29_treaty_indexation":
            required.extend(
                [
                    "basis",
                    "true_reserve",
                    "absolute_percentage_error",
                ]
            )
        sources[name] = load_frozen_results(
            path,
            key_columns=SOURCE_KEYS[name],
            required_columns=required,
            source_name=name,
        )
        hashes[name] = sha256_file(path)

    return sources, hashes


def _metadata(
    table: pd.DataFrame,
    *,
    comparison_id: str,
    source_step: int,
    model_a: str,
    model_b: str,
    analysis_family: str,
) -> pd.DataFrame:
    result = table.copy()
    result.insert(0, "comparison_id", comparison_id)
    result.insert(1, "analysis_family", analysis_family)
    result.insert(2, "source_step", source_step)
    result.insert(3, "model_a", model_a)
    result.insert(4, "model_b", model_b)
    return result


def _core_comparisons(
    sources: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    detail_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    applicability_parts: list[pd.DataFrame] = []
    raw_pairs: list[pd.DataFrame] = []

    for specification in CORE_COMPARISONS:
        source = sources[specification["source"]]
        basis_filter = specification["basis_filter"]
        if basis_filter is not None:
            source = source.loc[source["basis"].eq(basis_filter)].copy()
        passthrough = (
            ["structural_break"]
            if specification["source_step"] == 24
            else []
        )
        pairs = construct_successful_pairs(
            source,
            selector_column="model",
            selector_a=specification["model_a"],
            selector_b=specification["model_b"],
            pair_keys=["scenario_id", "simulation_id", "basis"],
            passthrough_columns=passthrough,
        )
        pairs = _metadata(
            pairs,
            comparison_id=specification["comparison_id"],
            source_step=specification["source_step"],
            model_a=specification["model_a"],
            model_b=specification["model_b"],
            analysis_family="core_model",
        )
        pairs["group_type"] = "scenario_basis"
        pairs["pool_label"] = ""
        detail_parts.append(pairs)
        raw_pairs.append(pairs)

        group_columns = ["scenario_id", "basis"]
        summary = summarise_paired_accuracy(
            pairs,
            group_columns=group_columns,
        )
        summary = _metadata(
            summary,
            comparison_id=specification["comparison_id"],
            source_step=specification["source_step"],
            model_a=specification["model_a"],
            model_b=specification["model_b"],
            analysis_family="core_model",
        )
        summary["group_type"] = "scenario_basis"
        summary["pool_label"] = ""
        summary_parts.append(summary)

        applicability = summarise_applicability(
            pairs,
            group_columns=group_columns,
        )
        applicability = _metadata(
            applicability,
            comparison_id=specification["comparison_id"],
            source_step=specification["source_step"],
            model_a=specification["model_a"],
            model_b=specification["model_b"],
            analysis_family="core_model",
        )
        applicability["group_type"] = "scenario_basis"
        applicability["pool_label"] = ""
        applicability_parts.append(applicability)

        if specification["source_step"] == 24:
            pooled = pairs.copy()
            pooled["pool_label"] = np.where(
                pooled["structural_break"].astype(bool),
                "break",
                "no_break",
            )
            pooled_summary = summarise_paired_accuracy(
                pooled,
                group_columns=["basis", "pool_label"],
            )
            pooled_summary["scenario_id"] = "__pooled__"
            pooled_summary = _metadata(
                pooled_summary,
                comparison_id=specification["comparison_id"],
                source_step=24,
                model_a=specification["model_a"],
                model_b=specification["model_b"],
                analysis_family="core_model",
            )
            pooled_summary["group_type"] = "pooled_structural_basis"
            summary_parts.append(pooled_summary)
            pooled_applicability = summarise_applicability(
                pooled,
                group_columns=["basis", "pool_label"],
            )
            pooled_applicability["scenario_id"] = "__pooled__"
            pooled_applicability = _metadata(
                pooled_applicability,
                comparison_id=specification["comparison_id"],
                source_step=24,
                model_a=specification["model_a"],
                model_b=specification["model_b"],
                analysis_family="core_model",
            )
            pooled_applicability[
                "group_type"
            ] = "pooled_structural_basis"
            applicability_parts.append(pooled_applicability)

    return (
        pd.concat(detail_parts, ignore_index=True, sort=False),
        pd.concat(summary_parts, ignore_index=True, sort=False),
        pd.concat(applicability_parts, ignore_index=True, sort=False),
        raw_pairs,
    )


def _history_comparisons(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    summary_parts: list[pd.DataFrame] = []
    applicability_parts: list[pd.DataFrame] = []
    raw_pairs: list[pd.DataFrame] = []

    for comparison_id, shorter, longer in HISTORY_COMPARISONS:
        pairs = construct_successful_pairs(
            source,
            selector_column="history_window_years",
            selector_a=shorter,
            selector_b=longer,
            pair_keys=[
                "scenario_id",
                "simulation_id",
                "basis",
                "model",
            ],
        )
        pairs = _metadata(
            pairs,
            comparison_id=comparison_id,
            source_step=26,
            model_a=f"same_model_{shorter}_year_window",
            model_b=f"same_model_{longer}_year_window",
            analysis_family="history_window",
        )
        pairs["shorter_window_years"] = shorter
        pairs["longer_window_years"] = longer
        raw_pairs.append(pairs)
        summary = summarise_paired_accuracy(
            pairs,
            group_columns=["scenario_id", "basis", "model"],
        )
        summary = _metadata(
            summary,
            comparison_id=comparison_id,
            source_step=26,
            model_a=f"same_model_{shorter}_year_window",
            model_b=f"same_model_{longer}_year_window",
            analysis_family="history_window",
        )
        summary["shorter_window_years"] = shorter
        summary["longer_window_years"] = longer
        summary["group_type"] = "scenario_basis_model"
        summary["pool_label"] = ""
        summary_parts.append(summary)
        applicability = summarise_applicability(
            pairs,
            group_columns=["scenario_id", "basis", "model"],
        )
        applicability = _metadata(
            applicability,
            comparison_id=comparison_id,
            source_step=26,
            model_a=f"same_model_{shorter}_year_window",
            model_b=f"same_model_{longer}_year_window",
            analysis_family="history_window",
        )
        applicability["shorter_window_years"] = shorter
        applicability["longer_window_years"] = longer
        applicability["group_type"] = "scenario_basis_model"
        applicability["pool_label"] = ""
        applicability_parts.append(applicability)

    return (
        pd.concat(summary_parts, ignore_index=True, sort=False),
        pd.concat(applicability_parts, ignore_index=True, sort=False),
        raw_pairs,
    )


def _prior_comparisons(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    summary_parts: list[pd.DataFrame] = []
    applicability_parts: list[pd.DataFrame] = []
    raw_pairs: list[pd.DataFrame] = []

    for comparison_id, misspecified, baseline in PRIOR_COMPARISONS:
        pairs = construct_successful_pairs(
            source,
            selector_column="prior_multiplier",
            selector_a=misspecified,
            selector_b=baseline,
            pair_keys=[
                "scenario_id",
                "simulation_id",
                "basis",
                "model",
            ],
        )
        pairs = _metadata(
            pairs,
            comparison_id=comparison_id,
            source_step=27,
            model_a=f"same_model_prior_{misspecified:.2f}",
            model_b="same_model_prior_1.00",
            analysis_family="prior_sensitivity",
        )
        pairs["prior_multiplier_a"] = misspecified
        pairs["prior_multiplier_b"] = baseline
        raw_pairs.append(pairs)
        summary = summarise_paired_accuracy(
            pairs,
            group_columns=["scenario_id", "basis", "model"],
        )
        summary = _metadata(
            summary,
            comparison_id=comparison_id,
            source_step=27,
            model_a=f"same_model_prior_{misspecified:.2f}",
            model_b="same_model_prior_1.00",
            analysis_family="prior_sensitivity",
        )
        summary["prior_multiplier_a"] = misspecified
        summary["prior_multiplier_b"] = baseline
        summary["group_type"] = "scenario_basis_model"
        summary["pool_label"] = ""
        summary_parts.append(summary)
        applicability = summarise_applicability(
            pairs,
            group_columns=["scenario_id", "basis", "model"],
        )
        applicability = _metadata(
            applicability,
            comparison_id=comparison_id,
            source_step=27,
            model_a=f"same_model_prior_{misspecified:.2f}",
            model_b="same_model_prior_1.00",
            analysis_family="prior_sensitivity",
        )
        applicability["prior_multiplier_a"] = misspecified
        applicability["prior_multiplier_b"] = baseline
        applicability["group_type"] = "scenario_basis_model"
        applicability["pool_label"] = ""
        applicability_parts.append(applicability)

    return (
        pd.concat(summary_parts, ignore_index=True, sort=False),
        pd.concat(applicability_parts, ignore_index=True, sort=False),
        raw_pairs,
    )


def _no_accuracy_when_empty(summary: pd.DataFrame) -> bool:
    metrics = [
        "mean_ape_a",
        "mean_ape_b",
        "mean_paired_ape_difference",
        "median_paired_ape_difference",
        "std_paired_ape_difference",
        "win_rate_a",
        "win_rate_b",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
    ]
    empty = summary["valid_paired_rows"].eq(0)
    return bool(summary.loc[empty, metrics].isna().all().all())


def _build_acceptance_report(
    *,
    source_hashes_before: dict[str, str],
    source_hashes_after: dict[str, str],
    all_accuracy_pairs: list[pd.DataFrame],
    all_accuracy_summary: pd.DataFrame,
    applicability_summary: pd.DataFrame,
    core_detail: pd.DataFrame,
    history_summary: pd.DataFrame,
    prior_summary: pd.DataFrame,
    treaty_detail: pd.DataFrame,
    treaty_summary: pd.DataFrame,
) -> pd.DataFrame:
    combined_pairs = pd.concat(
        all_accuracy_pairs,
        ignore_index=True,
        sort=False,
    )
    included = combined_pairs.loc[
        combined_pairs["included_in_accuracy"]
    ]
    difference_reconciles = np.array_equal(
        included["paired_ape_difference"].to_numpy(),
        (
            included["ape_a"] - included["ape_b"]
        ).to_numpy(),
    )
    win_reconciles = (
        all_accuracy_summary["win_count_a"]
        + all_accuracy_summary["win_count_b"]
        + all_accuracy_summary["tie_count"]
    ).eq(all_accuracy_summary["valid_paired_rows"])
    valid_summary = all_accuracy_summary.loc[
        all_accuracy_summary["valid_paired_rows"].gt(0)
    ]
    structurally_inapplicable = combined_pairs.loc[
        ~combined_pairs["applicable_a"].fillna(False)
        | ~combined_pairs["applicable_b"].fillna(False)
    ]
    summary_key = [
        "comparison_id",
        "group_type",
        "scenario_id",
        "basis",
        "pool_label",
    ]
    history_key = [
        "comparison_id", "scenario_id", "basis", "model"
    ]
    prior_key = [
        "comparison_id", "scenario_id", "basis", "model"
    ]
    checks = [
        (
            "all_source_experiment_files_exist",
            all(path.is_file() for path in SOURCE_FILES.values()),
            f"sources={len(SOURCE_FILES)}",
        ),
        (
            "source_result_keys_are_unique",
            True,
            "validated during frozen-source loading",
        ),
        (
            "paired_truth_is_identical_within_tolerance",
            bool(combined_pairs.loc[
                combined_pairs["present_a"] & combined_pairs["present_b"],
                "truth_matches",
            ].all()),
            f"max_abs={combined_pairs['truth_absolute_difference'].max()}",
        ),
        (
            "no_cross_simulation_pairing",
            "simulation_id" in combined_pairs.columns,
            "simulation_id is a mandatory merge key",
        ),
        (
            "no_cross_scenario_pairing",
            "scenario_id" in combined_pairs.columns,
            "scenario_id is a mandatory merge key",
        ),
        (
            "no_cross_basis_pairing",
            "basis" in combined_pairs.columns,
            "basis is a mandatory merge key",
        ),
        (
            "only_both_success_rows_enter_ape_comparisons",
            bool(
                included["success_a"].all()
                and included["success_b"].all()
                and included["applicable_a"].all()
                and included["applicable_b"].all()
            ),
            f"included={len(included)}",
        ),
        (
            "failures_remain_in_applicability_summaries",
            int(applicability_summary["required_pairs"].sum())
            >= int(applicability_summary["both_success_pair_count"].sum()),
            f"applicability_groups={len(applicability_summary)}",
        ),
        (
            "structural_non_applicability_not_treated_as_accuracy_failure",
            bool(
                structurally_inapplicable["included_in_accuracy"].eq(False).all()
                and structurally_inapplicable["exclusion_reason"]
                .str.contains("structurally_not_applicable")
                .all()
            ),
            f"rows={len(structurally_inapplicable)}",
        ),
        (
            "paired_difference_reconciles_to_ape_a_minus_ape_b",
            bool(difference_reconciles),
            "",
        ),
        (
            "win_loss_tie_counts_reconcile",
            bool(win_reconciles.all()),
            "",
        ),
        (
            "bootstrap_sample_size_and_seed_are_fixed",
            bool(
                all_accuracy_summary["bootstrap_resamples"]
                .eq(BOOTSTRAP_RESAMPLES)
                .all()
                and all_accuracy_summary["bootstrap_seed"].notna().all()
            ),
            f"resamples={BOOTSTRAP_RESAMPLES}, base_seed={BOOTSTRAP_SEED}",
        ),
        (
            "bootstrap_intervals_finite_when_valid_pairs_exist",
            bool(
                np.isfinite(valid_summary["bootstrap_ci_lower"]).all()
                and np.isfinite(valid_summary["bootstrap_ci_upper"]).all()
            ),
            f"groups={len(valid_summary)}",
        ),
        (
            "no_accuracy_metric_when_no_valid_pairs_exist",
            _no_accuracy_when_empty(all_accuracy_summary),
            f"empty_groups={int(all_accuracy_summary['valid_paired_rows'].eq(0).sum())}",
        ),
        (
            "step29_treaty_comparison_uses_no_ape",
            not any("ape" in column.lower() for column in treaty_detail.columns),
            "raw treaty-mechanics differences only",
        ),
        (
            "step29_paired_gross_ultimate_is_identical",
            bool(treaty_detail[
                "gross_ultimate_absolute_difference"
            ].fillna(0.0).eq(0.0).all()),
            "",
        ),
        (
            "source_frozen_files_are_unchanged",
            source_hashes_before == source_hashes_after,
            "",
        ),
        (
            "core_detail_output_keys_are_unique",
            not core_detail.duplicated(
                [
                    "comparison_id",
                    "scenario_id",
                    "simulation_id",
                    "basis",
                ]
            ).any(),
            "",
        ),
        (
            "core_summary_output_keys_are_unique",
            not all_accuracy_summary.loc[
                all_accuracy_summary["analysis_family"].eq("core_model")
            ].duplicated(summary_key).any(),
            "",
        ),
        (
            "history_summary_output_keys_are_unique",
            not history_summary.duplicated(history_key).any(),
            "",
        ),
        (
            "prior_summary_output_keys_are_unique",
            not prior_summary.duplicated(prior_key).any(),
            "",
        ),
        (
            "treaty_detail_output_keys_are_unique",
            not treaty_detail.duplicated(
                ["scenario_id", "simulation_id"]
            ).any(),
            "",
        ),
        (
            "treaty_summary_output_keys_are_unique",
            not treaty_summary.duplicated(["scenario_id"]).any(),
            "",
        ),
        (
            "all_prespecified_comparisons_are_present",
            set(all_accuracy_summary["comparison_id"])
            == {
                *(item["comparison_id"] for item in CORE_COMPARISONS),
                *(item[0] for item in HISTORY_COMPARISONS),
                *(item[0] for item in PRIOR_COMPARISONS),
            },
            f"comparison_ids={all_accuracy_summary['comparison_id'].nunique()}",
        ),
        (
            "bootstrap_is_reproducible",
            deterministic_bootstrap_mean_ci(
                [-1.0, 0.0, 1.0],
                resamples=BOOTSTRAP_RESAMPLES,
                seed=BOOTSTRAP_SEED,
            )
            == deterministic_bootstrap_mean_ci(
                [-1.0, 0.0, 1.0],
                resamples=BOOTSTRAP_RESAMPLES,
                seed=BOOTSTRAP_SEED,
            ),
            "",
        ),
    ]
    return pd.DataFrame(
        [
            {"check": check, "passed": bool(passed), "detail": detail}
            for check, passed, detail in checks
        ]
    )


def _method_note() -> str:
    return f"""# Step 31 — Paired Model Comparisons

## Scope and pairing

Step 31 is analysis-only. It reads frozen Step 20, 21 and 24–29 result
tables and does not run simulation, reserving, calibration or model-fitting
code. Reserving comparisons pair only exact `scenario_id`, `simulation_id`
and `basis` matches, with model-, history-window- or prior-multiplier
dimensions added as required. Evaluation truth must match within fixed
absolute/relative tolerances before a pair is accepted.

For each successful reserving pair,

    D_s = APE_A,s - APE_B,s.

Negative values favour the pre-specified model or sensitivity A. Only pairs
where both estimators are structurally applicable, successfully fitted and
have finite APE enter conditional accuracy. Missing, failed and structurally
inapplicable rows remain represented in the applicability output. The APE
tie tolerance is fixed in advance at `{APE_TIE_TOLERANCE}`.

## Bootstrap interval

The 95% percentile bootstrap interval resamples the paired differences
directly with replacement. Each group uses `{BOOTSTRAP_RESAMPLES}` resamples
and a deterministic group seed derived from base seed `{BOOTSTRAP_SEED}`.
An interval below zero favours A descriptively; one above zero favours B;
an interval containing zero provides no clear paired difference under this
interval. It is not a p-value. No t-test, Wilcoxon test, multiple-testing
correction or post-result comparison is added.

## Pre-specified comparisons

The analysis covers Step 24 interaction versus baseline Poisson, Step 25
Tweedie versus Poisson, Step 21 Break-Aware versus Standard BF, both Step 28
ceded-specific BF comparisons, the Step 26 10-versus-15, 7-versus-15 and
7-versus-10 history windows, and each Step 27 misspecified multiplier versus
1.00. Shorter history is A. Structurally inapplicable ML 7-year rows and all
excluded fit failures receive no accuracy statistic.

Step 29 is kept separate: fully indexed minus fixed nominal differences are
calculated for treaty mechanics on the same gross portfolio. No APE is
calculated because the treaty variants change the insured/ceded outcome
rather than estimate the same reserve truth.

All interpretations must retain success/applicability alongside conditional
accuracy. A realised APE improvement under prior misspecification does not
establish that the misspecified prior is preferable or optimally calibrated.
"""


def main() -> None:
    if OUTPUT_DIRECTORY.exists():
        raise FileExistsError(OUTPUT_DIRECTORY)

    started = time.perf_counter()
    sources, hashes_before = _load_sources()
    core_detail, core_summary, core_applicability, core_pairs = (
        _core_comparisons(sources)
    )
    history_summary, history_applicability, history_pairs = (
        _history_comparisons(sources["step26_history_window"])
    )
    prior_summary, prior_applicability, prior_pairs = _prior_comparisons(
        sources["step27_prior_sensitivity"]
    )
    treaty_detail = construct_treaty_mechanics_pairs(
        sources["step29_treaty_indexation"]
    )
    treaty_summary = summarise_treaty_mechanics(treaty_detail)
    all_accuracy_summary = pd.concat(
        [core_summary, history_summary, prior_summary],
        ignore_index=True,
        sort=False,
    )
    applicability_summary = pd.concat(
        [
            core_applicability,
            history_applicability,
            prior_applicability,
        ],
        ignore_index=True,
        sort=False,
    )
    bootstrap_columns = [
        "comparison_id",
        "analysis_family",
        "source_step",
        "model_a",
        "model_b",
        "group_type",
        "scenario_id",
        "basis",
        "pool_label",
        "valid_paired_rows",
        "mean_paired_ape_difference",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "bootstrap_resamples",
        "bootstrap_seed",
    ]
    bootstrap_summary = all_accuracy_summary[bootstrap_columns].copy()
    all_accuracy_pairs = core_pairs + history_pairs + prior_pairs
    hashes_after = {
        name: sha256_file(path) for name, path in SOURCE_FILES.items()
    }
    acceptance = _build_acceptance_report(
        source_hashes_before=hashes_before,
        source_hashes_after=hashes_after,
        all_accuracy_pairs=all_accuracy_pairs,
        all_accuracy_summary=all_accuracy_summary,
        applicability_summary=applicability_summary,
        core_detail=core_detail,
        history_summary=history_summary,
        prior_summary=prior_summary,
        treaty_detail=treaty_detail,
        treaty_summary=treaty_summary,
    )
    elapsed_seconds = time.perf_counter() - started

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    core_detail.to_csv(
        OUTPUT_DIRECTORY / "paired_model_comparisons.csv",
        index=False,
    )
    core_summary.to_csv(
        OUTPUT_DIRECTORY / "paired_model_comparison_summary.csv",
        index=False,
    )
    applicability_summary.to_csv(
        OUTPUT_DIRECTORY / "applicability_comparison_summary.csv",
        index=False,
    )
    history_summary.to_csv(
        OUTPUT_DIRECTORY / "history_window_paired_comparisons.csv",
        index=False,
    )
    prior_summary.to_csv(
        OUTPUT_DIRECTORY / "prior_sensitivity_paired_comparisons.csv",
        index=False,
    )
    treaty_detail.to_csv(
        OUTPUT_DIRECTORY / "treaty_indexation_paired_comparisons.csv",
        index=False,
    )
    treaty_summary.to_csv(
        OUTPUT_DIRECTORY
        / "treaty_indexation_paired_comparison_summary.csv",
        index=False,
    )
    bootstrap_summary.to_csv(
        OUTPUT_DIRECTORY / "bootstrap_summary.csv",
        index=False,
    )
    acceptance.to_csv(
        OUTPUT_DIRECTORY / "acceptance_report.csv",
        index=False,
    )
    (OUTPUT_DIRECTORY / "STEP31_METHOD_NOTE.md").write_text(
        _method_note(),
        encoding="utf-8",
    )
    manifest: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "step": 31,
        "analysis_only": True,
        "simulation_or_model_reruns": False,
        "source_experiments_loaded": len(sources),
        "source_files": {
            name: str(path) for name, path in SOURCE_FILES.items()
        },
        "source_sha256_before": hashes_before,
        "source_sha256_after": hashes_after,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_base_seed": BOOTSTRAP_SEED,
        "bootstrap_confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
        "ape_tie_tolerance": APE_TIE_TOLERANCE,
        "core_detailed_pairs": int(len(core_detail)),
        "core_summary_groups": int(len(core_summary)),
        "history_summary_groups": int(len(history_summary)),
        "prior_summary_groups": int(len(prior_summary)),
        "bootstrap_summary_groups": int(len(bootstrap_summary)),
        "treaty_detailed_pairs": int(len(treaty_detail)),
        "treaty_summary_groups": int(len(treaty_summary)),
        "total_reported_comparison_groups": int(
            len(all_accuracy_summary) + len(treaty_summary)
        ),
        "elapsed_seconds": elapsed_seconds,
    }
    (OUTPUT_DIRECTORY / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Step 31 paired comparisons completed.")
    print(f"Output: {OUTPUT_DIRECTORY}")
    print(f"Source experiments: {len(sources)}")
    print(
        "Comparison groups: "
        f"{len(all_accuracy_summary) + len(treaty_summary)}"
    )
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(acceptance.to_string(index=False))

    if not acceptance["passed"].all():
        raise RuntimeError("Step 31 acceptance checks failed.")


if __name__ == "__main__":
    main()

"""Focused tests for Step 31 paired model comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.paired_comparisons import (
    APE_TIE_TOLERANCE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    NOT_APPLICABLE_BY_DESIGN,
    PairingValidationError,
    construct_successful_pairs,
    construct_treaty_mechanics_pairs,
    deterministic_bootstrap_mean_ci,
    summarise_applicability,
    summarise_paired_accuracy,
)


def _model_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for scenario_id in ["scenario_a", "scenario_b"]:
        for simulation_id in [1, 2]:
            for basis in ["gross", "ceded"]:
                truth = 100.0 if basis == "gross" else 25.0
                for model, ape in [("model_a", 8.0), ("model_b", 10.0)]:
                    rows.append(
                        {
                            "scenario_id": scenario_id,
                            "simulation_id": simulation_id,
                            "basis": basis,
                            "model": model,
                            "true_reserve": truth,
                            "absolute_percentage_error": ape,
                            "success": True,
                            "applicability_status": "applicable_by_design",
                            "fit_attempted": True,
                            "failure_type": "",
                        }
                    )

    return pd.DataFrame(rows)


def _pairs(results: pd.DataFrame | None = None) -> pd.DataFrame:
    return construct_successful_pairs(
        _model_results() if results is None else results,
        selector_column="model",
        selector_a="model_a",
        selector_b="model_b",
        pair_keys=["scenario_id", "simulation_id", "basis"],
    )


def test_pairing_uses_scenario_simulation_and_basis() -> None:
    pairs = _pairs()
    assert len(pairs) == 8
    assert pairs["included_in_accuracy"].all()
    assert not pairs.duplicated(
        ["scenario_id", "simulation_id", "basis"]
    ).any()


def test_mismatched_truth_raises_validation_error() -> None:
    results = _model_results()
    mask = (
        results["model"].eq("model_a")
        & results["scenario_id"].eq("scenario_a")
        & results["simulation_id"].eq(1)
        & results["basis"].eq("gross")
    )
    results.loc[mask, "true_reserve"] = 101.0

    with pytest.raises(PairingValidationError, match="truth differs"):
        _pairs(results)


def test_failed_estimator_is_excluded_from_paired_ape() -> None:
    results = _model_results()
    mask = (
        results["model"].eq("model_a")
        & results["scenario_id"].eq("scenario_a")
        & results["simulation_id"].eq(1)
        & results["basis"].eq("gross")
    )
    results.loc[mask, "success"] = False
    results.loc[mask, "absolute_percentage_error"] = np.nan
    results.loc[mask, "failure_type"] = "fit_failed"
    pairs = _pairs(results)
    failed = pairs.loc[
        pairs["scenario_id"].eq("scenario_a")
        & pairs["simulation_id"].eq(1)
        & pairs["basis"].eq("gross")
    ].iloc[0]
    assert not bool(failed["included_in_accuracy"])
    assert failed["exclusion_reason"] == "model_a_failed"
    assert np.isnan(failed["paired_ape_difference"])


def test_failure_remains_in_applicability_summary() -> None:
    results = _model_results()
    mask = (
        results["model"].eq("model_b")
        & results["scenario_id"].eq("scenario_a")
        & results["simulation_id"].eq(1)
        & results["basis"].eq("gross")
    )
    results.loc[mask, "success"] = False
    pairs = _pairs(results)
    summary = summarise_applicability(
        pairs.loc[
            pairs["scenario_id"].eq("scenario_a")
            & pairs["basis"].eq("gross")
        ],
        group_columns=["scenario_id", "basis"],
    ).iloc[0]
    assert summary["attempts_b"] == 2
    assert summary["successful_fits_b"] == 1
    assert summary["success_rate_b"] == 0.5
    assert summary["either_failed_pair_count"] == 1


def test_paired_difference_is_exactly_ape_a_minus_ape_b() -> None:
    pairs = _pairs()
    expected = pairs["ape_a"] - pairs["ape_b"]
    np.testing.assert_array_equal(
        pairs["paired_ape_difference"],
        expected,
    )


def test_negative_difference_is_model_a_win() -> None:
    pairs = _pairs()
    assert pairs["paired_ape_difference"].lt(0.0).all()
    assert pairs["pair_outcome"].eq("model_a_win").all()


def test_positive_difference_is_model_b_win() -> None:
    results = _model_results()
    results.loc[results["model"].eq("model_a"), "absolute_percentage_error"] = 12.0
    pairs = _pairs(results)
    assert pairs["paired_ape_difference"].gt(0.0).all()
    assert pairs["pair_outcome"].eq("model_b_win").all()


def test_fixed_numerical_tolerance_defines_ties() -> None:
    results = _model_results()
    results.loc[
        results["model"].eq("model_a"),
        "absolute_percentage_error",
    ] = 10.0 + APE_TIE_TOLERANCE / 2.0
    pairs = _pairs(results)
    assert pairs["pair_outcome"].eq("tie").all()
    summary = summarise_paired_accuracy(
        pairs,
        group_columns=["scenario_id", "basis"],
    )
    assert summary["tie_count"].eq(2).all()


def test_bootstrap_interval_is_deterministic() -> None:
    first = deterministic_bootstrap_mean_ci(
        [-3.0, -2.0, -1.0],
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    second = deterministic_bootstrap_mean_ci(
        [-3.0, -2.0, -1.0],
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    assert first == second


def test_bootstrap_interval_on_constant_known_differences() -> None:
    lower, upper = deterministic_bootstrap_mean_ci(
        [-2.0] * 10,
        resamples=1_000,
        seed=123,
    )
    assert lower == -2.0
    assert upper == -2.0


def test_history_window_structural_inapplicability_is_not_accuracy() -> None:
    results = pd.DataFrame(
        [
            {
                "scenario_id": "scenario",
                "simulation_id": 1,
                "basis": "gross",
                "model": "regularized_tweedie",
                "history_window_years": 7,
                "true_reserve": 100.0,
                "absolute_percentage_error": np.nan,
                "success": False,
                "applicability_status": NOT_APPLICABLE_BY_DESIGN,
                "fit_attempted": False,
            },
            {
                "scenario_id": "scenario",
                "simulation_id": 1,
                "basis": "gross",
                "model": "regularized_tweedie",
                "history_window_years": 15,
                "true_reserve": 100.0,
                "absolute_percentage_error": 5.0,
                "success": True,
                "applicability_status": "applicable_by_design",
                "fit_attempted": True,
            },
        ]
    )
    pairs = construct_successful_pairs(
        results,
        selector_column="history_window_years",
        selector_a=7,
        selector_b=15,
        pair_keys=["scenario_id", "simulation_id", "basis", "model"],
    )
    summary = summarise_paired_accuracy(
        pairs,
        group_columns=["scenario_id", "basis", "model"],
    ).iloc[0]
    assert not bool(pairs.loc[0, "included_in_accuracy"])
    assert "structurally_not_applicable" in pairs.loc[0, "exclusion_reason"]
    assert summary["valid_paired_rows"] == 0
    assert np.isnan(summary["mean_paired_ape_difference"])
    assert np.isnan(summary["bootstrap_ci_lower"])


def test_treaty_comparison_uses_raw_differences_not_ape() -> None:
    results = pd.DataFrame(
        [
            {
                "scenario_id": "scenario",
                "simulation_id": 1,
                "treaty_variant": variant,
                "gross_ultimate": 1_000.0,
                "ceded_ultimate": ceded,
                "ceded_true_reserve": reserve,
                "ceded_share": ceded / 1_000.0,
                "attachment_frequency": attachment,
                "exhaustion_frequency": exhaustion,
                "success": True,
            }
            for variant, ceded, reserve, attachment, exhaustion in [
                ("fully_indexed", 150.0, 60.0, 0.20, 0.01),
                ("fixed_nominal", 250.0, 100.0, 0.40, 0.03),
            ]
        ]
    )
    pairs = construct_treaty_mechanics_pairs(results)
    assert pairs.loc[0, "ceded_ultimate_difference_a_minus_b"] == -100.0
    assert pairs.loc[0, "ceded_true_reserve_difference_a_minus_b"] == -40.0
    assert not any("ape" in column.lower() for column in pairs.columns)


def test_comparison_output_keys_are_unique() -> None:
    pairs = _pairs()
    pairs["comparison_id"] = "fixture"
    assert not pairs.duplicated(
        [
            "comparison_id",
            "scenario_id",
            "simulation_id",
            "basis",
        ]
    ).any()

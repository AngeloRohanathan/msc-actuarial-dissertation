"""Focused tests for Step 32 final-results consolidation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_final_analysis_dataset import (
    SOURCE_BY_ID,
    _data_dictionary,
    _portfolio_tables,
)
from src.final_results_consolidation import (
    BASELINE_MODELS,
    ConsolidationValidationError,
    canonical_model_name,
    classify_ci_direction,
    ensure_data_dictionary_complete,
    harmonise_portfolio_results,
    harmonise_treaty_sensitivity,
    reserve_metric_mismatch_counts,
    sha256_file,
    validate_unique_keys,
)


def _portfolio_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_id": "short_stable_no_break",
                "simulation_id": 1,
                "seed": 100,
                "model": "regularized_tweedie",
                "basis": "gross",
                "success": True,
                "success_or_failure": "success",
                "applicability_status": "applicable_by_design",
                "fit_attempted": True,
                "estimated_reserve": 110.0,
                "true_reserve": 100.0,
                "signed_error": 10.0,
                "percentage_error": 10.0,
                "absolute_percentage_error": 10.0,
                "runtime_seconds": 0.2,
                "tail_type": "short",
                "inflation_scenario": "stable",
                "structural_break": False,
                "history_window_years": 10,
                "history_start_ay": 2015,
                "selected_alpha": 0.1,
                "selected_power": 1.5,
            }
        ]
    )


def _harmonise(
    frame: pd.DataFrame,
    *,
    analysis_type: str = "baseline_model",
) -> pd.DataFrame:
    return harmonise_portfolio_results(
        frame,
        experiment_step=99,
        experiment_name="fixture",
        source_file="fixture.csv",
        analysis_type=analysis_type,
    )


def test_canonical_model_mapping() -> None:
    assert canonical_model_name("Regularised Poisson") == "regularized_poisson"
    assert canonical_model_name("regularized-tweedie") == "regularized_tweedie"
    with pytest.raises(ConsolidationValidationError, match="Unknown model"):
        canonical_model_name("oracle_model")


def test_baseline_duplicate_detection() -> None:
    frame = pd.DataFrame(
        [
            {"scenario_id": "s", "simulation_id": 1, "basis": "gross", "model": "m"},
            {"scenario_id": "s", "simulation_id": 1, "basis": "gross", "model": "m"},
        ]
    )
    with pytest.raises(ConsolidationValidationError, match="duplicate keys"):
        validate_unique_keys(
            frame,
            ["scenario_id", "simulation_id", "basis", "model"],
            source_name="fixture",
        )


def test_history_sensitivity_fields_are_preserved() -> None:
    result = _harmonise(_portfolio_fixture(), analysis_type="history_sensitivity")
    assert result.loc[0, "history_window_years"] == 10
    assert result.loc[0, "history_start_ay"] == 2015
    assert result.loc[0, "selected_power"] == 1.5
    assert np.isnan(result.loc[0, "prior_multiplier"])


def test_missing_sensitivity_fields_remain_null() -> None:
    result = _harmonise(_portfolio_fixture(), analysis_type="baseline_model")
    assert pd.isna(result.loc[0, "history_window_years"])
    assert pd.isna(result.loc[0, "prior_multiplier"])
    assert pd.isna(result.loc[0, "pattern_variant"])
    assert pd.isna(result.loc[0, "treaty_variant"])


def test_failure_and_structural_inapplicability_are_distinct() -> None:
    fixture = pd.concat([_portfolio_fixture()] * 2, ignore_index=True)
    fixture.loc[:, "model"] = "chain_ladder"
    fixture.loc[:, "success"] = False
    fixture.loc[:, "success_or_failure"] = "failed"
    fixture.loc[:, "estimated_reserve"] = np.nan
    fixture.loc[:, "signed_error"] = np.nan
    fixture.loc[:, "percentage_error"] = np.nan
    fixture.loc[:, "absolute_percentage_error"] = np.nan
    fixture.loc[:, "failure_type"] = ["fit_failed", "not_applicable_by_design"]
    fixture.loc[:, "failure_message"] = ["failed", "designed skip"]
    fixture.loc[1, "applicability_status"] = "not_applicable_by_design"
    fixture.loc[1, "fit_attempted"] = False
    result = _harmonise(fixture, analysis_type="history_sensitivity")
    assert result.loc[0, "status"] == "failed"
    assert bool(result.loc[0, "applicable"])
    assert result.loc[1, "status"] == "not_applicable_by_design"
    assert not bool(result.loc[1, "applicable"])
    assert not bool(result.loc[1, "fit_attempted"])


def test_reserve_error_reconciliation() -> None:
    result = _harmonise(_portfolio_fixture())
    counts = reserve_metric_mismatch_counts(result)
    assert counts["reserve_error_mismatches"] == 0
    result.loc[0, "reserve_error"] = 9.0
    counts = reserve_metric_mismatch_counts(result)
    assert counts["reserve_error_mismatches"] == 1


def test_percentage_error_and_ape_reconciliation() -> None:
    result = _harmonise(_portfolio_fixture())
    counts = reserve_metric_mismatch_counts(result)
    assert counts["percentage_error_mismatches"] == 0
    assert counts["absolute_percentage_error_mismatches"] == 0
    result.loc[0, "absolute_percentage_error"] = 11.0
    counts = reserve_metric_mismatch_counts(result)
    assert counts["absolute_percentage_error_mismatches"] == 1


def test_treaty_master_excludes_estimator_ape() -> None:
    portfolio = pd.DataFrame(
        [
            {
                "scenario_id": "short_stable_no_break",
                "simulation_id": 1,
                "seed": 100,
                "treaty_variant": "fully_indexed",
                "tail_type": "short",
                "inflation_scenario": "stable",
                "structural_break": False,
                "gross_ultimate": 1000.0,
                "ceded_ultimate": 200.0,
                "retained_ultimate": 800.0,
                "ceded_share": 0.2,
                "attachment_frequency": 0.1,
                "exhaustion_frequency": 0.01,
                "gross_true_reserve": 100.0,
                "ceded_true_reserve": 20.0,
                "retained_true_reserve": 80.0,
                "base_attachment": 2.0,
                "base_limit": 5.0,
                "index_reference_year": 2010,
                "success": True,
                "failure_message": "",
                "runtime_seconds": 0.1,
            }
        ]
    )
    detail = pd.DataFrame(
        [
            {
                "scenario_id": "short_stable_no_break",
                "simulation_id": 1,
                "treaty_variant": "fully_indexed",
                "accident_year": 2010,
                "applied_index_factor": 1.0,
            }
        ]
    )
    result = harmonise_treaty_sensitivity(
        portfolio,
        detail,
        source_file="fixture.csv",
    )
    assert not any("percentage_error" in column for column in result.columns)
    assert result.loc[0, "mean_applied_index_factor"] == 1.0


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (-2.0, -0.1, "favors_A"),
        (0.1, 2.0, "favors_B"),
        (-1.0, 1.0, "includes_zero"),
        (np.nan, np.nan, "not_available"),
    ],
)
def test_step31_ci_direction_classification(
    lower: float,
    upper: float,
    expected: str,
) -> None:
    assert classify_ci_direction(lower, upper) == expected


def test_source_hash_stability(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    before = sha256_file(source)
    pd.read_csv(source)
    after = sha256_file(source)
    assert before == after


def test_unique_keys_accept_distinct_sensitivity_rows() -> None:
    frame = pd.DataFrame(
        [
            {"scenario_id": "s", "simulation_id": 1, "window": 15},
            {"scenario_id": "s", "simulation_id": 1, "window": 10},
        ]
    )
    validate_unique_keys(
        frame,
        ["scenario_id", "simulation_id", "window"],
        source_name="fixture",
    )


def test_data_dictionary_contains_every_output_field() -> None:
    fixture = pd.DataFrame({"scenario_id": ["s"], "true_reserve": [100.0]})
    datasets = {"fixture.csv": fixture}
    dictionary = _data_dictionary(datasets)
    ensure_data_dictionary_complete(dictionary, datasets)
    assert set(dictionary["field_name"]) == {"scenario_id", "true_reserve"}


def test_authoritative_sources_prevent_duplicated_baselines() -> None:
    source_ids = [
        "step16_baseline_results",
        "step20_expected_loss_results",
        "step21_paid_bf_results",
        "step24_poisson_interaction_results",
        "step25_tweedie_results",
        "step26_history_results",
        "step27_prior_results",
        "step28_ceded_bf_results",
    ]
    frames = {
        source_id: pd.read_csv(SOURCE_BY_ID[source_id].path)
        for source_id in source_ids
    }
    _, baseline = _portfolio_tables(frames)
    key = ["scenario_id", "simulation_id", "basis", "model"]
    assert len(baseline) == 8100
    assert not baseline.duplicated(key).any()
    assert set(baseline["model"]) == set(BASELINE_MODELS)
    assert baseline.groupby("model").size().eq(900).all()

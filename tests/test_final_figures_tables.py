"""Focused tests for Step 33 final figures and tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.final_figures_tables import (
    BREAK_SCENARIOS,
    FIGURE_OUTPUTS,
    FINAL_ANALYSIS_DIRECTORY,
    MAIN_TABLE_FILES,
    MODEL_ORDER,
    SCENARIO_ORDER,
    STEP32_SOURCE_FILES,
    PresentationValidationError,
    baseline_scenario_summary,
    paired_interpretation,
    percentage_unit_for_field,
    treaty_scenario_summary,
    validate_nonapplicable_accuracy,
    validate_output_index,
    validate_step32_source_scope,
)


def _baseline_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_id": "short_stable_no_break",
                "simulation_id": 1,
                "model": "chain_ladder",
                "basis": "gross",
                "success": True,
                "percentage_error": -10.0,
                "absolute_percentage_error": 10.0,
            },
            {
                "scenario_id": "short_stable_no_break",
                "simulation_id": 2,
                "model": "chain_ladder",
                "basis": "gross",
                "success": True,
                "percentage_error": 20.0,
                "absolute_percentage_error": 20.0,
            },
            {
                "scenario_id": "short_stable_no_break",
                "simulation_id": 3,
                "model": "chain_ladder",
                "basis": "gross",
                "success": False,
                "percentage_error": np.nan,
                "absolute_percentage_error": np.nan,
            },
        ]
    )


def _valid_output_index(root: Path) -> pd.DataFrame:
    backing = root / "backing.csv"
    image = root / "figure.png"
    backing.write_text("x\n1\n", encoding="utf-8")
    image.write_bytes(b"png")
    return pd.DataFrame(
        [
            {
                "output_id": "figure01",
                "output_type": "main_figure",
                "title": "Title",
                "research_question": "Question",
                "key_message": "Message",
                "main_caveat": "Caveat",
                "backing_csv": "backing.csv",
                "image_or_table_path": "figure.png",
                "recommended_location": "Results",
            }
        ]
    )


def test_step33_sources_are_only_final_analysis_files() -> None:
    validate_step32_source_scope()
    assert all(path.parent == FINAL_ANALYSIS_DIRECTORY for path in STEP32_SOURCE_FILES.values())


def test_no_frozen_experiment_directories_are_declared() -> None:
    assert not any(
        any(part.startswith(f"step{step}") for step in range(16, 32) for part in path.parts)
        for path in STEP32_SOURCE_FILES.values()
    )


def test_canonical_model_order_is_fixed() -> None:
    assert MODEL_ORDER == (
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


def test_scenario_order_is_fixed() -> None:
    assert SCENARIO_ORDER[0] == "short_stable_no_break"
    assert SCENARIO_ORDER[-1] == "long_shock_break"
    assert len(SCENARIO_ORDER) == len(set(SCENARIO_ORDER)) == 9


def test_break_scenarios_exclude_no_break_labels() -> None:
    assert BREAK_SCENARIOS == (
        "long_stable_break",
        "long_emerging_break",
        "long_shock_break",
    )
    assert not any("no_break" in scenario for scenario in BREAK_SCENARIOS)


def test_all_expected_figure_backing_names_are_declared() -> None:
    assert set(FIGURE_OUTPUTS) == {
        "figure01",
        "figure02",
        "figure03",
        "figure04",
        "figure05",
        "figure06",
        "figure07",
        "figure08",
    }


def test_all_main_figure_stems_are_unique() -> None:
    assert len(FIGURE_OUTPUTS.values()) == len(set(FIGURE_OUTPUTS.values())) == 8


def test_all_main_table_files_are_declared_and_unique() -> None:
    assert len(MAIN_TABLE_FILES) == len(set(MAIN_TABLE_FILES)) == 7
    assert all(filename.endswith(".csv") for filename in MAIN_TABLE_FILES)


def test_conditional_backing_data_reconcile_to_step32_rows() -> None:
    summary = baseline_scenario_summary(_baseline_fixture()).iloc[0]
    assert summary["attempted_fits"] == 3
    assert summary["successful_fits"] == 2
    assert summary["success_rate_pct"] == pytest.approx(200.0 / 3.0)
    assert summary["mean_percentage_error_pct"] == 5.0
    assert summary["mean_absolute_percentage_error_pct"] == 15.0


def test_treaty_mechanics_reject_estimator_ape() -> None:
    treaty = pd.DataFrame(
        {
            "scenario_id": ["short_stable_no_break"],
            "treaty_variant": ["fixed_nominal"],
            "percentage_error": [1.0],
        }
    )
    with pytest.raises(PresentationValidationError, match="must not contain"):
        treaty_scenario_summary(treaty)


def test_nonapplicable_rows_are_not_assigned_accuracy() -> None:
    frame = pd.DataFrame(
        {
            "applicable": [False],
            "percentage_error": [np.nan],
            "absolute_percentage_error": [np.nan],
            "estimated_reserve": [np.nan],
        }
    )
    validate_nonapplicable_accuracy(frame)
    frame.loc[0, "absolute_percentage_error"] = 0.0
    with pytest.raises(PresentationValidationError, match="contain accuracy"):
        validate_nonapplicable_accuracy(frame)


@pytest.mark.parametrize(
    ("direction", "label"),
    [
        ("favors_A", "Favours A"),
        ("favors_B", "Favours B"),
        ("includes_zero", "Includes zero"),
        ("not_available", "Not available"),
    ],
)
def test_paired_ci_interpretation_labels(direction: str, label: str) -> None:
    assert paired_interpretation(direction) == label


def test_percentage_and_percentage_point_labels_are_distinct() -> None:
    assert percentage_unit_for_field("mape_pct") == "%"
    assert percentage_unit_for_field("mean_paired_difference") == "percentage points"
    assert percentage_unit_for_field("ceded_share_change") == "percentage points"


def test_output_index_references_valid_files(tmp_path: Path) -> None:
    index = _valid_output_index(tmp_path)
    validate_output_index(index, tmp_path)


def test_output_index_rejects_duplicate_ids(tmp_path: Path) -> None:
    index = _valid_output_index(tmp_path)
    index = pd.concat([index, index], ignore_index=True)
    with pytest.raises(PresentationValidationError, match="unique"):
        validate_output_index(index, tmp_path)


def test_output_index_rejects_empty_main_output(tmp_path: Path) -> None:
    index = _valid_output_index(tmp_path)
    (tmp_path / "figure.png").write_bytes(b"")
    with pytest.raises(PresentationValidationError, match="missing"):
        validate_output_index(index, tmp_path)

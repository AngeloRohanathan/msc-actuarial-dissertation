"""Focused tests for Step 26 historical data-window sensitivity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import (
    END_TO_END_SCENARIOS,
    ML_MIN_TRAINING_DIAGONALS,
    ML_MIN_VALIDATION_FOLDS,
    ML_POISSON_ALPHA_GRID,
    ML_TWEEDIE_ALPHA_GRID,
    ML_TWEEDIE_POWER_GRID,
)
from scripts import run_history_window_sensitivity as runner
from src.history_window_sensitivity import (
    APPLICABLE_BY_DESIGN,
    EVALUATION_ACCIDENT_YEARS,
    EVALUATION_END_AY,
    EVALUATION_START_AY,
    HISTORY_WINDOWS,
    NOT_APPLICABLE_BY_DESIGN,
    aggregate_accident_year_amount,
    assess_model_applicability,
    calculate_true_target_values,
    filter_triangle_to_history_window,
)


def create_triangle() -> pd.DataFrame:
    triangle = pd.DataFrame(
        np.nan,
        index=range(2010, 2025),
        columns=range(1, 11),
        dtype=float,
    )
    for accident_year in triangle.index:
        for development_year in triangle.columns:
            if accident_year + development_year - 1 <= 2024:
                triangle.loc[accident_year, development_year] = float(
                    accident_year + development_year
                )
    triangle.index.name = "accident_year"
    triangle.columns.name = "development_year"
    return triangle


def create_truth_table() -> pd.DataFrame:
    accident_years = list(range(2010, 2025))
    data = pd.DataFrame({"accident_year": accident_years})
    for basis, multiplier in (("gross", 10.0), ("ceded", 2.0)):
        observed = np.array(accident_years, dtype=float) * multiplier
        reserve = np.arange(1, 16, dtype=float) * multiplier
        data[f"observed_{basis}_paid"] = observed
        data[f"true_{basis}_reserve"] = reserve
        data[f"{basis}_ultimate"] = observed + reserve
    return data


@pytest.mark.parametrize(
    ("window_years", "expected_start"),
    [(15, 2010), (10, 2015), (7, 2018), (5, 2020)],
)
def test_frozen_history_window_retains_expected_accident_years(
    window_years: int,
    expected_start: int,
) -> None:
    filtered = filter_triangle_to_history_window(
        create_triangle(),
        HISTORY_WINDOWS[window_years],
        2024,
    )
    assert list(filtered.index) == list(range(expected_start, 2025))


def test_history_filter_does_not_mutate_input() -> None:
    triangle = create_triangle()
    original = triangle.copy(deep=True)
    filtered = filter_triangle_to_history_window(triangle, 2018, 2024)

    pd.testing.assert_frame_equal(triangle, original)
    assert filtered is not triangle


def test_evaluation_target_is_always_ay2020_2024() -> None:
    assert EVALUATION_START_AY == 2020
    assert EVALUATION_END_AY == 2024
    assert EVALUATION_ACCIDENT_YEARS == tuple(range(2020, 2025))


def test_truth_is_identical_across_history_windows() -> None:
    truth = create_truth_table()
    values = [
        calculate_true_target_values(truth, "gross")["true_target_reserve"]
        for _ in HISTORY_WINDOWS
    ]
    assert len(set(values)) == 1


def test_estimated_target_reserve_sums_only_ay2020_2024() -> None:
    reserve_table = pd.DataFrame(
        {
            "accident_year": range(2010, 2025),
            "estimated_reserve": [
                1_000_000.0 if year < 2020 else float(year - 2019)
                for year in range(2010, 2025)
            ],
        }
    )
    assert aggregate_accident_year_amount(
        reserve_table,
        "estimated_reserve",
    ) == 15.0


def test_model_fitting_receives_no_pre_window_accident_years(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_triangle = create_triangle().loc[:, range(1, 7)]
    incremental = filter_triangle_to_history_window(full_triangle, 2018, 2024)
    cumulative = incremental.cumsum(axis=1).where(incremental.notna())
    captured: dict[str, int] = {}

    def fake_fit_frozen_model(**kwargs: object) -> dict[str, object]:
        supplied = kwargs["incremental_triangle"]
        assert isinstance(supplied, pd.DataFrame)
        captured["minimum_ay"] = int(min(supplied.index))

        reserve_by_ay = pd.DataFrame(
            {
                "accident_year": list(range(2018, 2025)),
                "estimated_reserve": [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        future = pd.DataFrame(
            0.0,
            index=range(2018, 2025),
            columns=range(1, 7),
        )
        for position, accident_year in enumerate(range(2020, 2025), start=1):
            future.loc[accident_year, 1] = float(position)
        return {
            "reserve_by_accident_year": reserve_by_ay,
            "projected_future_incremental": future,
            "summary": pd.DataFrame(
                [{"total_estimated_reserve": 15.0}]
            ),
        }

    monkeypatch.setattr(runner, "fit_frozen_model", fake_fit_frozen_model)
    scenario = dict(END_TO_END_SCENARIOS[0])
    row = runner.run_model_attempt(
        model_name="chain_ladder",
        basis="gross",
        history_window_years=7,
        history_start_ay=2018,
        simulation_id=1,
        seed=123,
        scenario=scenario,
        inflation_index={year: 1.0 for year in range(2010, 2035)},
        incremental_triangle=incremental,
        cumulative_triangle=cumulative,
        true_reserves_by_accident_year=create_truth_table(),
        pipeline_reconciliation_passed=True,
    )

    assert captured["minimum_ay"] == 2018
    assert row["fit_received_pre_window_ay"] is False
    assert row["success"] is True


def test_failure_rows_remain_in_summary_and_failure_table() -> None:
    results = pd.DataFrame(
        [
            {
                "scenario_id": "scenario",
                "basis": "gross",
                "model": "chain_ladder",
                "history_window_years": 5,
                "history_start_ay": 2020,
                    "success": True,
                    "applicability_status": APPLICABLE_BY_DESIGN,
                    "fit_attempted": True,
                "runtime_seconds": 1.0,
                "signed_error": 1.0,
                "percentage_error": 2.0,
                "absolute_percentage_error": 2.0,
                "failure_type": "",
                "failure_message": "",
            },
            {
                "scenario_id": "scenario",
                "basis": "gross",
                "model": "chain_ladder",
                "history_window_years": 5,
                "history_start_ay": 2020,
                    "success": False,
                    "applicability_status": APPLICABLE_BY_DESIGN,
                    "fit_attempted": True,
                "runtime_seconds": 0.5,
                "signed_error": np.nan,
                "percentage_error": np.nan,
                "absolute_percentage_error": np.nan,
                "failure_type": "insufficient_development_history",
                "failure_message": "ValueError: no paired observations",
            },
        ]
    )
    summary = runner.build_history_window_summary(results)
    failures = runner.build_step26_failure_summary(results)

    assert summary.loc[0, "attempted_fits"] == 2
    assert summary.loc[0, "successful_fits"] == 1
    assert summary.loc[0, "success_rate"] == 0.5
    assert failures.loc[0, "failure_count"] == 1


@pytest.mark.parametrize(
    ("model_name", "window_years"),
    [
        ("regularized_poisson", 5),
        ("regularized_poisson", 7),
        ("regularized_poisson_break_interaction", 5),
        ("regularized_poisson_break_interaction", 7),
        ("regularized_tweedie", 5),
        ("regularized_tweedie", 7),
    ],
)
def test_ml_short_windows_are_structurally_inapplicable(
    model_name: str,
    window_years: int,
) -> None:
    triangle = filter_triangle_to_history_window(
        create_triangle(), HISTORY_WINDOWS[window_years], 2024
    )
    assessment = assess_model_applicability(
        model_name=model_name,
        triangle=triangle,
        minimum_training_diagonals=ML_MIN_TRAINING_DIAGONALS,
        minimum_validation_folds=ML_MIN_VALIDATION_FOLDS,
    )
    assert assessment.status == NOT_APPLICABLE_BY_DESIGN
    assert assessment.fit_should_be_attempted is False


@pytest.mark.parametrize(
    ("window_years", "maximum_development_year"),
    [
        (5, 6),
        (7, 6),
        (7, 15),
        (10, 6),
        (10, 15),
        (15, 15),
    ],
)
def test_classical_rows_are_not_pre_skipped(
    window_years: int,
    maximum_development_year: int,
) -> None:
    triangle = create_triangle().reindex(columns=range(1, maximum_development_year + 1))
    for accident_year in triangle.index:
        for development_year in triangle.columns:
            if accident_year + development_year - 1 <= 2024:
                triangle.loc[accident_year, development_year] = 1.0
    triangle = filter_triangle_to_history_window(
        triangle, HISTORY_WINDOWS[window_years], 2024
    )
    assessment = assess_model_applicability(
        model_name="chain_ladder",
        triangle=triangle,
        minimum_training_diagonals=ML_MIN_TRAINING_DIAGONALS,
        minimum_validation_folds=ML_MIN_VALIDATION_FOLDS,
    )
    assert assessment.status == APPLICABLE_BY_DESIGN
    assert assessment.fit_should_be_attempted is True


def test_not_applicable_row_is_retained_without_invoking_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incremental = filter_triangle_to_history_window(create_triangle(), 2018, 2024)
    cumulative = incremental.cumsum(axis=1).where(incremental.notna())

    def fail_if_called(**kwargs: object) -> dict[str, object]:
        raise AssertionError("Estimator must not be invoked.")

    monkeypatch.setattr(runner, "fit_frozen_model", fail_if_called)
    row = runner.run_model_attempt(
        model_name="regularized_tweedie",
        basis="gross",
        history_window_years=7,
        history_start_ay=2018,
        simulation_id=1,
        seed=123,
        scenario=dict(END_TO_END_SCENARIOS[0]),
        inflation_index={year: 1.0 for year in range(2010, 2035)},
        incremental_triangle=incremental,
        cumulative_triangle=cumulative,
        true_reserves_by_accident_year=create_truth_table(),
        pipeline_reconciliation_passed=True,
    )
    assert row["success"] is False
    assert row["success_or_failure"] == NOT_APPLICABLE_BY_DESIGN
    assert row["fit_attempted"] is False
    assert np.isnan(row["percentage_error"])


def test_data_dependent_failure_is_not_pre_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incremental = filter_triangle_to_history_window(
        create_triangle().loc[:, range(1, 7)], 2020, 2024
    )
    cumulative = incremental.cumsum(axis=1).where(incremental.notna())
    called = {"value": False}

    def fail_with_zero_denominator(**kwargs: object) -> dict[str, object]:
        called["value"] = True
        raise ValueError("The Chain Ladder denominator is zero for development 1 to 2.")

    monkeypatch.setattr(runner, "fit_frozen_model", fail_with_zero_denominator)
    row = runner.run_model_attempt(
        model_name="chain_ladder",
        basis="ceded",
        history_window_years=5,
        history_start_ay=2020,
        simulation_id=1,
        seed=123,
        scenario=dict(END_TO_END_SCENARIOS[0]),
        inflation_index={year: 1.0 for year in range(2010, 2035)},
        incremental_triangle=incremental,
        cumulative_triangle=cumulative,
        true_reserves_by_accident_year=create_truth_table(),
        pipeline_reconciliation_passed=True,
    )
    assert called["value"] is True
    assert row["fit_attempted"] is True
    assert row["applicability_status"] == APPLICABLE_BY_DESIGN
    assert row["success"] is False


def test_not_applicable_rows_are_in_denominator_but_not_accuracy() -> None:
    results = pd.DataFrame(
        [
            {
                "scenario_id": "scenario",
                "basis": "gross",
                "model": "regularized_tweedie",
                "history_window_years": 7,
                "history_start_ay": 2018,
                "success": True,
                "applicability_status": APPLICABLE_BY_DESIGN,
                "fit_attempted": True,
                "runtime_seconds": 1.0,
                "signed_error": 10.0,
                "percentage_error": 10.0,
                "absolute_percentage_error": 10.0,
            },
            {
                "scenario_id": "scenario",
                "basis": "gross",
                "model": "regularized_tweedie",
                "history_window_years": 7,
                "history_start_ay": 2018,
                "success": False,
                "applicability_status": APPLICABLE_BY_DESIGN,
                "fit_attempted": True,
                "runtime_seconds": 0.5,
                "signed_error": np.nan,
                "percentage_error": np.nan,
                "absolute_percentage_error": np.nan,
            },
            {
                "scenario_id": "scenario",
                "basis": "gross",
                "model": "regularized_tweedie",
                "history_window_years": 7,
                "history_start_ay": 2018,
                "success": False,
                "applicability_status": NOT_APPLICABLE_BY_DESIGN,
                "fit_attempted": False,
                "runtime_seconds": 0.0,
                "signed_error": np.nan,
                "percentage_error": np.nan,
                "absolute_percentage_error": np.nan,
            },
        ]
    )
    summary = runner.build_history_window_summary(results).iloc[0]
    assert summary["attempted_fits"] == 3
    assert summary["applicable_rows"] == 2
    assert summary["not_applicable_by_design"] == 1
    assert summary["failed_fits"] == 1
    assert summary["success_rate"] == pytest.approx(1 / 3)
    assert summary["mean_absolute_percentage_error"] == 10.0


def build_complete_synthetic_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in END_TO_END_SCENARIOS:
        for basis in runner.BASES:
            true_target = 100.0 if basis == "gross" else 25.0
            for window_years, start_ay in HISTORY_WINDOWS.items():
                for model in runner.MODEL_NAMES:
                    selected_alpha = (
                        ML_POISSON_ALPHA_GRID[0]
                        if model.startswith("regularized_poisson")
                        else np.nan
                    )
                    selected_power = np.nan
                    if model == "regularized_tweedie":
                        selected_alpha = ML_TWEEDIE_ALPHA_GRID[0]
                        selected_power = ML_TWEEDIE_POWER_GRID[0]
                    rows.append(
                        {
                            "scenario_id": scenario["scenario_id"],
                            "simulation_id": 1,
                            "basis": basis,
                            "model": model,
                            "history_window_years": window_years,
                            "history_start_ay": start_ay,
                            "evaluation_start_ay": 2020,
                            "evaluation_end_ay": 2024,
                            "success": True,
                            "applicability_status": APPLICABLE_BY_DESIGN,
                            "fit_attempted": True,
                            "runtime_seconds": 1.0,
                            "failure_type": "",
                            "failure_message": "",
                            "true_target_reserve": true_target,
                            "estimated_target_reserve": true_target,
                            "signed_error": 0.0,
                            "percentage_error": 0.0,
                            "absolute_percentage_error": 0.0,
                            "selected_alpha": selected_alpha,
                            "selected_power": selected_power,
                            "fit_received_pre_window_ay": False,
                            "fit_min_accident_year": start_ay,
                            "future_cells_hidden_from_fit": True,
                            "evaluation_truth_used_for_fitting": False,
                            "target_estimate_reconciliation_passed": True,
                            "target_truth_reconciliation_passed": True,
                            "pipeline_reconciliation_passed": True,
                            "model_reconciliation_passed": True,
                        }
                    )
    return pd.DataFrame(rows)


def test_same_portfolio_truth_is_used_across_models_and_windows() -> None:
    results = build_complete_synthetic_results()
    grouped = results.groupby(
        ["scenario_id", "simulation_id", "basis"]
    )["true_target_reserve"].nunique()
    assert grouped.eq(1).all()


def test_not_applicable_rows_preserve_total_expected_row_count() -> None:
    results = build_complete_synthetic_results()
    structurally_inapplicable = (
        results["model"].isin(
            {
                "regularized_poisson",
                "regularized_poisson_break_interaction",
                "regularized_tweedie",
            }
        )
        & results["history_window_years"].isin({5, 7})
    )
    results.loc[structurally_inapplicable, "success"] = False
    results.loc[
        structurally_inapplicable, "applicability_status"
    ] = NOT_APPLICABLE_BY_DESIGN
    results.loc[structurally_inapplicable, "fit_attempted"] = False
    results.loc[
        structurally_inapplicable, "failure_type"
    ] = NOT_APPLICABLE_BY_DESIGN
    results.loc[
        structurally_inapplicable, "failure_message"
    ] = "Frozen rolling-validation geometry is insufficient."
    results.loc[
        structurally_inapplicable,
        [
            "estimated_target_reserve",
            "signed_error",
            "percentage_error",
            "absolute_percentage_error",
            "selected_alpha",
            "selected_power",
        ],
    ] = np.nan

    report = runner.build_step26_acceptance_report(
        results=results,
        expected_rows=432,
        expected_scenario_ids={
            scenario["scenario_id"] for scenario in END_TO_END_SCENARIOS
        },
    ).set_index("check")

    assert len(results) == 432
    assert int(structurally_inapplicable.sum()) == 108
    for check in (
        "expected_row_count",
        "not_applicable_rows_retained_and_unsuccessful",
        "structurally_inapplicable_estimators_not_invoked",
        "not_applicable_rows_excluded_from_accuracy",
    ):
        assert bool(report.loc[check, "passed"])


def test_window_comparisons_exclude_structurally_inapplicable_pairs() -> None:
    results = build_complete_synthetic_results()
    structurally_inapplicable = (
        results["model"].eq("regularized_tweedie")
        & results["history_window_years"].isin({5, 7})
    )
    results.loc[structurally_inapplicable, "success"] = False
    results.loc[
        structurally_inapplicable, "applicability_status"
    ] = NOT_APPLICABLE_BY_DESIGN
    results.loc[structurally_inapplicable, "fit_attempted"] = False
    results.loc[
        structurally_inapplicable,
        ["signed_error", "percentage_error", "absolute_percentage_error"],
    ] = np.nan

    summary = runner.build_history_window_summary(results)
    comparisons = runner.build_history_window_comparisons(summary)
    tweedie = comparisons.loc[comparisons["model"].eq("regularized_tweedie")]
    assert set(tweedie["comparison"]) == {"15_vs_10"}


def test_applicability_summary_includes_not_applicable_denominator() -> None:
    results = pd.DataFrame(
        [
            {
                "model": "regularized_poisson",
                "history_window_years": 7,
                "applicability_status": NOT_APPLICABLE_BY_DESIGN,
                "fit_attempted": False,
                "success": False,
            },
            {
                "model": "regularized_poisson",
                "history_window_years": 7,
                "applicability_status": APPLICABLE_BY_DESIGN,
                "fit_attempted": True,
                "success": True,
            },
        ]
    )
    summary = runner.build_applicability_summary(results).iloc[0]
    assert summary["required_rows"] == 2
    assert summary["applicable_rows"] == 1
    assert summary["not_applicable_rows"] == 1
    assert summary["applicability_rate"] == 0.5
    assert summary["unconditional_success_rate"] == 0.5
    assert summary["conditional_fit_success_rate"] == 1.0


def test_successful_hyperparameters_remain_in_frozen_grids() -> None:
    results = build_complete_synthetic_results()
    report = runner.build_step26_acceptance_report(
        results=results,
        expected_rows=432,
        expected_scenario_ids={
            scenario["scenario_id"] for scenario in END_TO_END_SCENARIOS
        },
    )
    check = report.set_index("check").loc[
        "selected_hyperparameters_in_frozen_grids",
        "passed",
    ]
    assert bool(check)

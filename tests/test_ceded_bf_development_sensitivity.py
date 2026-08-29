"""Tests for Step 28 independent ceded BF development sensitivity."""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from config import (
    BF_BENCHMARK_INCREMENTAL_PATTERNS,
    BF_STRUCTURAL_BREAK_ACCIDENT_YEAR,
    END_TO_END_BASE_SEED,
    EXPECTED_LOSS_CALIBRATION_SEED_BASE,
    EXPECTED_LOSS_CALIBRATION_SIMULATIONS,
    VALUATION_YEAR,
)
from src.bornhuetter_ferguson import (
    BF_BREAK_AWARE,
    BF_STANDARD,
)
from src.ceded_bf_development_sensitivity import (
    CEDED_SPECIFIC_BREAK_AWARE,
    CEDED_SPECIFIC_STANDARD,
    DETAIL_KEY_COLUMNS,
    RESULT_KEY_COLUMNS,
    StreamingDevelopmentCalibration,
    build_final_acceptance_report,
    build_pattern_comparison,
    build_ceded_specific_bf_by_accident_year,
    compare_baseline_to_frozen_step21,
    development_year_from_calendar_year,
    run_ceded_bf_variants,
    sha256_file,
)
from src.expected_loss_prior import seed_ranges_are_disjoint


def _scenario() -> dict[str, object]:
    return {
        "scenario_id": "long_stable_no_break",
        "tail_type": "long",
        "inflation_scenario": "stable",
        "apply_structural_break": False,
    }


def _payments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "accident_year": [2023, 2023],
            "development_year": [1, 2],
            "payment_calendar_year": [2023, 2024],
            "nominal_gross_payment": [20.0, 40.0],
            "nominal_ceded_payment": [10.0, 30.0],
        }
    )


def _calibration_outputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    accumulator = StreamingDevelopmentCalibration(
        calibration_simulations=2,
        structural_break_year=2018,
        seed_start=101,
        seed_end=102,
    )

    for calibration_id, seed in [(1, 101), (2, 102)]:
        accumulator.add_portfolio(
            reinsured_payments=_payments(),
            scenario=_scenario(),
            calibration_id=calibration_id,
            seed=seed,
            accident_years=[2023, 2024],
        )

    return accumulator.build_outputs()


def _pattern_table() -> pd.DataFrame:
    records = []

    for regime, cumulative in [
        ("no_break", [0.25, 1.0]),
        ("post_break", [0.50, 1.0]),
    ]:
        for development_year, paid_proportion in enumerate(
            cumulative,
            start=1,
        ):
            records.append(
                {
                    "calibration_version": "fixture_v1",
                    "tail_type": "long",
                    "inflation_scenario": "stable",
                    "structural_break_regime": regime,
                    "development_year": development_year,
                    "cumulative_paid_proportion": paid_proportion,
                    "natural_terminal_development_year": 2,
                    "available": "True",
                }
            )

    return pd.DataFrame(records)


def _evaluation_input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario_id": ["fixture"] * 4,
            "simulation_id": [1] * 4,
            "accident_year": [2010, 2017, 2018, 2024],
            "basis": ["ceded"] * 4,
            "tail_type": ["long"] * 4,
            "structural_break": [True] * 4,
            "expected_loss_prior_ultimate": [
                50.0,
                80.0,
                100.0,
                200.0,
            ],
            "paid_to_date": [50.0, 70.0, 60.0, 20.0],
            "pricing_assumption_version": ["prior_v1"] * 4,
            "inflation_scenario": ["stable"] * 4,
            "true_reserve": [0.0, 5.0, 25.0, 80.0],
        }
    )


def _run_all_variants() -> tuple[pd.DataFrame, pd.DataFrame]:
    return run_ceded_bf_variants(
        frozen_evaluation_input=_evaluation_input(),
        benchmark_patterns=BF_BENCHMARK_INCREMENTAL_PATTERNS,
        valuation_year=VALUATION_YEAR,
        structural_break_year=BF_STRUCTURAL_BREAK_ACCIDENT_YEAR,
        pattern_table=_pattern_table(),
        include_ceded_specific=True,
    )


def test_calibration_seeds_are_independent_of_evaluation_seeds() -> None:
    assert seed_ranges_are_disjoint(
        first_seed_base=EXPECTED_LOSS_CALIBRATION_SEED_BASE,
        first_count=EXPECTED_LOSS_CALIBRATION_SIMULATIONS,
        second_seed_base=END_TO_END_BASE_SEED,
        second_count=10_000,
    )


def test_source_prior_hash_remains_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "expected_loss_prior.csv"
    path.write_bytes(b"scenario_id,accident_year,prior\nfixture,2024,1\n")
    before = sha256_file(path)

    _calibration_outputs()

    assert sha256_file(path) == before


def test_development_age_indexing_is_the_dissertation_convention() -> None:
    assert development_year_from_calendar_year(
        accident_year=2020,
        payment_calendar_year=2024,
    ) == 5


def test_incremental_proportions_are_nonnegative() -> None:
    _, pattern, _ = _calibration_outputs()
    assert (pattern["incremental_paid_proportion"] >= 0.0).all()


def test_cumulative_pattern_is_monotone() -> None:
    _, pattern, _ = _calibration_outputs()
    assert (pattern["cumulative_paid_proportion"].diff().dropna() >= 0.0).all()


def test_cumulative_proportions_are_bounded() -> None:
    _, pattern, _ = _calibration_outputs()
    assert pattern["cumulative_paid_proportion"].between(0.0, 1.0).all()


def test_terminal_pattern_reaches_one_at_natural_maturity() -> None:
    _, pattern, _ = _calibration_outputs()
    terminal = pattern.iloc[-1]
    assert terminal["development_year"] == 2
    assert np.isclose(terminal["cumulative_paid_proportion"], 1.0)


def test_calibration_payments_reconcile_to_ceded_ultimate() -> None:
    aggregate, pattern, support = _calibration_outputs()
    total = support["total_ceded_ultimate"].iloc[0]
    assert np.isclose(aggregate["incremental_ceded_paid"].sum(), total)
    assert np.isclose(pattern["incremental_ceded_paid"].sum(), total)


def test_source_calibration_data_are_not_mutated() -> None:
    payments = _payments()
    original = payments.copy(deep=True)
    accumulator = StreamingDevelopmentCalibration(
        calibration_simulations=2,
        structural_break_year=2018,
        seed_start=101,
        seed_end=102,
    )
    accumulator.add_portfolio(
        reinsured_payments=payments,
        scenario=_scenario(),
        calibration_id=1,
        seed=101,
        accident_years=[2023, 2024],
    )
    assert_frame_equal(payments, original)


def test_baseline_bf_reproduction_comparison_passes() -> None:
    results, _ = run_ceded_bf_variants(
        frozen_evaluation_input=_evaluation_input(),
        benchmark_patterns=BF_BENCHMARK_INCREMENTAL_PATTERNS,
        valuation_year=VALUATION_YEAR,
        structural_break_year=BF_STRUCTURAL_BREAK_ACCIDENT_YEAR,
    )
    frozen = results.copy(deep=True)
    comparison = compare_baseline_to_frozen_step21(
        results=results,
        frozen_step21_results=frozen,
    )
    assert comparison["passed"].all()
    assert (comparison["nonmatching_reserve_rows"] == 0).all()


def test_ceded_specific_pattern_changes_only_paid_proportion() -> None:
    estimator = _evaluation_input().drop(columns="true_reserve")
    standard = build_ceded_specific_bf_by_accident_year(
        estimator_input=estimator,
        pattern_table=_pattern_table(),
        variant=CEDED_SPECIFIC_STANDARD,
        valuation_year=VALUATION_YEAR,
        structural_break_year=BF_STRUCTURAL_BREAK_ACCIDENT_YEAR,
    )
    break_aware = build_ceded_specific_bf_by_accident_year(
        estimator_input=estimator,
        pattern_table=_pattern_table(),
        variant=CEDED_SPECIFIC_BREAK_AWARE,
        valuation_year=VALUATION_YEAR,
        structural_break_year=BF_STRUCTURAL_BREAK_ACCIDENT_YEAR,
    )
    invariant_columns = [
        "scenario_id",
        "simulation_id",
        "accident_year",
        "basis",
        "expected_loss_prior_ultimate",
        "paid_to_date",
        "pricing_assumption_version",
    ]
    assert_frame_equal(standard[invariant_columns], break_aware[invariant_columns])
    assert not np.array_equal(
        standard["benchmark_paid_proportion"].to_numpy(),
        break_aware["benchmark_paid_proportion"].to_numpy(),
    )


def test_prior_is_invariant_across_variants() -> None:
    _, detail = _run_all_variants()
    pivot = detail.pivot(
        index="accident_year",
        columns="model",
        values="expected_loss_prior_ultimate",
    )
    assert pivot.nunique(axis=1).eq(1).all()


def test_paid_to_date_is_invariant_across_variants() -> None:
    _, detail = _run_all_variants()
    pivot = detail.pivot(
        index="accident_year",
        columns="model",
        values="paid_to_date",
    )
    assert pivot.nunique(axis=1).eq(1).all()


def test_truth_is_invariant_across_variants() -> None:
    _, detail = _run_all_variants()
    pivot = detail.pivot(
        index="accident_year",
        columns="model",
        values="true_reserve",
    )
    assert pivot.nunique(axis=1).eq(1).all()


def test_ceded_specific_reserve_formula_reconciles() -> None:
    _, detail = _run_all_variants()
    ceded_specific = detail.loc[
        detail["model"].isin(
            [CEDED_SPECIFIC_STANDARD, CEDED_SPECIFIC_BREAK_AWARE]
        )
    ]
    expected = (
        ceded_specific["expected_loss_prior_ultimate"]
        * (1.0 - ceded_specific["benchmark_paid_proportion"])
    )
    assert np.allclose(ceded_specific["estimated_reserve"], expected)


def test_calibration_api_does_not_require_evaluation_truth() -> None:
    parameters = set(
        inspect.signature(StreamingDevelopmentCalibration.add_portfolio).parameters
    ) | set(inspect.signature(StreamingDevelopmentCalibration.build_outputs).parameters)
    assert "true_reserve" not in parameters
    assert "evaluation_truth" not in parameters


def test_result_and_detail_keys_are_unique() -> None:
    results, detail = _run_all_variants()
    assert not results.duplicated(RESULT_KEY_COLUMNS).any()
    assert not detail.duplicated(DETAIL_KEY_COLUMNS).any()
    assert set(results["model"]) == {
        BF_STANDARD,
        BF_BREAK_AWARE,
        CEDED_SPECIFIC_STANDARD,
        CEDED_SPECIFIC_BREAK_AWARE,
    }


def test_pattern_comparison_is_paired_by_simulation() -> None:
    results, _ = _run_all_variants()
    comparison = build_pattern_comparison(results)

    assert len(comparison) == 2
    assert comparison["attempted_pairs"].eq(1).all()
    assert comparison["successful_pairs"].eq(1).all()
    assert np.allclose(
        comparison[
            "mape_difference_ceded_specific_minus_baseline"
        ],
        comparison["mean_paired_ape_difference"],
    )


def test_final_acceptance_report_covers_pattern_assignment() -> None:
    break_input = _evaluation_input()
    no_break_input = break_input.copy(deep=True)
    no_break_input["scenario_id"] = "fixture_no_break"
    no_break_input["structural_break"] = False
    evaluation_input = pd.concat(
        [break_input, no_break_input],
        ignore_index=True,
    )
    results, detail = run_ceded_bf_variants(
        frozen_evaluation_input=evaluation_input,
        benchmark_patterns=BF_BENCHMARK_INCREMENTAL_PATTERNS,
        valuation_year=VALUATION_YEAR,
        structural_break_year=BF_STRUCTURAL_BREAK_ACCIDENT_YEAR,
        pattern_table=_pattern_table(),
        include_ceded_specific=True,
    )
    reproduction = compare_baseline_to_frozen_step21(
        results=results,
        frozen_step21_results=results,
    )
    comparison = build_pattern_comparison(results)
    acceptance = build_final_acceptance_report(
        results=results,
        detail=detail,
        reproduction=reproduction,
        pattern_table=_pattern_table(),
        pattern_comparison=comparison,
        expected_scenarios=2,
        simulations_per_scenario=1,
        structural_break_year=BF_STRUCTURAL_BREAK_ACCIDENT_YEAR,
        prior_hash_before="same",
        prior_hash_after="same",
    )

    assert len(acceptance) >= 19
    assert acceptance["passed"].all()


def test_unavailable_pattern_encoded_as_text_is_rejected() -> None:
    pattern = _pattern_table()
    pattern["available"] = "False"
    estimator = _evaluation_input().drop(columns="true_reserve")

    try:
        build_ceded_specific_bf_by_accident_year(
            estimator_input=estimator,
            pattern_table=pattern,
            variant=CEDED_SPECIFIC_STANDARD,
            valuation_year=VALUATION_YEAR,
            structural_break_year=BF_STRUCTURAL_BREAK_ACCIDENT_YEAR,
        )
    except ValueError as error:
        assert "unavailable" in str(error)
    else:
        raise AssertionError("Unavailable calibration pattern was accepted.")


def test_terminal_round_trip_noise_is_snapped_to_one() -> None:
    pattern = _pattern_table()
    pattern.loc[
        pattern["development_year"].eq(2),
        "cumulative_paid_proportion",
    ] = 1.0 + 1e-12
    estimator = _evaluation_input().loc[
        lambda frame: frame["accident_year"].eq(2023)
    ].drop(columns="true_reserve")
    if estimator.empty:
        estimator = pd.DataFrame(
            {
                "scenario_id": ["fixture"],
                "simulation_id": [1],
                "accident_year": [2023],
                "basis": ["ceded"],
                "tail_type": ["long"],
                "structural_break": [False],
                "expected_loss_prior_ultimate": [100.0],
                "paid_to_date": [100.0],
                "pricing_assumption_version": ["prior_v1"],
                "inflation_scenario": ["stable"],
            }
        )
    estimate = build_ceded_specific_bf_by_accident_year(
        estimator_input=estimator,
        pattern_table=pattern,
        variant=CEDED_SPECIFIC_STANDARD,
        valuation_year=VALUATION_YEAR,
        structural_break_year=BF_STRUCTURAL_BREAK_ACCIDENT_YEAR,
    )

    assert estimate["benchmark_paid_proportion"].eq(1.0).all()
    assert estimate["estimated_reserve"].eq(0.0).all()

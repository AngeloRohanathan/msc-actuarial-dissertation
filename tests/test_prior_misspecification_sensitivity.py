"""Tests for Step 27 prior-misspecification sensitivity."""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from config import (
    BF_BENCHMARK_INCREMENTAL_PATTERNS,
    BF_STRUCTURAL_BREAK_ACCIDENT_YEAR,
    VALUATION_YEAR,
)
from src.bornhuetter_ferguson import (
    BF_BREAK_AWARE,
    BF_STANDARD,
)
from src.prior_misspecification_sensitivity import (
    DETAIL_KEY_COLUMNS,
    RESULT_KEY_COLUMNS,
    PRIOR_MULTIPLIERS,
    apply_prior_multiplier,
    compare_multiplier_one_to_frozen_baselines,
    load_frozen_independent_prior,
    run_prior_misspecification_sensitivity,
    sha256_file,
)


def _prior_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario_id": ["fixture", "fixture"],
            "accident_year": [2023, 2024],
            "expected_gross_ultimate": [100.0, 200.0],
            "expected_ceded_ultimate": [40.0, 80.0],
            "expected_retained_ultimate": [60.0, 120.0],
            "pricing_assumption_version": [
                "pricing_mc_v1",
                "pricing_mc_v1",
            ],
        }
    )


def _evaluation_fixture() -> pd.DataFrame:
    records = []

    for basis, paid, truth in [
        (
            "gross",
            [60.0, 50.0],
            [30.0, 150.0],
        ),
        (
            "ceded",
            [20.0, 15.0],
            [10.0, 60.0],
        ),
    ]:
        for accident_year, paid_value, true_value in zip(
            [2023, 2024],
            paid,
            truth,
        ):
            records.append(
                {
                    "scenario_id": "fixture",
                    "simulation_id": 1,
                    "accident_year": accident_year,
                    "basis": basis,
                    "tail_type": "long",
                    "structural_break": True,
                    "expected_loss_prior_ultimate": (
                        100.0
                        if basis == "gross"
                        and accident_year == 2023
                        else 200.0
                        if basis == "gross"
                        else 40.0
                        if accident_year == 2023
                        else 80.0
                    ),
                    "paid_to_date": paid_value,
                    "pricing_assumption_version": (
                        "pricing_mc_v1"
                    ),
                    "true_reserve": true_value,
                    "inflation_scenario": "stable",
                    "clause_type": "none",
                    "seed": 123,
                }
            )

    return pd.DataFrame(records)


def _run(
    multipliers: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return run_prior_misspecification_sensitivity(
        expected_loss_prior=_prior_fixture(),
        evaluation_detail=_evaluation_fixture(),
        prior_multipliers=multipliers,
        valuation_year=VALUATION_YEAR,
        structural_break_year=(
            BF_STRUCTURAL_BREAK_ACCIDENT_YEAR
        ),
        benchmark_patterns=(
            BF_BENCHMARK_INCREMENTAL_PATTERNS
        ),
    )


def test_multiplier_one_leaves_prior_unchanged() -> None:
    prior = _prior_fixture()
    scaled = apply_prior_multiplier(
        prior,
        prior_multiplier=1.0,
    )

    for column in [
        "expected_gross_ultimate",
        "expected_ceded_ultimate",
        "expected_retained_ultimate",
    ]:
        assert np.array_equal(
            scaled[column].to_numpy(),
            prior[column].to_numpy(),
        )


def test_multiplier_point_eight_scales_prior_exactly() -> None:
    prior = _prior_fixture()
    scaled = apply_prior_multiplier(
        prior,
        prior_multiplier=0.8,
    )

    assert np.array_equal(
        scaled["expected_gross_ultimate"].to_numpy(),
        prior["expected_gross_ultimate"].to_numpy()
        * 0.8,
    )


def test_multiplier_one_point_two_scales_prior_exactly() -> None:
    prior = _prior_fixture()
    scaled = apply_prior_multiplier(
        prior,
        prior_multiplier=1.2,
    )

    assert np.array_equal(
        scaled["expected_ceded_ultimate"].to_numpy(),
        prior["expected_ceded_ultimate"].to_numpy()
        * 1.2,
    )


def test_retained_prior_is_scaled_consistently() -> None:
    prior = _prior_fixture()
    scaled = apply_prior_multiplier(
        prior,
        prior_multiplier=0.9,
    )

    assert np.array_equal(
        scaled["expected_retained_ultimate"].to_numpy(),
        prior["expected_retained_ultimate"].to_numpy()
        * 0.9,
    )


def test_source_prior_object_is_not_mutated() -> None:
    prior = _prior_fixture()
    original = prior.copy(deep=True)

    apply_prior_multiplier(
        prior,
        prior_multiplier=1.2,
    )

    assert_frame_equal(prior, original)


def test_source_prior_file_is_not_mutated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prior.csv"
    _prior_fixture().to_csv(path, index=False)
    before = sha256_file(path)

    prior = load_frozen_independent_prior(path)
    apply_prior_multiplier(
        prior,
        prior_multiplier=0.8,
    )

    assert sha256_file(path) == before


def test_expected_loss_formula_responds_to_scaling() -> None:
    results, _ = _run((0.8, 1.0))
    gross = results.loc[
        results["model"].eq("expected_loss")
        & results["basis"].eq("gross")
    ].set_index("prior_multiplier")

    assert np.isclose(
        gross.loc[0.8, "estimated_reserve"],
        130.0,
    )
    assert np.isclose(
        gross.loc[1.0, "estimated_reserve"],
        190.0,
    )


def test_standard_bf_formula_responds_to_scaling() -> None:
    results, _ = _run((1.0, 1.2))
    gross = results.loc[
        results["model"].eq(BF_STANDARD)
        & results["basis"].eq("gross")
    ].set_index("prior_multiplier")

    assert np.isclose(
        gross.loc[1.2, "estimated_reserve"],
        1.2 * gross.loc[1.0, "estimated_reserve"],
    )


def test_break_aware_bf_formula_responds_to_scaling() -> None:
    results, _ = _run((0.8, 1.0))
    ceded = results.loc[
        results["model"].eq(BF_BREAK_AWARE)
        & results["basis"].eq("ceded")
    ].set_index("prior_multiplier")

    assert np.isclose(
        ceded.loc[0.8, "estimated_reserve"],
        0.8 * ceded.loc[1.0, "estimated_reserve"],
    )


def test_true_reserve_is_invariant_across_multipliers() -> None:
    results, _ = _run((0.8, 1.0, 1.2))
    counts = results.groupby(
        [
            "scenario_id",
            "simulation_id",
            "basis",
        ]
    )["true_reserve"].nunique()

    assert counts.eq(1).all()


def test_paid_to_date_is_invariant_across_multipliers() -> None:
    _, detail = _run((0.8, 1.0, 1.2))
    counts = detail.groupby(
        [
            "scenario_id",
            "simulation_id",
            "basis",
            "accident_year",
        ]
    )["paid_to_date"].nunique()

    assert counts.eq(1).all()


def test_multiplier_one_baseline_reproduction_fixture() -> None:
    results, _ = _run((1.0,))
    frozen_step20 = results.loc[
        results["model"].eq("expected_loss")
    ].copy()
    frozen_step21 = results.loc[
        results["model"].isin(
            [BF_STANDARD, BF_BREAK_AWARE]
        )
    ].copy()

    comparison = compare_multiplier_one_to_frozen_baselines(
        results=results,
        frozen_step20_results=frozen_step20,
        frozen_step21_results=frozen_step21,
    )

    assert comparison["passed"].all()
    assert comparison[
        "max_absolute_reserve_difference"
    ].eq(0.0).all()


def test_scaling_api_does_not_require_evaluation_truth() -> None:
    parameters = set(
        inspect.signature(
            apply_prior_multiplier
        ).parameters
    )

    assert parameters == {
        "expected_loss_prior",
        "prior_multiplier",
    }
    assert not any(
        forbidden in parameter.lower()
        for parameter in parameters
        for forbidden in [
            "true",
            "future",
            "actual",
            "error",
        ]
    )


def test_unique_result_and_detail_keys_are_preserved() -> None:
    results, detail = _run(PRIOR_MULTIPLIERS)

    assert not results.duplicated(
        RESULT_KEY_COLUMNS
    ).any()
    assert not detail.duplicated(
        DETAIL_KEY_COLUMNS
    ).any()

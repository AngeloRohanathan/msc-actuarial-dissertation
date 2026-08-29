"""Tests for Step 29 treaty indexation mechanics."""

import inspect

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from config import (
    CLAUSE_INDEX_BASE_YEAR,
    PILOT_XOL_ATTACHMENT,
    PILOT_XOL_LIMIT,
    VALUATION_YEAR,
)
from src.reinsurance import apply_xol_to_payments, calculate_xol_ceded
from src.treaty_indexation_sensitivity import (
    FIXED_NOMINAL,
    FULLY_INDEXED,
    RESULT_KEY_COLUMNS,
    apply_treaty_variant_to_payments,
    build_accident_year_summary,
    build_paired_comparison,
    build_portfolio_result,
    build_scenario_summary,
    build_treaty_term_schedule,
    calibration_api_uses_only_approved_index_variables,
    treaty_terms_for_accident_year,
)


def _toy_payments(accident_year: int = 2018) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario_id": ["fixture"] * 4,
            "simulation_id": [1] * 4,
            "claim_id": ["CLAIM_1"] * 4,
            "accident_year": [accident_year] * 4,
            "report_year": [accident_year] * 4,
            "frequency_scenario": ["constant"] * 4,
            "tail_type": ["short"] * 4,
            "inflation_scenario": ["stable"] * 4,
            "apply_structural_break": [False] * 4,
            "structural_break_indicator": [0] * 4,
            "payment_pattern_name": ["short"] * 4,
            "payment_sequence": [1, 2, 3, 4],
            "development_year": [1, 2, 3, 4],
            "payment_calendar_year": [
                accident_year,
                accident_year + 1,
                accident_year + 2,
                accident_year + 3,
            ],
            "real_payment": [1_000_000.0, 1_500_000.0, 3_000_000.0, 4_000_000.0],
            "inflation_rate": [0.04] * 4,
            "inflation_index": [1.0] * 4,
            "nominal_gross_payment": [
                1_000_000.0,
                1_500_000.0,
                3_000_000.0,
                4_000_000.0,
            ],
        }
    )


def _scenario() -> dict[str, object]:
    return {
        "scenario_id": "fixture",
        "frequency_scenario": "constant",
        "tail_type": "short",
        "inflation_scenario": "stable",
        "apply_structural_break": False,
    }


def test_fixed_nominal_terms_are_two_million_and_five_million() -> None:
    terms = treaty_terms_for_accident_year(
        treaty_variant=FIXED_NOMINAL,
        accident_year=2024,
        inflation_scenario="shock",
    )
    assert terms["indexed_attachment"] == 2_000_000.0
    assert terms["indexed_limit"] == 5_000_000.0


def test_index_factor_one_returns_baseline_terms() -> None:
    terms = treaty_terms_for_accident_year(
        treaty_variant=FULLY_INDEXED,
        accident_year=CLAUSE_INDEX_BASE_YEAR,
        inflation_scenario="stable",
    )
    assert terms["applied_index_factor"] == 1.0
    assert terms["indexed_attachment"] == PILOT_XOL_ATTACHMENT
    assert terms["indexed_limit"] == PILOT_XOL_LIMIT


def test_indexed_terms_scale_attachment_and_limit_consistently() -> None:
    terms = treaty_terms_for_accident_year(
        treaty_variant=FULLY_INDEXED,
        accident_year=2024,
        inflation_scenario="emerging",
    )
    factor = terms["inflation_index_value"] / terms[
        "reference_inflation_index_value"
    ]
    assert np.isclose(
        terms["indexed_attachment"], PILOT_XOL_ATTACHMENT * factor
    )
    assert np.isclose(terms["indexed_limit"], PILOT_XOL_LIMIT * factor)
    assert np.isclose(
        terms["indexed_limit"] / terms["indexed_attachment"],
        PILOT_XOL_LIMIT / PILOT_XOL_ATTACHMENT,
    )


def test_all_indexed_terms_are_finite_and_positive() -> None:
    schedule = build_treaty_term_schedule(inflation_scenario="shock")
    assert np.isfinite(
        schedule[["indexed_attachment", "indexed_limit"]]
    ).all().all()
    assert (
        schedule[["indexed_attachment", "indexed_limit"]] > 0.0
    ).all().all()


def test_fixed_wrapper_reproduces_baseline_xol_exactly() -> None:
    payments = _toy_payments()
    baseline_payments, baseline_claims = apply_xol_to_payments(
        payments=payments,
        attachment=PILOT_XOL_ATTACHMENT,
        limit=PILOT_XOL_LIMIT,
    )
    fixed_payments, fixed_claims, _ = apply_treaty_variant_to_payments(
        payments=payments,
        treaty_variant=FIXED_NOMINAL,
        inflation_scenario="stable",
        scenario_id="fixture",
    )
    payment_columns = [
        "simulation_id",
        "claim_id",
        "payment_sequence",
        "nominal_gross_payment",
        "nominal_ceded_payment",
        "nominal_retained_payment",
        "cumulative_nominal_ceded_amount",
    ]
    claim_columns = [
        "simulation_id",
        "claim_id",
        "nominal_gross_ultimate",
        "nominal_ceded_ultimate",
        "nominal_retained_ultimate",
        "attachment_breached",
        "limit_exhausted",
    ]
    assert_frame_equal(
        fixed_payments[payment_columns], baseline_payments[payment_columns]
    )
    assert_frame_equal(
        fixed_claims[claim_columns], baseline_claims[claim_columns]
    )


def test_known_toy_claim_has_correct_fixed_ceded_amount() -> None:
    _, claims, _ = apply_treaty_variant_to_payments(
        payments=_toy_payments(),
        treaty_variant=FIXED_NOMINAL,
        inflation_scenario="stable",
    )
    assert np.isclose(claims.loc[0, "nominal_ceded_ultimate"], 5_000_000.0)


def test_known_toy_claim_has_correct_indexed_ceded_amount() -> None:
    payments = _toy_payments(accident_year=2018)
    _, claims, _ = apply_treaty_variant_to_payments(
        payments=payments,
        treaty_variant=FULLY_INDEXED,
        inflation_scenario="stable",
    )
    terms = treaty_terms_for_accident_year(
        treaty_variant=FULLY_INDEXED,
        accident_year=2018,
        inflation_scenario="stable",
    )
    expected = calculate_xol_ceded(
        claim_amount=9_500_000.0,
        attachment=terms["indexed_attachment"],
        limit=terms["indexed_limit"],
    )
    assert np.isclose(claims.loc[0, "nominal_ceded_ultimate"], expected)


def test_gross_claim_stream_is_identical_across_variants() -> None:
    payments = _toy_payments()
    fixed, _, _ = apply_treaty_variant_to_payments(
        payments=payments,
        treaty_variant=FIXED_NOMINAL,
        inflation_scenario="stable",
    )
    indexed, _, _ = apply_treaty_variant_to_payments(
        payments=payments,
        treaty_variant=FULLY_INDEXED,
        inflation_scenario="stable",
    )
    np.testing.assert_array_equal(
        fixed["nominal_gross_payment"], indexed["nominal_gross_payment"]
    )


def test_gross_equals_retained_plus_ceded() -> None:
    reinsured, claims, _ = apply_treaty_variant_to_payments(
        payments=_toy_payments(),
        treaty_variant=FULLY_INDEXED,
        inflation_scenario="stable",
    )
    np.testing.assert_allclose(
        reinsured["nominal_gross_payment"],
        reinsured["nominal_ceded_payment"]
        + reinsured["nominal_retained_payment"],
    )
    np.testing.assert_allclose(
        claims["nominal_gross_ultimate"],
        claims["nominal_ceded_ultimate"]
        + claims["nominal_retained_ultimate"],
    )


def test_ceded_amount_never_exceeds_applicable_limit() -> None:
    reinsured, claims, _ = apply_treaty_variant_to_payments(
        payments=_toy_payments(),
        treaty_variant=FULLY_INDEXED,
        inflation_scenario="stable",
    )
    assert (
        reinsured["cumulative_nominal_ceded_amount"]
        <= reinsured["indexed_limit"] + 1e-6
    ).all()
    assert (
        claims["nominal_ceded_ultimate"]
        <= claims["indexed_limit"] + 1e-6
    ).all()


def test_indexing_uses_accident_year_not_payment_year() -> None:
    original = _toy_payments()
    shifted = original.copy(deep=True)
    shifted["payment_calendar_year"] += 5
    first, _, _ = apply_treaty_variant_to_payments(
        payments=original,
        treaty_variant=FULLY_INDEXED,
        inflation_scenario="stable",
    )
    second, _, _ = apply_treaty_variant_to_payments(
        payments=shifted,
        treaty_variant=FULLY_INDEXED,
        inflation_scenario="stable",
    )
    np.testing.assert_array_equal(
        first["indexed_attachment"], second["indexed_attachment"]
    )
    np.testing.assert_array_equal(first["indexed_limit"], second["indexed_limit"])
    assert calibration_api_uses_only_approved_index_variables()


def test_source_payments_are_not_mutated() -> None:
    payments = _toy_payments()
    original = payments.copy(deep=True)
    apply_treaty_variant_to_payments(
        payments=payments,
        treaty_variant=FULLY_INDEXED,
        inflation_scenario="stable",
    )
    assert_frame_equal(payments, original)


def test_accident_year_aggregation_reconciles() -> None:
    reinsured, claims, _ = apply_treaty_variant_to_payments(
        payments=_toy_payments(),
        treaty_variant=FULLY_INDEXED,
        inflation_scenario="stable",
    )
    by_ay = build_accident_year_summary(
        reinsured_payments=reinsured,
        claim_summary=claims,
        scenario=_scenario(),
        simulation_id=1,
        seed=123,
        treaty_variant=FULLY_INDEXED,
        valuation_year=VALUATION_YEAR,
    )
    assert np.isclose(by_ay["gross_ultimate"].sum(), 9_500_000.0)
    assert np.isclose(
        by_ay["gross_ultimate"].sum(),
        by_ay["ceded_ultimate"].sum()
        + by_ay["retained_ultimate"].sum(),
    )


def test_result_keys_are_unique() -> None:
    rows = []

    for variant in [FIXED_NOMINAL, FULLY_INDEXED]:
        reinsured, claims, _ = apply_treaty_variant_to_payments(
            payments=_toy_payments(),
            treaty_variant=variant,
            inflation_scenario="stable",
        )
        by_ay = build_accident_year_summary(
            reinsured_payments=reinsured,
            claim_summary=claims,
            scenario=_scenario(),
            simulation_id=1,
            seed=123,
            treaty_variant=variant,
        )
        rows.append(
            build_portfolio_result(
                accident_year_summary=by_ay,
                claim_summary=claims,
                scenario=_scenario(),
                simulation_id=1,
                seed=123,
                treaty_variant=variant,
                runtime_seconds=0.1,
            )
        )

    results = pd.DataFrame(rows)
    assert not results.duplicated(RESULT_KEY_COLUMNS).any()


def test_terms_remain_fixed_during_claim_lifetime() -> None:
    reinsured, _, _ = apply_treaty_variant_to_payments(
        payments=_toy_payments(),
        treaty_variant=FULLY_INDEXED,
        inflation_scenario="stable",
    )
    assert reinsured["indexed_attachment"].nunique() == 1
    assert reinsured["indexed_limit"].nunique() == 1
    assert set(reinsured["index_timing"]) == {"accident_year"}


def test_term_selection_api_has_no_truth_or_payment_time_inputs() -> None:
    parameters = set(
        inspect.signature(build_treaty_term_schedule).parameters
    ) | set(inspect.signature(treaty_terms_for_accident_year).parameters)
    assert "true_reserve" not in parameters
    assert "payment_calendar_year" not in parameters


def _summary_fixture() -> pd.DataFrame:
    rows = []

    for simulation_id, fixed_ultimate, indexed_ultimate in [
        (1, 100.0, 80.0),
        (2, 120.0, 90.0),
    ]:
        for variant, ceded_ultimate in [
            (FIXED_NOMINAL, fixed_ultimate),
            (FULLY_INDEXED, indexed_ultimate),
        ]:
            rows.append(
                {
                    "scenario_id": "fixture",
                    "simulation_id": simulation_id,
                    "treaty_variant": variant,
                    "success": True,
                    "gross_ultimate": 500.0,
                    "ceded_ultimate": ceded_ultimate,
                    "retained_ultimate": 500.0 - ceded_ultimate,
                    "ceded_share": ceded_ultimate / 500.0,
                    "attaching_claims": 5,
                    "attachment_frequency": 0.5,
                    "exhausting_claims": 1,
                    "exhaustion_frequency": 0.1,
                    "ceded_true_reserve": ceded_ultimate / 2.0,
                    "retained_true_reserve": 100.0,
                    "gross_true_reserve": 150.0,
                    "ceded_paid_to_date": ceded_ultimate / 2.0,
                    "ceded_unpaid_proportion": 0.5,
                }
            )

    return pd.DataFrame(rows)


def test_scenario_summary_reports_both_variants() -> None:
    summary = build_scenario_summary(_summary_fixture())
    assert set(summary["treaty_variant"]) == {
        FIXED_NOMINAL,
        FULLY_INDEXED,
    }
    assert summary["attempted_simulations"].eq(2).all()
    assert summary["successful_simulations"].eq(2).all()
    assert summary["success_rate"].eq(1.0).all()


def test_paired_summary_uses_indexed_minus_fixed_direction() -> None:
    summary = build_paired_comparison(_summary_fixture()).iloc[0]
    assert summary["paired_simulations"] == 2
    assert np.isclose(summary["mean_ceded_ultimate_difference"], -25.0)
    assert np.isclose(summary["median_ceded_ultimate_difference"], -25.0)
    assert np.isclose(summary["mean_ceded_reserve_difference"], -12.5)
    assert summary["proportion_indexed_ceded_ultimate_lower"] == 1.0
    assert summary["proportion_indexed_ceded_reserve_lower"] == 1.0

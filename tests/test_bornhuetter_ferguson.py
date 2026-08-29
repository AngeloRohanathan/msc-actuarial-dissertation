"""Tests for paid Bornhuetter-Ferguson reserving."""

import inspect

import numpy as np

from config import (
    BF_BENCHMARK_INCREMENTAL_PATTERNS,
    BF_STRUCTURAL_BREAK_ACCIDENT_YEAR,
)

from src.bornhuetter_ferguson import (
    BF_BREAK_AWARE,
    BF_STANDARD,
    build_paid_bf_by_accident_year,
    cumulative_paid_proportion,
    paid_bf_reserve,
    select_bf_pattern_name,
)


def test_short_tail_cumulative_pattern() -> None:
    pattern = (
        BF_BENCHMARK_INCREMENTAL_PATTERNS[
            "short"
        ]
    )

    assert np.isclose(
        cumulative_paid_proportion(
            incremental_pattern=pattern,
            development_year=1,
        ),
        0.60,
    )

    assert np.isclose(
        cumulative_paid_proportion(
            incremental_pattern=pattern,
            development_year=2,
        ),
        0.85,
    )

    assert np.isclose(
        cumulative_paid_proportion(
            incremental_pattern=pattern,
            development_year=4,
        ),
        1.00,
    )


def test_accelerated_long_pattern() -> None:
    pattern = (
        BF_BENCHMARK_INCREMENTAL_PATTERNS[
            "accelerated_long"
        ]
    )

    assert np.isclose(
        cumulative_paid_proportion(
            incremental_pattern=pattern,
            development_year=1,
        ),
        0.08,
    )

    assert np.isclose(
        cumulative_paid_proportion(
            incremental_pattern=pattern,
            development_year=2,
        ),
        0.35,
    )


def test_bf_formula() -> None:
    reserve = paid_bf_reserve(
        prior_ultimate=100.0,
        cumulative_paid_proportion=0.60,
    )

    assert np.isclose(
        reserve,
        40.0,
    )


def test_fully_developed_short_tail_has_zero_reserve() -> None:
    proportion = (
        cumulative_paid_proportion(
            incremental_pattern=(
                BF_BENCHMARK_INCREMENTAL_PATTERNS[
                    "short"
                ]
            ),
            development_year=10,
        )
    )

    reserve = paid_bf_reserve(
        prior_ultimate=100.0,
        cumulative_paid_proportion=(
            proportion
        ),
    )

    assert np.isclose(
        reserve,
        0.0,
    )


def test_standard_bf_keeps_original_pattern_after_break() -> None:
    name = select_bf_pattern_name(
        tail_type="long",
        structural_break=True,
        accident_year=(
            BF_STRUCTURAL_BREAK_ACCIDENT_YEAR
        ),
        variant=BF_STANDARD,
        structural_break_year=(
            BF_STRUCTURAL_BREAK_ACCIDENT_YEAR
        ),
    )

    assert name == "long"


def test_break_aware_bf_uses_accelerated_pattern() -> None:
    name = select_bf_pattern_name(
        tail_type="long",
        structural_break=True,
        accident_year=(
            BF_STRUCTURAL_BREAK_ACCIDENT_YEAR
        ),
        variant=BF_BREAK_AWARE,
        structural_break_year=(
            BF_STRUCTURAL_BREAK_ACCIDENT_YEAR
        ),
    )

    assert name == "accelerated_long"


def test_bf_estimator_has_no_oracle_inputs() -> None:
    parameters = set(
        inspect.signature(
            build_paid_bf_by_accident_year
        ).parameters
    )

    forbidden = {
        "true",
        "future",
        "actual",
    }

    for parameter in parameters:
        assert not any(
            word in parameter.lower()
            for word in forbidden
        )

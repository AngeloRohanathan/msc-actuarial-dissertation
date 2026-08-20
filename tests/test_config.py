"""Tests for the central project configuration."""

from math import isclose

from config import (
    PAYMENT_PATTERNS,
    REPORTING_DELAY_PROBABILITIES,
    validate_config,
)


def test_configuration_is_valid() -> None:
    """The complete configuration should pass validation."""

    validate_config()


def test_payment_patterns_sum_to_one() -> None:
    """Every incremental payment pattern should sum to one."""

    for pattern in PAYMENT_PATTERNS.values():
        assert isclose(sum(pattern), 1.0, abs_tol=1e-12)


def test_reporting_probabilities_sum_to_one() -> None:
    """Every reporting-delay distribution should sum to one."""

    for probabilities in REPORTING_DELAY_PROBABILITIES.values():
        assert isclose(
            sum(probabilities.values()),
            1.0,
            abs_tol=1e-12,
        )
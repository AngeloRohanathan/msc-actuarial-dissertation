"""Tests for safe execution of reserving models."""

import pandas as pd

from scripts.run_reserving_pilot import (
    run_model_safely,
)


def failing_model(
    **kwargs,
) -> dict[str, object]:
    """Represent a model that cannot be fitted."""

    raise ValueError(
        "Test model failure."
    )


def successful_model(
    **kwargs,
) -> dict[str, object]:
    """Return a minimal valid model result."""

    return {
        "summary": pd.DataFrame(
            [
                {
                    "total_estimated_reserve": 123.0,
                }
            ]
        )
    }


def test_safe_runner_records_failure() -> None:
    result, status = run_model_safely(
        model_name="test_model",
        basis="ceded",
        model_function=failing_model,
    )

    assert result is None
    assert status["model_status"] == "failed"
    assert (
        status["failure_reason"]
        == "Test model failure."
    )


def test_safe_runner_records_success() -> None:
    result, status = run_model_safely(
        model_name="test_model",
        basis="gross",
        model_function=successful_model,
    )

    assert result is not None
    assert status["model_status"] == "success"
    assert status["estimated_reserve"] == 123.0
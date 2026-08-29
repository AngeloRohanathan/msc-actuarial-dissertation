"""Run Step 26 historical accident-year window sensitivity."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    CASHFLOW_UPLIFT_EMBEDDED_INFLATION,
    END_TO_END_BASE_SEED,
    END_TO_END_PILOT_SIMULATIONS,
    END_TO_END_SCENARIOS,
    EXPERIMENT_CLAUSE_TYPE,
    EXPERIMENT_STRUCTURAL_BREAK_YEAR,
    INFLATION_SCENARIOS,
    ML_AMOUNT_SCALE,
    ML_MIN_TRAINING_DIAGONALS,
    ML_MIN_VALIDATION_FOLDS,
    ML_POISSON_ALPHA_GRID,
    ML_TWEEDIE_ALPHA_GRID,
    ML_TWEEDIE_POWER_GRID,
    PILOT_XOL_ATTACHMENT,
    PILOT_XOL_LIMIT,
    VALUATION_YEAR,
    ensure_directories,
    validate_config,
)
from src.evaluation import calculate_error_metrics
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
    future_cells_are_hidden,
    validate_history_windows,
)
from src.ml_models import (
    fit_regularised_poisson_break_interaction_reserving_model,
    fit_regularised_poisson_reserving_model,
    fit_regularised_tweedie_reserving_model,
)
from src.reinsurance import apply_xol_to_payments
from src.reserving import (
    cashflow_uplift,
    chain_ladder,
    inflation_adjusted_chain_ladder,
)
from src.simulation import build_inflation_index, simulate_portfolio
from src.triangles import build_triangle_package


MODEL_NAMES = [
    "chain_ladder",
    "inflation_adjusted_chain_ladder",
    "cashflow_uplift",
    "regularized_poisson",
    "regularized_poisson_break_interaction",
    "regularized_tweedie",
]
BASES = ["gross", "ceded"]
STEP26_OUTPUT_DIR = Path("outputs/step26_history_window_sensitivity")

STEP26_METHOD_NOTE = """# Step 26: Historical Data-Window Sensitivity

## Objective

Step 26 investigates how the amount of historical accident-year experience
used for fitting affects reserve accuracy, responsiveness, stability, and
applicability, particularly under inflation and structural change.

## Frozen design

The four history windows are fixed at AY2010--2024 (15 years), AY2015--2024
(10 years), AY2018--2024 (7 years), and AY2020--2024 (5 years). Regardless of
the fitting window, every successful estimate and its independent truth are
evaluated only for AY2020--2024.

The six frozen models are Chain Ladder, inflation-adjusted Chain Ladder,
Cashflow Uplift, Regularized Poisson, Regularized Poisson with the existing
calendar-regime by development interaction, and Regularized Tweedie. Existing
model features, hyperparameter grids, validation requirements, links,
estimators, scenario definitions, and seeds are unchanged.

For each scenario and simulation, one portfolio is simulated and one gross and
ceded truth package is built. Every model and history window therefore uses the
same simulated portfolio, triangle basis, and AY2020--2024 evaluation truth.
Training-window filtering copies the triangle and supplies only accident years
at or after the frozen window start. Observed upper-triangle cells are retained;
future cells remain missing and never enter fitting or hyperparameter selection.

## Applicability and fitting status

With five minimum training diagonals and three minimum validation folds, a
7-year window supplies only two potential validation folds and a 5-year window
supplies none. The 7- and 5-year combinations are consequently marked
`not_applicable_by_design` for Regularized Poisson, Regularized Poisson with
break-development interactions, and Regularized Tweedie. Their estimator is
not invoked, but every required experiment row and its fixed evaluation truth
is retained. No estimate or accuracy statistic is calculated for these rows.

All classical combinations and all 10- and 15-year ML combinations invoke the
frozen estimator. Their failures are data-dependent and remain explicitly
reported. Applicability rate, unconditional success rate, and conditional fit
success rate are reported separately. Accuracy statistics use successful fits
only and are never used to redefine applicability or tune a model.

## Leakage safeguards and experiment size

No evaluation reserve truth enters model fitting or hyperparameter selection.
Rolling validation uses only historical observed cells within the requested
history window. All estimates and truths independently reconcile over the
fixed AY2020--2024 target.

The final design contains 9 scenarios x 50 simulations x 2 bases x 4 history
windows x 6 models = 21,600 required result rows. Of these, 5,400 rows are the
pre-specified structurally inapplicable ML 5- and 7-year combinations described
above. Accuracy improvement is a research result, not a technical acceptance
criterion.
"""


def parse_arguments() -> argparse.Namespace:
    """Read Step 26 experiment options."""

    parser = argparse.ArgumentParser(
        description="Run Step 26 historical data-window sensitivity."
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=END_TO_END_PILOT_SIMULATIONS,
        help="Number of simulations per scenario.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Run only the named scenario; may be repeated.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Optional output-folder label.",
    )
    return parser.parse_args()


def select_scenarios(
    requested_scenarios: list[str] | None,
) -> list[dict[str, Any]]:
    """Select requested frozen scenarios while preserving configured order."""

    scenarios = [dict(scenario) for scenario in END_TO_END_SCENARIOS]
    if requested_scenarios is None:
        return scenarios

    requested = set(requested_scenarios)
    selected = [
        scenario
        for scenario in scenarios
        if scenario["scenario_id"] in requested
    ]
    missing = requested - {scenario["scenario_id"] for scenario in selected}
    if missing:
        raise ValueError(f"Unknown requested scenarios: {sorted(missing)}")
    return selected


def build_scenario_inflation_index(
    inflation_scenario: str,
) -> dict[int, float]:
    """Build the configured claims-inflation index for one scenario."""

    table = build_inflation_index(INFLATION_SCENARIOS[inflation_scenario])
    return table.set_index("calendar_year")["inflation_index"].to_dict()


def verify_pipeline_reconciliations(
    reinsured_payments: pd.DataFrame,
    triangle_outputs: dict[str, pd.DataFrame],
) -> bool:
    """Check that gross, ceded, and retained triangle totals reconcile."""

    gross_total = float(reinsured_payments["nominal_gross_payment"].sum())
    ceded_total = float(reinsured_payments["nominal_ceded_payment"].sum())
    retained_total = float(
        reinsured_payments["nominal_retained_payment"].sum()
    )
    totals = triangle_outputs["true_reserve_totals"].iloc[0]

    return bool(
        np.isclose(
            gross_total,
            ceded_total + retained_total,
            rtol=1e-12,
            atol=1e-6,
        )
        and np.isclose(
            totals["total_observed_gross_paid"]
            + totals["total_true_gross_reserve"],
            totals["total_gross_ultimate"],
            rtol=1e-12,
            atol=1e-6,
        )
        and np.isclose(
            totals["total_observed_ceded_paid"]
            + totals["total_true_ceded_reserve"],
            totals["total_ceded_ultimate"],
            rtol=1e-12,
            atol=1e-6,
        )
    )


def classify_step26_failure(
    model_name: str,
    basis: str,
    failure_message: str,
) -> str:
    """Classify expected short-window limitations without hiding them."""

    message = failure_message.lower()
    if (
        "insufficient calendar-year diagonals" in message
        or "enough successful rolling validation folds" in message
        or "enough successful validation folds" in message
    ):
        return "insufficient_historical_validation"
    if "no paired observations are available" in message:
        return "insufficient_development_history"
    if "denominator is zero" in message:
        if basis == "ceded":
            return "known_sparse_ceded_chain_ladder_limitation"
        return "zero_development_denominator"
    if model_name in {
        "chain_ladder",
        "inflation_adjusted_chain_ladder",
        "cashflow_uplift",
    } and "missing development factor" in message:
        return "insufficient_development_history"
    return "unexplained_failure"


def fit_frozen_model(
    *,
    model_name: str,
    basis: str,
    incremental_triangle: pd.DataFrame,
    cumulative_triangle: pd.DataFrame,
    inflation_index: dict[int, float],
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch to one existing frozen reserving implementation."""

    structural_break_year = (
        EXPERIMENT_STRUCTURAL_BREAK_YEAR
        if scenario["apply_structural_break"]
        else None
    )

    if model_name == "chain_ladder":
        return chain_ladder(cumulative_triangle)
    if model_name == "inflation_adjusted_chain_ladder":
        return inflation_adjusted_chain_ladder(
            nominal_incremental_triangle=incremental_triangle,
            inflation_index=inflation_index,
            valuation_year=VALUATION_YEAR,
        )
    if model_name == "cashflow_uplift":
        return cashflow_uplift(
            nominal_cumulative_triangle=cumulative_triangle,
            forecast_inflation_index=inflation_index,
            valuation_year=VALUATION_YEAR,
            embedded_annual_inflation=CASHFLOW_UPLIFT_EMBEDDED_INFLATION,
        )
    if model_name == "regularized_poisson":
        return fit_regularised_poisson_reserving_model(
            incremental_triangle=incremental_triangle,
            inflation_index=inflation_index,
            valuation_year=VALUATION_YEAR,
            basis=basis,
            alpha_grid=ML_POISSON_ALPHA_GRID,
            minimum_training_diagonals=ML_MIN_TRAINING_DIAGONALS,
            minimum_validation_folds=ML_MIN_VALIDATION_FOLDS,
            amount_scale=ML_AMOUNT_SCALE,
            structural_break_year=structural_break_year,
        )
    if model_name == "regularized_poisson_break_interaction":
        return fit_regularised_poisson_break_interaction_reserving_model(
            incremental_triangle=incremental_triangle,
            inflation_index=inflation_index,
            valuation_year=VALUATION_YEAR,
            basis=basis,
            alpha_grid=ML_POISSON_ALPHA_GRID,
            minimum_training_diagonals=ML_MIN_TRAINING_DIAGONALS,
            minimum_validation_folds=ML_MIN_VALIDATION_FOLDS,
            amount_scale=ML_AMOUNT_SCALE,
            structural_break_year=structural_break_year,
        )
    if model_name == "regularized_tweedie":
        return fit_regularised_tweedie_reserving_model(
            incremental_triangle=incremental_triangle,
            inflation_index=inflation_index,
            valuation_year=VALUATION_YEAR,
            basis=basis,
            alpha_grid=ML_TWEEDIE_ALPHA_GRID,
            power_grid=ML_TWEEDIE_POWER_GRID,
            minimum_training_diagonals=ML_MIN_TRAINING_DIAGONALS,
            minimum_validation_folds=ML_MIN_VALIDATION_FOLDS,
            amount_scale=ML_AMOUNT_SCALE,
            structural_break_year=structural_break_year,
        )
    raise ValueError(f"Unknown model: {model_name}")


def calculate_model_target_prediction(
    result: dict[str, Any],
    model_name: str,
) -> float:
    """Independently sum predicted future cells for AY2020--2024."""

    if model_name == "inflation_adjusted_chain_ladder":
        future = result["projected_future_nominal_incremental"]
        return float(
            future.loc[list(EVALUATION_ACCIDENT_YEARS)].fillna(0.0).sum().sum()
        )
    if model_name in {"chain_ladder", "cashflow_uplift"}:
        future = result["projected_future_incremental"]
        return float(
            future.loc[list(EVALUATION_ACCIDENT_YEARS)].fillna(0.0).sum().sum()
        )

    future = result["future_predictions"]
    target = future.loc[
        future["accident_year"].isin(EVALUATION_ACCIDENT_YEARS)
    ]
    return float(target["predicted_incremental_paid"].sum())


def run_model_attempt(
    *,
    model_name: str,
    basis: str,
    history_window_years: int,
    history_start_ay: int,
    simulation_id: int,
    seed: int,
    scenario: dict[str, Any],
    inflation_index: dict[int, float],
    incremental_triangle: pd.DataFrame,
    cumulative_triangle: pd.DataFrame,
    true_reserves_by_accident_year: pd.DataFrame,
    pipeline_reconciliation_passed: bool,
) -> dict[str, Any]:
    """Run one frozen model and score only the fixed target AYs."""

    true_values = calculate_true_target_values(
        true_reserves_by_accident_year,
        basis,
    )
    input_future_cells_hidden = bool(
        future_cells_are_hidden(incremental_triangle, VALUATION_YEAR)
        and future_cells_are_hidden(cumulative_triangle, VALUATION_YEAR)
    )
    fit_min_accident_year = int(min(incremental_triangle.index))
    fit_max_accident_year = int(max(incremental_triangle.index))

    applicability = assess_model_applicability(
        model_name=model_name,
        triangle=incremental_triangle,
        minimum_training_diagonals=ML_MIN_TRAINING_DIAGONALS,
        minimum_validation_folds=ML_MIN_VALIDATION_FOLDS,
    )

    started = time.perf_counter()
    estimated_target_reserve: float | None = None
    selected_alpha: float | None = None
    selected_power: float | None = None
    model_reconciliation_passed = False
    target_estimate_reconciliation_passed = False
    status = "failed"
    failure_message = ""
    fit_attempted = False

    if not applicability.fit_should_be_attempted:
        status = NOT_APPLICABLE_BY_DESIGN
        failure_message = applicability.reason
    else:
        fit_attempted = True
        try:
            result = fit_frozen_model(
                model_name=model_name,
                basis=basis,
                incremental_triangle=incremental_triangle,
                cumulative_triangle=cumulative_triangle,
                inflation_index=inflation_index,
                scenario=scenario,
            )

            reserve_by_ay = result["reserve_by_accident_year"]
            estimated_target_reserve = aggregate_accident_year_amount(
                reserve_by_ay,
                "estimated_reserve",
            )
            future_target_total = calculate_model_target_prediction(
                result,
                model_name,
            )
            target_estimate_reconciliation_passed = bool(
                np.isclose(
                    estimated_target_reserve,
                    future_target_total,
                    rtol=1e-12,
                    atol=1e-6,
                )
            )

            summary_total = float(
                result["summary"].loc[0, "total_estimated_reserve"]
            )
            reserve_table_total = float(reserve_by_ay["estimated_reserve"].sum())
            model_reconciliation_passed = bool(
                np.isclose(
                    summary_total,
                    reserve_table_total,
                    rtol=1e-12,
                    atol=1e-6,
                )
            )

            if "selected_alpha" in result["summary"].columns:
                selected_alpha = float(result["summary"].loc[0, "selected_alpha"])
            if "selected_power" in result["summary"].columns:
                selected_power = float(result["summary"].loc[0, "selected_power"])

            if not np.isfinite(estimated_target_reserve):
                raise ValueError("Estimated target reserve is not finite.")
            if estimated_target_reserve < -1e-6:
                raise ValueError("Estimated target reserve is negative.")
            if not model_reconciliation_passed:
                raise ValueError(
                    "Model summary does not reconcile with its AY reserve table."
                )
            if not target_estimate_reconciliation_passed:
                raise ValueError(
                    "AY2020-2024 reserve does not reconcile with future cells."
                )

            status = "success"
        except Exception as error:
            failure_message = f"{type(error).__name__}: {error}"

    runtime_seconds = time.perf_counter() - started
    metrics = calculate_error_metrics(
        estimated_reserve=(
            estimated_target_reserve if status == "success" else None
        ),
        true_reserve=true_values["true_target_reserve"],
    )
    if status == "success":
        failure_type = ""
    elif status == NOT_APPLICABLE_BY_DESIGN:
        failure_type = NOT_APPLICABLE_BY_DESIGN
    else:
        failure_type = classify_step26_failure(model_name, basis, failure_message)

    row: dict[str, Any] = {
        "scenario_id": scenario["scenario_id"],
        "simulation_id": int(simulation_id),
        "seed": int(seed),
        "frequency_scenario": scenario["frequency_scenario"],
        "tail_type": scenario["tail_type"],
        "inflation_scenario": scenario["inflation_scenario"],
        "structural_break": bool(scenario["apply_structural_break"]),
        "clause_type": EXPERIMENT_CLAUSE_TYPE,
        "basis": basis,
        "model": model_name,
        "history_window_years": int(history_window_years),
        "history_start_ay": int(history_start_ay),
        "evaluation_start_ay": EVALUATION_START_AY,
        "evaluation_end_ay": EVALUATION_END_AY,
        "success": status == "success",
        "success_or_failure": status,
        "applicability_status": applicability.status,
        "applicability_reason": applicability.reason,
        "fit_attempted": fit_attempted,
        "available_calendar_diagonals": (
            applicability.available_calendar_diagonals
        ),
        "potential_validation_folds": (
            applicability.potential_validation_folds
            if applicability.potential_validation_folds is not None
            else np.nan
        ),
        "maximum_development_year": applicability.maximum_development_year,
        "failure_type": failure_type,
        "failure_category": failure_type,
        "failure_message": failure_message,
        "failure_reason": failure_message,
        "fit_min_accident_year": fit_min_accident_year,
        "fit_max_accident_year": fit_max_accident_year,
        "fit_received_pre_window_ay": (
            fit_min_accident_year < int(history_start_ay)
        ),
        "future_cells_hidden_from_fit": input_future_cells_hidden,
        "evaluation_truth_used_for_fitting": False,
        "observed_target_paid": true_values["observed_target_paid"],
        "true_target_reserve": true_values["true_target_reserve"],
        "true_target_ultimate": true_values["true_target_ultimate"],
        "estimated_target_reserve": (
            float(estimated_target_reserve)
            if status == "success"
            else np.nan
        ),
        "true_reserve": true_values["true_target_reserve"],
        "estimated_reserve": (
            float(estimated_target_reserve)
            if status == "success"
            else np.nan
        ),
        "selected_alpha": (
            float(selected_alpha) if selected_alpha is not None else np.nan
        ),
        "selected_power": (
            float(selected_power) if selected_power is not None else np.nan
        ),
        "runtime_seconds": float(runtime_seconds),
        "pipeline_reconciliation_passed": bool(
            pipeline_reconciliation_passed
        ),
        "model_reconciliation_passed": bool(model_reconciliation_passed),
        "target_estimate_reconciliation_passed": bool(
            target_estimate_reconciliation_passed
        ),
        "target_truth_reconciliation_passed": True,
    }
    row.update(metrics)
    return row


def run_one_portfolio(
    *,
    simulation_id: int,
    seed: int,
    scenario: dict[str, Any],
) -> list[dict[str, Any]]:
    """Simulate once, then apply every model and history window."""

    _, simulated_payments = simulate_portfolio(
        simulation_id=simulation_id,
        frequency_scenario=scenario["frequency_scenario"],
        tail_type=scenario["tail_type"],
        inflation_scenario=scenario["inflation_scenario"],
        apply_structural_break=scenario["apply_structural_break"],
        seed=seed,
    )
    payments = simulated_payments.copy()
    if "scenario_id" not in payments.columns:
        payments["scenario_id"] = scenario["scenario_id"]

    reinsured_payments, _ = apply_xol_to_payments(
        payments=payments,
        attachment=PILOT_XOL_ATTACHMENT,
        limit=PILOT_XOL_LIMIT,
    )
    triangle_outputs = build_triangle_package(
        payments=reinsured_payments,
        valuation_year=VALUATION_YEAR,
    )
    pipeline_reconciliation_passed = verify_pipeline_reconciliations(
        reinsured_payments,
        triangle_outputs,
    )
    inflation_index = build_scenario_inflation_index(
        scenario["inflation_scenario"]
    )

    rows: list[dict[str, Any]] = []
    for basis in BASES:
        full_incremental = triangle_outputs[f"{basis}_incremental"]
        full_cumulative = triangle_outputs[f"{basis}_cumulative"]

        for history_window_years, history_start_ay in HISTORY_WINDOWS.items():
            incremental = filter_triangle_to_history_window(
                full_incremental,
                history_start_ay,
                VALUATION_YEAR,
            )
            cumulative = filter_triangle_to_history_window(
                full_cumulative,
                history_start_ay,
                VALUATION_YEAR,
            )

            for model_name in MODEL_NAMES:
                rows.append(
                    run_model_attempt(
                        model_name=model_name,
                        basis=basis,
                        history_window_years=history_window_years,
                        history_start_ay=history_start_ay,
                        simulation_id=simulation_id,
                        seed=seed,
                        scenario=scenario,
                        inflation_index=inflation_index,
                        incremental_triangle=incremental,
                        cumulative_triangle=cumulative,
                        true_reserves_by_accident_year=triangle_outputs[
                            "true_reserves_by_accident_year"
                        ],
                        pipeline_reconciliation_passed=(
                            pipeline_reconciliation_passed
                        ),
                    )
                )
    return rows


def build_history_window_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise applicability and conditional accuracy by window."""

    grouping = [
        "scenario_id",
        "basis",
        "model",
        "history_window_years",
        "history_start_ay",
    ]
    rows: list[dict[str, Any]] = []

    for values, group in results.groupby(grouping, dropna=False, sort=False):
        information = dict(zip(grouping, values))
        successful = group.loc[group["success"]]
        attempted = int(len(group))
        successful_fits = int(len(successful))
        if "applicability_status" in group.columns:
            not_applicable = group["applicability_status"].eq(
                NOT_APPLICABLE_BY_DESIGN
            )
        else:
            not_applicable = pd.Series(False, index=group.index)
        if "fit_attempted" in group.columns:
            fit_attempted = group["fit_attempted"].astype(bool)
        else:
            fit_attempted = pd.Series(True, index=group.index)
        not_applicable_count = int(not_applicable.sum())
        applicable_count = attempted - not_applicable_count
        estimator_invocations = int(fit_attempted.sum())
        data_dependent_failures = int(
            (fit_attempted & ~group["success"]).sum()
        )
        row: dict[str, Any] = {
            **information,
            "attempted_fits": attempted,
            "required_result_rows": attempted,
            "required_rows": attempted,
            "applicable_rows": applicable_count,
            "not_applicable_by_design": not_applicable_count,
            "not_applicable_rows": not_applicable_count,
            "estimator_invocations": estimator_invocations,
            "successful_fits": successful_fits,
            "failed_fits": data_dependent_failures,
            "data_dependent_failures": data_dependent_failures,
            "unsuccessful_or_not_applicable": attempted - successful_fits,
            "applicability_rate": applicable_count / attempted,
            "success_rate": successful_fits / attempted,
            "unconditional_success_rate": successful_fits / attempted,
            "conditional_fit_success_rate": (
                successful_fits / applicable_count
                if applicable_count > 0
                else np.nan
            ),
            "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
        }

        if successful.empty:
            row.update(
                {
                    "mean_percentage_error": np.nan,
                    "median_percentage_error": np.nan,
                    "mean_absolute_percentage_error": np.nan,
                    "root_mean_squared_error": np.nan,
                    "rmse": np.nan,
                    "standard_deviation_percentage_error": np.nan,
                    "percentage_error_std": np.nan,
                }
            )
        else:
            signed_error = successful["signed_error"].to_numpy(dtype=float)
            row.update(
                {
                    "mean_percentage_error": float(
                        successful["percentage_error"].mean()
                    ),
                    "median_percentage_error": float(
                        successful["percentage_error"].median()
                    ),
                    "mean_absolute_percentage_error": float(
                        successful["absolute_percentage_error"].mean()
                    ),
                    "root_mean_squared_error": float(
                        np.sqrt(np.mean(signed_error**2))
                    ),
                    "rmse": float(np.sqrt(np.mean(signed_error**2))),
                    "standard_deviation_percentage_error": float(
                        successful["percentage_error"].std()
                    ),
                    "percentage_error_std": float(
                        successful["percentage_error"].std()
                    ),
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


def build_applicability_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise applicability and fitting success by model and window."""

    rows: list[dict[str, Any]] = []
    for (model, window), group in results.groupby(
        ["model", "history_window_years"], sort=False
    ):
        required = int(len(group))
        not_applicable = int(
            group["applicability_status"].eq(NOT_APPLICABLE_BY_DESIGN).sum()
        )
        applicable = required - not_applicable
        invocations = int(group["fit_attempted"].sum())
        successful = int(group["success"].sum())
        failures = int((group["fit_attempted"] & ~group["success"]).sum())
        rows.append(
            {
                "model": model,
                "history_window_years": int(window),
                "required_rows": required,
                "applicable_rows": applicable,
                "not_applicable_rows": not_applicable,
                "estimator_invocations": invocations,
                "successful_fits": successful,
                "data_dependent_failures": failures,
                "applicability_rate": applicable / required,
                "conditional_fit_success_rate": (
                    successful / applicable if applicable else np.nan
                ),
                "unconditional_success_rate": successful / required,
            }
        )
    return pd.DataFrame(rows)


def build_history_window_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    """Create descriptive comparisons only where both windows apply."""

    comparison_pairs = [(15, 10), (15, 7), (15, 5), (10, 7), (10, 5)]
    rows: list[dict[str, Any]] = []
    grouping = ["scenario_id", "basis", "model"]
    for values, group in summary.groupby(grouping, sort=False):
        information = dict(zip(grouping, values))
        by_window = group.set_index("history_window_years")
        for first_window, second_window in comparison_pairs:
            if first_window not in by_window.index or second_window not in by_window.index:
                continue
            first = by_window.loc[first_window]
            second = by_window.loc[second_window]
            if int(first["applicable_rows"]) == 0 or int(second["applicable_rows"]) == 0:
                continue
            first_mape = float(first["mean_absolute_percentage_error"])
            second_mape = float(second["mean_absolute_percentage_error"])
            accuracy_available = bool(
                np.isfinite(first_mape) and np.isfinite(second_mape)
            )
            rows.append(
                {
                    **information,
                    "first_window_years": first_window,
                    "second_window_years": second_window,
                    "comparison": f"{first_window}_vs_{second_window}",
                    "first_mape": first_mape,
                    "second_mape": second_mape,
                    "first_mpe": float(first["mean_percentage_error"]),
                    "second_mpe": float(second["mean_percentage_error"]),
                    "first_applicability_rate": float(first["applicability_rate"]),
                    "second_applicability_rate": float(second["applicability_rate"]),
                    "first_unconditional_success_rate": float(
                        first["unconditional_success_rate"]
                    ),
                    "second_unconditional_success_rate": float(
                        second["unconditional_success_rate"]
                    ),
                    "first_conditional_fit_success_rate": float(
                        first["conditional_fit_success_rate"]
                    ),
                    "second_conditional_fit_success_rate": float(
                        second["conditional_fit_success_rate"]
                    ),
                    "difference_in_mape_second_minus_first": (
                        second_mape - first_mape if accuracy_available else np.nan
                    ),
                    "accuracy_comparison_available": accuracy_available,
                }
            )
    return pd.DataFrame(rows)


def build_step26_failure_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise failed or structurally skipped Step 26 model attempts."""

    failed = results.loc[~results["success"]]
    columns = [
        "scenario_id",
        "model",
        "basis",
        "history_window_years",
        "history_start_ay",
        "applicability_status",
        "failure_type",
        "failure_message",
        "failure_count",
    ]
    if failed.empty:
        return pd.DataFrame(columns=columns)
    return (
        failed.groupby(columns[:-1], dropna=False)
        .size()
        .reset_index(name="failure_count")
    )


def build_step26_acceptance_report(
    *,
    results: pd.DataFrame,
    expected_rows: int,
    expected_scenario_ids: set[str],
) -> pd.DataFrame:
    """Evaluate the frozen Step 26 structural requirements."""

    successful = results.loc[results["success"]]
    failed = results.loc[~results["success"]]
    not_applicable = results.loc[
        results["applicability_status"].eq(NOT_APPLICABLE_BY_DESIGN)
    ]
    applicable_failures = results.loc[
        ~results["success"]
        & results["applicability_status"].eq(APPLICABLE_BY_DESIGN)
    ]
    poisson_models = {
        "regularized_poisson",
        "regularized_poisson_break_interaction",
    }
    classical_models = {
        "chain_ladder",
        "inflation_adjusted_chain_ladder",
        "cashflow_uplift",
    }

    truth_across_windows = results.groupby(
        ["scenario_id", "simulation_id", "basis", "model"],
        dropna=False,
    )["true_target_reserve"].nunique(dropna=False)
    truth_across_models = results.groupby(
        [
            "scenario_id",
            "simulation_id",
            "basis",
            "history_window_years",
        ],
        dropna=False,
    )["true_target_reserve"].nunique(dropna=False)

    poisson_success = successful.loc[
        successful["model"].isin(poisson_models)
    ]
    tweedie_success = successful.loc[
        successful["model"].eq("regularized_tweedie")
    ]
    classical_success = successful.loc[
        successful["model"].isin(classical_models)
    ]

    failures_complete = bool(
        failed.empty
        or (
            failed["failure_type"].fillna("").str.len().gt(0).all()
            and failed["failure_message"].fillna("").str.len().gt(0).all()
        )
    )
    actual_mapping = (
        results[["history_window_years", "history_start_ay"]]
        .drop_duplicates()
        .set_index("history_window_years")["history_start_ay"]
        .to_dict()
    )

    checks = [
        (
            "expected_row_count",
            len(results) == expected_rows,
            f"actual={len(results)}, expected={expected_rows}",
        ),
        (
            "all_requested_history_windows_present",
            set(results["history_window_years"]) == set(HISTORY_WINDOWS),
            "",
        ),
        (
            "history_windows_map_to_frozen_start_years",
            actual_mapping == HISTORY_WINDOWS,
            str(actual_mapping),
        ),
        (
            "fixed_evaluation_target_ay2020_2024",
            bool(
                results["evaluation_start_ay"].eq(EVALUATION_START_AY).all()
                and results["evaluation_end_ay"].eq(EVALUATION_END_AY).all()
            ),
            "",
        ),
        (
            "all_nine_scenarios_present",
            set(results["scenario_id"]) == expected_scenario_ids,
            "",
        ),
        ("both_bases_present", set(results["basis"]) == set(BASES), ""),
        ("all_six_models_present", set(results["model"]) == set(MODEL_NAMES), ""),
        (
            "same_truth_across_history_windows",
            bool(truth_across_windows.eq(1).all()),
            "",
        ),
        (
            "same_truth_across_models",
            bool(truth_across_models.eq(1).all()),
            "",
        ),
        (
            "no_pre_window_accident_years_supplied",
            bool(
                ~results["fit_received_pre_window_ay"].any()
                and results["fit_min_accident_year"]
                .eq(results["history_start_ay"])
                .all()
            ),
            "",
        ),
        (
            "no_future_cells_enter_model_fitting",
            bool(
                results["future_cells_hidden_from_fit"].all()
                and ~results["evaluation_truth_used_for_fitting"].any()
            ),
            "",
        ),
        (
            "estimated_target_reserve_uses_ay2020_2024_only",
            bool(successful["target_estimate_reconciliation_passed"].all()),
            "",
        ),
        (
            "true_target_reserve_uses_ay2020_2024_only",
            bool(results["target_truth_reconciliation_passed"].all()),
            "",
        ),
        (
            "target_ay_reserve_aggregation_reconciles",
            bool(
                results["target_truth_reconciliation_passed"].all()
                and successful["target_estimate_reconciliation_passed"].all()
            ),
            "",
        ),
        (
            "successful_estimates_finite_and_nonnegative",
            bool(
                np.isfinite(successful["estimated_target_reserve"]).all()
                and successful["estimated_target_reserve"].ge(0.0).all()
            ),
            "",
        ),
        (
            "selected_hyperparameters_in_frozen_grids",
            bool(
                poisson_success["selected_alpha"]
                .isin(ML_POISSON_ALPHA_GRID)
                .all()
                and poisson_success["selected_power"].isna().all()
                and tweedie_success["selected_alpha"]
                .isin(ML_TWEEDIE_ALPHA_GRID)
                .all()
                and tweedie_success["selected_power"]
                .isin(ML_TWEEDIE_POWER_GRID)
                .all()
                and classical_success["selected_alpha"].isna().all()
                and classical_success["selected_power"].isna().all()
            ),
            "",
        ),
        (
            "failures_completely_reported",
            failures_complete,
            f"failed_rows={len(failed)}",
        ),
        (
            "not_applicable_rows_retained_and_unsuccessful",
            bool(
                not_applicable["success"].eq(False).all()
                and not_applicable["failure_type"]
                .eq(NOT_APPLICABLE_BY_DESIGN)
                .all()
            ),
            f"not_applicable_rows={len(not_applicable)}",
        ),
        (
            "structurally_inapplicable_estimators_not_invoked",
            bool(~not_applicable["fit_attempted"].any()),
            "",
        ),
        (
            "not_applicable_rows_excluded_from_accuracy",
            bool(
                not_applicable[
                    [
                        "estimated_target_reserve",
                        "signed_error",
                        "percentage_error",
                        "absolute_percentage_error",
                    ]
                ]
                .isna()
                .all()
                .all()
            ),
            "",
        ),
        (
            "data_dependent_failures_were_attempted",
            bool(applicable_failures["fit_attempted"].all()),
            f"data_dependent_failures={len(applicable_failures)}",
        ),
        (
            "row_count_reconciles_successes_and_failures",
            len(results) == len(successful) + len(failed),
            f"successes={len(successful)}, failures={len(failed)}",
        ),
        (
            "unique_scenario_simulation_basis_model_window_key",
            not results.duplicated(
                [
                    "scenario_id",
                    "simulation_id",
                    "basis",
                    "model",
                    "history_window_years",
                ]
            ).any(),
            "",
        ),
        (
            "pipeline_reconciliations_passed",
            bool(results["pipeline_reconciliation_passed"].all()),
            "",
        ),
        (
            "successful_model_reconciliations_passed",
            bool(successful["model_reconciliation_passed"].all()),
            "",
        ),
    ]
    return pd.DataFrame(
        [
            {"check": check, "passed": bool(passed), "detail": detail}
            for check, passed, detail in checks
        ]
    )


def main() -> None:
    """Run and validate the frozen Step 26 history-window sensitivity."""

    arguments = parse_arguments()
    ensure_directories()
    validate_config()
    validate_history_windows()

    if arguments.simulations < 1:
        raise ValueError("--simulations must be positive.")

    scenarios = select_scenarios(arguments.scenario)
    run_label = arguments.run_label or (
        f"step26_history_window_{arguments.simulations}_simulations"
    )
    output_directory = STEP26_OUTPUT_DIR / run_label
    output_directory.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    experiment_start = time.perf_counter()
    total_portfolios = arguments.simulations * len(scenarios)
    completed_portfolios = 0

    for simulation_id in range(1, arguments.simulations + 1):
        seed = END_TO_END_BASE_SEED + simulation_id
        for scenario in scenarios:
            completed_portfolios += 1
            print(
                f"[{completed_portfolios}/{total_portfolios}] "
                f"simulation={simulation_id}, "
                f"scenario={scenario['scenario_id']}"
            )
            all_rows.extend(
                run_one_portfolio(
                    simulation_id=simulation_id,
                    seed=seed,
                    scenario=scenario,
                )
            )
            pd.DataFrame(all_rows).to_csv(
                output_directory / "results_checkpoint.csv",
                index=False,
            )

    results = pd.DataFrame(all_rows)
    expected_rows = (
        arguments.simulations
        * len(scenarios)
        * len(BASES)
        * len(HISTORY_WINDOWS)
        * len(MODEL_NAMES)
    )
    summary = build_history_window_summary(results)
    applicability_summary = build_applicability_summary(results)
    history_window_comparisons = build_history_window_comparisons(summary)
    failure_summary = build_step26_failure_summary(results)
    acceptance_report = build_step26_acceptance_report(
        results=results,
        expected_rows=expected_rows,
        expected_scenario_ids={
            scenario["scenario_id"] for scenario in scenarios
        },
    )

    results.to_csv(output_directory / "results.csv", index=False)
    summary.to_csv(output_directory / "summary.csv", index=False)
    applicability_summary.to_csv(
        output_directory / "applicability_summary.csv", index=False
    )
    history_window_comparisons.to_csv(
        output_directory / "history_window_comparisons.csv", index=False
    )
    failure_summary.to_csv(
        output_directory / "failure_summary.csv",
        index=False,
    )
    acceptance_report.to_csv(
        output_directory / "acceptance_report.csv",
        index=False,
    )

    elapsed_seconds = time.perf_counter() - experiment_start
    manifest = {
        "run_label": run_label,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment": "historical_data_window_sensitivity",
        "simulations_per_scenario": arguments.simulations,
        "scenario_ids": [scenario["scenario_id"] for scenario in scenarios],
        "number_of_scenarios": len(scenarios),
        "expected_result_rows": expected_rows,
        "actual_result_rows": int(len(results)),
        "base_seed": END_TO_END_BASE_SEED,
        "valuation_year": VALUATION_YEAR,
        "history_windows": HISTORY_WINDOWS,
        "evaluation_start_ay": EVALUATION_START_AY,
        "evaluation_end_ay": EVALUATION_END_AY,
        "models": MODEL_NAMES,
        "bases": BASES,
        "treaty_attachment": PILOT_XOL_ATTACHMENT,
        "treaty_limit": PILOT_XOL_LIMIT,
        "clause_type": EXPERIMENT_CLAUSE_TYPE,
        "minimum_training_diagonals": ML_MIN_TRAINING_DIAGONALS,
        "minimum_validation_folds": ML_MIN_VALIDATION_FOLDS,
        "poisson_alpha_grid": list(ML_POISSON_ALPHA_GRID),
        "tweedie_alpha_grid": list(ML_TWEEDIE_ALPHA_GRID),
        "tweedie_power_grid": list(ML_TWEEDIE_POWER_GRID),
        "evaluation_truth_used_for_fitting": False,
        "deterministic_inapplicability_optimization": True,
        "elapsed_seconds": elapsed_seconds,
    }
    with (output_directory / "manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=2)
    (output_directory / "STEP26_METHOD_NOTE.md").write_text(
        STEP26_METHOD_NOTE,
        encoding="utf-8",
    )

    print("\nStep 26 smoke experiment completed.")
    print(f"Result rows: {len(results):,}")
    print("\nAcceptance report:")
    print(acceptance_report.to_string(index=False))
    print("\nOutputs saved to:")
    print(output_directory)


if __name__ == "__main__":
    main()

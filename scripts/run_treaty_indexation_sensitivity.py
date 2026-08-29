"""Run Step 29 fixed-versus-indexed treaty mechanics sensitivity."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from config import (
    CLAUSE_INDEX_BASE_YEAR,
    END_TO_END_BASE_SEED,
    END_TO_END_SCENARIOS,
    PILOT_XOL_ATTACHMENT,
    PILOT_XOL_LIMIT,
    validate_config,
)
from src.reinsurance import apply_xol_to_payments
from src.simulation import simulate_portfolio
from src.treaty_indexation_sensitivity import (
    ACCIDENT_YEAR_KEY_COLUMNS,
    FIXED_NOMINAL,
    FULLY_INDEXED,
    NUMERICAL_ATOL,
    NUMERICAL_RTOL,
    RESULT_KEY_COLUMNS,
    TREATY_VARIANTS,
    apply_treaty_variant_to_payments,
    build_accident_year_summary,
    build_comparison_summary,
    build_development_summary,
    build_paired_comparison,
    build_portfolio_result,
    build_scenario_summary,
    calibration_api_uses_only_approved_index_variables,
    compare_fixed_nominal_to_baseline,
    sha256_directory,
    sha256_file,
)


DEFAULT_OUTPUT_ROOT = Path(
    "outputs/step29_treaty_indexation_sensitivity"
)
DEFAULT_FROZEN_STEP16 = Path("data/final/baseline_step16")
DEFAULT_GATE_REPORT = (
    DEFAULT_OUTPUT_ROOT / "baseline_gate_2" / "acceptance_report.csv"
)


def parse_arguments() -> argparse.Namespace:
    """Read Step 29 treaty sensitivity options."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["baseline-gate", "smoke", "final"],
        required=True,
    )
    parser.add_argument("--simulations", type=int, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--frozen-step16",
        type=Path,
        default=DEFAULT_FROZEN_STEP16,
    )
    parser.add_argument(
        "--baseline-gate-report",
        type=Path,
        default=DEFAULT_GATE_REPORT,
    )
    return parser.parse_args()


def _selected_scenarios() -> list[dict[str, Any]]:
    return [dict(scenario) for scenario in END_TO_END_SCENARIOS]


def _load_frozen_step16_truth(directory: Path) -> pd.DataFrame:
    results_path = directory / "results.csv"

    if not results_path.exists():
        raise FileNotFoundError(results_path)

    results = pd.read_csv(results_path)
    keys = ["scenario_id", "simulation_id", "basis"]
    columns = keys + ["observed_paid", "true_reserve", "true_ultimate"]
    truth = results[columns].drop_duplicates()

    if truth.duplicated(keys).any():
        raise ValueError("Frozen Step 16 truth is not unique.")

    return truth


def _truth_reproduction_differences(
    *,
    fixed_result: dict[str, Any],
    frozen_truth: pd.DataFrame,
) -> tuple[float, float, int]:
    scenario_id = fixed_result["scenario_id"]
    simulation_id = int(fixed_result["simulation_id"])
    frozen = frozen_truth.loc[
        frozen_truth["scenario_id"].eq(scenario_id)
        & frozen_truth["simulation_id"].eq(simulation_id)
    ].set_index("basis")

    if set(frozen.index) != {"gross", "ceded"}:
        raise ValueError("Frozen Step 16 truth pair is missing.")

    comparisons = [
        (
            float(fixed_result["gross_ultimate"]),
            float(frozen.loc["gross", "true_ultimate"]),
        ),
        (
            float(fixed_result["gross_true_reserve"]),
            float(frozen.loc["gross", "true_reserve"]),
        ),
        (
            float(fixed_result["ceded_ultimate"]),
            float(frozen.loc["ceded", "true_ultimate"]),
        ),
        (
            float(fixed_result["ceded_true_reserve"]),
            float(frozen.loc["ceded", "true_reserve"]),
        ),
        (
            float(fixed_result["ceded_paid_to_date"]),
            float(frozen.loc["ceded", "observed_paid"]),
        ),
    ]
    absolute = np.asarray(
        [abs(actual - expected) for actual, expected in comparisons]
    )
    relative = np.asarray(
        [
            difference / max(abs(expected), NUMERICAL_ATOL)
            for difference, (_, expected) in zip(absolute, comparisons)
        ]
    )
    matching = [
        np.isclose(
            actual,
            expected,
            rtol=NUMERICAL_RTOL,
            atol=NUMERICAL_ATOL,
        )
        for actual, expected in comparisons
    ]

    return float(absolute.max()), float(relative.max()), int(
        (~np.asarray(matching)).sum()
    )


def _failure_result(
    *,
    scenario: dict[str, Any],
    simulation_id: int,
    seed: int,
    treaty_variant: str,
    runtime_seconds: float,
    error: Exception,
) -> dict[str, Any]:
    result = {
        "scenario_id": scenario["scenario_id"],
        "simulation_id": int(simulation_id),
        "seed": int(seed),
        "treaty_variant": treaty_variant,
        "tail_type": scenario["tail_type"],
        "inflation_scenario": scenario["inflation_scenario"],
        "structural_break": bool(
            scenario["apply_structural_break"]
        ),
        "base_attachment": float(PILOT_XOL_ATTACHMENT),
        "base_limit": float(PILOT_XOL_LIMIT),
        "index_reference_year": int(CLAUSE_INDEX_BASE_YEAR),
    }
    for column in [
        "gross_ultimate",
        "ceded_ultimate",
        "retained_ultimate",
        "ceded_share",
        "claim_count",
        "attaching_claims",
        "attachment_frequency",
        "exhausting_claims",
        "exhaustion_frequency",
        "mean_ceded_per_ceded_claim",
        "ceded_paid_to_date",
        "ceded_true_reserve",
        "retained_true_reserve",
        "gross_true_reserve",
        "ceded_unpaid_proportion",
    ]:
        result[column] = np.nan
    result.update(
        {
            "success": False,
            "failure_message": f"{type(error).__name__}: {error}",
            "runtime_seconds": float(runtime_seconds),
        }
    )
    return result


def _run_portfolios(
    *,
    simulations: int,
    include_indexed: bool,
    frozen_truth: pd.DataFrame,
) -> dict[str, Any]:
    result_rows: list[dict[str, Any]] = []
    accident_year_parts: list[pd.DataFrame] = []
    development_parts: list[pd.DataFrame] = []
    schedule_parts: list[pd.DataFrame] = []
    reproduction_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    variants = TREATY_VARIANTS if include_indexed else (FIXED_NOMINAL,)

    for simulation_id in range(1, simulations + 1):
        seed = END_TO_END_BASE_SEED + simulation_id

        for scenario in _selected_scenarios():
            _, gross_payments = simulate_portfolio(
                simulation_id=simulation_id,
                frequency_scenario=scenario["frequency_scenario"],
                tail_type=scenario["tail_type"],
                inflation_scenario=scenario["inflation_scenario"],
                apply_structural_break=scenario[
                    "apply_structural_break"
                ],
                seed=seed,
            )
            gross_payments = gross_payments.copy()
            gross_payments["scenario_id"] = scenario["scenario_id"]
            baseline_payments, baseline_claims = apply_xol_to_payments(
                payments=gross_payments,
                attachment=PILOT_XOL_ATTACHMENT,
                limit=PILOT_XOL_LIMIT,
            )
            variant_payments: dict[str, pd.DataFrame] = {}
            variant_claims: dict[str, pd.DataFrame] = {}

            for treaty_variant in variants:
                started = time.perf_counter()

                try:
                    reinsured, claim_summary, schedule = (
                        apply_treaty_variant_to_payments(
                            payments=gross_payments,
                            treaty_variant=treaty_variant,
                            inflation_scenario=scenario[
                                "inflation_scenario"
                            ],
                            scenario_id=scenario["scenario_id"],
                        )
                    )
                    accident_year = build_accident_year_summary(
                        reinsured_payments=reinsured,
                        claim_summary=claim_summary,
                        scenario=scenario,
                        simulation_id=simulation_id,
                        seed=seed,
                        treaty_variant=treaty_variant,
                    )
                    runtime_seconds = time.perf_counter() - started
                    result = build_portfolio_result(
                        accident_year_summary=accident_year,
                        claim_summary=claim_summary,
                        scenario=scenario,
                        simulation_id=simulation_id,
                        seed=seed,
                        treaty_variant=treaty_variant,
                        runtime_seconds=runtime_seconds,
                    )
                    development = build_development_summary(
                        reinsured_payments=reinsured,
                        scenario_id=scenario["scenario_id"],
                        simulation_id=simulation_id,
                        treaty_variant=treaty_variant,
                    )
                    result_rows.append(result)
                    accident_year_parts.append(accident_year)
                    development_parts.append(development)
                    variant_payments[treaty_variant] = reinsured
                    variant_claims[treaty_variant] = claim_summary

                    if simulation_id == 1:
                        schedule_parts.append(schedule)

                    payment_reconciliation = np.allclose(
                        reinsured["nominal_gross_payment"],
                        reinsured["nominal_ceded_payment"]
                        + reinsured["nominal_retained_payment"],
                        rtol=NUMERICAL_RTOL,
                        atol=NUMERICAL_ATOL,
                    )
                    claim_reconciliation = np.allclose(
                        claim_summary["nominal_gross_ultimate"],
                        claim_summary["nominal_ceded_ultimate"]
                        + claim_summary["nominal_retained_ultimate"],
                        rtol=NUMERICAL_RTOL,
                        atol=NUMERICAL_ATOL,
                    )
                    validation_rows.append(
                        {
                            "scenario_id": scenario["scenario_id"],
                            "simulation_id": simulation_id,
                            "treaty_variant": treaty_variant,
                            "gross_stream_unchanged": bool(
                                np.array_equal(
                                    reinsured[
                                        "nominal_gross_payment"
                                    ].to_numpy(),
                                    baseline_payments[
                                        "nominal_gross_payment"
                                    ].to_numpy(),
                                )
                            ),
                            "payment_reconciliation": bool(
                                payment_reconciliation
                            ),
                            "claim_reconciliation": bool(
                                claim_reconciliation
                            ),
                            "claim_ceded_nonnegative": bool(
                                claim_summary[
                                    "nominal_ceded_ultimate"
                                ].ge(0.0).all()
                            ),
                            "claim_ceded_within_limit": bool(
                                (
                                    claim_summary[
                                        "nominal_ceded_ultimate"
                                    ]
                                    <= claim_summary["indexed_limit"]
                                    + NUMERICAL_ATOL
                                ).all()
                            ),
                        }
                    )
                except Exception as error:
                    result_rows.append(
                        _failure_result(
                            scenario=scenario,
                            simulation_id=simulation_id,
                            seed=seed,
                            treaty_variant=treaty_variant,
                            runtime_seconds=(
                                time.perf_counter() - started
                            ),
                            error=error,
                        )
                    )

            if FIXED_NOMINAL in variant_payments:
                reproduction = compare_fixed_nominal_to_baseline(
                    fixed_payments=variant_payments[FIXED_NOMINAL],
                    fixed_claims=variant_claims[FIXED_NOMINAL],
                    baseline_payments=baseline_payments,
                    baseline_claims=baseline_claims,
                    scenario_id=scenario["scenario_id"],
                    simulation_id=simulation_id,
                )
                fixed_result = result_rows[-len(variants)]
                maximum_absolute, maximum_relative, nonmatching = (
                    _truth_reproduction_differences(
                        fixed_result=fixed_result,
                        frozen_truth=frozen_truth,
                    )
                )
                reproduction.update(
                    {
                        "max_absolute_frozen_step16_truth_difference": (
                            maximum_absolute
                        ),
                        "max_relative_frozen_step16_truth_difference": (
                            maximum_relative
                        ),
                        "nonmatching_frozen_step16_truth_values": (
                            nonmatching
                        ),
                    }
                )
                reproduction["passed"] = bool(
                    reproduction["passed"] and nonmatching == 0
                )
                reproduction_rows.append(reproduction)

    return {
        "results": pd.DataFrame(result_rows),
        "by_accident_year": (
            pd.concat(accident_year_parts, ignore_index=True)
            if accident_year_parts
            else pd.DataFrame()
        ),
        "development_summary": (
            pd.concat(development_parts, ignore_index=True)
            if development_parts
            else pd.DataFrame()
        ),
        "treaty_term_schedule": (
            pd.concat(schedule_parts, ignore_index=True)
            .drop_duplicates(
                ["scenario_id", "treaty_variant", "accident_year"]
            )
            .sort_values(
                ["scenario_id", "treaty_variant", "accident_year"]
            )
            .reset_index(drop=True)
            if schedule_parts
            else pd.DataFrame()
        ),
        "baseline_reproduction": pd.DataFrame(reproduction_rows),
        "validation": pd.DataFrame(validation_rows),
    }


def _build_acceptance_report(
    *,
    mode: str,
    simulations: int,
    experiment: dict[str, Any],
    frozen_hash_before: str,
    frozen_hash_after: str,
    gate_approved: bool,
) -> pd.DataFrame:
    results = experiment["results"]
    by_ay = experiment["by_accident_year"]
    schedules = experiment["treaty_term_schedule"]
    reproduction = experiment["baseline_reproduction"]
    validation = experiment["validation"]
    expected_variants = (
        {FIXED_NOMINAL, FULLY_INDEXED}
        if mode != "baseline-gate"
        else {FIXED_NOMINAL}
    )
    expected_rows = 9 * simulations * len(expected_variants)
    indexed_schedule = schedules.loc[
        schedules["treaty_variant"].eq(FULLY_INDEXED)
    ]
    fixed_schedule = schedules.loc[
        schedules["treaty_variant"].eq(FIXED_NOMINAL)
    ]
    result_reconciliation = np.isclose(
        results["gross_ultimate"],
        results["ceded_ultimate"] + results["retained_ultimate"],
        rtol=NUMERICAL_RTOL,
        atol=NUMERICAL_ATOL,
    )
    ay_reconciliation = np.isclose(
        by_ay["gross_ultimate"],
        by_ay["ceded_ultimate"] + by_ay["retained_ultimate"],
        rtol=NUMERICAL_RTOL,
        atol=NUMERICAL_ATOL,
    )
    portfolio_from_ay = (
        by_ay.groupby(RESULT_KEY_COLUMNS, as_index=False)[
            [
                "gross_ultimate",
                "ceded_ultimate",
                "retained_ultimate",
                "gross_true_reserve",
                "ceded_true_reserve",
                "retained_true_reserve",
            ]
        ]
        .sum()
        .merge(
            results[
                RESULT_KEY_COLUMNS
                + [
                    "gross_ultimate",
                    "ceded_ultimate",
                    "retained_ultimate",
                    "gross_true_reserve",
                    "ceded_true_reserve",
                    "retained_true_reserve",
                ]
            ],
            on=RESULT_KEY_COLUMNS,
            suffixes=("_ay", "_result"),
            validate="one_to_one",
        )
    )
    ay_portfolio_match = all(
        np.allclose(
            portfolio_from_ay[f"{column}_ay"],
            portfolio_from_ay[f"{column}_result"],
            rtol=NUMERICAL_RTOL,
            atol=NUMERICAL_ATOL,
        )
        for column in [
            "gross_ultimate",
            "ceded_ultimate",
            "retained_ultimate",
            "gross_true_reserve",
            "ceded_true_reserve",
            "retained_true_reserve",
        ]
    )
    gross_invariant = results.groupby(
        ["scenario_id", "simulation_id"]
    )[["gross_ultimate", "gross_true_reserve"]].nunique().le(1).all().all()
    failures = results.loc[~results["success"]]
    failures_reported = failures.empty or failures[
        "failure_message"
    ].fillna("").str.len().gt(0).all()
    full_formula = np.allclose(
        indexed_schedule["indexed_attachment"],
        indexed_schedule["base_attachment"]
        * indexed_schedule["inflation_index_value"]
        / indexed_schedule["reference_inflation_index_value"],
        rtol=NUMERICAL_RTOL,
        atol=NUMERICAL_ATOL,
    ) and np.allclose(
        indexed_schedule["indexed_limit"],
        indexed_schedule["base_limit"]
        * indexed_schedule["inflation_index_value"]
        / indexed_schedule["reference_inflation_index_value"],
        rtol=NUMERICAL_RTOL,
        atol=NUMERICAL_ATOL,
    )
    simulation_counts = results.groupby(
        ["scenario_id", "treaty_variant"]
    )["simulation_id"].nunique()

    checks = [
        ("exact_expected_result_row_count", len(results) == expected_rows,
         f"actual={len(results)}, expected={expected_rows}"),
        ("expected_treaty_variants_present", set(results["treaty_variant"])
         == expected_variants, ""),
        ("all_nine_scenarios_present", results["scenario_id"].nunique() == 9,
         f"actual={results['scenario_id'].nunique()}"),
        ("expected_simulations_per_scenario_and_variant",
         bool(simulation_counts.eq(simulations).all()),
         f"expected={simulations}"),
        ("same_gross_portfolio_used_across_variants",
         bool(validation["gross_stream_unchanged"].all()), ""),
        ("identical_gross_ultimate_across_variants", bool(gross_invariant), ""),
        ("fixed_nominal_reproduces_baseline_xol",
         bool(reproduction["passed"].all()),
         f"rows={len(reproduction)}"),
        ("indexed_attachment_is_finite_and_positive",
         bool(np.isfinite(indexed_schedule["indexed_attachment"]).all()
              and indexed_schedule["indexed_attachment"].gt(0.0).all())
         if not indexed_schedule.empty else mode == "baseline-gate", ""),
        ("indexed_limit_is_finite_and_positive",
         bool(np.isfinite(indexed_schedule["indexed_limit"]).all()
              and indexed_schedule["indexed_limit"].gt(0.0).all())
         if not indexed_schedule.empty else mode == "baseline-gate", ""),
        ("fixed_nominal_attachment_is_exactly_two_million",
         bool(fixed_schedule["indexed_attachment"].eq(
             PILOT_XOL_ATTACHMENT).all()), ""),
        ("fixed_nominal_limit_is_exactly_five_million",
         bool(fixed_schedule["indexed_limit"].eq(PILOT_XOL_LIMIT).all()), ""),
        ("indexed_terms_follow_prespecified_rule",
         bool(full_formula) if not indexed_schedule.empty
         else mode == "baseline-gate", ""),
        ("no_future_truth_enters_index_calculation",
         calibration_api_uses_only_approved_index_variables(), ""),
        ("gross_equals_ceded_plus_retained",
         bool(result_reconciliation.all() and ay_reconciliation.all()
              and validation["payment_reconciliation"].all()
              and validation["claim_reconciliation"].all()), ""),
        ("claim_level_ceded_amounts_are_nonnegative",
         bool(validation["claim_ceded_nonnegative"].all()), ""),
        ("claim_level_ceded_does_not_exceed_limit",
         bool(validation["claim_ceded_within_limit"].all()), ""),
        ("portfolio_totals_reconcile_to_claim_results",
         bool(validation["claim_reconciliation"].all()), ""),
        ("accident_year_totals_reconcile_to_portfolio",
         bool(ay_portfolio_match), ""),
        ("gross_truth_is_identical_before_reinsurance",
         bool(gross_invariant), ""),
        ("result_keys_are_unique",
         not results.duplicated(RESULT_KEY_COLUMNS).any(), ""),
        ("accident_year_keys_are_unique",
         not by_ay.duplicated(ACCIDENT_YEAR_KEY_COLUMNS).any(), ""),
        ("failures_are_retained_and_reported", bool(failures_reported),
         f"failures={len(failures)}"),
        ("frozen_step16_outputs_are_unchanged",
         frozen_hash_before == frozen_hash_after,
         f"sha256={frozen_hash_after}"),
        ("approved_baseline_gate_available",
         bool(gate_approved or mode == "baseline-gate"), ""),
    ]

    return pd.DataFrame(
        [
            {"check": check, "passed": bool(passed), "detail": detail}
            for check, passed, detail in checks
        ]
    )


def _method_note(mode: str, simulations: int) -> str:
    return f"""# Step 29 — Simplified Treaty Indexation Sensitivity

## Scope

This `{mode}` run is an optional reinsurance-mechanics sensitivity. It
does not alter the frozen baseline treaty, simulation scenarios, payment
mechanisms, reserving models, or Steps 16–28 outputs. No reserving model
is fitted in Step 29.

## Monetary and timing convention

Claim ultimate severity is generated in real 2010-base money. Each real
payment is converted to nominal money using the existing scenario-specific
claims-inflation index for its payment calendar year. XoL recovery is then
calculated from cumulative nominal paid amounts for each claim.

The simulation specification assigns treaty terms by claim accident year.
The terms are fixed for that claim's subsequent lifetime. This run therefore
uses accident-year indexation rather than payment-year or report-year
indexation. It requires no future payment or evaluation-truth information.

For accident year i, with reference year 2010:

    A_i = £2,000,000 × I_i / I_2010
    L_i = £5,000,000 × I_i / I_2010

Both terms use the same factor, preserving the real layer shape. The fixed
nominal comparator retains £2m attachment and £5m limit in every accident
year. This is a simplified stabilisation/indexation sensitivity, not a full
legal implementation of every market stabilisation clause.

## Pairing and evaluation

The run contains {simulations} simulations per scenario. Each gross portfolio
is simulated once per scenario/simulation and both treaty variants are applied
to that exact payment stream. Primary outputs describe ceded volume, claim
penetration, limit exhaustion, reserve truth, accident-year emergence, and
development-age payments. The Step 16 archive is read only for the fixed
nominal reproduction check.

The fixed-nominal reproduction gate was completed before the indexed
sensitivity was authorised. The `{mode}` design is 9 scenarios ×
{simulations} simulations × {1 if mode == "baseline-gate" else 2} treaty
variant(s). This remains a sensitivity analysis rather than treaty
optimisation or a complete legal representation of a market clause.
"""


def main() -> None:
    """Run and validate the requested Step 29 treaty sensitivity mode."""

    arguments = parse_arguments()
    validate_config()

    if arguments.simulations < 1:
        raise ValueError("--simulations must be positive.")

    output_directory = arguments.output_root / arguments.run_label

    if output_directory.exists():
        raise FileExistsError(output_directory)

    if not arguments.frozen_step16.is_dir():
        raise FileNotFoundError(arguments.frozen_step16)

    gate_approved = False

    if arguments.mode in {"smoke", "final"}:
        if not arguments.baseline_gate_report.exists():
            raise FileNotFoundError(arguments.baseline_gate_report)

        gate_report = pd.read_csv(arguments.baseline_gate_report)
        gate_approved = bool(gate_report["passed"].all())

        if not gate_approved:
            raise RuntimeError("The fixed-nominal baseline gate failed.")

    frozen_hash_before = sha256_directory(arguments.frozen_step16)
    frozen_results_hash_before = sha256_file(
        arguments.frozen_step16 / "results.csv"
    )
    frozen_truth = _load_frozen_step16_truth(arguments.frozen_step16)
    started = time.perf_counter()
    experiment = _run_portfolios(
        simulations=arguments.simulations,
        include_indexed=arguments.mode != "baseline-gate",
        frozen_truth=frozen_truth,
    )
    frozen_hash_after = sha256_directory(arguments.frozen_step16)
    frozen_results_hash_after = sha256_file(
        arguments.frozen_step16 / "results.csv"
    )
    acceptance = _build_acceptance_report(
        mode=arguments.mode,
        simulations=arguments.simulations,
        experiment=experiment,
        frozen_hash_before=frozen_hash_before,
        frozen_hash_after=frozen_hash_after,
        gate_approved=gate_approved,
    )
    results = experiment["results"].sort_values(
        RESULT_KEY_COLUMNS
    ).reset_index(drop=True)
    comparison = (
        build_comparison_summary(results)
        if arguments.mode != "baseline-gate"
        else pd.DataFrame()
    )
    scenario_summary = (
        build_scenario_summary(results)
        if arguments.mode != "baseline-gate"
        else pd.DataFrame()
    )
    paired_comparison = (
        build_paired_comparison(results)
        if arguments.mode != "baseline-gate"
        else pd.DataFrame()
    )
    failures = (
        results.loc[~results["success"]]
        .groupby(
            ["scenario_id", "treaty_variant", "failure_message"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "failures"})
    )
    elapsed_seconds = time.perf_counter() - started

    output_directory.mkdir(parents=True, exist_ok=False)
    results.to_csv(output_directory / "results.csv", index=False)
    experiment["by_accident_year"].to_csv(
        output_directory / "by_accident_year.csv",
        index=False,
    )
    experiment["development_summary"].to_csv(
        output_directory / "development_summary.csv",
        index=False,
    )
    experiment["treaty_term_schedule"].to_csv(
        output_directory / "treaty_term_schedule.csv",
        index=False,
    )
    experiment["baseline_reproduction"].to_csv(
        output_directory / "baseline_reproduction.csv",
        index=False,
    )
    comparison.to_csv(
        output_directory / "comparison_summary.csv",
        index=False,
    )
    scenario_summary.to_csv(
        output_directory / "scenario_summary.csv",
        index=False,
    )
    paired_comparison.to_csv(
        output_directory / "paired_comparison.csv",
        index=False,
    )
    acceptance.to_csv(
        output_directory / "acceptance_report.csv",
        index=False,
    )
    failures.to_csv(
        output_directory / "failure_summary.csv",
        index=False,
    )
    (output_directory / "STEP29_METHOD_NOTE.md").write_text(
        _method_note(arguments.mode, arguments.simulations),
        encoding="utf-8",
    )
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "step": 29,
        "mode": arguments.mode,
        "simulations_per_scenario": arguments.simulations,
        "scenario_ids": [
            scenario["scenario_id"] for scenario in _selected_scenarios()
        ],
        "treaty_variants": sorted(results["treaty_variant"].unique()),
        "base_seed": END_TO_END_BASE_SEED,
        "base_attachment": PILOT_XOL_ATTACHMENT,
        "base_limit": PILOT_XOL_LIMIT,
        "index_reference_year": CLAUSE_INDEX_BASE_YEAR,
        "index_timing": "accident_year",
        "gross_portfolios_simulated_once_per_scenario_simulation": True,
        "reserving_models_fitted": [],
        "result_rows": int(len(results)),
        "accident_year_rows": int(len(experiment["by_accident_year"])),
        "scenario_summary_rows": int(len(scenario_summary)),
        "paired_comparison_rows": int(len(paired_comparison)),
        "frozen_step16_directory_sha256_before": frozen_hash_before,
        "frozen_step16_directory_sha256_after": frozen_hash_after,
        "frozen_step16_results_sha256_before": (
            frozen_results_hash_before
        ),
        "frozen_step16_results_sha256_after": frozen_results_hash_after,
        "elapsed_seconds": elapsed_seconds,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Step 29 {arguments.mode} completed.")
    print(f"Output: {output_directory}")
    print(f"Result rows: {len(results)}")
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(acceptance.to_string(index=False))

    if not acceptance["passed"].all():
        raise RuntimeError("Step 29 acceptance checks failed.")


if __name__ == "__main__":
    main()

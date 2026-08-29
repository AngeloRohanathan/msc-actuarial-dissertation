"""Run Step 28 independent ceded BF development sensitivity."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import pandas as pd

from config import (
    BF_BENCHMARK_INCREMENTAL_PATTERNS,
    BF_STRUCTURAL_BREAK_ACCIDENT_YEAR,
    END_TO_END_BASE_SEED,
    END_TO_END_SCENARIOS,
    EXPECTED_LOSS_ACCIDENT_YEARS,
    EXPECTED_LOSS_CALIBRATION_SEED_BASE,
    EXPECTED_LOSS_CALIBRATION_SIMULATIONS,
    EXPECTED_LOSS_PRIOR_VERSION,
    PILOT_XOL_ATTACHMENT,
    PILOT_XOL_LIMIT,
    VALUATION_YEAR,
    validate_config,
)
from src.ceded_bf_development_sensitivity import (
    BASELINE_REPRODUCTION_ATOL,
    BASELINE_REPRODUCTION_RTOL,
    CALIBRATION_VERSION,
    StreamingDevelopmentCalibration,
    build_baseline_gate_acceptance_report,
    build_calibration_acceptance_report,
    build_final_acceptance_report,
    build_inflation_pattern_comparison,
    build_natural_horizon_summary,
    build_pattern_comparison,
    build_result_summary,
    compare_baseline_to_frozen_step21,
    run_ceded_bf_variants,
    select_frozen_ceded_evaluation_input,
    sha256_file,
)
from src.expected_loss_prior import (
    build_calibration_seed_schedule,
)
from src.reinsurance import apply_xol_to_payments
from src.simulation import simulate_portfolio


DEFAULT_OUTPUT_ROOT = Path(
    "outputs/step28_ceded_bf_development_sensitivity"
)
DEFAULT_PRIOR = Path(
    "data/calibration/expected_loss_prior/final_2000/"
    "expected_loss_prior.csv"
)
DEFAULT_PATTERN_FILE = (
    DEFAULT_OUTPUT_ROOT
    / "calibration"
    / "final_2000"
    / "ceded_development_pattern.csv"
)
DEFAULT_STEP21_DETAIL = Path(
    "outputs/step21_paid_bf/final_50/"
    "bf_by_accident_year.csv"
)
DEFAULT_STEP21_RESULTS = Path(
    "outputs/step21_paid_bf/final_50/bf_results.csv"
)
DEFAULT_BASELINE_GATE_REPORT = (
    DEFAULT_OUTPUT_ROOT
    / "baseline_gate_50"
    / "acceptance_report.csv"
)


def parse_arguments() -> argparse.Namespace:
    """Read Step 28 command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "calibrate",
            "baseline-gate",
            "final",
        ],
        required=True,
    )
    parser.add_argument(
        "--run-label",
        required=True,
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--prior",
        type=Path,
        default=DEFAULT_PRIOR,
    )
    parser.add_argument(
        "--pattern-file",
        type=Path,
        default=DEFAULT_PATTERN_FILE,
    )
    parser.add_argument(
        "--step21-detail",
        type=Path,
        default=DEFAULT_STEP21_DETAIL,
    )
    parser.add_argument(
        "--step21-results",
        type=Path,
        default=DEFAULT_STEP21_RESULTS,
    )
    parser.add_argument(
        "--baseline-gate-report",
        type=Path,
        default=DEFAULT_BASELINE_GATE_REPORT,
    )
    return parser.parse_args()


def _calibration_method_note() -> str:
    return r"""# Step 28 — Independent Ceded Development Calibration

## Objective

This calibration regenerates the frozen Step 19 independent pricing
portfolios solely to estimate ceded paid-development patterns. It does
not alter the Step 19 expected-loss prior or use evaluation portfolios.

## Frozen simulation design

The calibration uses the original nine scenarios, 2,000 simulations
per scenario, accident years 2010–2024, calibration seed range
91,000,001–91,002,000, and the frozen £5m xs £2m per-claim XoL treaty.

## Development age

The dissertation convention is retained exactly:

\[
j = \text{payment calendar year} - \text{accident year} + 1.
\]

Reporting delay therefore contributes to development age. The
calibration retains the complete natural payment horizon and does not
truncate or renormalise at development year 10.

## Volume-weighted calibration

For calibration stratum (h) and development age (j):

\[
q^{ceded}_{h,j}
=
\frac{\sum_{m,i} C^{ceded}_{m,i,j}}
{\sum_{m,i} U^{ceded}_{m,i}},
\qquad
p^{ceded}_{h,j}
=
\sum_{k\leq j} q^{ceded}_{h,k}.
\]

Zero-ceded accident years remain in the calibration population and
contribute zero volume. No individual AY ratios are averaged.

## Strata

Short-tail and long-tail business are separate. Stable, emerging and
shock inflation scenarios remain separate because nominal inflation
and cumulative-paid XoL attachment can affect ceded payment timing.

No-break patterns use all 2010–2024 AYs in the corresponding independent
no-break scenario. Post-break patterns use only AY2018–2024 from the
corresponding independently simulated structural-break scenario.
Pre-break evaluation cohorts use the matched no-break calibration.

## Stability

Every usable pattern must have at least 1,000 calibration simulations,
5,000 AY observations, positive ceded volume and a maximum first-half
versus second-half cumulative-pattern difference no greater than 0.02.
All patterns must reconcile to their full lifetime ceded ultimates and
reach cumulative proportion one at natural terminal maturity.
"""


def _evaluation_method_note(
    *,
    mode: str,
) -> str:
    return rf"""# Step 28 — Ceded BF Development Sensitivity

## Run type

`{mode}`

## Motivation and independent calibration

Step 22 identified development-pattern misspecification as the main
driver of the Break-Aware BF ceded under-reserving result. Step 28 tests
that diagnosis by replacing the gross-derived BF paid pattern with a
ceded-specific pattern calibrated independently from the regenerated
Step 19 pricing portfolios. Calibration seeds 91,000,001–91,002,000 are
disjoint from the frozen Step 21 evaluation seeds.

The independent pattern is volume weighted: ceded payments at each
development age are divided by total lifetime ceded ultimate for the
same calibration population. Development age remains payment calendar
year minus accident year plus one. Short-tail patterns reach natural
maturity at DY6 and long-tail patterns at DY14. Nothing is truncated or
renormalised at DY10.

Stable, emerging and shock inflation strata remain separate. Long-tail
post-break AY2018–2024 cohorts have independently calibrated post-break
patterns. Pre-break cohorts use the corresponding no-break pattern.

## Frozen evaluation

The evaluation reuses frozen Step 21 ceded paid-to-date, independent
Step 19 expected-loss prior ultimates and true reserves. No calibration
or evaluation portfolio is resimulated. The Expected Loss prior and BF
formula are frozen.

The paid BF formula remains:

\[
\widehat R_i = U_i^{{prior}}(1-p_i).
\]

Standard and Break-Aware BF retain the original Step 21 patterns.
Ceded-Specific BF uses the matched independent no-break pattern.
Break-Aware Ceded-Specific BF uses the matched independent post-break
pattern only for post-break AYs in structural-break scenarios. In
no-break scenarios the two ceded-specific variants are identical by
design. For valuation ages beyond natural maturity, paid proportion is
one; no extrapolation is performed.

Only (p_i) changes between corresponding variants. Prior, paid-to-date,
truth, valuation year and accident year remain fixed. Pattern calibration
does not use evaluation truth. Accuracy and paired APE comparisons are
descriptive only; formal paired inference is deferred to Step 31.
"""


def run_calibration(
    arguments: argparse.Namespace,
) -> None:
    """Regenerate frozen Step 19 portfolios and calibrate patterns."""

    validate_config()
    simulations = (
        EXPECTED_LOSS_CALIBRATION_SIMULATIONS
        if arguments.simulations is None
        else int(arguments.simulations)
    )

    if simulations != EXPECTED_LOSS_CALIBRATION_SIMULATIONS:
        raise ValueError(
            "Final Step 28 calibration must use the frozen 2,000 "
            "Step 19 simulations per scenario."
        )

    if not arguments.prior.exists():
        raise FileNotFoundError(arguments.prior)

    output_directory = (
        arguments.output_root
        / "calibration"
        / arguments.run_label
    )

    if output_directory.exists():
        raise FileExistsError(output_directory)

    prior_hash_before = sha256_file(arguments.prior)
    seeds = build_calibration_seed_schedule(
        number_of_simulations=simulations,
        seed_base=EXPECTED_LOSS_CALIBRATION_SEED_BASE,
    )
    accumulator = StreamingDevelopmentCalibration(
        calibration_simulations=simulations,
        structural_break_year=(
            BF_STRUCTURAL_BREAK_ACCIDENT_YEAR
        ),
        seed_start=min(seeds),
        seed_end=max(seeds),
    )
    total_portfolios = (
        len(END_TO_END_SCENARIOS) * simulations
    )
    completed = 0
    started = time.perf_counter()

    for scenario in END_TO_END_SCENARIOS:
        for calibration_id, seed in enumerate(
            seeds,
            start=1,
        ):
            completed += 1

            if (
                completed == 1
                or completed % 100 == 0
                or completed == total_portfolios
            ):
                print(
                    f"[{completed}/{total_portfolios}] "
                    f"scenario={scenario['scenario_id']}, "
                    f"calibration={calibration_id}",
                    flush=True,
                )

            _, payments = simulate_portfolio(
                simulation_id=calibration_id,
                frequency_scenario=(
                    scenario["frequency_scenario"]
                ),
                tail_type=scenario["tail_type"],
                inflation_scenario=(
                    scenario["inflation_scenario"]
                ),
                apply_structural_break=(
                    scenario["apply_structural_break"]
                ),
                seed=seed,
            )
            reinsured_payments, _ = apply_xol_to_payments(
                payments=payments,
                attachment=PILOT_XOL_ATTACHMENT,
                limit=PILOT_XOL_LIMIT,
            )
            accumulator.add_portfolio(
                reinsured_payments=reinsured_payments,
                scenario=scenario,
                calibration_id=calibration_id,
                seed=seed,
                accident_years=(
                    EXPECTED_LOSS_ACCIDENT_YEARS
                ),
            )

    aggregate_table, pattern_table, support_summary = (
        accumulator.build_outputs()
    )
    horizon_summary = build_natural_horizon_summary(
        aggregate_table=aggregate_table
    )
    inflation_comparison = (
        build_inflation_pattern_comparison(
            pattern_table=pattern_table
        )
    )
    prior_hash_after = sha256_file(arguments.prior)
    acceptance_report = (
        build_calibration_acceptance_report(
            pattern_table=pattern_table,
            aggregate_table=aggregate_table,
            support_summary=support_summary,
            expected_strata=9,
            calibration_seed_base=(
                EXPECTED_LOSS_CALIBRATION_SEED_BASE
            ),
            calibration_simulations=simulations,
            evaluation_seed_base=END_TO_END_BASE_SEED,
            prior_hash_before=prior_hash_before,
            prior_hash_after=prior_hash_after,
        )
    )
    elapsed_seconds = time.perf_counter() - started

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )
    pattern_table.to_csv(
        output_directory / "ceded_development_pattern.csv",
        index=False,
    )
    aggregate_table.to_csv(
        output_directory / "development_aggregates.csv",
        index=False,
    )
    support_summary.to_csv(
        output_directory / "calibration_support_summary.csv",
        index=False,
    )
    horizon_summary.to_csv(
        output_directory / "natural_horizon_summary.csv",
        index=False,
    )
    inflation_comparison.to_csv(
        output_directory
        / "inflation_pattern_comparison.csv",
        index=False,
    )
    acceptance_report.to_csv(
        output_directory
        / "calibration_acceptance_report.csv",
        index=False,
    )
    (
        output_directory / "STEP28_CALIBRATION_METHOD_NOTE.md"
    ).write_text(
        _calibration_method_note(),
        encoding="utf-8",
    )
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "step": 28,
        "mode": "independent_ceded_development_calibration",
        "calibration_version": CALIBRATION_VERSION,
        "calibration_simulations_per_scenario": simulations,
        "scenario_ids": [
            scenario["scenario_id"]
            for scenario in END_TO_END_SCENARIOS
        ],
        "accident_years": list(
            EXPECTED_LOSS_ACCIDENT_YEARS
        ),
        "calibration_seed_start": min(seeds),
        "calibration_seed_end": max(seeds),
        "evaluation_seed_base": END_TO_END_BASE_SEED,
        "attachment": PILOT_XOL_ATTACHMENT,
        "limit": PILOT_XOL_LIMIT,
        "step19_prior_path": str(arguments.prior),
        "step19_prior_sha256_before": prior_hash_before,
        "step19_prior_sha256_after": prior_hash_after,
        "evaluation_files_used_for_calibration": [],
        "portfolio_resimulation_scope": (
            "exact_frozen_step19_independent_calibration_only"
        ),
        "development_year_definition": (
            "payment_calendar_year - accident_year + 1"
        ),
        "truncation_or_renormalisation_applied": False,
        "calibration_strata": int(len(support_summary)),
        "pattern_rows": int(len(pattern_table)),
        "elapsed_seconds": elapsed_seconds,
    }

    with (
        output_directory / "manifest.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("\nStep 28 independent calibration completed.")
    print(f"Output: {output_directory}")
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print("\nNatural horizons:")
    print(horizon_summary.to_string(index=False))
    print("\nSupport summary:")
    print(support_summary.to_string(index=False))
    print("\nAcceptance report:")
    print(acceptance_report.to_string(index=False))

    if not acceptance_report["passed"].all():
        raise RuntimeError(
            "Step 28 calibration acceptance failed."
        )


def _filter_frozen_results(
    *,
    frozen_results: pd.DataFrame,
    selected_input: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "scenario_id",
        "simulation_id",
        "basis",
    ]
    return frozen_results.merge(
        selected_input[keys].drop_duplicates(),
        on=keys,
        how="inner",
        validate="many_to_one",
    )


def run_evaluation(
    arguments: argparse.Namespace,
) -> None:
    """Run the baseline gate or later approved final sensitivity."""

    simulations = (
        50
        if arguments.simulations is None
        else int(arguments.simulations)
    )

    for path in [
        arguments.step21_detail,
        arguments.step21_results,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    if arguments.mode == "final":
        if not arguments.prior.exists():
            raise FileNotFoundError(arguments.prior)

        if not arguments.pattern_file.exists():
            raise FileNotFoundError(arguments.pattern_file)

        if not arguments.baseline_gate_report.exists():
            raise FileNotFoundError(
                "An approved baseline gate report is required: "
                f"{arguments.baseline_gate_report}"
            )

        gate_report = pd.read_csv(
            arguments.baseline_gate_report
        )

        if not gate_report["passed"].all():
            raise RuntimeError(
                "The Step 28 baseline gate did not pass."
            )

    output_directory = (
        arguments.output_root / arguments.run_label
    )

    if output_directory.exists():
        raise FileExistsError(output_directory)

    source_detail = pd.read_csv(
        arguments.step21_detail
    )
    selected_input = select_frozen_ceded_evaluation_input(
        step21_detail=source_detail,
        simulations_per_scenario=simulations,
    )
    frozen_results = _filter_frozen_results(
        frozen_results=pd.read_csv(
            arguments.step21_results
        ),
        selected_input=selected_input,
    )
    prior_hash_before = (
        sha256_file(arguments.prior)
        if arguments.mode == "final"
        else None
    )
    pattern_hash_before = (
        sha256_file(arguments.pattern_file)
        if arguments.mode == "final"
        else None
    )
    pattern_table = (
        pd.read_csv(arguments.pattern_file)
        if arguments.mode == "final"
        else None
    )
    started = time.perf_counter()
    results, detail = run_ceded_bf_variants(
        frozen_evaluation_input=selected_input,
        benchmark_patterns=(
            BF_BENCHMARK_INCREMENTAL_PATTERNS
        ),
        valuation_year=VALUATION_YEAR,
        structural_break_year=(
            BF_STRUCTURAL_BREAK_ACCIDENT_YEAR
        ),
        pattern_table=pattern_table,
        include_ceded_specific=(
            arguments.mode == "final"
        ),
    )
    reproduction = compare_baseline_to_frozen_step21(
        results=results,
        frozen_step21_results=frozen_results,
        rtol=BASELINE_REPRODUCTION_RTOL,
        atol=BASELINE_REPRODUCTION_ATOL,
    )
    pattern_comparison = (
        build_pattern_comparison(results)
        if arguments.mode == "final"
        else pd.DataFrame()
    )
    prior_hash_after = (
        sha256_file(arguments.prior)
        if arguments.mode == "final"
        else None
    )
    pattern_hash_after = (
        sha256_file(arguments.pattern_file)
        if arguments.mode == "final"
        else None
    )

    if arguments.mode == "final":
        acceptance_report = build_final_acceptance_report(
            results=results,
            detail=detail,
            reproduction=reproduction,
            pattern_table=pattern_table,
            pattern_comparison=pattern_comparison,
            expected_scenarios=(
                selected_input["scenario_id"].nunique()
            ),
            simulations_per_scenario=simulations,
            structural_break_year=(
                BF_STRUCTURAL_BREAK_ACCIDENT_YEAR
            ),
            prior_hash_before=str(prior_hash_before),
            prior_hash_after=str(prior_hash_after),
        )
    else:
        acceptance_report = build_baseline_gate_acceptance_report(
            results=results.loc[
                results["model"].isin(
                    [
                        "bornhuetter_ferguson_standard",
                        "bornhuetter_ferguson_break_aware",
                    ]
                )
            ],
            detail=detail.loc[
                detail["model"].isin(
                    [
                        "bornhuetter_ferguson_standard",
                        "bornhuetter_ferguson_break_aware",
                    ]
                )
            ],
            reproduction=reproduction,
            expected_scenarios=(
                selected_input["scenario_id"].nunique()
            ),
            simulations_per_scenario=simulations,
        )
    summary = build_result_summary(results)
    failure_summary = pd.DataFrame(
        columns=[
            "scenario_id",
            "model",
            "failure_type",
            "failure_message",
            "failures",
        ]
    )
    elapsed_seconds = time.perf_counter() - started

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )
    results.to_csv(
        output_directory / "results.csv",
        index=False,
    )
    detail.to_csv(
        output_directory / "by_accident_year.csv",
        index=False,
    )
    summary.to_csv(
        output_directory / "summary.csv",
        index=False,
    )
    if arguments.mode == "final":
        pattern_comparison.to_csv(
            output_directory / "pattern_comparison.csv",
            index=False,
        )
    reproduction.to_csv(
        output_directory / "baseline_reproduction.csv",
        index=False,
    )
    acceptance_report.to_csv(
        output_directory / "acceptance_report.csv",
        index=False,
    )
    failure_summary.to_csv(
        output_directory / "failure_summary.csv",
        index=False,
    )
    (
        output_directory / "STEP28_METHOD_NOTE.md"
    ).write_text(
        _evaluation_method_note(mode=arguments.mode),
        encoding="utf-8",
    )
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "step": 28,
        "mode": arguments.mode,
        "simulations_per_scenario": simulations,
        "scenario_count": int(
            selected_input["scenario_id"].nunique()
        ),
        "basis": "ceded",
        "result_rows": int(len(results)),
        "detail_rows": int(len(detail)),
        "step21_detail_source": str(
            arguments.step21_detail
        ),
        "step21_results_source": str(
            arguments.step21_results
        ),
        "pattern_file": (
            str(arguments.pattern_file)
            if pattern_table is not None
            else None
        ),
        "pattern_sha256_before": pattern_hash_before,
        "pattern_sha256_after": pattern_hash_after,
        "step19_prior_sha256_before": prior_hash_before,
        "step19_prior_sha256_after": prior_hash_after,
        "evaluation_portfolios_resimulated": False,
        "calibration_portfolios_resimulated": False,
        "truth_used_for": "evaluation_only_after_estimation",
        "baseline_reproduction_rtol": (
            BASELINE_REPRODUCTION_RTOL
        ),
        "baseline_reproduction_atol": (
            BASELINE_REPRODUCTION_ATOL
        ),
        "elapsed_seconds": elapsed_seconds,
    }

    with (
        output_directory / "manifest.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"\nStep 28 {arguments.mode} completed.")
    print(f"Output: {output_directory}")
    print(f"Results: {len(results):,}")
    print("\nBaseline reproduction:")
    print(reproduction.to_string(index=False))
    print("\nAcceptance report:")
    print(acceptance_report.to_string(index=False))
    print(f"\nElapsed seconds: {elapsed_seconds:.3f}")

    if not acceptance_report["passed"].all():
        raise RuntimeError(
            "Step 28 evaluation acceptance failed."
        )


def main() -> None:
    arguments = parse_arguments()

    if arguments.mode == "calibrate":
        run_calibration(arguments)
    else:
        run_evaluation(arguments)


if __name__ == "__main__":
    main()

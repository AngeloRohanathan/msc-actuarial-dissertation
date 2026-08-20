"""Build an independent pricing Monte Carlo expected-loss prior."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    END_TO_END_BASE_SEED,
    END_TO_END_SCENARIOS,
    EXPECTED_LOSS_ACCIDENT_YEARS,
    EXPECTED_LOSS_CALIBRATION_SEED_BASE,
    EXPECTED_LOSS_CALIBRATION_SIMULATIONS,
    EXPECTED_LOSS_CEDED_RELATIVE_MCSE_TARGET,
    EXPECTED_LOSS_GROSS_RELATIVE_MCSE_TARGET,
    EXPECTED_LOSS_PRIOR_VERSION,
    PILOT_XOL_ATTACHMENT,
    PILOT_XOL_LIMIT,
    validate_config,
)

from src.expected_loss_prior import (
    build_calibration_rows,
    build_calibration_seed_schedule,
    build_precision_summary,
    build_prior_acceptance_report,
    summarise_expected_loss_prior,
)

from src.reinsurance import (
    apply_xol_to_payments,
)

from src.simulation import (
    simulate_portfolio,
)


DEFAULT_OUTPUT_ROOT = Path(
    "data/calibration/expected_loss_prior"
)


def parse_arguments() -> argparse.Namespace:
    """Read command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Build an independent expected-loss "
            "pricing prior."
        )
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=(
            EXPECTED_LOSS_CALIBRATION_SIMULATIONS
        ),
        help=(
            "Calibration simulations per scenario."
        ),
    )

    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help=(
            "Run only a named scenario. "
            "May be supplied more than once."
        ),
    )

    parser.add_argument(
        "--run-label",
        required=True,
        type=str,
        help=(
            "Unique folder label for this "
            "calibration run."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    return parser.parse_args()


def select_scenarios(
    requested: list[str] | None,
) -> list[dict[str, Any]]:
    """Select all scenarios or a requested subset."""

    scenarios = [
        dict(scenario)
        for scenario in END_TO_END_SCENARIOS
    ]

    if requested is None:
        return scenarios

    requested_set = set(
        requested
    )

    selected = [
        scenario
        for scenario in scenarios
        if scenario[
            "scenario_id"
        ]
        in requested_set
    ]

    found = {
        scenario[
            "scenario_id"
        ]
        for scenario in selected
    }

    missing = (
        requested_set
        - found
    )

    if missing:
        raise ValueError(
            "Unknown scenarios requested: "
            f"{sorted(missing)}"
        )

    return selected


def main() -> None:
    """Run the independent prior calibration."""

    arguments = parse_arguments()

    validate_config()

    if arguments.simulations < 2:
        raise ValueError(
            "--simulations must be at least 2."
        )

    scenarios = select_scenarios(
        arguments.scenario
    )

    output_directory = (
        arguments.output_root
        / arguments.run_label
    )

    if output_directory.exists():
        raise FileExistsError(
            "Output directory already exists: "
            f"{output_directory}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    seeds = build_calibration_seed_schedule(
        number_of_simulations=(
            arguments.simulations
        ),
        seed_base=(
            EXPECTED_LOSS_CALIBRATION_SEED_BASE
        ),
    )

    all_records: list[
        pd.DataFrame
    ] = []

    total_portfolios = (
        arguments.simulations
        * len(scenarios)
    )

    completed_portfolios = 0

    run_start = time.perf_counter()

    for scenario in scenarios:
        scenario_records: list[
            pd.DataFrame
        ] = []

        for calibration_id, seed in enumerate(
            seeds,
            start=1,
        ):
            completed_portfolios += 1

            if (
                completed_portfolios == 1
                or completed_portfolios % 50 == 0
                or completed_portfolios
                == total_portfolios
            ):
                print(
                    f"[{completed_portfolios}/"
                    f"{total_portfolios}] "
                    f"scenario="
                    f"{scenario['scenario_id']}, "
                    f"calibration="
                    f"{calibration_id}"
                )

            _, simulated_payments = (
                simulate_portfolio(
                    simulation_id=(
                        calibration_id
                    ),
                    frequency_scenario=(
                        scenario[
                            "frequency_scenario"
                        ]
                    ),
                    tail_type=(
                        scenario[
                            "tail_type"
                        ]
                    ),
                    inflation_scenario=(
                        scenario[
                            "inflation_scenario"
                        ]
                    ),
                    apply_structural_break=(
                        scenario[
                            "apply_structural_break"
                        ]
                    ),
                    seed=seed,
                )
            )

            (
                reinsured_payments,
                _,
            ) = apply_xol_to_payments(
                payments=(
                    simulated_payments
                ),
                attachment=(
                    PILOT_XOL_ATTACHMENT
                ),
                limit=(
                    PILOT_XOL_LIMIT
                ),
            )

            rows = build_calibration_rows(
                reinsured_payments=(
                    reinsured_payments
                ),
                scenario=scenario,
                calibration_id=(
                    calibration_id
                ),
                seed=seed,
                accident_years=(
                    EXPECTED_LOSS_ACCIDENT_YEARS
                ),
            )

            scenario_records.append(
                rows
            )

        scenario_frame = pd.concat(
            scenario_records,
            ignore_index=True,
        )

        all_records.append(
            scenario_frame
        )

        checkpoint = pd.concat(
            all_records,
            ignore_index=True,
        )

        checkpoint.to_csv(
            output_directory
            / "calibration_records_checkpoint.csv",
            index=False,
        )

    calibration_records = pd.concat(
        all_records,
        ignore_index=True,
    )

    expected_loss_prior = (
        summarise_expected_loss_prior(
            calibration_records=(
                calibration_records
            ),
            pricing_assumption_version=(
                EXPECTED_LOSS_PRIOR_VERSION
            ),
            calibration_simulations=(
                arguments.simulations
            ),
            calibration_seed_base=(
                EXPECTED_LOSS_CALIBRATION_SEED_BASE
            ),
            attachment=(
                PILOT_XOL_ATTACHMENT
            ),
            limit=(
                PILOT_XOL_LIMIT
            ),
        )
    )

    precision_summary = (
        build_precision_summary(
            expected_loss_prior=(
                expected_loss_prior
            ),
            gross_relative_mcse_target=(
                EXPECTED_LOSS_GROSS_RELATIVE_MCSE_TARGET
            ),
            ceded_relative_mcse_target=(
                EXPECTED_LOSS_CEDED_RELATIVE_MCSE_TARGET
            ),
        )
    )

    acceptance_report = (
        build_prior_acceptance_report(
            calibration_records=(
                calibration_records
            ),
            expected_loss_prior=(
                expected_loss_prior
            ),
            number_of_scenarios=(
                len(scenarios)
            ),
            number_of_accident_years=(
                len(
                    EXPECTED_LOSS_ACCIDENT_YEARS
                )
            ),
            calibration_simulations=(
                arguments.simulations
            ),
            calibration_seed_base=(
                EXPECTED_LOSS_CALIBRATION_SEED_BASE
            ),
            evaluation_seed_base=(
                END_TO_END_BASE_SEED
            ),
        )
    )

    calibration_records.to_csv(
        output_directory
        / "calibration_records.csv",
        index=False,
    )

    expected_loss_prior.to_csv(
        output_directory
        / "expected_loss_prior.csv",
        index=False,
    )

    precision_summary.to_csv(
        output_directory
        / "precision_summary.csv",
        index=False,
    )

    acceptance_report.to_csv(
        output_directory
        / "acceptance_report.csv",
        index=False,
    )

    elapsed_seconds = (
        time.perf_counter()
        - run_start
    )

    manifest = {
        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "run_label": (
            arguments.run_label
        ),
        "method": (
            "independent_pricing_monte_carlo"
        ),
        "pricing_assumption_version": (
            EXPECTED_LOSS_PRIOR_VERSION
        ),
        "calibration_simulations_per_scenario": (
            arguments.simulations
        ),
        "number_of_scenarios": (
            len(scenarios)
        ),
        "scenario_ids": [
            scenario[
                "scenario_id"
            ]
            for scenario in scenarios
        ],
        "accident_years": list(
            EXPECTED_LOSS_ACCIDENT_YEARS
        ),
        "calibration_seed_base": (
            EXPECTED_LOSS_CALIBRATION_SEED_BASE
        ),
        "calibration_seed_minimum": (
            min(seeds)
        ),
        "calibration_seed_maximum": (
            max(seeds)
        ),
        "evaluation_seed_base": (
            END_TO_END_BASE_SEED
        ),
        "evaluation_results_used": False,
        "realised_evaluation_ultimates_used": False,
        "attachment": (
            PILOT_XOL_ATTACHMENT
        ),
        "limit": (
            PILOT_XOL_LIMIT
        ),
        "calibration_record_rows": int(
            len(
                calibration_records
            )
        ),
        "prior_rows": int(
            len(
                expected_loss_prior
            )
        ),
        "elapsed_seconds": float(
            elapsed_seconds
        ),
    }

    with (
        output_directory
        / "manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print(
        "\nExpected-loss prior calibration completed."
    )

    print(
        f"Calibration records: "
        f"{len(calibration_records):,}"
    )

    print(
        f"Prior rows: "
        f"{len(expected_loss_prior):,}"
    )

    print(
        "\nAcceptance report:"
    )

    print(
        acceptance_report.to_string(
            index=False
        )
    )

    print(
        "\nPrecision summary:"
    )

    print(
        precision_summary.to_string(
            index=False
        )
    )

    print(
        "\nOutputs saved to:"
    )

    print(
        output_directory
    )

    if not acceptance_report[
        "passed"
    ].all():
        raise RuntimeError(
            "At least one expected-loss prior "
            "acceptance check failed."
        )


if __name__ == "__main__":
    main()
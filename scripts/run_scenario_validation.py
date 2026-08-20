"""Run and save the permitted Step 11 scenario combinations."""

from __future__ import annotations

import pandas as pd

from config import (
    MASTER_RANDOM_SEED,
    PILOT_DATA_DIR,
    ensure_directories,
    validate_config,
)

from src.simulation import (
    build_scenario_metadata,
    simulate_portfolio,
)


OUTPUT_DIR = (
    PILOT_DATA_DIR / "scenario_engine"
)

CLAIMS_OUTPUT_PATH = (
    OUTPUT_DIR
    / "claims_scenario_validation.parquet"
)

PAYMENTS_OUTPUT_PATH = (
    OUTPUT_DIR
    / "payments_scenario_validation.parquet"
)

METADATA_OUTPUT_PATH = (
    OUTPUT_DIR / "scenario_metadata.csv"
)

SUMMARY_OUTPUT_PATH = (
    OUTPUT_DIR / "scenario_summary.csv"
)


SCENARIOS = [
    {
        "scenario_id": "short_stable_no_break",
        "tail_type": "short",
        "inflation_scenario": "stable",
        "apply_structural_break": False,
    },
    {
        "scenario_id": "short_emerging_no_break",
        "tail_type": "short",
        "inflation_scenario": "emerging",
        "apply_structural_break": False,
    },
    {
        "scenario_id": "short_shock_no_break",
        "tail_type": "short",
        "inflation_scenario": "shock",
        "apply_structural_break": False,
    },
    {
        "scenario_id": "long_stable_no_break",
        "tail_type": "long",
        "inflation_scenario": "stable",
        "apply_structural_break": False,
    },
    {
        "scenario_id": "long_emerging_no_break",
        "tail_type": "long",
        "inflation_scenario": "emerging",
        "apply_structural_break": False,
    },
    {
        "scenario_id": "long_shock_no_break",
        "tail_type": "long",
        "inflation_scenario": "shock",
        "apply_structural_break": False,
    },
    {
        "scenario_id": "long_stable_break",
        "tail_type": "long",
        "inflation_scenario": "stable",
        "apply_structural_break": True,
    },
    {
        "scenario_id": "long_emerging_break",
        "tail_type": "long",
        "inflation_scenario": "emerging",
        "apply_structural_break": True,
    },
    {
        "scenario_id": "long_shock_break",
        "tail_type": "long",
        "inflation_scenario": "shock",
        "apply_structural_break": True,
    },
]


def main() -> None:
    """Run all permitted scenarios and save combined outputs."""

    ensure_directories()
    validate_config()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    claim_tables: list[pd.DataFrame] = []
    payment_tables: list[pd.DataFrame] = []
    metadata_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for scenario in SCENARIOS:
        scenario_id = str(
            scenario["scenario_id"]
        )

        tail_type = str(
            scenario["tail_type"]
        )

        inflation_scenario = str(
            scenario["inflation_scenario"]
        )

        apply_structural_break = bool(
            scenario["apply_structural_break"]
        )

        claims, payments = simulate_portfolio(
            simulation_id=1,
            frequency_scenario="constant",
            tail_type=tail_type,
            inflation_scenario=inflation_scenario,
            apply_structural_break=(
                apply_structural_break
            ),
            seed=MASTER_RANDOM_SEED,
        )

        claims = claims.copy()
        payments = payments.copy()

        claims.insert(
            0,
            "scenario_id",
            scenario_id,
        )

        payments.insert(
            0,
            "scenario_id",
            scenario_id,
        )

        metadata = build_scenario_metadata(
            claims=claims,
            simulation_id=1,
            seed=MASTER_RANDOM_SEED,
            tail_type=tail_type,
            frequency_scenario="constant",
            inflation_scenario=(
                inflation_scenario
            ),
            apply_structural_break=(
                apply_structural_break
            ),
        )

        metadata.insert(
            0,
            "scenario_id",
            scenario_id,
        )

        pattern_names = ", ".join(
            sorted(
                payments[
                    "payment_pattern_name"
                ].unique()
            )
        )

        summary_rows.append(
            {
                "scenario_id": scenario_id,
                "tail_type": tail_type,
                "inflation_scenario": (
                    inflation_scenario
                ),
                "structural_break": (
                    apply_structural_break
                ),
                "claim_count": len(claims),
                "payment_count": len(payments),
                "payment_patterns": (
                    pattern_names
                ),
                "minimum_payment_year": int(
                    payments[
                        "payment_calendar_year"
                    ].min()
                ),
                "maximum_payment_year": int(
                    payments[
                        "payment_calendar_year"
                    ].max()
                ),
            }
        )

        claim_tables.append(claims)
        payment_tables.append(payments)
        metadata_tables.append(metadata)

    all_claims = pd.concat(
        claim_tables,
        ignore_index=True,
    )

    all_payments = pd.concat(
        payment_tables,
        ignore_index=True,
    )

    all_metadata = pd.concat(
        metadata_tables,
        ignore_index=True,
    )

    scenario_summary = pd.DataFrame(
        summary_rows
    )

    all_claims.to_parquet(
        CLAIMS_OUTPUT_PATH,
        index=False,
        engine="pyarrow",
    )

    all_payments.to_parquet(
        PAYMENTS_OUTPUT_PATH,
        index=False,
        engine="pyarrow",
    )

    all_metadata.to_csv(
        METADATA_OUTPUT_PATH,
        index=False,
    )

    scenario_summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    print(
        "Scenario-engine validation completed."
    )

    print(
        f"Scenarios generated: "
        f"{len(SCENARIOS)}"
    )

    print(
        f"Combined claim rows: "
        f"{len(all_claims):,}"
    )

    print(
        f"Combined payment rows: "
        f"{len(all_payments):,}"
    )

    print("\nScenario summary:")
    print(
        scenario_summary.to_string(
            index=False
        )
    )

    print(
        "\nMetadata saved to: "
        f"{METADATA_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
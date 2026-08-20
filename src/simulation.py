"""Claim and payment simulation for the dissertation model.

This module generates:

1. Claim-level data.
2. Payment-level gross data.

Reinsurance is deliberately excluded at this stage.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from config import (
    ACCIDENT_YEARS,
    BASE_CLAIM_FREQUENCY,
    CLAIMS_INFLATION_BASE_YEAR,
    CORE_FREQUENCY_SCENARIO,
    FREQUENCY_SCENARIOS,
    INFLATION_SCENARIOS,
    MASTER_RANDOM_SEED,
    PARETO_SCALE,
    PARETO_SHAPE,
    PAYMENT_PATTERNS,
    REPORTING_DELAY_PROBABILITIES,
    STRUCTURAL_BREAK_ACCIDENT_YEAR,
)


CLAIM_COLUMNS = [
    "simulation_id",
    "claim_id",
    "accident_year",
    "frequency_scenario",
    "tail_type",
    "ultimate_real_severity",
    "report_delay",
    "report_year",
    "structural_break_indicator",
]

CLAIM_OUTPUT_COLUMNS = CLAIM_COLUMNS + [
    "inflation_scenario",
    "apply_structural_break",
]


PAYMENT_COLUMNS = [
    "simulation_id",
    "claim_id",
    "accident_year",
    "report_year",
    "frequency_scenario",
    "tail_type",
    "inflation_scenario",
    "apply_structural_break",
    "structural_break_indicator",
    "payment_pattern_name",
    "payment_sequence",
    "development_year",
    "payment_calendar_year",
    "real_payment",
    "inflation_rate",
    "inflation_index",
    "nominal_gross_payment",
]

def validate_scenario_inputs(
    frequency_scenario: str,
    tail_type: str,
    inflation_scenario: str,
    apply_structural_break: bool,
) -> None:
    """Validate a requested simulation scenario.

    Structural change is currently defined only for the
    long-tail portfolio. Short-tail structural-break scenarios
    are therefore excluded from the permitted scenario grid.
    """

    if frequency_scenario not in FREQUENCY_SCENARIOS:
        raise ValueError(
            "Unknown frequency scenario: "
            f"{frequency_scenario}"
        )

    if tail_type not in REPORTING_DELAY_PROBABILITIES:
        raise ValueError(
            f"Unknown tail type: {tail_type}"
        )

    if inflation_scenario not in INFLATION_SCENARIOS:
        raise ValueError(
            "Unknown inflation scenario: "
            f"{inflation_scenario}"
        )

    if not isinstance(apply_structural_break, bool):
        raise TypeError(
            "apply_structural_break must be True or False."
        )

    if apply_structural_break and tail_type != "long":
        raise ValueError(
            "The structural-break scenario is currently "
            "defined only for long-tail business."
        )


def calculate_frequency_mean(
    accident_year: int,
    frequency_scenario: str,
) -> float:
    """Calculate the expected claim frequency for an accident year.

    Parameters
    ----------
    accident_year:
        The accident year for which the Poisson mean is required.

    frequency_scenario:
        One of the scenarios defined in FREQUENCY_SCENARIOS, such as
        "constant", "decreasing" or "increasing".

    Returns
    -------
    float
        The Poisson mean for the selected accident year.
    """

    if accident_year not in ACCIDENT_YEARS:
        raise ValueError(
            f"Accident year {accident_year} is not configured."
        )

    if frequency_scenario not in FREQUENCY_SCENARIOS:
        raise ValueError(
            f"Unknown frequency scenario: {frequency_scenario}"
        )

    first_accident_year = min(ACCIDENT_YEARS)

    year_index = accident_year - first_accident_year

    annual_multiplier = FREQUENCY_SCENARIOS[
        frequency_scenario
    ]

    frequency_mean = (
        BASE_CLAIM_FREQUENCY
        * annual_multiplier**year_index
    )

    return float(frequency_mean)


def build_inflation_index(
    annual_rates: Mapping[int, float],
    base_year: int = CLAIMS_INFLATION_BASE_YEAR,
) -> pd.DataFrame:
    """Convert annual inflation rates into a cumulative inflation index.

    The index is normalised so that it equals 1.0 in the base year.

    For years after the base year:

        I_t = I_(t-1) * (1 + inflation_rate_t)

    Parameters
    ----------
    annual_rates:
        Dictionary-like object mapping calendar years to annual
        inflation rates.

    base_year:
        Calendar year in which the inflation index equals 1.0.

    Returns
    -------
    pandas.DataFrame
        A table containing calendar year, annual inflation rate and
        cumulative inflation index.
    """

    if not annual_rates:
        raise ValueError("annual_rates cannot be empty.")

    years = sorted(int(year) for year in annual_rates)

    if base_year not in annual_rates:
        raise ValueError(
            f"Base year {base_year} is missing from annual_rates."
        )

    expected_years = list(
        range(years[0], years[-1] + 1)
    )

    if years != expected_years:
        raise ValueError(
            "Inflation years must be consecutive."
        )

    if any(
        float(rate) <= -1.0
        for rate in annual_rates.values()
    ):
        raise ValueError(
            "Every inflation rate must exceed -100%."
        )

    inflation_index_by_year = {
        base_year: 1.0
    }

    for year in range(base_year + 1, years[-1] + 1):
        previous_index = inflation_index_by_year[
            year - 1
        ]

        current_rate = float(annual_rates[year])

        inflation_index_by_year[year] = (
            previous_index * (1.0 + current_rate)
        )

    output_years = list(
        range(base_year, years[-1] + 1)
    )

    inflation_table = pd.DataFrame(
        {
            "calendar_year": output_years,
            "inflation_rate": [
                float(annual_rates[year])
                for year in output_years
            ],
            "inflation_index": [
                float(inflation_index_by_year[year])
                for year in output_years
            ],
        }
    )

    return inflation_table


def simulate_claims(
    simulation_id: int = 1,
    frequency_scenario: str = CORE_FREQUENCY_SCENARIO,
    tail_type: str = "long",
    apply_structural_break: bool = False,
    seed: int = MASTER_RANDOM_SEED,
) -> pd.DataFrame:
    """Simulate claim-level data for one portfolio.

    For every accident year, the function:

    1. Calculates the Poisson frequency mean.
    2. Simulates the number of claims.
    3. Simulates Pareto claim severities.
    4. Simulates reporting delays.
    5. Calculates report years.

    Parameters
    ----------
    simulation_id:
        Identifier for the Monte Carlo simulation.

    frequency_scenario:
        Claim-frequency scenario.

    tail_type:
        Either "short" or "long".

    seed:
        Random seed used to make the simulation reproducible.

    Returns
    -------
    pandas.DataFrame
        One row per simulated claim.
    """

    if tail_type not in REPORTING_DELAY_PROBABILITIES:
        raise ValueError(
            f"Unknown tail type: {tail_type}"
        )

    rng = np.random.default_rng(seed)

    delay_distribution = (
        REPORTING_DELAY_PROBABILITIES[tail_type]
    )

    delay_values = np.asarray(
        tuple(delay_distribution.keys()),
        dtype=int,
    )

    delay_probabilities = np.asarray(
        tuple(delay_distribution.values()),
        dtype=float,
    )

    rows: list[dict[str, object]] = []

    next_claim_number = 1

    for accident_year in ACCIDENT_YEARS:
        frequency_mean = calculate_frequency_mean(
            accident_year=accident_year,
            frequency_scenario=frequency_scenario,
        )

        claim_count = int(
            rng.poisson(frequency_mean)
        )

        if claim_count == 0:
            continue

        # NumPy's pareto() produces values starting at zero.
        # Multiplying 1 + Y by PARETO_SCALE gives a standard
        # Pareto Type I variable with minimum PARETO_SCALE.
        severities = PARETO_SCALE * (
            1.0
            + rng.pareto(
                PARETO_SHAPE,
                size=claim_count,
            )
        )

        report_delays = rng.choice(
            delay_values,
            size=claim_count,
            p=delay_probabilities,
        )

        for severity, report_delay in zip(
            severities,
            report_delays,
        ):
            report_delay = int(report_delay)

            report_year = (
                int(accident_year)
                + report_delay
            )

            structural_break_indicator = int(
                apply_structural_break
                and tail_type == "long"
                and accident_year
                >= STRUCTURAL_BREAK_ACCIDENT_YEAR
            )

            claim_id = (
                f"S{simulation_id:04d}_"
                f"C{next_claim_number:06d}"
            )

            rows.append(
                {
                    "simulation_id": int(
                        simulation_id
                    ),
                    "claim_id": claim_id,
                    "accident_year": int(
                        accident_year
                    ),
                    "frequency_scenario": (
                        frequency_scenario
                    ),
                    "tail_type": tail_type,
                    "ultimate_real_severity": float(
                        severity
                    ),
                    "report_delay": report_delay,
                    "report_year": report_year,
                    "structural_break_indicator": (
                        structural_break_indicator
                    ),
                }
            )

            next_claim_number += 1

    claims = pd.DataFrame.from_records(
        rows,
        columns=CLAIM_COLUMNS,
    )

    return claims


def expand_claims_to_payments(
    claims: pd.DataFrame,
    inflation_scenario: str = "stable",
    apply_structural_break: bool = False,
) -> pd.DataFrame:
    """Expand each simulated claim into individual payments.

    Payment sequence is measured from the claim's report year.

    Payment calendar year is:

        report_year + payment_sequence - 1

    Triangle development year is measured from accident year:

        payment_calendar_year - accident_year + 1

    Parameters
    ----------
    claims:
        Claim-level DataFrame produced by simulate_claims().

    inflation_scenario:
        Inflation scenario used to convert real payments into nominal
        gross payments.

    Returns
    -------
    pandas.DataFrame
        One row per claim payment.
    """

    missing_columns = (
        set(CLAIM_COLUMNS)
        - set(claims.columns)
    )

    if missing_columns:
        raise ValueError(
            "Claims data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if inflation_scenario not in INFLATION_SCENARIOS:
        raise ValueError(
            f"Unknown inflation scenario: "
            f"{inflation_scenario}"
        )

    inflation_table = build_inflation_index(
        INFLATION_SCENARIOS[inflation_scenario]
    )

    rate_by_year = (
        inflation_table
        .set_index("calendar_year")[
            "inflation_rate"
        ]
        .to_dict()
    )

    index_by_year = (
        inflation_table
        .set_index("calendar_year")[
            "inflation_index"
        ]
        .to_dict()
    )

    rows: list[dict[str, object]] = []

    for claim in claims.itertuples(index=False):
        if claim.tail_type == "short":
            pattern_name = "short"

        elif claim.tail_type == "long":
            if claim.structural_break_indicator == 1:
                pattern_name = "accelerated_long"
            else:
                pattern_name = "long"

        else:
            raise ValueError(
                f"Unknown tail type for claim "
                f"{claim.claim_id}: "
                f"{claim.tail_type}"
            )

        payment_pattern = PAYMENT_PATTERNS[pattern_name]

        real_payments = [
            float(claim.ultimate_real_severity)
            * float(proportion)
            for proportion in payment_pattern
        ]

        # Use the residual for the final payment.
        # This prevents tiny floating-point differences from causing
        # payments not to reconcile exactly with the ultimate severity.
        real_payments[-1] = (
            float(claim.ultimate_real_severity)
            - sum(real_payments[:-1])
        )

        for payment_sequence, real_payment in enumerate(
            real_payments,
            start=1,
        ):
            payment_calendar_year = (
                int(claim.report_year)
                + payment_sequence
                - 1
            )

            development_year = (
                payment_calendar_year
                - int(claim.accident_year)
                + 1
            )

            if payment_calendar_year not in index_by_year:
                raise ValueError(
                    "The inflation scenario does not cover "
                    f"payment year {payment_calendar_year}."
                )

            inflation_rate = float(
                rate_by_year[payment_calendar_year]
            )

            inflation_index = float(
                index_by_year[payment_calendar_year]
            )

            nominal_gross_payment = (
                float(real_payment)
                * inflation_index
            )

            rows.append(
                {
                    "simulation_id": int(
                        claim.simulation_id
                    ),
                    "claim_id": claim.claim_id,
                    "accident_year": int(
                        claim.accident_year
                    ),
                    "report_year": int(
                        claim.report_year
                    ),
                    "frequency_scenario": (
                        claim.frequency_scenario
                    ),
                    "tail_type": claim.tail_type,
                    "inflation_scenario": (
                        inflation_scenario
                    ),
                    "apply_structural_break": bool(
                        apply_structural_break
                    ),
                    "structural_break_indicator": int(
                        claim.structural_break_indicator
                    ),
                    "payment_pattern_name": pattern_name,
                    "payment_sequence": int(
                        payment_sequence
                    ),
                    "development_year": int(
                        development_year
                    ),
                    "payment_calendar_year": int(
                        payment_calendar_year
                    ),
                    "real_payment": float(
                        real_payment
                    ),
                    "inflation_rate": inflation_rate,
                    "inflation_index": inflation_index,
                    "nominal_gross_payment": float(
                        nominal_gross_payment
                    ),
                }
            )

    payments = pd.DataFrame.from_records(
        rows,
        columns=PAYMENT_COLUMNS,
    )

    return payments


def _validate_simulated_data(
    claims: pd.DataFrame,
    payments: pd.DataFrame,
) -> None:
    """Perform internal validation checks on simulated data."""

    if claims.isna().any().any():
        raise ValueError(
            "The claim-level data contains missing values."
        )

    if payments.isna().any().any():
        raise ValueError(
            "The payment-level data contains missing values."
        )

    if not (
        claims["ultimate_real_severity"]
        >= PARETO_SCALE
    ).all():
        raise ValueError(
            "At least one severity is below PARETO_SCALE."
        )

    expected_report_year = (
        claims["accident_year"]
        + claims["report_delay"]
    )

    if not (
        claims["report_year"]
        == expected_report_year
    ).all():
        raise ValueError(
            "At least one report year is inconsistent."
        )

    if not (
        payments["payment_calendar_year"]
        >= payments["report_year"]
    ).all():
        raise ValueError(
            "At least one payment occurs before reporting."
        )

    if not claims.empty:
        actual_real_totals = (
            payments
            .groupby("claim_id")["real_payment"]
            .sum()
            .sort_index()
        )

        expected_real_totals = (
            claims
            .set_index("claim_id")[
                "ultimate_real_severity"
            ]
            .sort_index()
        )

        if not actual_real_totals.index.equals(
            expected_real_totals.index
        ):
            raise ValueError(
                "Claim IDs and payment IDs do not reconcile."
            )

        if not np.allclose(
            actual_real_totals.to_numpy(),
            expected_real_totals.to_numpy(),
            rtol=1e-12,
            atol=1e-6,
        ):
            raise ValueError(
                "Real payments do not sum to "
                "ultimate real severities."
            )

    expected_nominal_payments = (
        payments["real_payment"]
        * payments["inflation_index"]
    )

    if not np.allclose(
        payments["nominal_gross_payment"],
        expected_nominal_payments,
        rtol=1e-12,
        atol=1e-6,
    ):
        raise ValueError(
            "Nominal payments do not reconcile with "
            "real payments and inflation indices."
        )


def simulate_portfolio(
    simulation_id: int = 1,
    frequency_scenario: str = CORE_FREQUENCY_SCENARIO,
    tail_type: str = "long",
    inflation_scenario: str = "stable",
    apply_structural_break: bool = False,
    seed: int = MASTER_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one complete simulated portfolio.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        The claim-level table followed by the payment-level table.
    """
    
    validate_scenario_inputs(
        frequency_scenario=frequency_scenario,
        tail_type=tail_type,
        inflation_scenario=inflation_scenario,
        apply_structural_break=apply_structural_break,
    )
    
    claims = simulate_claims(
        simulation_id=simulation_id,
        frequency_scenario=frequency_scenario,
        tail_type=tail_type,
        apply_structural_break=apply_structural_break,
        seed=seed,
    )
    
    claims = claims.copy()

    claims["inflation_scenario"] = (
        inflation_scenario
    )

    claims["apply_structural_break"] = bool(
        apply_structural_break
    )

    claims = claims[CLAIM_OUTPUT_COLUMNS]

    payments = expand_claims_to_payments(
        claims=claims,
        inflation_scenario=inflation_scenario,
        apply_structural_break=apply_structural_break,
    )

    _validate_simulated_data(
        claims=claims,
        payments=payments,
    )

    return claims, payments


SCENARIO_METADATA_COLUMNS = [
    "simulation_id",
    "seed",
    "tail_type",
    "frequency_scenario",
    "inflation_scenario",
    "structural_break",
    "structural_break_year",
    "pareto_scale",
    "pareto_shape",
    "claim_count",
]


def build_scenario_metadata(
    claims: pd.DataFrame,
    simulation_id: int,
    seed: int,
    tail_type: str,
    frequency_scenario: str,
    inflation_scenario: str,
    apply_structural_break: bool,
) -> pd.DataFrame:
    """Create a one-row reproducibility record for a scenario."""

    validate_scenario_inputs(
        frequency_scenario=frequency_scenario,
        tail_type=tail_type,
        inflation_scenario=inflation_scenario,
        apply_structural_break=apply_structural_break,
    )

    metadata = pd.DataFrame(
        [
            {
                "simulation_id": int(simulation_id),
                "seed": int(seed),
                "tail_type": tail_type,
                "frequency_scenario": frequency_scenario,
                "inflation_scenario": inflation_scenario,
                "structural_break": bool(
                    apply_structural_break
                ),
                "structural_break_year": int(
                    STRUCTURAL_BREAK_ACCIDENT_YEAR
                ),
                "pareto_scale": float(
                    PARETO_SCALE
                ),
                "pareto_shape": float(
                    PARETO_SHAPE
                ),
                "claim_count": int(
                    len(claims)
                ),
            }
        ],
        columns=SCENARIO_METADATA_COLUMNS,
    )

    return metadata

if __name__ == "__main__":
    pilot_claims, pilot_payments = simulate_portfolio(
        simulation_id=1,
        frequency_scenario="constant",
        tail_type="long",
        inflation_scenario="stable",
        apply_structural_break=False,
        seed=MASTER_RANDOM_SEED,
    )

    print(
        f"Number of claims: "
        f"{len(pilot_claims):,}"
    )

    print(
        f"Number of payments: "
        f"{len(pilot_payments):,}"
    )

    print("\nClaim-level sample:")
    print(pilot_claims.head())

    print("\nPayment-level sample:")
    print(pilot_payments.head())
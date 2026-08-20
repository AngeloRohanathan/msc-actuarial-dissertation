"""Central configuration for the dissertation simulation.

The values in this file are provisional pilot assumptions.
Final experiment parameters should be frozen before the main
Monte Carlo results are generated.
"""

from __future__ import annotations

from math import isclose
from pathlib import Path


# ---------------------------------------------------------------------
# Project directories
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
PILOT_DATA_DIR = DATA_DIR / "pilot"
FINAL_DATA_DIR = DATA_DIR / "final"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
LOGS_DIR = OUTPUTS_DIR / "logs"


def ensure_directories() -> None:
    """Create the project output directories if they do not exist."""

    directories = (
        DATA_DIR,
        PILOT_DATA_DIR,
        FINAL_DATA_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        LOGS_DIR,
    )

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

MASTER_RANDOM_SEED = 27072002


# ---------------------------------------------------------------------
# Time structure
# ---------------------------------------------------------------------

ACCIDENT_YEARS = tuple(range(2010, 2025))
EARLIEST_ACCIDENT_YEAR = min(ACCIDENT_YEARS)
LATEST_ACCIDENT_YEAR = max(ACCIDENT_YEARS)

VALUATION_YEAR = 2024

CLAIMS_INFLATION_BASE_YEAR = 2010
CLAUSE_INDEX_BASE_YEAR = 2010

MAX_REPORTING_DELAY = 4
MAX_PAYMENT_PERIOD = 10

FINAL_PAYMENT_YEAR = (
    LATEST_ACCIDENT_YEAR
    + MAX_REPORTING_DELAY
    + MAX_PAYMENT_PERIOD
    - 1
)

CALENDAR_YEARS = tuple(
    range(CLAIMS_INFLATION_BASE_YEAR, FINAL_PAYMENT_YEAR + 1)
)

# ---------------------------------------------------------------------
# Independent expected-loss prior calibration
# ---------------------------------------------------------------------

# The prior is estimated from a separate pricing Monte Carlo sample.
EXPECTED_LOSS_PRIOR_VERSION = "pricing_mc_v1"

# Final number of independent calibration simulations per scenario.
EXPECTED_LOSS_CALIBRATION_SIMULATIONS = 2_000

# This seed range must remain separate from the evaluation experiment.
EXPECTED_LOSS_CALIBRATION_SEED_BASE = 91_000_000

# Accident years represented in the reserving triangles.
EXPECTED_LOSS_ACCIDENT_YEARS = tuple(
    range(
        2010,
        VALUATION_YEAR + 1,
    )
)

# Precision targets used to assess the final pricing prior.
EXPECTED_LOSS_GROSS_RELATIVE_MCSE_TARGET = 0.02
EXPECTED_LOSS_CEDED_RELATIVE_MCSE_TARGET = 0.05


# ---------------------------------------------------------------------
# Claim frequency
# ---------------------------------------------------------------------

BASE_CLAIM_FREQUENCY = 50.0

FREQUENCY_SCENARIOS = {
    "constant": 1.00,
    "decreasing": 0.95,
    "increasing": 1.05,
}

CORE_FREQUENCY_SCENARIO = "constant"


# ---------------------------------------------------------------------
# Claim severity
# ---------------------------------------------------------------------

PARETO_SCALE = 1_000_000.0
PARETO_SHAPE = 2.5


# ---------------------------------------------------------------------
# Reporting delays
# ---------------------------------------------------------------------

REPORTING_DELAY_PROBABILITIES = {
    "short": {
        0: 0.85,
        1: 0.12,
        2: 0.03,
    },
    "long": {
        0: 0.55,
        1: 0.25,
        2: 0.12,
        3: 0.05,
        4: 0.03,
    },
}


# ---------------------------------------------------------------------
# Incremental payment patterns
# ---------------------------------------------------------------------

PAYMENT_PATTERNS = {
    "short": (
        0.60,
        0.25,
        0.10,
        0.05,
    ),
    "long": (
        0.04,
        0.20,
        0.19,
        0.14,
        0.09,
        0.07,
        0.06,
        0.05,
        0.06,
        0.10,
    ),
    "accelerated_long": (
        0.08,
        0.27,
        0.23,
        0.16,
        0.10,
        0.06,
        0.04,
        0.03,
        0.02,
        0.01,
    ),
}

STRUCTURAL_BREAK_ACCIDENT_YEAR = 2018


# ---------------------------------------------------------------------
# Inflation scenarios
# ---------------------------------------------------------------------

EMERGING_INFLATION_BREAK_YEAR = 2021

SHOCK_INFLATION_RATES = {
    2023: 0.12,
    2024: 0.10,
    2025: 0.08,
    2026: 0.07,
}


def shock_inflation_rate(year: int) -> float:
    """Return the pilot shock-scenario inflation rate."""

    if year < 2021:
        return 0.04

    if year < 2023:
        return 0.06

    return SHOCK_INFLATION_RATES.get(year, 0.06)


INFLATION_SCENARIOS = {
    "stable": {
        year: 0.04
        for year in CALENDAR_YEARS
    },
    "emerging": {
        year: 0.04 if year < EMERGING_INFLATION_BREAK_YEAR else 0.06
        for year in CALENDAR_YEARS
    },
    "shock": {
        year: shock_inflation_rate(year)
        for year in CALENDAR_YEARS
    },
}


# ---------------------------------------------------------------------
# Basic Excess-of-Loss treaty assumptions
# ---------------------------------------------------------------------

# Pilot treaty: £5 million excess of £2 million.
#
# These are provisional baseline assumptions. They will later be
# supplemented with indexed attachment and limit structures.
PILOT_XOL_ATTACHMENT = 2_000_000.0
PILOT_XOL_LIMIT = 5_000_000.0




# ---------------------------------------------------------------------
# Reinsurance pilot calibration
# ---------------------------------------------------------------------

TREATY_CALIBRATION = {
    "treaty_1": {
        "attachment_quantile": 0.90,
        "limit": 3_000_000.0,
    },
    "treaty_2": {
        "attachment_quantile": 0.95,
        "limit": 5_000_000.0,
    },
}

INDEXATION_CLAUSES = (
    "none",
    "full",
    "single_trigger",
)

INDEXATION_THRESHOLD = 0.10


# ---------------------------------------------------------------------
# Monte Carlo experiment sizes
# ---------------------------------------------------------------------

PILOT_SIMULATIONS = 10
INTERMEDIATE_SIMULATIONS = 100
FINAL_SIMULATIONS = 1_000


# ---------------------------------------------------------------------
# Classical reserving assumptions
# ---------------------------------------------------------------------

# The standard Chain Ladder factors are treated as embedding the
# baseline stable inflation assumption of 4% per year.
#
# Cashflow Uplift will adjust projected future cashflows when the
# forecast inflation path differs from this embedded assumption.
CASHFLOW_UPLIFT_EMBEDDED_INFLATION = 0.04


# ---------------------------------------------------------------------
# Machine-learning reserving assumptions
# ---------------------------------------------------------------------

# Candidate L2 regularisation strengths for Poisson regression.
#
# Alpha must remain positive because the model includes encoded
# categorical variables that may be collinear.
ML_POISSON_ALPHA_GRID = (
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
)

# At least five historical calendar-year diagonals must be
# available before the following diagonal is used for validation.
ML_MIN_TRAINING_DIAGONALS = 5

# Require at least three valid rolling validation folds when
# selecting the regularisation parameter.
ML_MIN_VALIDATION_FOLDS = 3

# Payments are modelled in millions of pounds to improve numerical
# stability. Predictions are converted back to pounds afterwards.
ML_AMOUNT_SCALE = 1_000_000.0


# ---------------------------------------------------------------------
# End-to-end experiment assumptions
# ---------------------------------------------------------------------

# Initial and extended pilot sizes. These are simulations per
# scenario, rather than total simulations across all scenarios.
END_TO_END_PILOT_SIMULATIONS = 10
END_TO_END_EXTENDED_SIMULATIONS = 50

# The same simulation seed is used across scenarios for each
# simulation ID. This creates a paired scenario comparison.
END_TO_END_BASE_SEED = 20260816

# Structural-break year used in the existing scenario engine.
EXPERIMENT_STRUCTURAL_BREAK_YEAR = 2018

# Current Step 16 experiment uses the unindexed baseline treaty.
EXPERIMENT_CLAUSE_TYPE = "none"

END_TO_END_SCENARIOS = (
    {
        "scenario_id": "short_stable_no_break",
        "frequency_scenario": "constant",
        "tail_type": "short",
        "inflation_scenario": "stable",
        "apply_structural_break": False,
    },
    {
        "scenario_id": "short_emerging_no_break",
        "frequency_scenario": "constant",
        "tail_type": "short",
        "inflation_scenario": "emerging",
        "apply_structural_break": False,
    },
    {
        "scenario_id": "short_shock_no_break",
        "frequency_scenario": "constant",
        "tail_type": "short",
        "inflation_scenario": "shock",
        "apply_structural_break": False,
    },
    {
        "scenario_id": "long_stable_no_break",
        "frequency_scenario": "constant",
        "tail_type": "long",
        "inflation_scenario": "stable",
        "apply_structural_break": False,
    },
    {
        "scenario_id": "long_emerging_no_break",
        "frequency_scenario": "constant",
        "tail_type": "long",
        "inflation_scenario": "emerging",
        "apply_structural_break": False,
    },
    {
        "scenario_id": "long_shock_no_break",
        "frequency_scenario": "constant",
        "tail_type": "long",
        "inflation_scenario": "shock",
        "apply_structural_break": False,
    },
    {
        "scenario_id": "long_stable_break",
        "frequency_scenario": "constant",
        "tail_type": "long",
        "inflation_scenario": "stable",
        "apply_structural_break": True,
    },
    {
        "scenario_id": "long_emerging_break",
        "frequency_scenario": "constant",
        "tail_type": "long",
        "inflation_scenario": "emerging",
        "apply_structural_break": True,
    },
    {
        "scenario_id": "long_shock_break",
        "frequency_scenario": "constant",
        "tail_type": "long",
        "inflation_scenario": "shock",
        "apply_structural_break": True,
    },
)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

PERCENTAGE_ERROR_TOLERANCE = 1_000.0


# ---------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------

def validate_config() -> None:
    """Check that the configuration is internally consistent."""

    if not ACCIDENT_YEARS:
        raise ValueError("ACCIDENT_YEARS cannot be empty.")

    if tuple(sorted(ACCIDENT_YEARS)) != ACCIDENT_YEARS:
        raise ValueError("ACCIDENT_YEARS must be sorted.")

    expected_years = tuple(
        range(EARLIEST_ACCIDENT_YEAR, LATEST_ACCIDENT_YEAR + 1)
    )

    if ACCIDENT_YEARS != expected_years:
        raise ValueError("ACCIDENT_YEARS must be consecutive.")

    if VALUATION_YEAR != LATEST_ACCIDENT_YEAR:
        raise ValueError(
            "For the pilot, VALUATION_YEAR should equal the latest "
            "accident year."
        )

    if CLAIMS_INFLATION_BASE_YEAR > EARLIEST_ACCIDENT_YEAR:
        raise ValueError(
            "The claims-inflation base year must not be later than "
            "the earliest accident year."
        )

    if STRUCTURAL_BREAK_ACCIDENT_YEAR not in ACCIDENT_YEARS:
        raise ValueError(
            "The structural-break year must be an accident year."
        )

    if PARETO_SCALE <= 0:
        raise ValueError("PARETO_SCALE must be positive.")

    if PARETO_SHAPE <= 2:
        raise ValueError(
            "PARETO_SHAPE must exceed 2 if a finite variance is required."
        )
    if PILOT_XOL_ATTACHMENT < 0:
        raise ValueError(
            "PILOT_XOL_ATTACHMENT cannot be negative."
        )
    if PILOT_XOL_LIMIT <= 0:
        raise ValueError(
            "PILOT_XOL_LIMIT must be positive."
        )
        
    if CASHFLOW_UPLIFT_EMBEDDED_INFLATION <= -1.0:
        raise ValueError(
            "CASHFLOW_UPLIFT_EMBEDDED_INFLATION "
            "must exceed -100%."
        )
    if not ML_POISSON_ALPHA_GRID:
        raise ValueError(
            "ML_POISSON_ALPHA_GRID cannot be empty."
        )

    if any(
        alpha <= 0.0
        for alpha in ML_POISSON_ALPHA_GRID
    ):
        raise ValueError(
            "All ML Poisson alpha values must be positive."
        )

    if ML_MIN_TRAINING_DIAGONALS < 2:
        raise ValueError(
            "ML_MIN_TRAINING_DIAGONALS must be at least 2."
        )

    if ML_MIN_VALIDATION_FOLDS < 1:
        raise ValueError(
            "ML_MIN_VALIDATION_FOLDS must be positive."
        )

    if ML_AMOUNT_SCALE <= 0.0:
        raise ValueError(
            "ML_AMOUNT_SCALE must be positive."
        )
    
    if END_TO_END_PILOT_SIMULATIONS < 1:
        raise ValueError(
            "END_TO_END_PILOT_SIMULATIONS must be positive."
        )

    if END_TO_END_EXTENDED_SIMULATIONS < 1:
        raise ValueError(
            "END_TO_END_EXTENDED_SIMULATIONS must be positive."
        )

    if END_TO_END_BASE_SEED < 0:
        raise ValueError(
            "END_TO_END_BASE_SEED cannot be negative."
        )

    if not END_TO_END_SCENARIOS:
        raise ValueError(
            "END_TO_END_SCENARIOS cannot be empty."
        )

    scenario_ids = [
        scenario["scenario_id"]
        for scenario in END_TO_END_SCENARIOS
    ]

    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError(
            "END_TO_END_SCENARIOS contains duplicate scenario IDs."
        )

    valid_tail_types = {
        "short",
        "long",
    }

    valid_inflation_scenarios = set(
        INFLATION_SCENARIOS
    )

    for scenario in END_TO_END_SCENARIOS:
        if scenario["tail_type"] not in valid_tail_types:
            raise ValueError(
                "Invalid tail type in experiment scenario: "
                f"{scenario}"
            )

        if (
            scenario["inflation_scenario"]
            not in valid_inflation_scenarios
        ):
            raise ValueError(
                "Invalid inflation scenario: "
                f"{scenario}"
            )

        if not isinstance(
            scenario["apply_structural_break"],
            bool,
        ):
            raise ValueError(
                "apply_structural_break must be Boolean."
            )

    for tail_type, probabilities in REPORTING_DELAY_PROBABILITIES.items():
        total_probability = sum(probabilities.values())

        if not isclose(total_probability, 1.0, abs_tol=1e-12):
            raise ValueError(
                f"Reporting probabilities for {tail_type} sum to "
                f"{total_probability}, not 1."
            )

        if any(delay < 0 for delay in probabilities):
            raise ValueError("Reporting delays cannot be negative.")

        if any(probability < 0 for probability in probabilities.values()):
            raise ValueError(
                "Reporting-delay probabilities cannot be negative."
            )

    for pattern_name, pattern in PAYMENT_PATTERNS.items():
        if not isclose(sum(pattern), 1.0, abs_tol=1e-12):
            raise ValueError(
                f"Payment pattern {pattern_name} does not sum to 1."
            )

        if any(proportion < 0 for proportion in pattern):
            raise ValueError(
                f"Payment pattern {pattern_name} contains a negative value."
            )

    for scenario_name, rates in INFLATION_SCENARIOS.items():
        missing_years = set(CALENDAR_YEARS) - set(rates)

        if missing_years:
            raise ValueError(
                f"Inflation scenario {scenario_name} is missing years: "
                f"{sorted(missing_years)}"
            )

        if any(rate <= -1 for rate in rates.values()):
            raise ValueError(
                f"Inflation scenario {scenario_name} contains an "
                "invalid rate."
            )
    if EXPECTED_LOSS_CALIBRATION_SIMULATIONS < 2:
        raise ValueError(
            "EXPECTED_LOSS_CALIBRATION_SIMULATIONS "
            "must be at least 2."
        )

    if EXPECTED_LOSS_CALIBRATION_SEED_BASE < 0:
        raise ValueError(
            "EXPECTED_LOSS_CALIBRATION_SEED_BASE "
            "cannot be negative."
        )

    if not EXPECTED_LOSS_ACCIDENT_YEARS:
        raise ValueError(
            "EXPECTED_LOSS_ACCIDENT_YEARS "
            "cannot be empty."
        )

    if len(
        EXPECTED_LOSS_ACCIDENT_YEARS
    ) != len(
        set(
            EXPECTED_LOSS_ACCIDENT_YEARS
        )
    ):
        raise ValueError(
            "EXPECTED_LOSS_ACCIDENT_YEARS "
            "contains duplicates."
        )

    if (
        EXPECTED_LOSS_GROSS_RELATIVE_MCSE_TARGET
        <= 0
    ):
        raise ValueError(
            "The gross relative MCSE target "
            "must be positive."
        )

    if (
        EXPECTED_LOSS_CEDED_RELATIVE_MCSE_TARGET
        <= 0
    ):
        raise ValueError(
            "The ceded relative MCSE target "
            "must be positive."
        )

    # Reserve a generous range for future evaluation simulations and
    # verify that it does not overlap the pricing-calibration seeds.
    evaluation_seed_min = (
        END_TO_END_BASE_SEED + 1
    )

    evaluation_seed_max = (
        END_TO_END_BASE_SEED + 10_000
    )

    calibration_seed_min = (
        EXPECTED_LOSS_CALIBRATION_SEED_BASE + 1
    )

    calibration_seed_max = (
        EXPECTED_LOSS_CALIBRATION_SEED_BASE
        + EXPECTED_LOSS_CALIBRATION_SIMULATIONS
    )

    seed_ranges_overlap = not (
        calibration_seed_max
        < evaluation_seed_min
        or calibration_seed_min
        > evaluation_seed_max
    )

    if seed_ranges_overlap:
        raise ValueError(
            "Expected-loss calibration seeds overlap "
            "the evaluation experiment seed range."
        )


if __name__ == "__main__":
    ensure_directories()
    validate_config()
    print("Configuration is valid.")
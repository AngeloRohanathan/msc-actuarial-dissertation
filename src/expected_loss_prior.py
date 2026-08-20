"""Independent pricing prior for expected-loss reserving methods."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


PAYMENT_REQUIRED_COLUMNS = {
    "accident_year",
    "claim_id",
    "nominal_gross_payment",
    "nominal_ceded_payment",
    "nominal_retained_payment",
}


CALIBRATION_KEY_COLUMNS = [
    "scenario_id",
    "calibration_id",
    "accident_year",
]


PRIOR_KEY_COLUMNS = [
    "scenario_id",
    "accident_year",
]


def build_calibration_seed_schedule(
    *,
    number_of_simulations: int,
    seed_base: int,
) -> list[int]:
    """Create a deterministic calibration seed schedule."""

    if number_of_simulations < 1:
        raise ValueError(
            "number_of_simulations must be positive."
        )

    if seed_base < 0:
        raise ValueError(
            "seed_base cannot be negative."
        )

    return [
        seed_base + calibration_id
        for calibration_id in range(
            1,
            number_of_simulations + 1,
        )
    ]


def seed_ranges_are_disjoint(
    *,
    first_seed_base: int,
    first_count: int,
    second_seed_base: int,
    second_count: int,
) -> bool:
    """Return whether two generated seed ranges do not overlap."""

    first_minimum = first_seed_base + 1
    first_maximum = first_seed_base + first_count

    second_minimum = second_seed_base + 1
    second_maximum = second_seed_base + second_count

    return bool(
        first_maximum < second_minimum
        or second_maximum < first_minimum
    )


def build_calibration_rows(
    *,
    reinsured_payments: pd.DataFrame,
    scenario: dict[str, Any],
    calibration_id: int,
    seed: int,
    accident_years: Iterable[int],
) -> pd.DataFrame:
    """Aggregate one independent portfolio to accident-year ultimates."""

    missing_columns = (
        PAYMENT_REQUIRED_COLUMNS
        - set(
            reinsured_payments.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Reinsured payment data are missing "
            f"columns: {sorted(missing_columns)}"
        )

    accident_years = [
        int(year)
        for year in accident_years
    ]

    if not accident_years:
        raise ValueError(
            "accident_years cannot be empty."
        )

    grouped = (
        reinsured_payments
        .groupby(
            "accident_year",
            as_index=True,
        )
        .agg(
            gross_ultimate=(
                "nominal_gross_payment",
                "sum",
            ),
            ceded_ultimate=(
                "nominal_ceded_payment",
                "sum",
            ),
            retained_ultimate=(
                "nominal_retained_payment",
                "sum",
            ),
            claim_count=(
                "claim_id",
                "nunique",
            ),
        )
        .reindex(
            accident_years,
            fill_value=0.0,
        )
        .rename_axis(
            "accident_year"
        )
        .reset_index()
    )

    grouped[
        "claim_count"
    ] = grouped[
        "claim_count"
    ].astype(int)

    grouped.insert(
        0,
        "seed",
        int(seed),
    )

    grouped.insert(
        0,
        "calibration_id",
        int(calibration_id),
    )

    grouped.insert(
        0,
        "scenario_id",
        scenario["scenario_id"],
    )

    grouped[
        "frequency_scenario"
    ] = scenario[
        "frequency_scenario"
    ]

    grouped[
        "tail_type"
    ] = scenario[
        "tail_type"
    ]

    grouped[
        "inflation_scenario"
    ] = scenario[
        "inflation_scenario"
    ]

    grouped[
        "structural_break"
    ] = bool(
        scenario[
            "apply_structural_break"
        ]
    )

    return grouped


def _sample_standard_deviation(
    values: pd.Series,
) -> float:
    """Return the sample standard deviation."""

    if len(values) < 2:
        return np.nan

    return float(
        values.std(
            ddof=1
        )
    )


def _monte_carlo_standard_error(
    values: pd.Series,
) -> float:
    """Return standard error of an independently simulated mean."""

    if len(values) < 2:
        return np.nan

    return float(
        values.std(
            ddof=1
        )
        / np.sqrt(
            len(values)
        )
    )


def _relative_mcse(
    *,
    mean_value: float,
    mcse: float,
) -> float:
    """Calculate MCSE relative to the estimated mean."""

    if (
        not np.isfinite(mean_value)
        or not np.isfinite(mcse)
        or np.isclose(
            mean_value,
            0.0,
            atol=1e-12,
        )
    ):
        return np.nan

    return float(
        mcse
        / abs(
            mean_value
        )
    )


def summarise_expected_loss_prior(
    *,
    calibration_records: pd.DataFrame,
    pricing_assumption_version: str,
    calibration_simulations: int,
    calibration_seed_base: int,
    attachment: float,
    limit: float,
) -> pd.DataFrame:
    """Create expected gross and ceded ultimates by scenario and AY."""

    required_columns = {
        "scenario_id",
        "frequency_scenario",
        "tail_type",
        "inflation_scenario",
        "structural_break",
        "calibration_id",
        "accident_year",
        "gross_ultimate",
        "ceded_ultimate",
        "retained_ultimate",
        "claim_count",
    }

    missing_columns = (
        required_columns
        - set(
            calibration_records.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Calibration records are missing "
            f"columns: {sorted(missing_columns)}"
        )

    grouping_columns = [
        "scenario_id",
        "frequency_scenario",
        "tail_type",
        "inflation_scenario",
        "structural_break",
        "accident_year",
    ]

    rows: list[
        dict[str, Any]
    ] = []

    for group_values, group in (
        calibration_records.groupby(
            grouping_columns,
            dropna=False,
            sort=True,
        )
    ):
        group_information = dict(
            zip(
                grouping_columns,
                group_values,
            )
        )

        gross = pd.to_numeric(
            group[
                "gross_ultimate"
            ],
            errors="raise",
        ).astype(float)

        ceded = pd.to_numeric(
            group[
                "ceded_ultimate"
            ],
            errors="raise",
        ).astype(float)

        retained = pd.to_numeric(
            group[
                "retained_ultimate"
            ],
            errors="raise",
        ).astype(float)

        claim_count = pd.to_numeric(
            group[
                "claim_count"
            ],
            errors="raise",
        ).astype(float)

        number_of_records = int(
            len(group)
        )

        unique_calibrations = int(
            group[
                "calibration_id"
            ].nunique()
        )

        expected_gross = float(
            gross.mean()
        )

        expected_ceded = float(
            ceded.mean()
        )

        expected_retained = float(
            retained.mean()
        )

        gross_mcse = (
            _monte_carlo_standard_error(
                gross
            )
        )

        ceded_mcse = (
            _monte_carlo_standard_error(
                ceded
            )
        )

        retained_mcse = (
            _monte_carlo_standard_error(
                retained
            )
        )

        exposure_measure = float(
            claim_count.mean()
        )

        rows.append(
            {
                **group_information,
                "expected_gross_ultimate": (
                    expected_gross
                ),
                "expected_ceded_ultimate": (
                    expected_ceded
                ),
                "expected_retained_ultimate": (
                    expected_retained
                ),
                "exposure_measure": (
                    exposure_measure
                ),
                "expected_gross_loss_per_claim": (
                    expected_gross
                    / exposure_measure
                    if exposure_measure > 0
                    else np.nan
                ),
                "expected_ceded_loss_per_claim": (
                    expected_ceded
                    / exposure_measure
                    if exposure_measure > 0
                    else np.nan
                ),
                "gross_standard_deviation": (
                    _sample_standard_deviation(
                        gross
                    )
                ),
                "ceded_standard_deviation": (
                    _sample_standard_deviation(
                        ceded
                    )
                ),
                "retained_standard_deviation": (
                    _sample_standard_deviation(
                        retained
                    )
                ),
                "gross_mcse": gross_mcse,
                "ceded_mcse": ceded_mcse,
                "retained_mcse": (
                    retained_mcse
                ),
                "gross_relative_mcse": (
                    _relative_mcse(
                        mean_value=expected_gross,
                        mcse=gross_mcse,
                    )
                ),
                "ceded_relative_mcse": (
                    _relative_mcse(
                        mean_value=expected_ceded,
                        mcse=ceded_mcse,
                    )
                ),
                "calibration_records": (
                    number_of_records
                ),
                "unique_calibrations": (
                    unique_calibrations
                ),
                "pricing_assumption_version": (
                    pricing_assumption_version
                ),
                "calibration_simulations": int(
                    calibration_simulations
                ),
                "calibration_seed_base": int(
                    calibration_seed_base
                ),
                "attachment": float(
                    attachment
                ),
                "limit": float(
                    limit
                ),
            }
        )

    prior = pd.DataFrame(
        rows
    )

    return (
        prior.sort_values(
            PRIOR_KEY_COLUMNS
        )
        .reset_index(
            drop=True
        )
    )


def build_precision_summary(
    *,
    expected_loss_prior: pd.DataFrame,
    gross_relative_mcse_target: float,
    ceded_relative_mcse_target: float,
) -> pd.DataFrame:
    """Summarise Monte Carlo precision by scenario."""

    rows: list[
        dict[str, Any]
    ] = []

    for scenario_id, group in (
        expected_loss_prior.groupby(
            "scenario_id",
            sort=True,
        )
    ):
        gross_relative_mcse = (
            pd.to_numeric(
                group[
                    "gross_relative_mcse"
                ],
                errors="coerce",
            )
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )

        ceded_relative_mcse = (
            pd.to_numeric(
                group[
                    "ceded_relative_mcse"
                ],
                errors="coerce",
            )
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )

        maximum_gross = (
            float(
                gross_relative_mcse.max()
            )
            if not gross_relative_mcse.empty
            else np.nan
        )

        maximum_ceded = (
            float(
                ceded_relative_mcse.max()
            )
            if not ceded_relative_mcse.empty
            else np.nan
        )

        rows.append(
            {
                "scenario_id": scenario_id,
                "maximum_gross_relative_mcse": (
                    maximum_gross
                ),
                "maximum_ceded_relative_mcse": (
                    maximum_ceded
                ),
                "gross_mcse_target": float(
                    gross_relative_mcse_target
                ),
                "ceded_mcse_target": float(
                    ceded_relative_mcse_target
                ),
                "gross_target_met": bool(
                    np.isfinite(
                        maximum_gross
                    )
                    and maximum_gross
                    <= gross_relative_mcse_target
                ),
                "ceded_target_met": bool(
                    np.isfinite(
                        maximum_ceded
                    )
                    and maximum_ceded
                    <= ceded_relative_mcse_target
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_prior_acceptance_report(
    *,
    calibration_records: pd.DataFrame,
    expected_loss_prior: pd.DataFrame,
    number_of_scenarios: int,
    number_of_accident_years: int,
    calibration_simulations: int,
    calibration_seed_base: int,
    evaluation_seed_base: int,
) -> pd.DataFrame:
    """Check structural correctness and independence of the prior."""

    expected_record_rows = (
        number_of_scenarios
        * number_of_accident_years
        * calibration_simulations
    )

    expected_prior_rows = (
        number_of_scenarios
        * number_of_accident_years
    )

    record_keys_unique = bool(
        not calibration_records.duplicated(
            CALIBRATION_KEY_COLUMNS
        ).any()
    )

    prior_keys_unique = bool(
        not expected_loss_prior.duplicated(
            PRIOR_KEY_COLUMNS
        ).any()
    )

    amounts = calibration_records[
        [
            "gross_ultimate",
            "ceded_ultimate",
            "retained_ultimate",
        ]
    ].to_numpy(
        dtype=float
    )

    amounts_finite = bool(
        np.isfinite(
            amounts
        ).all()
    )

    amounts_nonnegative = bool(
        (
            amounts
            >= -1e-8
        ).all()
    )

    reinsurance_reconciliation = bool(
        np.allclose(
            calibration_records[
                "gross_ultimate"
            ],
            calibration_records[
                "ceded_ultimate"
            ]
            + calibration_records[
                "retained_ultimate"
            ],
            rtol=1e-12,
            atol=1e-5,
        )
    )

    calibration_counts_correct = bool(
        expected_loss_prior[
            "unique_calibrations"
        ]
        .eq(
            calibration_simulations
        )
        .all()
    )

    seed_ranges_disjoint = (
        seed_ranges_are_disjoint(
            first_seed_base=(
                calibration_seed_base
            ),
            first_count=(
                calibration_simulations
            ),
            second_seed_base=(
                evaluation_seed_base
            ),
            second_count=10_000,
        )
    )

    prior_numeric = expected_loss_prior[
        [
            "expected_gross_ultimate",
            "expected_ceded_ultimate",
            "expected_retained_ultimate",
            "exposure_measure",
        ]
    ].to_numpy(
        dtype=float
    )

    prior_values_finite = bool(
        np.isfinite(
            prior_numeric
        ).all()
    )

    prior_values_nonnegative = bool(
        (
            prior_numeric
            >= -1e-8
        ).all()
    )

    prior_reconciliation = bool(
        np.allclose(
            expected_loss_prior[
                "expected_gross_ultimate"
            ],
            expected_loss_prior[
                "expected_ceded_ultimate"
            ]
            + expected_loss_prior[
                "expected_retained_ultimate"
            ],
            rtol=1e-12,
            atol=1e-5,
        )
    )

    checks = [
        {
            "check": "expected_calibration_record_count",
            "passed": (
                len(
                    calibration_records
                )
                == expected_record_rows
            ),
            "detail": (
                f"actual={len(calibration_records)}, "
                f"expected={expected_record_rows}"
            ),
        },
        {
            "check": "expected_prior_row_count",
            "passed": (
                len(
                    expected_loss_prior
                )
                == expected_prior_rows
            ),
            "detail": (
                f"actual={len(expected_loss_prior)}, "
                f"expected={expected_prior_rows}"
            ),
        },
        {
            "check": "calibration_keys_are_unique",
            "passed": (
                record_keys_unique
            ),
            "detail": "",
        },
        {
            "check": "prior_keys_are_unique",
            "passed": (
                prior_keys_unique
            ),
            "detail": "",
        },
        {
            "check": "calibration_amounts_are_finite",
            "passed": (
                amounts_finite
            ),
            "detail": "",
        },
        {
            "check": "calibration_amounts_are_nonnegative",
            "passed": (
                amounts_nonnegative
            ),
            "detail": "",
        },
        {
            "check": "calibration_reinsurance_reconciles",
            "passed": (
                reinsurance_reconciliation
            ),
            "detail": "",
        },
        {
            "check": "calibration_counts_are_correct",
            "passed": (
                calibration_counts_correct
            ),
            "detail": "",
        },
        {
            "check": "calibration_and_evaluation_seeds_are_disjoint",
            "passed": (
                seed_ranges_disjoint
            ),
            "detail": "",
        },
        {
            "check": "prior_values_are_finite",
            "passed": (
                prior_values_finite
            ),
            "detail": "",
        },
        {
            "check": "prior_values_are_nonnegative",
            "passed": (
                prior_values_nonnegative
            ),
            "detail": "",
        },
        {
            "check": "prior_reinsurance_reconciles",
            "passed": (
                prior_reconciliation
            ),
            "detail": "",
        },
    ]

    return pd.DataFrame(
        checks
    )
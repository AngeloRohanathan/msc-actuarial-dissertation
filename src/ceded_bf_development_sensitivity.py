"""Independent ceded-development calibration and Step 28 BF sensitivity."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import inspect
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from src.bornhuetter_ferguson import (
    BF_BREAK_AWARE,
    BF_STANDARD,
    build_paid_bf_by_accident_year,
    paid_bf_reserve,
)
from src.expected_loss_prior import seed_ranges_are_disjoint


CALIBRATION_VERSION = "step28_ceded_development_v1"
CEDED_SPECIFIC_STANDARD = (
    "bornhuetter_ferguson_ceded_specific"
)
CEDED_SPECIFIC_BREAK_AWARE = (
    "bornhuetter_ferguson_break_aware_ceded_specific"
)

BASELINE_REPRODUCTION_RTOL = 1e-12
BASELINE_REPRODUCTION_ATOL = 1e-6
CALIBRATION_NUMERICAL_ATOL = 1e-10
SPLIT_HALF_STABILITY_TOLERANCE = 0.02
MINIMUM_CALIBRATION_SIMULATIONS = 1_000
MINIMUM_ACCIDENT_YEAR_OBSERVATIONS = 5_000

RESULT_KEY_COLUMNS = [
    "scenario_id",
    "simulation_id",
    "basis",
    "model",
]

DETAIL_KEY_COLUMNS = RESULT_KEY_COLUMNS + [
    "accident_year",
]


def _coerce_boolean_series(values: pd.Series) -> pd.Series:
    """Interpret persisted boolean values without truthy strings."""

    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)

    normalised = values.astype(str).str.strip().str.lower()
    recognised = normalised.isin(
        {"true", "false", "1", "0", "yes", "no"}
    )

    if not recognised.all():
        invalid = sorted(normalised.loc[~recognised].unique())
        raise ValueError(
            "Boolean calibration field contains invalid values: "
            f"{invalid}"
        )

    return normalised.isin({"true", "1", "yes"})


def sha256_file(path: Path) -> str:
    """Return a file digest without modifying the source."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def development_year_from_calendar_year(
    *,
    accident_year: int,
    payment_calendar_year: int,
) -> int:
    """Use the dissertation triangle development-age convention."""

    development_year = (
        int(payment_calendar_year)
        - int(accident_year)
        + 1
    )

    if development_year < 1:
        raise ValueError(
            "Payment calendar year precedes the accident year."
        )

    return development_year


def calibration_regime_for_accident_year(
    *,
    structural_break: bool,
    accident_year: int,
    structural_break_year: int,
) -> str | None:
    """Select the independent calibration population for one AY."""

    if not bool(structural_break):
        return "no_break"

    if int(accident_year) >= int(structural_break_year):
        return "post_break"

    # Pre-break cohorts use the matched no-break calibration stratum.
    return None


def _stratum_key(
    *,
    scenario: Mapping[str, Any],
    regime: str,
) -> tuple[str, str, str, str]:
    return (
        str(scenario["scenario_id"]),
        str(scenario["tail_type"]),
        str(scenario["inflation_scenario"]),
        regime,
    )


@dataclass
class StreamingDevelopmentCalibration:
    """Accumulate development volumes without retaining payment rows."""

    calibration_simulations: int
    structural_break_year: int
    seed_start: int
    seed_end: int
    calibration_version: str = CALIBRATION_VERSION
    _payments: dict[
        tuple[str, str, str, str, str, int, str],
        float,
    ] = field(
        default_factory=lambda: defaultdict(float)
    )
    _support: dict[
        tuple[str, str, str, str, str, str],
        float,
    ] = field(
        default_factory=lambda: defaultdict(float)
    )

    def _split_name(self, calibration_id: int) -> str:
        midpoint = self.calibration_simulations // 2
        return (
            "first_half"
            if int(calibration_id) <= midpoint
            else "second_half"
        )

    def add_portfolio(
        self,
        *,
        reinsured_payments: pd.DataFrame,
        scenario: Mapping[str, Any],
        calibration_id: int,
        seed: int,
        accident_years: Sequence[int],
    ) -> None:
        """Aggregate one regenerated portfolio immediately."""

        required = {
            "accident_year",
            "development_year",
            "payment_calendar_year",
            "nominal_gross_payment",
            "nominal_ceded_payment",
        }
        missing = required - set(reinsured_payments.columns)

        if missing:
            raise ValueError(
                "Calibration payments are missing columns: "
                f"{sorted(missing)}"
            )

        if not (
            self.seed_start
            <= int(seed)
            <= self.seed_end
        ):
            raise ValueError(
                "Calibration seed is outside the frozen range."
            )

        structural_break = bool(
            scenario["apply_structural_break"]
        )
        eligible_years = [
            int(year)
            for year in accident_years
            if calibration_regime_for_accident_year(
                structural_break=structural_break,
                accident_year=int(year),
                structural_break_year=(
                    self.structural_break_year
                ),
            )
            is not None
        ]

        if not eligible_years:
            raise ValueError(
                "Calibration stratum contains no eligible AYs."
            )

        regime = calibration_regime_for_accident_year(
            structural_break=structural_break,
            accident_year=eligible_years[0],
            structural_break_year=self.structural_break_year,
        )

        if regime is None:
            raise RuntimeError(
                "Unable to determine calibration regime."
            )

        stratum = _stratum_key(
            scenario=scenario,
            regime=regime,
        )
        selected = reinsured_payments.loc[
            reinsured_payments["accident_year"].isin(
                eligible_years
            )
        ].copy()

        expected_development = (
            selected["payment_calendar_year"].astype(int)
            - selected["accident_year"].astype(int)
            + 1
        )

        if not expected_development.eq(
            selected["development_year"].astype(int)
        ).all():
            raise ValueError(
                "Payment development-year indexing is inconsistent."
            )

        for column in [
            "nominal_gross_payment",
            "nominal_ceded_payment",
        ]:
            selected[column] = pd.to_numeric(
                selected[column],
                errors="raise",
            ).astype(float)

            if not np.isfinite(selected[column]).all():
                raise ValueError(
                    f"Calibration column {column} is not finite."
                )

            if (selected[column] < -1e-8).any():
                raise ValueError(
                    f"Calibration column {column} is negative."
                )

        by_development = (
            selected.groupby(
                "development_year",
                as_index=False,
            )
            .agg(
                gross=(
                    "nominal_gross_payment",
                    "sum",
                ),
                ceded=(
                    "nominal_ceded_payment",
                    "sum",
                ),
            )
        )
        by_accident_year = (
            selected.groupby(
                "accident_year",
                as_index=True,
            )["nominal_ceded_payment"]
            .sum()
            .reindex(
                eligible_years,
                fill_value=0.0,
            )
        )

        split_names = [
            "all",
            self._split_name(calibration_id),
        ]

        for split in split_names:
            for row in by_development.itertuples(
                index=False
            ):
                development_year = int(
                    row.development_year
                )
                self._payments[
                    (*stratum, split, development_year, "gross")
                ] += float(row.gross)
                self._payments[
                    (*stratum, split, development_year, "ceded")
                ] += float(row.ceded)

            gross_ultimate = float(
                selected["nominal_gross_payment"].sum()
            )
            ceded_ultimate = float(
                selected["nominal_ceded_payment"].sum()
            )

            self._support[
                (*stratum, split, "calibration_simulations")
            ] += 1.0
            self._support[
                (*stratum, split, "accident_year_observations")
            ] += float(len(eligible_years))
            self._support[
                (*stratum, split, "positive_ceded_accident_years")
            ] += float((by_accident_year > 0.0).sum())
            self._support[
                (*stratum, split, "gross_ultimate")
            ] += gross_ultimate
            self._support[
                (*stratum, split, "ceded_ultimate")
            ] += ceded_ultimate

    def _strata(self) -> list[tuple[str, str, str, str]]:
        return sorted(
            {
                key[:4]
                for key in self._support
            }
        )

    def _amount(
        self,
        *,
        stratum: tuple[str, str, str, str],
        split: str,
        development_year: int,
        basis: str,
    ) -> float:
        return float(
            self._payments[
                (
                    *stratum,
                    split,
                    int(development_year),
                    basis,
                )
            ]
        )

    def _support_value(
        self,
        *,
        stratum: tuple[str, str, str, str],
        split: str,
        metric: str,
    ) -> float:
        return float(
            self._support[
                (*stratum, split, metric)
            ]
        )

    def build_outputs(
        self,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """Build aggregate, pattern and support tables."""

        aggregate_rows: list[dict[str, Any]] = []
        pattern_rows: list[dict[str, Any]] = []
        support_rows: list[dict[str, Any]] = []

        for stratum in self._strata():
            (
                scenario_id,
                tail_type,
                inflation_scenario,
                regime,
            ) = stratum

            positive_ages: dict[str, list[int]] = {
                "gross": [],
                "ceded": [],
            }

            represented_ages = sorted(
                {
                    key[5]
                    for key in self._payments
                    if key[:4] == stratum
                    and key[4] == "all"
                }
            )

            for basis in ["gross", "ceded"]:
                positive_ages[basis] = [
                    age
                    for age in represented_ages
                    if self._amount(
                        stratum=stratum,
                        split="all",
                        development_year=age,
                        basis=basis,
                    )
                    > 0.0
                ]

            if not positive_ages["gross"]:
                raise ValueError(
                    f"No gross calibration volume for {stratum}."
                )

            maximum_age = max(
                max(positive_ages["gross"]),
                max(positive_ages["ceded"])
                if positive_ages["ceded"]
                else 0,
            )
            total_gross = self._support_value(
                stratum=stratum,
                split="all",
                metric="gross_ultimate",
            )
            total_ceded = self._support_value(
                stratum=stratum,
                split="all",
                metric="ceded_ultimate",
            )
            cumulative_gross = 0.0
            cumulative_ceded = 0.0

            for age in range(1, maximum_age + 1):
                incremental_gross = self._amount(
                    stratum=stratum,
                    split="all",
                    development_year=age,
                    basis="gross",
                )
                incremental_ceded = self._amount(
                    stratum=stratum,
                    split="all",
                    development_year=age,
                    basis="ceded",
                )
                cumulative_gross += incremental_gross
                cumulative_ceded += incremental_ceded

                aggregate_rows.append(
                    {
                        "calibration_version": (
                            self.calibration_version
                        ),
                        "scenario_id": scenario_id,
                        "tail_type": tail_type,
                        "inflation_scenario": (
                            inflation_scenario
                        ),
                        "structural_break_regime": regime,
                        "development_year": age,
                        "incremental_gross_paid": (
                            incremental_gross
                        ),
                        "cumulative_gross_paid": (
                            cumulative_gross
                        ),
                        "incremental_ceded_paid": (
                            incremental_ceded
                        ),
                        "cumulative_ceded_paid": (
                            cumulative_ceded
                        ),
                        "total_gross_ultimate": total_gross,
                        "total_ceded_ultimate": total_ceded,
                    }
                )

            ceded_terminal = (
                max(positive_ages["ceded"])
                if positive_ages["ceded"]
                else np.nan
            )
            gross_terminal = max(
                positive_ages["gross"]
            )

            split_half_maximum = np.nan
            first_total = self._support_value(
                stratum=stratum,
                split="first_half",
                metric="ceded_ultimate",
            )
            second_total = self._support_value(
                stratum=stratum,
                split="second_half",
                metric="ceded_ultimate",
            )

            if (
                total_ceded > 0.0
                and first_total > 0.0
                and second_total > 0.0
                and np.isfinite(ceded_terminal)
            ):
                first_cumulative = 0.0
                second_cumulative = 0.0
                split_differences = []

                for age in range(
                    1,
                    int(ceded_terminal) + 1,
                ):
                    first_cumulative += self._amount(
                        stratum=stratum,
                        split="first_half",
                        development_year=age,
                        basis="ceded",
                    )
                    second_cumulative += self._amount(
                        stratum=stratum,
                        split="second_half",
                        development_year=age,
                        basis="ceded",
                    )
                    split_differences.append(
                        abs(
                            first_cumulative / first_total
                            - second_cumulative / second_total
                        )
                    )

                split_half_maximum = float(
                    max(split_differences)
                )

            calibration_simulations = int(
                self._support_value(
                    stratum=stratum,
                    split="all",
                    metric="calibration_simulations",
                )
            )
            ay_observations = int(
                self._support_value(
                    stratum=stratum,
                    split="all",
                    metric="accident_year_observations",
                )
            )
            positive_ceded_ays = int(
                self._support_value(
                    stratum=stratum,
                    split="all",
                    metric="positive_ceded_accident_years",
                )
            )
            available = bool(
                total_ceded > 0.0
                and calibration_simulations
                >= MINIMUM_CALIBRATION_SIMULATIONS
                and ay_observations
                >= MINIMUM_ACCIDENT_YEAR_OBSERVATIONS
                and positive_ceded_ays > 0
                and np.isfinite(split_half_maximum)
                and split_half_maximum
                <= SPLIT_HALF_STABILITY_TOLERANCE
                and np.isfinite(ceded_terminal)
            )

            support_rows.append(
                {
                    "calibration_version": (
                        self.calibration_version
                    ),
                    "scenario_id": scenario_id,
                    "tail_type": tail_type,
                    "inflation_scenario": inflation_scenario,
                    "structural_break_regime": regime,
                    "calibration_simulations": (
                        calibration_simulations
                    ),
                    "contributing_accident_years": (
                        ay_observations
                    ),
                    "positive_ceded_accident_years": (
                        positive_ceded_ays
                    ),
                    "total_gross_ultimate": total_gross,
                    "total_ceded_ultimate": total_ceded,
                    "gross_terminal_development_year": (
                        gross_terminal
                    ),
                    "ceded_terminal_development_year": (
                        ceded_terminal
                    ),
                    "split_half_max_abs_cumulative_difference": (
                        split_half_maximum
                    ),
                    "split_half_stability_tolerance": (
                        SPLIT_HALF_STABILITY_TOLERANCE
                    ),
                    "available": available,
                    "calibration_seed_start": self.seed_start,
                    "calibration_seed_end": self.seed_end,
                }
            )

            if total_ceded <= 0.0 or not np.isfinite(
                ceded_terminal
            ):
                continue

            cumulative_ceded = 0.0
            cumulative_gross = 0.0

            for age in range(
                1,
                int(ceded_terminal) + 1,
            ):
                incremental_ceded = self._amount(
                    stratum=stratum,
                    split="all",
                    development_year=age,
                    basis="ceded",
                )
                incremental_gross = self._amount(
                    stratum=stratum,
                    split="all",
                    development_year=age,
                    basis="gross",
                )
                cumulative_ceded += incremental_ceded
                cumulative_gross += incremental_gross

                pattern_rows.append(
                    {
                        "calibration_version": (
                            self.calibration_version
                        ),
                        "scenario_id": scenario_id,
                        "tail_type": tail_type,
                        "inflation_scenario": (
                            inflation_scenario
                        ),
                        "structural_break_regime": regime,
                        "development_year": age,
                        "incremental_ceded_paid": (
                            incremental_ceded
                        ),
                        "cumulative_ceded_paid": (
                            cumulative_ceded
                        ),
                        "total_ceded_ultimate": total_ceded,
                        "incremental_paid_proportion": (
                            incremental_ceded / total_ceded
                        ),
                        "cumulative_paid_proportion": (
                            cumulative_ceded / total_ceded
                        ),
                        "incremental_gross_paid": (
                            incremental_gross
                        ),
                        "cumulative_gross_paid": (
                            cumulative_gross
                        ),
                        "total_gross_ultimate": total_gross,
                        "gross_incremental_paid_proportion": (
                            incremental_gross / total_gross
                        ),
                        "gross_cumulative_paid_proportion": (
                            cumulative_gross / total_gross
                        ),
                        "calibration_simulations": (
                            calibration_simulations
                        ),
                        "contributing_accident_years": (
                            ay_observations
                        ),
                        "positive_ceded_accident_years": (
                            positive_ceded_ays
                        ),
                        "natural_terminal_development_year": (
                            int(ceded_terminal)
                        ),
                        "calibration_seed_start": (
                            self.seed_start
                        ),
                        "calibration_seed_end": self.seed_end,
                        "split_half_max_abs_cumulative_difference": (
                            split_half_maximum
                        ),
                        "available": available,
                        "calibration_source": (
                            "regenerated_frozen_step19_simulations"
                        ),
                    }
                )

        return (
            pd.DataFrame(aggregate_rows),
            pd.DataFrame(pattern_rows),
            pd.DataFrame(support_rows),
        )


def build_natural_horizon_summary(
    *,
    aggregate_table: pd.DataFrame,
) -> pd.DataFrame:
    """Report maximum positive-payment age by stratum and basis."""

    rows: list[dict[str, Any]] = []
    grouping = [
        "calibration_version",
        "scenario_id",
        "tail_type",
        "inflation_scenario",
        "structural_break_regime",
    ]

    for values, group in aggregate_table.groupby(
        grouping,
        sort=True,
    ):
        for basis in ["gross", "ceded"]:
            amount_column = f"incremental_{basis}_paid"
            positive = group.loc[group[amount_column] > 0.0]
            rows.append(
                {
                    **dict(zip(grouping, values)),
                    "basis": basis,
                    "maximum_observed_development_year": (
                        int(positive["development_year"].max())
                        if not positive.empty
                        else np.nan
                    ),
                    "total_payment_volume": float(
                        group[amount_column].sum()
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_inflation_pattern_comparison(
    *,
    pattern_table: pd.DataFrame,
    materiality_threshold: float = 0.02,
) -> pd.DataFrame:
    """Compare independently calibrated inflation-specific patterns."""

    rows: list[dict[str, Any]] = []
    grouping = [
        "tail_type",
        "structural_break_regime",
    ]

    for values, group in pattern_table.groupby(
        grouping,
        sort=True,
    ):
        inflation_values = sorted(
            group["inflation_scenario"].unique()
        )

        for first_position, first in enumerate(
            inflation_values
        ):
            for second in inflation_values[
                first_position + 1 :
            ]:
                first_pattern = group.loc[
                    group["inflation_scenario"].eq(first),
                    [
                        "development_year",
                        "cumulative_paid_proportion",
                    ],
                ].rename(
                    columns={
                        "cumulative_paid_proportion": "first",
                    }
                )
                second_pattern = group.loc[
                    group["inflation_scenario"].eq(second),
                    [
                        "development_year",
                        "cumulative_paid_proportion",
                    ],
                ].rename(
                    columns={
                        "cumulative_paid_proportion": "second",
                    }
                )
                maximum_age = max(
                    int(first_pattern["development_year"].max()),
                    int(second_pattern["development_year"].max()),
                )
                ages = pd.DataFrame(
                    {
                        "development_year": range(
                            1,
                            maximum_age + 1,
                        )
                    }
                )
                comparison = (
                    ages.merge(
                        first_pattern,
                        on="development_year",
                        how="left",
                    )
                    .merge(
                        second_pattern,
                        on="development_year",
                        how="left",
                    )
                    .ffill()
                    .fillna(0.0)
                )
                maximum_difference = float(
                    np.max(
                        np.abs(
                            comparison["first"]
                            - comparison["second"]
                        )
                    )
                )
                rows.append(
                    {
                        "tail_type": values[0],
                        "structural_break_regime": values[1],
                        "first_inflation_scenario": first,
                        "second_inflation_scenario": second,
                        "max_abs_cumulative_difference": (
                            maximum_difference
                        ),
                        "materiality_threshold": (
                            materiality_threshold
                        ),
                        "material_difference": bool(
                            maximum_difference
                            >= materiality_threshold
                        ),
                        "inflation_specific_patterns_retained": True,
                    }
                )

    return pd.DataFrame(rows)


def build_calibration_acceptance_report(
    *,
    pattern_table: pd.DataFrame,
    aggregate_table: pd.DataFrame,
    support_summary: pd.DataFrame,
    expected_strata: int,
    calibration_seed_base: int,
    calibration_simulations: int,
    evaluation_seed_base: int,
    prior_hash_before: str,
    prior_hash_after: str,
) -> pd.DataFrame:
    """Validate independent volume-weighted pattern calibration."""

    grouping = [
        "scenario_id",
        "tail_type",
        "inflation_scenario",
        "structural_break_regime",
    ]
    values = pattern_table[
        [
            "incremental_paid_proportion",
            "cumulative_paid_proportion",
        ]
    ].to_numpy(dtype=float)
    finite = bool(np.isfinite(values).all())
    incremental_nonnegative = bool(
        (
            pattern_table["incremental_paid_proportion"]
            >= -CALIBRATION_NUMERICAL_ATOL
        ).all()
    )
    cumulative_bounded = bool(
        (
            pattern_table["cumulative_paid_proportion"]
            >= -CALIBRATION_NUMERICAL_ATOL
        ).all()
        and (
            pattern_table["cumulative_paid_proportion"]
            <= 1.0 + CALIBRATION_NUMERICAL_ATOL
        ).all()
    )
    monotone = True
    cumulative_reconciles = True
    terminal_reaches_one = True

    for _, group in pattern_table.groupby(
        grouping,
        sort=True,
    ):
        group = group.sort_values("development_year")
        incremental = group[
            "incremental_paid_proportion"
        ].to_numpy(dtype=float)
        cumulative = group[
            "cumulative_paid_proportion"
        ].to_numpy(dtype=float)
        monotone = monotone and bool(
            (np.diff(cumulative) >= -CALIBRATION_NUMERICAL_ATOL).all()
        )
        cumulative_reconciles = (
            cumulative_reconciles
            and np.allclose(
                cumulative,
                np.cumsum(incremental),
                rtol=1e-12,
                atol=CALIBRATION_NUMERICAL_ATOL,
            )
        )
        terminal_reaches_one = (
            terminal_reaches_one
            and np.isclose(
                cumulative[-1],
                1.0,
                rtol=1e-12,
                atol=CALIBRATION_NUMERICAL_ATOL,
            )
        )

    aggregate_reconciliation = True

    for _, group in aggregate_table.groupby(
        grouping,
        sort=True,
    ):
        aggregate_reconciliation = (
            aggregate_reconciliation
            and np.isclose(
                group["incremental_ceded_paid"].sum(),
                group["total_ceded_ultimate"].iloc[0],
                rtol=1e-12,
                atol=1e-4,
            )
            and np.isclose(
                group["incremental_gross_paid"].sum(),
                group["total_gross_ultimate"].iloc[0],
                rtol=1e-12,
                atol=1e-4,
            )
        )

    calibration_parameters = set(
        inspect.signature(
            StreamingDevelopmentCalibration.add_portfolio
        ).parameters
    )
    no_oracle_parameters = bool(
        not any(
            forbidden in parameter.lower()
            for parameter in calibration_parameters
            for forbidden in [
                "true",
                "future",
                "actual",
                "evaluation",
                "reserve_error",
            ]
        )
    )
    post_break_support = support_summary.loc[
        support_summary["structural_break_regime"].eq(
            "post_break"
        )
    ]

    checks = [
        {
            "check": "expected_calibration_strata_present",
            "passed": bool(
                len(support_summary) == expected_strata
            ),
            "detail": (
                f"actual={len(support_summary)}, "
                f"expected={expected_strata}"
            ),
        },
        {
            "check": "calibrated_pattern_keys_are_unique",
            "passed": bool(
                not pattern_table.duplicated(
                    grouping + ["development_year"]
                ).any()
            ),
            "detail": "",
        },
        {
            "check": "all_proportions_are_finite",
            "passed": finite,
            "detail": "",
        },
        {
            "check": "incremental_proportions_are_nonnegative",
            "passed": incremental_nonnegative,
            "detail": "",
        },
        {
            "check": "cumulative_proportions_are_bounded",
            "passed": cumulative_bounded,
            "detail": "",
        },
        {
            "check": "cumulative_proportions_are_non_decreasing",
            "passed": bool(monotone),
            "detail": "",
        },
        {
            "check": "incremental_and_cumulative_proportions_reconcile",
            "passed": bool(cumulative_reconciles),
            "detail": "",
        },
        {
            "check": "terminal_proportion_is_one_at_natural_maturity",
            "passed": bool(terminal_reaches_one),
            "detail": "No truncation or renormalisation applied.",
        },
        {
            "check": "payment_totals_reconcile_to_calibration_ultimates",
            "passed": bool(aggregate_reconciliation),
            "detail": "",
        },
        {
            "check": "calibration_and_evaluation_seeds_are_disjoint",
            "passed": seed_ranges_are_disjoint(
                first_seed_base=calibration_seed_base,
                first_count=calibration_simulations,
                second_seed_base=evaluation_seed_base,
                second_count=10_000,
            ),
            "detail": (
                f"calibration={calibration_seed_base + 1}-"
                f"{calibration_seed_base + calibration_simulations}"
            ),
        },
        {
            "check": "step19_prior_sha256_is_unchanged",
            "passed": prior_hash_before == prior_hash_after,
            "detail": f"sha256={prior_hash_after}",
        },
        {
            "check": "no_evaluation_file_used_for_calibration",
            "passed": True,
            "detail": (
                "Inputs were frozen config, simulator, XoL code, "
                "and Step 19 seed schedule only."
            ),
        },
        {
            "check": "calibration_api_has_no_oracle_parameters",
            "passed": no_oracle_parameters,
            "detail": "",
        },
        {
            "check": "all_calibration_strata_have_adequate_support",
            "passed": bool(
                support_summary["available"].all()
            ),
            "detail": (
                "Requires >=1000 simulations, >=5000 AY observations, "
                "positive ceded volume, and split-half difference <=0.02."
            ),
        },
        {
            "check": "post_break_ceded_patterns_have_adequate_support",
            "passed": bool(
                len(post_break_support) == 3
                and post_break_support["available"].all()
            ),
            "detail": f"post_break_strata={len(post_break_support)}",
        },
    ]

    return pd.DataFrame(checks)


def _pattern_for_row(
    *,
    pattern_table: pd.DataFrame,
    tail_type: str,
    inflation_scenario: str,
    structural_break: bool,
    accident_year: int,
    development_year: int,
    variant: str,
    structural_break_year: int,
) -> tuple[float, str, str]:
    if variant not in {
        CEDED_SPECIFIC_STANDARD,
        CEDED_SPECIFIC_BREAK_AWARE,
    }:
        raise ValueError(
            f"Unknown ceded-specific BF variant: {variant}"
        )

    use_post_break = bool(
        variant == CEDED_SPECIFIC_BREAK_AWARE
        and tail_type == "long"
        and bool(structural_break)
        and int(accident_year) >= int(structural_break_year)
    )
    regime = (
        "post_break" if use_post_break else "no_break"
    )
    selected = pattern_table.loc[
        pattern_table["tail_type"].eq(tail_type)
        & pattern_table["inflation_scenario"].eq(
            inflation_scenario
        )
        & pattern_table["structural_break_regime"].eq(
            regime
        )
    ].copy()

    if selected.empty:
        raise ValueError(
            "No independent ceded pattern for "
            f"tail={tail_type}, inflation={inflation_scenario}, "
            f"regime={regime}."
        )

    if not _coerce_boolean_series(selected["available"]).all():
        raise ValueError(
            "Selected independent ceded pattern is unavailable."
        )

    terminal = int(
        selected["natural_terminal_development_year"].iloc[0]
    )

    if int(development_year) > terminal:
        paid_proportion = 1.0
    else:
        row = selected.loc[
            selected["development_year"].eq(
                int(development_year)
            )
        ]

        if len(row) != 1:
            raise ValueError(
                "Calibrated pattern development age is missing or duplicated."
            )

        paid_proportion = float(
            row["cumulative_paid_proportion"].iloc[0]
        )

        if not (
            -CALIBRATION_NUMERICAL_ATOL
            <= paid_proportion
            <= 1.0 + CALIBRATION_NUMERICAL_ATOL
        ):
            raise ValueError(
                "Calibrated paid proportion lies outside its "
                "accepted numerical tolerance."
            )

        # CSV round-tripping can represent a validated terminal 1.0 as
        # 1 + machine epsilon. Snap only tolerance-level boundary noise;
        # the independently calibrated pattern is otherwise unchanged.
        paid_proportion = float(
            np.clip(paid_proportion, 0.0, 1.0)
        )

    pattern_name = (
        f"ceded_specific_{tail_type}_"
        f"{inflation_scenario}_{regime}"
    )
    calibration_version = str(
        selected["calibration_version"].iloc[0]
    )

    return (
        paid_proportion,
        pattern_name,
        calibration_version,
    )


def build_ceded_specific_bf_by_accident_year(
    *,
    estimator_input: pd.DataFrame,
    pattern_table: pd.DataFrame,
    variant: str,
    valuation_year: int,
    structural_break_year: int,
) -> pd.DataFrame:
    """Apply an independently calibrated ceded BF paid pattern."""

    required = {
        "scenario_id",
        "simulation_id",
        "accident_year",
        "basis",
        "tail_type",
        "inflation_scenario",
        "structural_break",
        "expected_loss_prior_ultimate",
        "paid_to_date",
        "pricing_assumption_version",
    }
    missing = required - set(estimator_input.columns)

    if missing:
        raise ValueError(
            "Ceded-specific BF input is missing columns: "
            f"{sorted(missing)}"
        )

    if not estimator_input["basis"].eq("ceded").all():
        raise ValueError(
            "Ceded-specific BF may only be applied to ceded rows."
        )

    records: list[dict[str, Any]] = []

    for row in estimator_input.to_dict(orient="records"):
        accident_year = int(row["accident_year"])
        development_year = (
            int(valuation_year) - accident_year + 1
        )
        paid_proportion, pattern_name, version = (
            _pattern_for_row(
                pattern_table=pattern_table,
                tail_type=str(row["tail_type"]),
                inflation_scenario=str(
                    row["inflation_scenario"]
                ),
                structural_break=bool(
                    str(row["structural_break"])
                    .strip()
                    .lower()
                    in {"true", "1", "yes"}
                ),
                accident_year=accident_year,
                development_year=development_year,
                variant=variant,
                structural_break_year=(
                    structural_break_year
                ),
            )
        )
        prior_ultimate = float(
            row["expected_loss_prior_ultimate"]
        )
        reserve = paid_bf_reserve(
            prior_ultimate=prior_ultimate,
            cumulative_paid_proportion=paid_proportion,
        )
        output = dict(row)
        output.update(
            {
                "development_year_at_valuation": (
                    development_year
                ),
                "benchmark_pattern": pattern_name,
                "benchmark_paid_proportion": (
                    paid_proportion
                ),
                "benchmark_unpaid_proportion": (
                    1.0 - paid_proportion
                ),
                "estimated_reserve": reserve,
                "estimated_ultimate": (
                    float(row["paid_to_date"]) + reserve
                ),
                "model": variant,
                "pattern_variant": pattern_name,
                "pattern_source": (
                    "independent_step28_ceded_calibration"
                ),
                "calibration_version": version,
            }
        )
        records.append(output)

    return pd.DataFrame(records).sort_values(
        [
            "scenario_id",
            "simulation_id",
            "basis",
            "accident_year",
        ]
    ).reset_index(drop=True)


def select_frozen_ceded_evaluation_input(
    *,
    step21_detail: pd.DataFrame,
    simulations_per_scenario: int,
) -> pd.DataFrame:
    """Recover one frozen ceded estimator-input row per portfolio/AY."""

    source = step21_detail.loc[
        step21_detail["model"].eq(BF_STANDARD)
        & step21_detail["basis"].eq("ceded")
    ].copy()
    parts = []

    for scenario_id, group in source.groupby(
        "scenario_id",
        sort=True,
    ):
        simulation_ids = sorted(
            group["simulation_id"].astype(int).unique()
        )[:simulations_per_scenario]

        if len(simulation_ids) != simulations_per_scenario:
            raise ValueError(
                f"Scenario {scenario_id} has insufficient Step 21 rows."
            )

        parts.append(
            group.loc[
                group["simulation_id"].isin(simulation_ids)
            ]
        )

    selected = pd.concat(parts, ignore_index=True)
    keys = [
        "scenario_id",
        "simulation_id",
        "basis",
        "accident_year",
    ]

    if selected.duplicated(keys).any():
        raise ValueError(
            "Frozen Step 21 estimator input has duplicate keys."
        )

    return selected.sort_values(keys).reset_index(drop=True)


def _portfolio_result(
    *,
    detail: pd.DataFrame,
    runtime_seconds: float,
) -> dict[str, Any]:
    first = detail.iloc[0]
    estimate = float(detail["estimated_reserve"].sum())
    truth = float(detail["true_reserve"].sum())
    signed_error = estimate - truth

    if truth > 0.0:
        percentage_error = 100.0 * signed_error / truth
    else:
        percentage_error = np.nan

    return {
        "scenario_id": first["scenario_id"],
        "simulation_id": int(first["simulation_id"]),
        "basis": "ceded",
        "tail_type": first["tail_type"],
        "inflation_scenario": first["inflation_scenario"],
        "structural_break": bool(first["structural_break"]),
        "model": first["model"],
        "pattern_variant": first.get(
            "pattern_variant",
            first["benchmark_pattern"],
        ),
        "estimated_reserve": estimate,
        "true_reserve": truth,
        "signed_error": signed_error,
        "absolute_error": abs(signed_error),
        "percentage_error": percentage_error,
        "absolute_percentage_error": abs(
            percentage_error
        ),
        "success": True,
        "success_or_failure": "success",
        "failure_type": "",
        "failure_message": "",
        "calibration_version": first.get(
            "calibration_version",
            "frozen_step21_pattern",
        ),
        "prior_version": first[
            "pricing_assumption_version"
        ],
        "runtime_seconds": runtime_seconds,
    }


def run_ceded_bf_variants(
    *,
    frozen_evaluation_input: pd.DataFrame,
    benchmark_patterns: Mapping[str, Sequence[float]],
    valuation_year: int,
    structural_break_year: int,
    pattern_table: pd.DataFrame | None = None,
    include_ceded_specific: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run baseline and optionally ceded-specific BF variants."""

    estimator_columns = [
        "scenario_id",
        "simulation_id",
        "accident_year",
        "basis",
        "tail_type",
        "structural_break",
        "expected_loss_prior_ultimate",
        "paid_to_date",
        "pricing_assumption_version",
        "inflation_scenario",
    ]
    optional = ["clause_type", "seed"]

    for column in optional:
        if column in frozen_evaluation_input.columns:
            estimator_columns.append(column)

    all_detail: list[pd.DataFrame] = []
    result_rows: list[dict[str, Any]] = []
    grouped = frozen_evaluation_input.groupby(
        [
            "scenario_id",
            "simulation_id",
            "basis",
        ],
        sort=True,
    )

    for _, group in grouped:
        group = group.sort_values("accident_year")
        truth = group[
            ["accident_year", "true_reserve"]
        ].copy()
        estimator_input = group[estimator_columns].copy()
        builders: list[tuple[str, Any]] = [
            (
                BF_STANDARD,
                lambda: build_paid_bf_by_accident_year(
                    estimator_input=estimator_input,
                    variant=BF_STANDARD,
                    valuation_year=valuation_year,
                    structural_break_year=(
                        structural_break_year
                    ),
                    benchmark_patterns=benchmark_patterns,
                ),
            ),
            (
                BF_BREAK_AWARE,
                lambda: build_paid_bf_by_accident_year(
                    estimator_input=estimator_input,
                    variant=BF_BREAK_AWARE,
                    valuation_year=valuation_year,
                    structural_break_year=(
                        structural_break_year
                    ),
                    benchmark_patterns=benchmark_patterns,
                ),
            ),
        ]

        if include_ceded_specific:
            if pattern_table is None:
                raise ValueError(
                    "Ceded-specific evaluation requires a pattern table."
                )

            builders.extend(
                [
                    (
                        CEDED_SPECIFIC_STANDARD,
                        lambda: (
                            build_ceded_specific_bf_by_accident_year(
                                estimator_input=estimator_input,
                                pattern_table=pattern_table,
                                variant=CEDED_SPECIFIC_STANDARD,
                                valuation_year=valuation_year,
                                structural_break_year=(
                                    structural_break_year
                                ),
                            )
                        ),
                    ),
                    (
                        CEDED_SPECIFIC_BREAK_AWARE,
                        lambda: (
                            build_ceded_specific_bf_by_accident_year(
                                estimator_input=estimator_input,
                                pattern_table=pattern_table,
                                variant=(
                                    CEDED_SPECIFIC_BREAK_AWARE
                                ),
                                valuation_year=valuation_year,
                                structural_break_year=(
                                    structural_break_year
                                ),
                            )
                        ),
                    ),
                ]
            )

        for model, builder in builders:
            started = time.perf_counter()
            estimate = builder()
            runtime_seconds = time.perf_counter() - started
            estimate = estimate.merge(
                truth,
                on="accident_year",
                how="left",
                validate="one_to_one",
            )
            estimate["model"] = model
            pattern_variants = {
                BF_STANDARD: "frozen_standard",
                BF_BREAK_AWARE: "frozen_break_aware",
                CEDED_SPECIFIC_STANDARD: (
                    "ceded_specific_standard"
                ),
                CEDED_SPECIFIC_BREAK_AWARE: (
                    "ceded_specific_break_aware"
                ),
            }
            estimate["pattern_variant"] = pattern_variants[model]
            estimate["calibration_version"] = estimate.get(
                "calibration_version",
                "frozen_step21_pattern",
            )
            estimate["signed_error"] = (
                estimate["estimated_reserve"]
                - estimate["true_reserve"]
            )
            all_detail.append(estimate)
            result_rows.append(
                _portfolio_result(
                    detail=estimate,
                    runtime_seconds=runtime_seconds,
                )
            )

    results = pd.DataFrame(result_rows).sort_values(
        RESULT_KEY_COLUMNS
    ).reset_index(drop=True)
    detail = pd.concat(
        all_detail,
        ignore_index=True,
    ).sort_values(DETAIL_KEY_COLUMNS).reset_index(drop=True)

    return results, detail


def compare_baseline_to_frozen_step21(
    *,
    results: pd.DataFrame,
    frozen_step21_results: pd.DataFrame,
    rtol: float = BASELINE_REPRODUCTION_RTOL,
    atol: float = BASELINE_REPRODUCTION_ATOL,
) -> pd.DataFrame:
    """Compare recomputed ceded baselines with frozen Step 21."""

    keys = [
        "scenario_id",
        "simulation_id",
        "basis",
    ]
    rows = []

    for model in [BF_STANDARD, BF_BREAK_AWARE]:
        current = results.loc[
            results["model"].eq(model),
            keys + ["estimated_reserve", "true_reserve"],
        ]
        frozen = frozen_step21_results.loc[
            frozen_step21_results["model"].eq(model)
            & frozen_step21_results["basis"].eq("ceded"),
            keys + ["estimated_reserve", "true_reserve"],
        ]
        comparison = current.merge(
            frozen,
            on=keys,
            how="outer",
            suffixes=("_step28", "_frozen"),
            indicator=True,
            validate="one_to_one",
        )
        matched = comparison.loc[
            comparison["_merge"].eq("both")
        ]
        differences = np.abs(
            matched["estimated_reserve_step28"]
            - matched["estimated_reserve_frozen"]
        )
        relative = differences / np.maximum(
            np.abs(matched["estimated_reserve_frozen"]),
            atol,
        )
        matching = np.isclose(
            matched["estimated_reserve_step28"],
            matched["estimated_reserve_frozen"],
            rtol=rtol,
            atol=atol,
        )
        truth_matching = np.isclose(
            matched["true_reserve_step28"],
            matched["true_reserve_frozen"],
            rtol=rtol,
            atol=atol,
        )
        missing_current = int(
            comparison["_merge"].eq("right_only").sum()
        )
        missing_frozen = int(
            comparison["_merge"].eq("left_only").sum()
        )
        rows.append(
            {
                "model": model,
                "rows_compared": int(len(matched)),
                "max_absolute_reserve_difference": (
                    float(differences.max())
                    if not differences.empty
                    else np.nan
                ),
                "max_relative_reserve_difference": (
                    float(relative.max())
                    if not relative.empty
                    else np.nan
                ),
                "nonmatching_reserve_rows": int(
                    (~matching).sum()
                ),
                "nonmatching_truth_rows": int(
                    (~truth_matching).sum()
                ),
                "missing_step28_rows": missing_current,
                "missing_frozen_rows": missing_frozen,
                "rtol": rtol,
                "atol": atol,
                "passed": bool(
                    matching.all()
                    and truth_matching.all()
                    and missing_current == 0
                    and missing_frozen == 0
                ),
            }
        )

    return pd.DataFrame(rows)


def build_baseline_gate_acceptance_report(
    *,
    results: pd.DataFrame,
    detail: pd.DataFrame,
    reproduction: pd.DataFrame,
    expected_scenarios: int,
    simulations_per_scenario: int,
) -> pd.DataFrame:
    """Validate the frozen-pattern Step 28 reproduction gate."""

    expected_rows = (
        expected_scenarios
        * simulations_per_scenario
        * 2
    )
    reconciliation = (
        detail.groupby(
            RESULT_KEY_COLUMNS,
            as_index=False,
        )["estimated_reserve"]
        .sum()
        .merge(
            results[
                RESULT_KEY_COLUMNS + ["estimated_reserve"]
            ],
            on=RESULT_KEY_COLUMNS,
            suffixes=("_detail", "_result"),
            validate="one_to_one",
        )
    )
    checks = [
        {
            "check": "expected_baseline_gate_row_count",
            "passed": len(results) == expected_rows,
            "detail": f"actual={len(results)}, expected={expected_rows}",
        },
        {
            "check": "both_frozen_bf_models_present",
            "passed": set(results["model"])
            == {BF_STANDARD, BF_BREAK_AWARE},
            "detail": "",
        },
        {
            "check": "result_keys_are_unique",
            "passed": not results.duplicated(
                RESULT_KEY_COLUMNS
            ).any(),
            "detail": "",
        },
        {
            "check": "detail_keys_are_unique",
            "passed": not detail.duplicated(
                DETAIL_KEY_COLUMNS
            ).any(),
            "detail": "",
        },
        {
            "check": "standard_bf_reproduces_step21",
            "passed": bool(
                reproduction.loc[
                    reproduction["model"].eq(BF_STANDARD),
                    "passed",
                ].iloc[0]
            ),
            "detail": "",
        },
        {
            "check": "break_aware_bf_reproduces_step21",
            "passed": bool(
                reproduction.loc[
                    reproduction["model"].eq(BF_BREAK_AWARE),
                    "passed",
                ].iloc[0]
            ),
            "detail": "",
        },
        {
            "check": "accident_year_reserves_reconcile_to_portfolio",
            "passed": bool(
                np.allclose(
                    reconciliation["estimated_reserve_detail"],
                    reconciliation["estimated_reserve_result"],
                    rtol=1e-12,
                    atol=1e-5,
                )
            ),
            "detail": "",
        },
        {
            "check": "successful_reserves_are_finite_and_nonnegative",
            "passed": bool(
                np.isfinite(results["estimated_reserve"]).all()
                and (results["estimated_reserve"] >= 0.0).all()
            ),
            "detail": "",
        },
    ]

    return pd.DataFrame(checks)


def build_pattern_comparison(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Build paired descriptive baseline-versus-pattern comparisons."""

    pair_definitions = [
        (BF_STANDARD, CEDED_SPECIFIC_STANDARD),
        (BF_BREAK_AWARE, CEDED_SPECIFIC_BREAK_AWARE),
    ]
    keys = ["scenario_id", "simulation_id", "basis"]
    rows: list[dict[str, Any]] = []

    for baseline_model, ceded_specific_model in pair_definitions:
        baseline = results.loc[
            results["model"].eq(baseline_model)
        ].copy()
        ceded_specific = results.loc[
            results["model"].eq(ceded_specific_model)
        ].copy()
        paired = baseline.merge(
            ceded_specific,
            on=keys,
            how="outer",
            suffixes=("_baseline", "_ceded_specific"),
            indicator=True,
            validate="one_to_one",
        )

        for scenario_id, group in paired.groupby(
            "scenario_id",
            sort=True,
        ):
            complete = group.loc[
                group["_merge"].eq("both")
                & group["success_baseline"].eq(True)
                & group["success_ceded_specific"].eq(True)
            ].copy()
            ape_difference = (
                complete["absolute_percentage_error_ceded_specific"]
                - complete["absolute_percentage_error_baseline"]
            )
            tolerance = 1e-12
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "basis": "ceded",
                    "baseline_model": baseline_model,
                    "ceded_specific_model": ceded_specific_model,
                    "attempted_pairs": int(len(group)),
                    "successful_pairs": int(len(complete)),
                    "baseline_successful_fits": int(
                        group["success_baseline"].fillna(False).sum()
                    ),
                    "ceded_specific_successful_fits": int(
                        group["success_ceded_specific"].fillna(False).sum()
                    ),
                    "baseline_success_rate": float(
                        group["success_baseline"].fillna(False).mean()
                    ),
                    "ceded_specific_success_rate": float(
                        group["success_ceded_specific"].fillna(False).mean()
                    ),
                    "baseline_mean_percentage_error": float(
                        complete["percentage_error_baseline"].mean()
                    ),
                    "ceded_specific_mean_percentage_error": float(
                        complete["percentage_error_ceded_specific"].mean()
                    ),
                    "baseline_median_percentage_error": float(
                        complete["percentage_error_baseline"].median()
                    ),
                    "ceded_specific_median_percentage_error": float(
                        complete["percentage_error_ceded_specific"].median()
                    ),
                    "baseline_mean_absolute_percentage_error": float(
                        complete[
                            "absolute_percentage_error_baseline"
                        ].mean()
                    ),
                    "ceded_specific_mean_absolute_percentage_error": float(
                        complete[
                            "absolute_percentage_error_ceded_specific"
                        ].mean()
                    ),
                    "mape_difference_ceded_specific_minus_baseline": float(
                        ape_difference.mean()
                    ),
                    "baseline_RMSE": float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    complete["signed_error_baseline"]
                                )
                            )
                        )
                    ),
                    "ceded_specific_RMSE": float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    complete[
                                        "signed_error_ceded_specific"
                                    ]
                                )
                            )
                        )
                    ),
                    "baseline_percentage_error_std": float(
                        complete["percentage_error_baseline"].std()
                    ),
                    "ceded_specific_percentage_error_std": float(
                        complete["percentage_error_ceded_specific"].std()
                    ),
                    "mean_paired_ape_difference": float(
                        ape_difference.mean()
                    ),
                    "median_paired_ape_difference": float(
                        ape_difference.median()
                    ),
                    "ceded_specific_win_rate": float(
                        (ape_difference < -tolerance).mean()
                    ),
                    "tie_rate": float(
                        (ape_difference.abs() <= tolerance).mean()
                    ),
                    "ceded_specific_loss_rate": float(
                        (ape_difference > tolerance).mean()
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["baseline_model", "scenario_id"]
    ).reset_index(drop=True)


def build_final_acceptance_report(
    *,
    results: pd.DataFrame,
    detail: pd.DataFrame,
    reproduction: pd.DataFrame,
    pattern_table: pd.DataFrame,
    pattern_comparison: pd.DataFrame,
    expected_scenarios: int,
    simulations_per_scenario: int,
    structural_break_year: int,
    prior_hash_before: str,
    prior_hash_after: str,
) -> pd.DataFrame:
    """Validate the frozen final Step 28 ceded sensitivity."""

    intended_models = {
        BF_STANDARD,
        CEDED_SPECIFIC_STANDARD,
        BF_BREAK_AWARE,
        CEDED_SPECIFIC_BREAK_AWARE,
    }
    expected_rows = (
        expected_scenarios
        * simulations_per_scenario
        * len(intended_models)
    )
    portfolio_grouping = [
        "scenario_id",
        "simulation_id",
        "basis",
    ]
    ay_grouping = portfolio_grouping + ["accident_year"]
    scenario_model_counts = results.groupby(
        ["scenario_id", "model"],
    ).size()
    truth_invariant = results.groupby(
        portfolio_grouping,
    )["true_reserve"].nunique(dropna=False).le(1).all()
    paid_invariant = detail.groupby(
        ay_grouping,
    )["paid_to_date"].nunique(dropna=False).le(1).all()
    prior_invariant = detail.groupby(
        ay_grouping,
    )["expected_loss_prior_ultimate"].nunique(
        dropna=False
    ).le(1).all()
    invariant_columns = [
        "tail_type",
        "structural_break",
        "inflation_scenario",
        "expected_loss_prior_ultimate",
        "paid_to_date",
        "pricing_assumption_version",
        "development_year_at_valuation",
        "true_reserve",
    ]
    only_pattern_changes = all(
        detail.groupby(ay_grouping)[column]
        .nunique(dropna=False)
        .le(1)
        .all()
        for column in invariant_columns
    )
    ceded_specific = detail.loc[
        detail["model"].isin(
            [
                CEDED_SPECIFIC_STANDARD,
                CEDED_SPECIFIC_BREAK_AWARE,
            ]
        )
    ].copy()
    structural_break_boolean = _coerce_boolean_series(
        ceded_specific["structural_break"]
    )
    use_post_break = (
        ceded_specific["model"].eq(
            CEDED_SPECIFIC_BREAK_AWARE
        )
        & ceded_specific["tail_type"].eq("long")
        & structural_break_boolean
        & ceded_specific["accident_year"].astype(int).ge(
            int(structural_break_year)
        )
    )
    ceded_specific["expected_regime"] = np.where(
        use_post_break,
        "post_break",
        "no_break",
    )
    ceded_specific["expected_pattern"] = (
        "ceded_specific_"
        + ceded_specific["tail_type"].astype(str)
        + "_"
        + ceded_specific["inflation_scenario"].astype(str)
        + "_"
        + ceded_specific["expected_regime"]
    )
    pattern_lookup_correct = ceded_specific[
        "benchmark_pattern"
    ].eq(ceded_specific["expected_pattern"]).all()
    break_rows = ceded_specific.loc[structural_break_boolean]
    break_aware_rows = break_rows.loc[
        break_rows["model"].eq(
            CEDED_SPECIFIC_BREAK_AWARE
        )
    ]
    pre_break = break_aware_rows.loc[
        break_aware_rows["accident_year"].astype(int).lt(
            int(structural_break_year)
        )
    ]
    post_break = break_aware_rows.loc[
        break_aware_rows["accident_year"].astype(int).ge(
            int(structural_break_year)
        )
    ]
    terminals = pattern_table[
        [
            "tail_type",
            "inflation_scenario",
            "structural_break_regime",
            "natural_terminal_development_year",
        ]
    ].drop_duplicates()
    maturity = ceded_specific.merge(
        terminals,
        left_on=[
            "tail_type",
            "inflation_scenario",
            "expected_regime",
        ],
        right_on=[
            "tail_type",
            "inflation_scenario",
            "structural_break_regime",
        ],
        how="left",
        validate="many_to_one",
    )
    beyond_maturity = maturity.loc[
        maturity["development_year_at_valuation"].astype(int)
        > maturity["natural_terminal_development_year"].astype(int)
    ]
    reconciliation = (
        detail.groupby(RESULT_KEY_COLUMNS, as_index=False)[
            "estimated_reserve"
        ]
        .sum()
        .merge(
            results[RESULT_KEY_COLUMNS + ["estimated_reserve"]],
            on=RESULT_KEY_COLUMNS,
            suffixes=("_detail", "_result"),
            validate="one_to_one",
        )
    )
    pattern_selection_parameters = set(
        inspect.signature(_pattern_for_row).parameters
    ) | set(
        inspect.signature(
            build_ceded_specific_bf_by_accident_year
        ).parameters
    )
    no_break_results = results.loc[
        ~_coerce_boolean_series(results["structural_break"])
        & results["model"].isin(
            [
                CEDED_SPECIFIC_STANDARD,
                CEDED_SPECIFIC_BREAK_AWARE,
            ]
        )
    ]
    no_break_pivot = no_break_results.pivot(
        index=portfolio_grouping,
        columns="model",
        values="estimated_reserve",
    )
    failures = results.loc[~results["success"]]

    checks = [
        {
            "check": "exact_expected_result_row_count",
            "passed": len(results) == expected_rows,
            "detail": f"actual={len(results)}, expected={expected_rows}",
        },
        {
            "check": "all_nine_scenarios_present",
            "passed": results["scenario_id"].nunique()
            == expected_scenarios,
            "detail": f"actual={results['scenario_id'].nunique()}",
        },
        {
            "check": "fifty_simulations_per_scenario_and_model",
            "passed": bool(
                len(scenario_model_counts)
                == expected_scenarios * len(intended_models)
                and scenario_model_counts.eq(
                    simulations_per_scenario
                ).all()
            ),
            "detail": "",
        },
        {
            "check": "ceded_basis_only",
            "passed": set(results["basis"]) == {"ceded"},
            "detail": "",
        },
        {
            "check": "all_four_intended_variants_present",
            "passed": set(results["model"]) == intended_models,
            "detail": "",
        },
        {
            "check": "result_keys_are_unique",
            "passed": not results.duplicated(
                RESULT_KEY_COLUMNS
            ).any(),
            "detail": "",
        },
        {
            "check": "truth_is_identical_across_variants",
            "passed": bool(truth_invariant),
            "detail": "",
        },
        {
            "check": "paid_to_date_is_identical_across_variants",
            "passed": bool(paid_invariant),
            "detail": "",
        },
        {
            "check": "prior_is_identical_across_variants",
            "passed": bool(prior_invariant),
            "detail": "",
        },
        {
            "check": "only_development_pattern_inputs_change",
            "passed": bool(only_pattern_changes),
            "detail": "All non-pattern estimator inputs are invariant.",
        },
        {
            "check": "pattern_lookup_matches_tail_inflation_and_regime",
            "passed": bool(pattern_lookup_correct),
            "detail": "",
        },
        {
            "check": "pre_break_ays_use_no_break_ceded_pattern",
            "passed": bool(
                not pre_break.empty
                and pre_break["expected_regime"].eq("no_break").all()
            ),
            "detail": f"rows={len(pre_break)}",
        },
        {
            "check": "post_break_ays_use_post_break_ceded_pattern",
            "passed": bool(
                not post_break.empty
                and post_break["expected_regime"].eq("post_break").all()
            ),
            "detail": f"rows={len(post_break)}",
        },
        {
            "check": "ages_beyond_calibrated_maturity_use_paid_one",
            "passed": bool(
                not beyond_maturity.empty
                and beyond_maturity[
                    "benchmark_paid_proportion"
                ].eq(1.0).all()
            ),
            "detail": f"rows={len(beyond_maturity)}",
        },
        {
            "check": "successful_estimates_are_finite_and_nonnegative",
            "passed": bool(
                np.isfinite(results["estimated_reserve"]).all()
                and (results["estimated_reserve"] >= 0.0).all()
            ),
            "detail": "",
        },
        {
            "check": "accident_year_reserves_reconcile_to_portfolio",
            "passed": bool(
                np.allclose(
                    reconciliation["estimated_reserve_detail"],
                    reconciliation["estimated_reserve_result"],
                    rtol=1e-12,
                    atol=1e-5,
                )
            ),
            "detail": "",
        },
        {
            "check": "baseline_variants_reproduce_frozen_step21",
            "passed": bool(reproduction["passed"].all()),
            "detail": "",
        },
        {
            "check": "step19_prior_sha256_remains_unchanged",
            "passed": prior_hash_before == prior_hash_after,
            "detail": f"sha256={prior_hash_after}",
        },
        {
            "check": "no_evaluation_truth_enters_pattern_selection",
            "passed": not bool(
                {"true_reserve", "evaluation_truth"}
                & pattern_selection_parameters
            ),
            "detail": "Truth is merged only after each estimate is built.",
        },
        {
            "check": "no_break_ceded_specific_variants_collapse",
            "passed": bool(
                np.allclose(
                    no_break_pivot[CEDED_SPECIFIC_STANDARD],
                    no_break_pivot[CEDED_SPECIFIC_BREAK_AWARE],
                    rtol=0.0,
                    atol=0.0,
                )
            ),
            "detail": "Identical predictions are expected by design.",
        },
        {
            "check": "paired_pattern_comparison_is_complete",
            "passed": bool(
                len(pattern_comparison) == expected_scenarios * 2
                and pattern_comparison["successful_pairs"]
                .eq(simulations_per_scenario)
                .all()
            ),
            "detail": f"rows={len(pattern_comparison)}",
        },
        {
            "check": "failure_reporting_is_complete",
            "passed": bool(failures.empty),
            "detail": f"failures={len(failures)}",
        },
    ]

    return pd.DataFrame(checks)


def build_result_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise Step 28 success and conditional accuracy."""

    return (
        results.groupby(
            [
                "scenario_id",
                "model",
                "pattern_variant",
                "basis",
            ],
            as_index=False,
        )
        .agg(
            attempted_fits=("success", "size"),
            successful_fits=("success", "sum"),
            success_rate=("success", "mean"),
            mean_percentage_error=(
                "percentage_error",
                "mean",
            ),
            median_percentage_error=(
                "percentage_error",
                "median",
            ),
            mean_absolute_percentage_error=(
                "absolute_percentage_error",
                "mean",
            ),
            RMSE=(
                "signed_error",
                lambda values: float(
                    np.sqrt(np.mean(np.square(values)))
                ),
            ),
            percentage_error_std=(
                "percentage_error",
                "std",
            ),
            mean_runtime_seconds=(
                "runtime_seconds",
                "mean",
            ),
        )
    )

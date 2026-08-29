"""Step 29 simplified XoL treaty-indexation sensitivity.

The frozen baseline reinsurance implementation remains unchanged. This
module assigns claim-specific treaty terms using the existing claims
inflation index and then reuses the cumulative-paid XoL helper.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    ACCIDENT_YEARS,
    CLAIMS_INFLATION_BASE_YEAR,
    CLAUSE_INDEX_BASE_YEAR,
    INFLATION_SCENARIOS,
    PILOT_XOL_ATTACHMENT,
    PILOT_XOL_LIMIT,
    VALUATION_YEAR,
)
from src.reinsurance import apply_xol_to_payments
from src.simulation import build_inflation_index


FIXED_NOMINAL = "fixed_nominal"
FULLY_INDEXED = "fully_indexed"
TREATY_VARIANTS = (FIXED_NOMINAL, FULLY_INDEXED)
INDEX_TIMING = "accident_year"
NUMERICAL_ATOL = 1e-6
NUMERICAL_RTOL = 1e-12

RESULT_KEY_COLUMNS = [
    "scenario_id",
    "simulation_id",
    "treaty_variant",
]

ACCIDENT_YEAR_KEY_COLUMNS = RESULT_KEY_COLUMNS + [
    "accident_year",
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    """Return a deterministic digest of a directory tree."""

    root = Path(path)

    if not root.is_dir():
        raise FileNotFoundError(root)

    digest = hashlib.sha256()

    for file_path in sorted(
        item for item in root.rglob("*") if item.is_file()
    ):
        relative = file_path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\n")

    return digest.hexdigest()


def build_treaty_term_schedule(
    *,
    inflation_scenario: str,
    scenario_id: str | None = None,
    accident_years: Sequence[int] = ACCIDENT_YEARS,
    base_attachment: float = PILOT_XOL_ATTACHMENT,
    base_limit: float = PILOT_XOL_LIMIT,
    index_reference_year: int = CLAUSE_INDEX_BASE_YEAR,
) -> pd.DataFrame:
    """Build fixed and accident-year-indexed treaty terms.

    Both attachment and limit use the same claims-inflation factor:

        term_i = base_term * I_i / I_reference.

    The accident-year terms are then fixed for the claim's lifetime.
    """

    if inflation_scenario not in INFLATION_SCENARIOS:
        raise ValueError(
            f"Unknown inflation scenario: {inflation_scenario}"
        )

    if not np.isfinite(base_attachment) or base_attachment <= 0.0:
        raise ValueError("base_attachment must be finite and positive.")

    if not np.isfinite(base_limit) or base_limit <= 0.0:
        raise ValueError("base_limit must be finite and positive.")

    inflation_table = build_inflation_index(
        INFLATION_SCENARIOS[inflation_scenario],
        base_year=CLAIMS_INFLATION_BASE_YEAR,
    )
    index_by_year = inflation_table.set_index("calendar_year")[
        "inflation_index"
    ].to_dict()

    if int(index_reference_year) not in index_by_year:
        raise ValueError(
            "index_reference_year is outside the inflation index."
        )

    reference_index = float(index_by_year[int(index_reference_year)])

    if not np.isfinite(reference_index) or reference_index <= 0.0:
        raise ValueError("Reference inflation index must be positive.")

    rows: list[dict[str, Any]] = []

    for accident_year in sorted(int(year) for year in accident_years):
        if accident_year not in index_by_year:
            raise ValueError(
                f"Accident year {accident_year} is outside the index."
            )

        inflation_index_value = float(index_by_year[accident_year])
        full_factor = inflation_index_value / reference_index

        for treaty_variant in TREATY_VARIANTS:
            applied_factor = (
                1.0 if treaty_variant == FIXED_NOMINAL else full_factor
            )
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "inflation_scenario": inflation_scenario,
                    "treaty_variant": treaty_variant,
                    "accident_year": accident_year,
                    "base_attachment": float(base_attachment),
                    "base_limit": float(base_limit),
                    "index_reference_year": int(index_reference_year),
                    "reference_inflation_index_value": reference_index,
                    "inflation_index_value": inflation_index_value,
                    "full_index_factor": full_factor,
                    "applied_index_factor": applied_factor,
                    "indexed_attachment": float(
                        base_attachment * applied_factor
                    ),
                    "indexed_limit": float(base_limit * applied_factor),
                    "index_timing": INDEX_TIMING,
                    "terms_fixed_for_claim_lifetime": True,
                }
            )

    schedule = pd.DataFrame(rows)
    key_columns = ["treaty_variant", "accident_year"]

    if schedule.duplicated(key_columns).any():
        raise ValueError("Treaty-term schedule contains duplicate keys.")

    if not np.isfinite(
        schedule[["indexed_attachment", "indexed_limit"]]
    ).all().all():
        raise ValueError("Indexed treaty terms must be finite.")

    if not (
        schedule[["indexed_attachment", "indexed_limit"]] > 0.0
    ).all().all():
        raise ValueError("Indexed treaty terms must be positive.")

    return schedule


def treaty_terms_for_accident_year(
    *,
    treaty_variant: str,
    accident_year: int,
    inflation_scenario: str,
    base_attachment: float = PILOT_XOL_ATTACHMENT,
    base_limit: float = PILOT_XOL_LIMIT,
    index_reference_year: int = CLAUSE_INDEX_BASE_YEAR,
) -> dict[str, Any]:
    """Return the pre-specified treaty terms for one claim cohort."""

    if treaty_variant not in TREATY_VARIANTS:
        raise ValueError(f"Unknown treaty variant: {treaty_variant}")

    schedule = build_treaty_term_schedule(
        inflation_scenario=inflation_scenario,
        accident_years=[int(accident_year)],
        base_attachment=base_attachment,
        base_limit=base_limit,
        index_reference_year=index_reference_year,
    )
    row = schedule.loc[
        schedule["treaty_variant"].eq(treaty_variant)
    ]

    if len(row) != 1:
        raise RuntimeError("Treaty terms are missing or duplicated.")

    return row.iloc[0].to_dict()


def _annotate_reinsurance_outputs(
    *,
    reinsured_payments: pd.DataFrame,
    claim_summary: pd.DataFrame,
    schedule_row: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    payment_output = reinsured_payments.copy()
    claim_output = claim_summary.copy()

    annotations = {
        "treaty_variant": schedule_row["treaty_variant"],
        "base_attachment": float(schedule_row["base_attachment"]),
        "base_limit": float(schedule_row["base_limit"]),
        "index_reference_year": int(
            schedule_row["index_reference_year"]
        ),
        "reference_inflation_index_value": float(
            schedule_row["reference_inflation_index_value"]
        ),
        "inflation_index_value": float(
            schedule_row["inflation_index_value"]
        ),
        "full_index_factor": float(schedule_row["full_index_factor"]),
        "applied_index_factor": float(
            schedule_row["applied_index_factor"]
        ),
        "indexed_attachment": float(
            schedule_row["indexed_attachment"]
        ),
        "indexed_limit": float(schedule_row["indexed_limit"]),
        "index_timing": INDEX_TIMING,
        "terms_fixed_for_claim_lifetime": True,
    }

    for column, value in annotations.items():
        payment_output[column] = value
        claim_output[column] = value

    payment_output["treaty_attachment"] = annotations[
        "indexed_attachment"
    ]
    payment_output["treaty_limit"] = annotations["indexed_limit"]
    claim_output["treaty_attachment"] = annotations[
        "indexed_attachment"
    ]
    claim_output["treaty_limit"] = annotations["indexed_limit"]

    if schedule_row["treaty_variant"] == FULLY_INDEXED:
        payment_output["clause_type"] = "full"
        claim_output["clause_type"] = "full"

    return payment_output, claim_output


def apply_treaty_variant_to_payments(
    *,
    payments: pd.DataFrame,
    treaty_variant: str,
    inflation_scenario: str,
    base_attachment: float = PILOT_XOL_ATTACHMENT,
    base_limit: float = PILOT_XOL_LIMIT,
    index_reference_year: int = CLAUSE_INDEX_BASE_YEAR,
    scenario_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply one fixed or fully indexed cumulative-paid XoL layer."""

    if treaty_variant not in TREATY_VARIANTS:
        raise ValueError(f"Unknown treaty variant: {treaty_variant}")

    if "accident_year" not in payments.columns:
        raise ValueError("payments must contain accident_year.")

    source = payments.copy(deep=True)
    schedule = build_treaty_term_schedule(
        inflation_scenario=inflation_scenario,
        scenario_id=scenario_id,
        accident_years=sorted(source["accident_year"].astype(int).unique()),
        base_attachment=base_attachment,
        base_limit=base_limit,
        index_reference_year=index_reference_year,
    )
    selected_schedule = schedule.loc[
        schedule["treaty_variant"].eq(treaty_variant)
    ].copy()
    payment_parts: list[pd.DataFrame] = []
    claim_parts: list[pd.DataFrame] = []

    if treaty_variant == FIXED_NOMINAL:
        baseline_payments, baseline_claims = apply_xol_to_payments(
            payments=source,
            attachment=base_attachment,
            limit=base_limit,
        )

        for accident_year, payment_group in baseline_payments.groupby(
            "accident_year",
            sort=True,
        ):
            schedule_row = selected_schedule.loc[
                selected_schedule["accident_year"].eq(int(accident_year))
            ].iloc[0]
            claim_group = baseline_claims.loc[
                baseline_claims["accident_year"].eq(int(accident_year))
            ]
            annotated_payments, annotated_claims = (
                _annotate_reinsurance_outputs(
                    reinsured_payments=payment_group,
                    claim_summary=claim_group,
                    schedule_row=schedule_row,
                )
            )
            payment_parts.append(annotated_payments)
            claim_parts.append(annotated_claims)
    else:
        for accident_year, payment_group in source.groupby(
            "accident_year",
            sort=True,
        ):
            schedule_row = selected_schedule.loc[
                selected_schedule["accident_year"].eq(int(accident_year))
            ].iloc[0]
            indexed_payments, indexed_claims = apply_xol_to_payments(
                payments=payment_group,
                attachment=float(schedule_row["indexed_attachment"]),
                limit=float(schedule_row["indexed_limit"]),
            )
            annotated_payments, annotated_claims = (
                _annotate_reinsurance_outputs(
                    reinsured_payments=indexed_payments,
                    claim_summary=indexed_claims,
                    schedule_row=schedule_row,
                )
            )
            payment_parts.append(annotated_payments)
            claim_parts.append(annotated_claims)

    reinsured = pd.concat(payment_parts, ignore_index=True).sort_values(
        ["simulation_id", "claim_id", "payment_calendar_year", "payment_sequence"]
    ).reset_index(drop=True)
    claims = pd.concat(claim_parts, ignore_index=True).sort_values(
        ["simulation_id", "claim_id"]
    ).reset_index(drop=True)
    claims["attachment_breached"] = claims[
        "nominal_ceded_ultimate"
    ].gt(0.0)
    claims["limit_exhausted"] = np.isclose(
        claims["nominal_ceded_ultimate"],
        claims["indexed_limit"],
        rtol=1e-10,
        atol=NUMERICAL_ATOL,
    )
    claims["ceded_share"] = np.where(
        claims["nominal_gross_ultimate"].gt(0.0),
        claims["nominal_ceded_ultimate"]
        / claims["nominal_gross_ultimate"],
        0.0,
    )

    validate_treaty_variant_outputs(
        source_payments=source,
        reinsured_payments=reinsured,
        claim_summary=claims,
        treaty_variant=treaty_variant,
    )

    return reinsured, claims, selected_schedule.reset_index(drop=True)


def validate_treaty_variant_outputs(
    *,
    source_payments: pd.DataFrame,
    reinsured_payments: pd.DataFrame,
    claim_summary: pd.DataFrame,
    treaty_variant: str,
) -> None:
    """Validate indexed XoL payment and claim accounting."""

    if treaty_variant not in TREATY_VARIANTS:
        raise ValueError("Unknown treaty variant.")

    source_total = float(source_payments["nominal_gross_payment"].sum())
    output_total = float(
        reinsured_payments["nominal_gross_payment"].sum()
    )

    if not np.isclose(
        source_total,
        output_total,
        rtol=NUMERICAL_RTOL,
        atol=NUMERICAL_ATOL,
    ):
        raise ValueError("Gross payment stream changed under reinsurance.")

    expected_gross = (
        reinsured_payments["nominal_ceded_payment"]
        + reinsured_payments["nominal_retained_payment"]
    )

    if not np.allclose(
        reinsured_payments["nominal_gross_payment"],
        expected_gross,
        rtol=NUMERICAL_RTOL,
        atol=NUMERICAL_ATOL,
    ):
        raise ValueError("Gross does not equal ceded plus retained.")

    if (reinsured_payments["nominal_ceded_payment"] < -NUMERICAL_ATOL).any():
        raise ValueError("A ceded payment is negative.")

    if (
        reinsured_payments["cumulative_nominal_ceded_amount"]
        > reinsured_payments["indexed_limit"] + NUMERICAL_ATOL
    ).any():
        raise ValueError("Cumulative ceded exceeds the applicable limit.")

    if (
        claim_summary["nominal_ceded_ultimate"]
        > claim_summary["indexed_limit"] + NUMERICAL_ATOL
    ).any():
        raise ValueError("Claim ceded ultimate exceeds its limit.")

    claim_totals = claim_summary[
        [
            "nominal_gross_ultimate",
            "nominal_ceded_ultimate",
            "nominal_retained_ultimate",
        ]
    ].sum()
    payment_totals = reinsured_payments[
        [
            "nominal_gross_payment",
            "nominal_ceded_payment",
            "nominal_retained_payment",
        ]
    ].sum()

    if not np.allclose(
        claim_totals.to_numpy(),
        payment_totals.to_numpy(),
        rtol=NUMERICAL_RTOL,
        atol=NUMERICAL_ATOL,
    ):
        raise ValueError("Claim and payment totals do not reconcile.")

    term_counts = reinsured_payments.groupby(
        ["simulation_id", "claim_id"]
    )[["indexed_attachment", "indexed_limit"]].nunique()

    if not term_counts.eq(1).all().all():
        raise ValueError("Treaty terms change during a claim's lifetime.")


def build_accident_year_summary(
    *,
    reinsured_payments: pd.DataFrame,
    claim_summary: pd.DataFrame,
    scenario: dict[str, Any],
    simulation_id: int,
    seed: int,
    treaty_variant: str,
    valuation_year: int = VALUATION_YEAR,
) -> pd.DataFrame:
    """Summarise treaty outcomes and truth by accident year."""

    payment_rows = []

    for accident_year, group in reinsured_payments.groupby(
        "accident_year",
        sort=True,
    ):
        observed = group["payment_calendar_year"].le(int(valuation_year))
        payment_rows.append(
            {
                "accident_year": int(accident_year),
                "gross_ultimate": float(
                    group["nominal_gross_payment"].sum()
                ),
                "ceded_ultimate": float(
                    group["nominal_ceded_payment"].sum()
                ),
                "retained_ultimate": float(
                    group["nominal_retained_payment"].sum()
                ),
                "gross_paid_to_date": float(
                    group.loc[observed, "nominal_gross_payment"].sum()
                ),
                "ceded_paid_to_date": float(
                    group.loc[observed, "nominal_ceded_payment"].sum()
                ),
                "retained_paid_to_date": float(
                    group.loc[observed, "nominal_retained_payment"].sum()
                ),
                "gross_true_reserve": float(
                    group.loc[~observed, "nominal_gross_payment"].sum()
                ),
                "ceded_true_reserve": float(
                    group.loc[~observed, "nominal_ceded_payment"].sum()
                ),
                "retained_true_reserve": float(
                    group.loc[~observed, "nominal_retained_payment"].sum()
                ),
                "indexed_attachment": float(
                    group["indexed_attachment"].iloc[0]
                ),
                "indexed_limit": float(group["indexed_limit"].iloc[0]),
                "inflation_index_value": float(
                    group["inflation_index_value"].iloc[0]
                ),
                "applied_index_factor": float(
                    group["applied_index_factor"].iloc[0]
                ),
            }
        )

    payments_by_year = pd.DataFrame(payment_rows)
    claims_by_year = (
        claim_summary.groupby("accident_year", as_index=False)
        .agg(
            claim_count=("claim_id", "nunique"),
            attaching_claims=("attachment_breached", "sum"),
            exhausting_claims=("limit_exhausted", "sum"),
        )
    )
    output = payments_by_year.merge(
        claims_by_year,
        on="accident_year",
        how="left",
        validate="one_to_one",
    )
    output.insert(0, "scenario_id", scenario["scenario_id"])
    output.insert(1, "simulation_id", int(simulation_id))
    output.insert(2, "seed", int(seed))
    output.insert(3, "treaty_variant", treaty_variant)
    output["tail_type"] = scenario["tail_type"]
    output["inflation_scenario"] = scenario["inflation_scenario"]
    output["structural_break"] = bool(
        scenario["apply_structural_break"]
    )
    output["base_attachment"] = float(PILOT_XOL_ATTACHMENT)
    output["base_limit"] = float(PILOT_XOL_LIMIT)
    output["index_reference_year"] = int(CLAUSE_INDEX_BASE_YEAR)
    output["ceded_share"] = np.where(
        output["gross_ultimate"].gt(0.0),
        output["ceded_ultimate"] / output["gross_ultimate"],
        0.0,
    )
    output["attachment_frequency"] = np.where(
        output["claim_count"].gt(0),
        output["attaching_claims"] / output["claim_count"],
        0.0,
    )
    output["exhaustion_frequency"] = np.where(
        output["claim_count"].gt(0),
        output["exhausting_claims"] / output["claim_count"],
        0.0,
    )
    output["ceded_unpaid_proportion"] = np.where(
        output["ceded_ultimate"].gt(0.0),
        output["ceded_true_reserve"] / output["ceded_ultimate"],
        0.0,
    )

    return output.sort_values("accident_year").reset_index(drop=True)


def build_portfolio_result(
    *,
    accident_year_summary: pd.DataFrame,
    claim_summary: pd.DataFrame,
    scenario: dict[str, Any],
    simulation_id: int,
    seed: int,
    treaty_variant: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Build one portfolio/treaty result row."""

    gross = float(accident_year_summary["gross_ultimate"].sum())
    ceded = float(accident_year_summary["ceded_ultimate"].sum())
    retained = float(accident_year_summary["retained_ultimate"].sum())
    attaching = int(claim_summary["attachment_breached"].sum())
    exhausting = int(claim_summary["limit_exhausted"].sum())
    claim_count = int(claim_summary["claim_id"].nunique())
    ceded_claims = int(
        claim_summary["nominal_ceded_ultimate"].gt(0.0).sum()
    )
    ceded_reserve = float(
        accident_year_summary["ceded_true_reserve"].sum()
    )

    return {
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
        "gross_ultimate": gross,
        "ceded_ultimate": ceded,
        "retained_ultimate": retained,
        "ceded_share": ceded / gross if gross > 0.0 else 0.0,
        "claim_count": claim_count,
        "attaching_claims": attaching,
        "attachment_frequency": (
            attaching / claim_count if claim_count else 0.0
        ),
        "exhausting_claims": exhausting,
        "exhaustion_frequency": (
            exhausting / claim_count if claim_count else 0.0
        ),
        "mean_ceded_per_ceded_claim": (
            ceded / ceded_claims if ceded_claims else 0.0
        ),
        "ceded_paid_to_date": float(
            accident_year_summary["ceded_paid_to_date"].sum()
        ),
        "ceded_true_reserve": ceded_reserve,
        "retained_true_reserve": float(
            accident_year_summary["retained_true_reserve"].sum()
        ),
        "gross_true_reserve": float(
            accident_year_summary["gross_true_reserve"].sum()
        ),
        "ceded_unpaid_proportion": (
            ceded_reserve / ceded if ceded > 0.0 else 0.0
        ),
        "success": True,
        "failure_message": "",
        "runtime_seconds": float(runtime_seconds),
    }


def build_development_summary(
    *,
    reinsured_payments: pd.DataFrame,
    scenario_id: str,
    simulation_id: int,
    treaty_variant: str,
) -> pd.DataFrame:
    """Summarise complete ceded payments by development age."""

    output = (
        reinsured_payments.groupby("development_year", as_index=False)
        .agg(
            incremental_gross_paid=("nominal_gross_payment", "sum"),
            incremental_ceded_paid=("nominal_ceded_payment", "sum"),
            incremental_retained_paid=("nominal_retained_payment", "sum"),
        )
        .sort_values("development_year")
        .reset_index(drop=True)
    )
    output["cumulative_ceded_paid"] = output[
        "incremental_ceded_paid"
    ].cumsum()
    total_ceded = float(output["incremental_ceded_paid"].sum())
    output["cumulative_ceded_proportion"] = (
        output["cumulative_ceded_paid"] / total_ceded
        if total_ceded > 0.0
        else 0.0
    )
    output.insert(0, "scenario_id", scenario_id)
    output.insert(1, "simulation_id", int(simulation_id))
    output.insert(2, "treaty_variant", treaty_variant)

    return output


def compare_fixed_nominal_to_baseline(
    *,
    fixed_payments: pd.DataFrame,
    fixed_claims: pd.DataFrame,
    baseline_payments: pd.DataFrame,
    baseline_claims: pd.DataFrame,
    scenario_id: str,
    simulation_id: int,
) -> dict[str, Any]:
    """Compare the Step 29 fixed wrapper with the frozen XoL helper."""

    payment_keys = ["simulation_id", "claim_id", "payment_sequence"]
    claim_keys = ["simulation_id", "claim_id"]
    payment_comparison = fixed_payments[
        payment_keys
        + [
            "nominal_gross_payment",
            "nominal_ceded_payment",
            "nominal_retained_payment",
        ]
    ].merge(
        baseline_payments[
            payment_keys
            + [
                "nominal_gross_payment",
                "nominal_ceded_payment",
                "nominal_retained_payment",
            ]
        ],
        on=payment_keys,
        how="outer",
        suffixes=("_step29", "_baseline"),
        indicator=True,
        validate="one_to_one",
    )
    claim_comparison = fixed_claims[
        claim_keys
        + [
            "nominal_gross_ultimate",
            "nominal_ceded_ultimate",
            "nominal_retained_ultimate",
            "attachment_breached",
            "limit_exhausted",
        ]
    ].merge(
        baseline_claims[
            claim_keys
            + [
                "nominal_gross_ultimate",
                "nominal_ceded_ultimate",
                "nominal_retained_ultimate",
                "attachment_breached",
                "limit_exhausted",
            ]
        ],
        on=claim_keys,
        how="outer",
        suffixes=("_step29", "_baseline"),
        indicator=True,
        validate="one_to_one",
    )
    matched_payments = payment_comparison.loc[
        payment_comparison["_merge"].eq("both")
    ]
    ceded_difference = (
        matched_payments["nominal_ceded_payment_step29"]
        - matched_payments["nominal_ceded_payment_baseline"]
    ).abs()
    ceded_relative = ceded_difference / np.maximum(
        matched_payments["nominal_ceded_payment_baseline"].abs(),
        NUMERICAL_ATOL,
    )
    payment_matching = np.isclose(
        matched_payments["nominal_ceded_payment_step29"],
        matched_payments["nominal_ceded_payment_baseline"],
        rtol=NUMERICAL_RTOL,
        atol=NUMERICAL_ATOL,
    )
    matched_claims = claim_comparison.loc[
        claim_comparison["_merge"].eq("both")
    ]
    claim_matching = np.isclose(
        matched_claims["nominal_ceded_ultimate_step29"],
        matched_claims["nominal_ceded_ultimate_baseline"],
        rtol=NUMERICAL_RTOL,
        atol=NUMERICAL_ATOL,
    )
    flags_matching = (
        matched_claims["attachment_breached_step29"].eq(
            matched_claims["attachment_breached_baseline"]
        )
        & matched_claims["limit_exhausted_step29"].eq(
            matched_claims["limit_exhausted_baseline"]
        )
    )

    return {
        "scenario_id": scenario_id,
        "simulation_id": int(simulation_id),
        "payment_rows_compared": int(len(matched_payments)),
        "claims_compared": int(len(matched_claims)),
        "max_absolute_ceded_payment_difference": float(
            ceded_difference.max() if not ceded_difference.empty else np.nan
        ),
        "max_relative_ceded_payment_difference": float(
            ceded_relative.max() if not ceded_relative.empty else np.nan
        ),
        "nonmatching_payment_rows": int((~payment_matching).sum()),
        "nonmatching_claim_rows": int((~claim_matching).sum()),
        "nonmatching_attachment_or_exhaustion_flags": int(
            (~flags_matching).sum()
        ),
        "missing_step29_payment_rows": int(
            payment_comparison["_merge"].eq("right_only").sum()
        ),
        "missing_baseline_payment_rows": int(
            payment_comparison["_merge"].eq("left_only").sum()
        ),
        "missing_step29_claim_rows": int(
            claim_comparison["_merge"].eq("right_only").sum()
        ),
        "missing_baseline_claim_rows": int(
            claim_comparison["_merge"].eq("left_only").sum()
        ),
        "passed": bool(
            payment_matching.all()
            and claim_matching.all()
            and flags_matching.all()
            and payment_comparison["_merge"].eq("both").all()
            and claim_comparison["_merge"].eq("both").all()
        ),
    }


def build_comparison_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise fully indexed minus fixed nominal by scenario."""

    metrics = [
        "gross_ultimate",
        "ceded_ultimate",
        "retained_ultimate",
        "ceded_share",
        "attaching_claims",
        "attachment_frequency",
        "exhausting_claims",
        "exhaustion_frequency",
        "ceded_paid_to_date",
        "ceded_true_reserve",
        "retained_true_reserve",
        "gross_true_reserve",
        "ceded_unpaid_proportion",
    ]
    rows: list[dict[str, Any]] = []

    for scenario_id, scenario_results in results.groupby(
        "scenario_id",
        sort=True,
    ):
        successful = scenario_results.loc[scenario_results["success"]]

        for metric in metrics:
            pivot = successful.pivot(
                index="simulation_id",
                columns="treaty_variant",
                values=metric,
            ).dropna()
            fixed_mean = float(pivot[FIXED_NOMINAL].mean())
            indexed_mean = float(pivot[FULLY_INDEXED].mean())
            difference = indexed_mean - fixed_mean
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "metric": metric,
                    "paired_simulations": int(len(pivot)),
                    "fixed_nominal_mean": fixed_mean,
                    "fully_indexed_mean": indexed_mean,
                    "difference_fully_indexed_minus_fixed": difference,
                    "percentage_difference_relative_to_fixed": (
                        100.0 * difference / fixed_mean
                        if abs(fixed_mean) > NUMERICAL_ATOL
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_scenario_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise successful treaty-mechanics results by scenario/variant."""

    metrics = [
        "gross_ultimate",
        "ceded_ultimate",
        "retained_ultimate",
        "ceded_share",
        "attaching_claims",
        "attachment_frequency",
        "exhausting_claims",
        "exhaustion_frequency",
        "ceded_true_reserve",
        "retained_true_reserve",
        "gross_true_reserve",
        "ceded_paid_to_date",
        "ceded_unpaid_proportion",
    ]
    rows: list[dict[str, Any]] = []

    for keys, group in results.groupby(
        ["scenario_id", "treaty_variant"],
        sort=True,
    ):
        scenario_id, treaty_variant = keys
        successful = group.loc[group["success"]]
        row: dict[str, Any] = {
            "scenario_id": scenario_id,
            "treaty_variant": treaty_variant,
            "attempted_simulations": int(len(group)),
            "successful_simulations": int(len(successful)),
            "success_rate": float(len(successful) / len(group)),
        }
        row.update(
            {
                f"mean_{metric}": float(successful[metric].mean())
                if not successful.empty
                else np.nan
                for metric in metrics
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def build_paired_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise paired indexed-minus-fixed differences by scenario."""

    rows: list[dict[str, Any]] = []

    for scenario_id, scenario_results in results.groupby(
        "scenario_id",
        sort=True,
    ):
        successful = scenario_results.loc[scenario_results["success"]]
        ultimate = successful.pivot(
            index="simulation_id",
            columns="treaty_variant",
            values="ceded_ultimate",
        ).dropna(subset=[FIXED_NOMINAL, FULLY_INDEXED])
        reserve = successful.pivot(
            index="simulation_id",
            columns="treaty_variant",
            values="ceded_true_reserve",
        ).dropna(subset=[FIXED_NOMINAL, FULLY_INDEXED])
        paired_ids = ultimate.index.intersection(reserve.index)
        ultimate = ultimate.loc[paired_ids]
        reserve = reserve.loc[paired_ids]
        ultimate_difference = (
            ultimate[FULLY_INDEXED] - ultimate[FIXED_NOMINAL]
        )
        reserve_difference = (
            reserve[FULLY_INDEXED] - reserve[FIXED_NOMINAL]
        )
        ultimate_percentage = 100.0 * ultimate_difference.div(
            ultimate[FIXED_NOMINAL].replace(0.0, np.nan)
        )
        reserve_percentage = 100.0 * reserve_difference.div(
            reserve[FIXED_NOMINAL].replace(0.0, np.nan)
        )
        rows.append(
            {
                "scenario_id": scenario_id,
                "paired_simulations": int(len(paired_ids)),
                "mean_ceded_ultimate_difference": float(
                    ultimate_difference.mean()
                ),
                "median_ceded_ultimate_difference": float(
                    ultimate_difference.median()
                ),
                "mean_percentage_ceded_ultimate_difference": float(
                    ultimate_percentage.mean()
                ),
                "mean_ceded_reserve_difference": float(
                    reserve_difference.mean()
                ),
                "median_ceded_reserve_difference": float(
                    reserve_difference.median()
                ),
                "mean_percentage_ceded_reserve_difference": float(
                    reserve_percentage.mean()
                ),
                "proportion_indexed_ceded_ultimate_lower": float(
                    ultimate_difference.lt(0.0).mean()
                ),
                "proportion_indexed_ceded_reserve_lower": float(
                    reserve_difference.lt(0.0).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def calibration_api_uses_only_approved_index_variables() -> bool:
    """Check that treaty-term selection has no truth/payment arguments."""

    parameters = set(
        inspect.signature(build_treaty_term_schedule).parameters
    ) | set(
        inspect.signature(treaty_terms_for_accident_year).parameters
    )
    forbidden = {
        "true_reserve",
        "evaluation_truth",
        "payment_calendar_year",
        "report_year",
        "paid_to_date",
    }

    return not bool(parameters & forbidden)

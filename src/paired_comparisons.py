"""Analysis-only paired comparisons over frozen dissertation results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


BOOTSTRAP_SEED = 20260831
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
PAIRING_ATOL = 1e-6
PAIRING_RTOL = 1e-12
APE_TIE_TOLERANCE = 1e-12
NOT_APPLICABLE_BY_DESIGN = "not_applicable_by_design"


class PairingValidationError(ValueError):
    """Raised when frozen rows cannot be paired without ambiguity."""


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest without modifying the source file."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def validate_unique_keys(
    results: pd.DataFrame,
    key_columns: Sequence[str],
    *,
    source_name: str,
) -> None:
    """Require one frozen result row per declared experiment key."""

    missing = [column for column in key_columns if column not in results]

    if missing:
        raise PairingValidationError(
            f"{source_name} is missing key columns: {missing}."
        )

    duplicated = results.duplicated(list(key_columns), keep=False)

    if duplicated.any():
        example = results.loc[duplicated, list(key_columns)].iloc[0].to_dict()
        raise PairingValidationError(
            f"{source_name} contains duplicate keys; example={example}."
        )


def load_frozen_results(
    path: Path,
    *,
    key_columns: Sequence[str],
    required_columns: Iterable[str] = (),
    source_name: str | None = None,
) -> pd.DataFrame:
    """Load and structurally validate one frozen experiment table."""

    source_path = Path(path)

    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    results = pd.read_csv(source_path)
    missing = [
        column
        for column in required_columns
        if column not in results.columns
    ]

    if missing:
        raise PairingValidationError(
            f"{source_name or source_path} is missing columns: {missing}."
        )

    validate_unique_keys(
        results,
        key_columns,
        source_name=source_name or str(source_path),
    )
    return results


def _as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def success_mask(results: pd.DataFrame) -> pd.Series:
    """Normalise the project success conventions to one boolean mask."""

    if "success" in results.columns:
        return _as_boolean(results["success"])

    if "success_or_failure" in results.columns:
        return results["success_or_failure"].astype(str).eq("success")

    raise PairingValidationError("Results have no success-status column.")


def applicability_mask(results: pd.DataFrame) -> pd.Series:
    """Return structural applicability separately from fit success."""

    if "applicability_status" not in results.columns:
        return pd.Series(True, index=results.index, dtype=bool)

    return ~results["applicability_status"].astype(str).eq(
        NOT_APPLICABLE_BY_DESIGN
    )


def fit_attempted_mask(results: pd.DataFrame) -> pd.Series:
    """Identify estimator attempts without counting designed skips."""

    if "fit_attempted" not in results.columns:
        return pd.Series(True, index=results.index, dtype=bool)

    return _as_boolean(results["fit_attempted"])


def _failure_detail(results: pd.DataFrame) -> pd.Series:
    candidates = [
        "failure_type",
        "failure_category",
        "failure_reason",
        "failure_message",
    ]
    available = [column for column in candidates if column in results]

    if not available:
        return pd.Series("", index=results.index, dtype=object)

    detail = results[available[0]].fillna("").astype(str)

    for column in available[1:]:
        value = results[column].fillna("").astype(str)
        detail = detail.where(detail.str.len().gt(0), value)

    return detail


def _normalise_estimator_side(
    results: pd.DataFrame,
    *,
    pair_keys: Sequence[str],
    truth_column: str,
    ape_column: str,
    passthrough_columns: Sequence[str],
) -> pd.DataFrame:
    required = list(pair_keys) + [truth_column, ape_column]
    missing = [column for column in required if column not in results]

    if missing:
        raise PairingValidationError(f"Estimator rows are missing: {missing}.")

    side = results[list(pair_keys)].copy()
    side["truth"] = pd.to_numeric(results[truth_column], errors="coerce")
    side["ape"] = pd.to_numeric(results[ape_column], errors="coerce")
    side["success"] = success_mask(results).to_numpy()
    side["applicable"] = applicability_mask(results).to_numpy()
    side["fit_attempted"] = fit_attempted_mask(results).to_numpy()
    side["failure_detail"] = _failure_detail(results).to_numpy()

    for column in passthrough_columns:
        if column not in results:
            raise PairingValidationError(
                f"Estimator rows are missing passthrough column {column}."
            )
        side[column] = results[column].to_numpy()

    return side


def _exclusion_reason(row: pd.Series) -> str:
    if not bool(row["present_a"]) and not bool(row["present_b"]):
        return "both_rows_missing"
    if not bool(row["present_a"]):
        return "model_a_row_missing"
    if not bool(row["present_b"]):
        return "model_b_row_missing"
    if not bool(row["applicable_a"]) and not bool(row["applicable_b"]):
        return "both_structurally_not_applicable"
    if not bool(row["applicable_a"]):
        return "model_a_structurally_not_applicable"
    if not bool(row["applicable_b"]):
        return "model_b_structurally_not_applicable"
    if not bool(row["success_a"]) and not bool(row["success_b"]):
        return "both_estimators_failed"
    if not bool(row["success_a"]):
        return "model_a_failed"
    if not bool(row["success_b"]):
        return "model_b_failed"
    if not np.isfinite(row["ape_a"]):
        return "model_a_ape_nonfinite"
    if not np.isfinite(row["ape_b"]):
        return "model_b_ape_nonfinite"
    return "included"


def construct_successful_pairs(
    results: pd.DataFrame,
    *,
    selector_column: str,
    selector_a: Any,
    selector_b: Any,
    pair_keys: Sequence[str],
    truth_column: str = "true_reserve",
    ape_column: str = "absolute_percentage_error",
    passthrough_columns: Sequence[str] = (),
    truth_atol: float = PAIRING_ATOL,
    truth_rtol: float = PAIRING_RTOL,
) -> pd.DataFrame:
    """Construct all required pairs and flag the accuracy-eligible subset."""

    if selector_column not in results:
        raise PairingValidationError(
            f"Results are missing selector column {selector_column}."
        )

    rows_a = results.loc[results[selector_column].eq(selector_a)].copy()
    rows_b = results.loc[results[selector_column].eq(selector_b)].copy()
    validate_unique_keys(rows_a, pair_keys, source_name=f"selector A={selector_a}")
    validate_unique_keys(rows_b, pair_keys, source_name=f"selector B={selector_b}")
    side_a = _normalise_estimator_side(
        rows_a,
        pair_keys=pair_keys,
        truth_column=truth_column,
        ape_column=ape_column,
        passthrough_columns=passthrough_columns,
    )
    side_b = _normalise_estimator_side(
        rows_b,
        pair_keys=pair_keys,
        truth_column=truth_column,
        ape_column=ape_column,
        passthrough_columns=passthrough_columns,
    )
    paired = side_a.merge(
        side_b,
        on=list(pair_keys),
        how="outer",
        suffixes=("_a", "_b"),
        indicator=True,
        validate="one_to_one",
    )
    paired["present_a"] = paired["_merge"].isin(["left_only", "both"])
    paired["present_b"] = paired["_merge"].isin(["right_only", "both"])
    both_present = paired["present_a"] & paired["present_b"]
    paired["truth_absolute_difference"] = np.where(
        both_present,
        np.abs(paired["truth_a"] - paired["truth_b"]),
        np.nan,
    )
    paired["truth_matches"] = False
    paired.loc[both_present, "truth_matches"] = np.isclose(
        paired.loc[both_present, "truth_a"],
        paired.loc[both_present, "truth_b"],
        rtol=truth_rtol,
        atol=truth_atol,
    )

    if not paired.loc[both_present, "truth_matches"].all():
        maximum = paired.loc[
            both_present & ~paired["truth_matches"],
            "truth_absolute_difference",
        ].max()
        raise PairingValidationError(
            "Paired evaluation truth differs; "
            f"maximum absolute difference={maximum}."
        )

    for column in passthrough_columns:
        mismatch = both_present & (
            paired[f"{column}_a"].astype(str)
            != paired[f"{column}_b"].astype(str)
        )
        if mismatch.any():
            raise PairingValidationError(
                f"Paired passthrough field {column} differs."
            )
        paired[column] = paired[f"{column}_a"].where(
            paired["present_a"], paired[f"{column}_b"]
        )

    paired["exclusion_reason"] = paired.apply(_exclusion_reason, axis=1)
    paired["included_in_accuracy"] = paired["exclusion_reason"].eq(
        "included"
    )
    paired["paired_ape_difference"] = np.where(
        paired["included_in_accuracy"],
        paired["ape_a"] - paired["ape_b"],
        np.nan,
    )
    paired["pair_outcome"] = "excluded"
    included = paired["included_in_accuracy"]
    difference = paired["paired_ape_difference"]
    paired.loc[
        included & difference.lt(-APE_TIE_TOLERANCE),
        "pair_outcome",
    ] = "model_a_win"
    paired.loc[
        included & difference.gt(APE_TIE_TOLERANCE),
        "pair_outcome",
    ] = "model_b_win"
    paired.loc[
        included & difference.abs().le(APE_TIE_TOLERANCE),
        "pair_outcome",
    ] = "tie"
    return paired.drop(columns="_merge")


def deterministic_bootstrap_mean_ci(
    differences: Sequence[float] | np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """Bootstrap the paired differences directly with a fixed seed."""

    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan
    if resamples < 1:
        raise ValueError("resamples must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one.")

    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        values.size,
        size=(resamples, values.size),
    )
    bootstrap_means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(
        bootstrap_means,
        [alpha, 1.0 - alpha],
    )
    return float(lower), float(upper)


def seed_for_group(base_seed: int, group_values: Sequence[Any]) -> int:
    """Derive a stable group-specific seed independent of Python hashing."""

    encoded = json.dumps(
        [str(value) for value in group_values],
        separators=(",", ":"),
    ).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")
    return int((base_seed + offset) % np.iinfo(np.uint32).max)


def _empty_accuracy_statistics() -> dict[str, float]:
    return {
        "mean_ape_a": np.nan,
        "mean_ape_b": np.nan,
        "mean_paired_ape_difference": np.nan,
        "median_paired_ape_difference": np.nan,
        "std_paired_ape_difference": np.nan,
        "win_rate_a": np.nan,
        "win_rate_b": np.nan,
        "bootstrap_ci_lower": np.nan,
        "bootstrap_ci_upper": np.nan,
    }


def summarise_paired_accuracy(
    pairs: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    tie_tolerance: float = APE_TIE_TOLERANCE,
) -> pd.DataFrame:
    """Summarise conditional paired accuracy without dropping failures."""

    rows: list[dict[str, Any]] = []

    for values, group in pairs.groupby(
        list(group_columns),
        sort=True,
        dropna=False,
    ):
        if not isinstance(values, tuple):
            values = (values,)
        information = dict(zip(group_columns, values))
        valid = group.loc[group["included_in_accuracy"]].copy()
        excluded_counts = (
            group.loc[~group["included_in_accuracy"], "exclusion_reason"]
            .value_counts()
            .sort_index()
            .to_dict()
        )
        row: dict[str, Any] = {
            **information,
            "required_paired_rows": int(len(group)),
            "valid_paired_rows": int(len(valid)),
            "excluded_pairs": int(len(group) - len(valid)),
            "exclusion_reason_counts": json.dumps(
                excluded_counts,
                sort_keys=True,
            ),
            "maximum_absolute_truth_difference": float(
                group["truth_absolute_difference"].max()
            )
            if group["truth_absolute_difference"].notna().any()
            else np.nan,
            "win_count_a": 0,
            "win_count_b": 0,
            "tie_count": 0,
            "bootstrap_resamples": int(bootstrap_resamples),
            "bootstrap_seed": int(
                seed_for_group(bootstrap_seed, values)
            ),
            "tie_tolerance": float(tie_tolerance),
        }
        row.update(_empty_accuracy_statistics())

        if not valid.empty:
            differences = valid["paired_ape_difference"].to_numpy(dtype=float)
            win_a = int(np.sum(differences < -tie_tolerance))
            win_b = int(np.sum(differences > tie_tolerance))
            ties = int(len(differences) - win_a - win_b)
            lower, upper = deterministic_bootstrap_mean_ci(
                differences,
                resamples=bootstrap_resamples,
                seed=row["bootstrap_seed"],
            )
            row.update(
                {
                    "mean_ape_a": float(valid["ape_a"].mean()),
                    "mean_ape_b": float(valid["ape_b"].mean()),
                    "mean_paired_ape_difference": float(
                        np.mean(differences)
                    ),
                    "median_paired_ape_difference": float(
                        np.median(differences)
                    ),
                    "std_paired_ape_difference": float(
                        np.std(differences, ddof=1)
                    )
                    if len(differences) > 1
                    else 0.0,
                    "win_count_a": win_a,
                    "win_count_b": win_b,
                    "tie_count": ties,
                    "win_rate_a": float(win_a / len(valid)),
                    "win_rate_b": float(win_b / len(valid)),
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def summarise_applicability(
    pairs: pd.DataFrame,
    *,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Report attempts and applicability alongside conditional accuracy."""

    rows: list[dict[str, Any]] = []

    for values, group in pairs.groupby(
        list(group_columns),
        sort=True,
        dropna=False,
    ):
        if not isinstance(values, tuple):
            values = (values,)
        attempted_a = int(
            (group["present_a"] & group["fit_attempted_a"].fillna(False)).sum()
        )
        attempted_b = int(
            (group["present_b"] & group["fit_attempted_b"].fillna(False)).sum()
        )
        successful_a = int(
            (group["present_a"] & group["success_a"].fillna(False)).sum()
        )
        successful_b = int(
            (group["present_b"] & group["success_b"].fillna(False)).sum()
        )
        applicable_a = group["applicable_a"].fillna(False)
        applicable_b = group["applicable_b"].fillna(False)
        structurally_not_applicable = ~(applicable_a & applicable_b)
        applicable_pairs = (
            group["present_a"]
            & group["present_b"]
            & applicable_a
            & applicable_b
        )
        either_failed = applicable_pairs & ~(
            group["success_a"].fillna(False)
            & group["success_b"].fillna(False)
        )
        rows.append(
            {
                **dict(zip(group_columns, values)),
                "required_pairs": int(len(group)),
                "rows_present_a": int(group["present_a"].sum()),
                "rows_present_b": int(group["present_b"].sum()),
                "attempts_a": attempted_a,
                "successful_fits_a": successful_a,
                "success_rate_a": float(successful_a / attempted_a)
                if attempted_a
                else np.nan,
                "attempts_b": attempted_b,
                "successful_fits_b": successful_b,
                "success_rate_b": float(successful_b / attempted_b)
                if attempted_b
                else np.nan,
                "unconditional_success_rate_a": float(
                    successful_a / len(group)
                ),
                "unconditional_success_rate_b": float(
                    successful_b / len(group)
                ),
                "both_success_pair_count": int(
                    group["included_in_accuracy"].sum()
                ),
                "either_failed_pair_count": int(either_failed.sum()),
                "structurally_not_applicable_count": int(
                    structurally_not_applicable.sum()
                ),
                "not_applicable_a": int((~applicable_a).sum()),
                "not_applicable_b": int((~applicable_b).sum()),
            }
        )

    return pd.DataFrame(rows)


def construct_treaty_mechanics_pairs(
    results: pd.DataFrame,
    *,
    variant_a: str = "fully_indexed",
    variant_b: str = "fixed_nominal",
    pair_keys: Sequence[str] = ("scenario_id", "simulation_id"),
) -> pd.DataFrame:
    """Compare treaty mechanics on paired gross portfolios without APE."""

    metrics = [
        "ceded_ultimate",
        "ceded_true_reserve",
        "ceded_share",
        "attachment_frequency",
        "exhaustion_frequency",
    ]
    required = list(pair_keys) + [
        "treaty_variant",
        "gross_ultimate",
        "success",
        *metrics,
    ]
    missing = [column for column in required if column not in results]

    if missing:
        raise PairingValidationError(
            f"Treaty results are missing columns: {missing}."
        )

    rows_a = results.loc[results["treaty_variant"].eq(variant_a)].copy()
    rows_b = results.loc[results["treaty_variant"].eq(variant_b)].copy()
    validate_unique_keys(rows_a, pair_keys, source_name=f"treaty A={variant_a}")
    validate_unique_keys(rows_b, pair_keys, source_name=f"treaty B={variant_b}")
    columns = list(pair_keys) + ["gross_ultimate", "success", *metrics]
    paired = rows_a[columns].merge(
        rows_b[columns],
        on=list(pair_keys),
        how="outer",
        suffixes=("_a", "_b"),
        indicator=True,
        validate="one_to_one",
    )
    paired["present_a"] = paired["_merge"].isin(["left_only", "both"])
    paired["present_b"] = paired["_merge"].isin(["right_only", "both"])
    paired["success_a"] = _as_boolean(paired["success_a"])
    paired["success_b"] = _as_boolean(paired["success_b"])
    both = paired["present_a"] & paired["present_b"]
    paired["gross_ultimate_absolute_difference"] = np.where(
        both,
        np.abs(paired["gross_ultimate_a"] - paired["gross_ultimate_b"]),
        np.nan,
    )
    gross_matches = np.isclose(
        paired.loc[both, "gross_ultimate_a"],
        paired.loc[both, "gross_ultimate_b"],
        rtol=PAIRING_RTOL,
        atol=PAIRING_ATOL,
    )

    if not gross_matches.all():
        raise PairingValidationError("Paired treaty gross ultimate differs.")

    paired["included_in_mechanics"] = (
        both & paired["success_a"] & paired["success_b"]
    )
    paired["exclusion_reason"] = "included"
    paired.loc[~paired["present_a"], "exclusion_reason"] = "variant_a_missing"
    paired.loc[~paired["present_b"], "exclusion_reason"] = "variant_b_missing"
    paired.loc[
        both & ~paired["success_a"] & ~paired["success_b"],
        "exclusion_reason",
    ] = "both_variants_failed"
    paired.loc[
        both & ~paired["success_a"] & paired["success_b"],
        "exclusion_reason",
    ] = "variant_a_failed"
    paired.loc[
        both & paired["success_a"] & ~paired["success_b"],
        "exclusion_reason",
    ] = "variant_b_failed"

    for metric in metrics:
        paired[f"{metric}_difference_a_minus_b"] = np.where(
            paired["included_in_mechanics"],
            paired[f"{metric}_a"] - paired[f"{metric}_b"],
            np.nan,
        )

    paired["variant_a"] = variant_a
    paired["variant_b"] = variant_b
    return paired.drop(columns="_merge")


def summarise_treaty_mechanics(
    pairs: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("scenario_id",),
) -> pd.DataFrame:
    """Summarise raw indexed-minus-fixed paired treaty outcomes."""

    difference_columns = [
        column
        for column in pairs.columns
        if column.endswith("_difference_a_minus_b")
    ]
    rows: list[dict[str, Any]] = []

    for values, group in pairs.groupby(list(group_columns), sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        valid = group.loc[group["included_in_mechanics"]]
        row: dict[str, Any] = {
            **dict(zip(group_columns, values)),
            "required_pairs": int(len(group)),
            "valid_pairs": int(len(valid)),
            "excluded_pairs": int(len(group) - len(valid)),
            "maximum_absolute_gross_ultimate_difference": float(
                group["gross_ultimate_absolute_difference"].max()
            ),
        }

        for column in difference_columns:
            row[f"mean_{column}"] = (
                float(valid[column].mean()) if not valid.empty else np.nan
            )
            row[f"median_{column}"] = (
                float(valid[column].median()) if not valid.empty else np.nan
            )

        row["proportion_indexed_ceded_ultimate_lower"] = (
            float(
                valid["ceded_ultimate_difference_a_minus_b"].lt(0.0).mean()
            )
            if not valid.empty
            else np.nan
        )
        row["proportion_indexed_ceded_reserve_lower"] = (
            float(
                valid["ceded_true_reserve_difference_a_minus_b"]
                .lt(0.0)
                .mean()
            )
            if not valid.empty
            else np.nan
        )
        rows.append(row)

    return pd.DataFrame(rows)

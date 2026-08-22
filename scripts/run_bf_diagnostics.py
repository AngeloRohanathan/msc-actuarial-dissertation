"""Run Step 22 BF diagnostic analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BF_STRUCTURAL_BREAK_ACCIDENT_YEAR,
)

from src.bf_diagnostics import (
    BF_BREAK_AWARE,
    BF_STANDARD,
    add_bf_error_decomposition,
    build_decomposition_summary,
    build_pattern_adequacy_summary,
    build_prior_adequacy_summary,
)


INPUT_PATH = Path(
    "outputs/step21_paid_bf/"
    "final_50/"
    "bf_by_accident_year.csv"
)

OUTPUT_DIR = Path(
    "outputs/step22_bf_diagnostics"
)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing Step 21 input: {INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = pd.read_csv(
        INPUT_PATH
    )

    source = source.loc[
        source["model"].isin(
            [
                BF_STANDARD,
                BF_BREAK_AWARE,
            ]
        )
    ].copy()

    diagnostics = add_bf_error_decomposition(
        source,
        structural_break_year=(
            BF_STRUCTURAL_BREAK_ACCIDENT_YEAR
        ),
    )

    prior_summary = (
        build_prior_adequacy_summary(
            diagnostics
        )
    )

    pattern_summary = (
        build_pattern_adequacy_summary(
            diagnostics
        )
    )

    decomposition_summary = (
        build_decomposition_summary(
            diagnostics
        )
    )

    # ---------------------------------------------------------
    # Acceptance checks
    # ---------------------------------------------------------

    residual_tolerance = 1e-5

    max_abs_residual = float(
        diagnostics[
            "decomposition_residual"
        ].abs().max()
    )

    true_paid_valid = bool(
        diagnostics.loc[
            diagnostics[
                "true_ultimate"
            ]
            > 0,
            "true_paid_proportion",
        ]
        .between(
            0.0,
            1.0,
            inclusive="both",
        )
        .all()
    )

    true_unpaid_valid = bool(
        diagnostics.loc[
            diagnostics[
                "true_ultimate"
            ]
            > 0,
            "true_unpaid_proportion",
        ]
        .between(
            0.0,
            1.0,
            inclusive="both",
        )
        .all()
    )

    bf_models_complete = (
        diagnostics[
            "model"
        ].nunique()
        == 2
    )

    expected_models = {
        BF_STANDARD,
        BF_BREAK_AWARE,
    }

    models_correct = (
        set(
            diagnostics[
                "model"
            ].unique()
        )
        == expected_models
    )

    # Both BF variants should use exactly
    # the same realised truth.
    truth_key = [
        "scenario_id",
        "simulation_id",
        "accident_year",
        "basis",
    ]

    standard_truth = (
        diagnostics.loc[
            diagnostics["model"]
            == BF_STANDARD,
            truth_key
            + [
                "paid_to_date",
                "true_reserve",
                "true_ultimate",
            ],
        ]
        .rename(
            columns={
                "paid_to_date":
                    "standard_paid",
                "true_reserve":
                    "standard_true_reserve",
                "true_ultimate":
                    "standard_true_ultimate",
            }
        )
    )

    aware_truth = (
        diagnostics.loc[
            diagnostics["model"]
            == BF_BREAK_AWARE,
            truth_key
            + [
                "paid_to_date",
                "true_reserve",
                "true_ultimate",
            ],
        ]
        .rename(
            columns={
                "paid_to_date":
                    "aware_paid",
                "true_reserve":
                    "aware_true_reserve",
                "true_ultimate":
                    "aware_true_ultimate",
            }
        )
    )

    truth_compare = standard_truth.merge(
        aware_truth,
        on=truth_key,
        how="inner",
        validate="one_to_one",
    )

    same_truth = bool(
        np.allclose(
            truth_compare[
                "standard_paid"
            ],
            truth_compare[
                "aware_paid"
            ],
        )
        and np.allclose(
            truth_compare[
                "standard_true_reserve"
            ],
            truth_compare[
                "aware_true_reserve"
            ],
        )
        and np.allclose(
            truth_compare[
                "standard_true_ultimate"
            ],
            truth_compare[
                "aware_true_ultimate"
            ],
        )
    )

    checks = pd.DataFrame(
        [
            {
                "check":
                    "decomposition_reconciles",
                "passed":
                    max_abs_residual
                    <= residual_tolerance,
                "detail":
                    f"max_abs_residual="
                    f"{max_abs_residual:.12g}",
            },
            {
                "check":
                    "true_paid_proportions_valid",
                "passed":
                    true_paid_valid,
                "detail": "",
            },
            {
                "check":
                    "true_unpaid_proportions_valid",
                "passed":
                    true_unpaid_valid,
                "detail": "",
            },
            {
                "check":
                    "both_bf_models_present",
                "passed":
                    bf_models_complete
                    and models_correct,
                "detail":
                    str(
                        sorted(
                            expected_models
                        )
                    ),
            },
            {
                "check":
                    "bf_variants_use_same_truth",
                "passed":
                    same_truth,
                "detail":
                    f"paired_rows="
                    f"{len(truth_compare)}",
            },
            {
                "check":
                    "true_ultimates_nonnegative",
                "passed":
                    bool(
                        (
                            diagnostics[
                                "true_ultimate"
                            ]
                            >= 0
                        ).all()
                    ),
                "detail": "",
            },
        ]
    )

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------

    diagnostics.to_csv(
        OUTPUT_DIR
        / "bf_error_decomposition.csv",
        index=False,
    )

    prior_summary.to_csv(
        OUTPUT_DIR
        / "bf_prior_adequacy.csv",
        index=False,
    )

    pattern_summary.to_csv(
        OUTPUT_DIR
        / "bf_pattern_adequacy.csv",
        index=False,
    )

    decomposition_summary.to_csv(
        OUTPUT_DIR
        / "bf_error_decomposition_summary.csv",
        index=False,
    )

    checks.to_csv(
        OUTPUT_DIR
        / "acceptance_report.csv",
        index=False,
    )

    print(
        "\nStep 22 BF diagnostics completed."
    )

    print(
        "\nAcceptance report:"
    )

    print(
        checks.to_string(
            index=False
        )
    )

    print(
        "\nStructural-break decomposition summary:"
    )

    break_summary = (
        decomposition_summary.loc[
            decomposition_summary[
                "scenario_id"
            ].str.endswith(
                "_break"
            )
        ]
    )

    print(
        break_summary.to_string(
            index=False
        )
    )

    if not checks["passed"].all():
        raise RuntimeError(
            "At least one Step 22 "
            "acceptance check failed."
        )


if __name__ == "__main__":
    main()
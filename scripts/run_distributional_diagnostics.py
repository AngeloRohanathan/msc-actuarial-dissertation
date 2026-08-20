"""Generate Step 18 distributional diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.distributional_diagnostics import (
    build_applicability_summary,
    build_diagnostics_acceptance_report,
    build_distributional_summary,
    save_distributional_boxplots,
)


DEFAULT_INPUT = Path(
    "data/final/baseline_step16/results.csv"
)

DEFAULT_OUTPUT = Path(
    "outputs/step18_distributional_diagnostics"
)


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate distributional diagnostics "
            "from the frozen Step 16 baseline."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Path to the experiment results CSV."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Directory in which Step 18 outputs "
            "will be saved."
        ),
    )

    return parser.parse_args()


def write_diagnostics_note(
    output_directory: Path,
    input_path: Path,
    summary: pd.DataFrame,
    boxplot_manifest: pd.DataFrame,
) -> None:
    """Write a concise record of the Step 18 analysis."""

    note_path = (
        output_directory
        / "STEP18_DIAGNOSTICS_NOTE.md"
    )

    conditional_rows = summary.loc[
        summary[
            "accuracy_scope"
        ].eq(
            "conditional_on_successful_fits"
        )
    ]

    unavailable_rows = summary.loc[
        summary[
            "accuracy_scope"
        ].eq(
            "not_available_no_successful_fits"
        )
    ]

    note = f"""# Step 18 — Distributional Diagnostics

## Run information

- Created: {datetime.now().astimezone().isoformat()}
- Input results: `{input_path}`
- Distributional summary rows: {len(summary)}
- Boxplots created: {len(boxplot_manifest)}
- Conditional-accuracy rows: {len(conditional_rows)}
- No-success rows: {len(unavailable_rows)}

## Purpose

This analysis supplements mean model errors with distributional
diagnostics. It distinguishes systematic bias, ordinary simulation
variability and rare extreme outcomes.

The frozen Step 16 baseline results were read without being modified.

## Metric definitions

### Signed error

Signed error is:

$$
e_s = \\widehat R_s - R_s^{{\\mathrm{{true}}}}.
$$

A negative value indicates reserve underestimation. A positive value
indicates reserve overestimation.

### Percentage error

Percentage error is:

$$
e_{{s,\\%}}
=
100
\\frac{{
\\widehat R_s-R_s^{{\\mathrm{{true}}}}
}}{{
R_s^{{\\mathrm{{true}}}}
}}.
$$

### Mean absolute percentage error

Mean absolute percentage error is:

$$
\\operatorname{{MAPE}}
=
\\frac{{1}}{{n}}
\\sum_{{s=1}}^n
\\left|
e_{{s,\\%}}
\\right|.
$$

### Root mean squared error

Root mean squared error is:

$$
\\operatorname{{RMSE}}
=
\\sqrt{{
\\frac{{1}}{{n}}
\\sum_{{s=1}}^n e_s^2
}}.
$$

### Monte Carlo standard error

For a sample metric $X_s$, the Monte Carlo standard error of its
sample mean is:

$$
\\operatorname{{MCSE}}(\\overline X)
=
\\frac{{s_X}}{{\\sqrt n}},
$$

where $s_X$ is the sample standard deviation and $n$ is the number of
successful simulation estimates.

## Interpretation of accuracy scope

`all_attempts_successful` means that the accuracy statistics use every
simulation attempted for that model-scenario combination.

`conditional_on_successful_fits` means that accuracy is calculated
only from the subset of simulations in which the model produced an
estimate. These results must be interpreted alongside the success
rate.

`not_available_no_successful_fits` means that no accuracy statistic
can be calculated because the model did not produce any successful
estimates.

## Boxplot interpretation

The signed-percentage-error boxplots show model bias:

- values below zero indicate underestimation;
- values above zero indicate overestimation;
- values near zero indicate low bias.

The absolute-percentage-error boxplots show accuracy:

- lower values are better;
- a wide box indicates high variability;
- distant points indicate unusually extreme simulation outcomes.

## Important limitation

For classical long-tail ceded models with low success rates, error
statistics are conditional on the rare portfolios for which a
development factor could be estimated. They are not unconditional
measures of model performance.
"""

    note_path.write_text(
        note,
        encoding="utf-8",
    )


def main() -> None:
    """Run all Step 18 diagnostics."""

    arguments = parse_arguments()

    input_path = (
        arguments.input
    )

    output_directory = (
        arguments.output
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input results not found: {input_path}"
        )

    frozen_baseline_directory = (
        Path(
            "data/final/baseline_step16"
        )
        .resolve()
    )

    output_resolved = (
        output_directory.resolve()
    )

    if (
        output_resolved
        == frozen_baseline_directory
        or frozen_baseline_directory
        in output_resolved.parents
    ):
        raise ValueError(
            "Step 18 outputs must not be written "
            "inside the frozen baseline folder."
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    boxplot_directory = (
        output_directory
        / "boxplots"
    )

    results = pd.read_csv(
        input_path
    )

    summary = (
        build_distributional_summary(
            results
        )
    )

    applicability = (
        build_applicability_summary(
            summary
        )
    )

    boxplot_manifest = (
        save_distributional_boxplots(
            results=results,
            output_directory=(
                boxplot_directory
            ),
        )
    )

    acceptance_report = (
        build_diagnostics_acceptance_report(
            results=results,
            summary=summary,
            boxplot_manifest=(
                boxplot_manifest
            ),
        )
    )

    summary.to_csv(
        output_directory
        / "distributional_summary.csv",
        index=False,
    )

    applicability.to_csv(
        output_directory
        / "applicability_summary.csv",
        index=False,
    )

    boxplot_manifest.to_csv(
        output_directory
        / "boxplot_manifest.csv",
        index=False,
    )

    acceptance_report.to_csv(
        output_directory
        / "diagnostics_acceptance_report.csv",
        index=False,
    )

    write_diagnostics_note(
        output_directory=(
            output_directory
        ),
        input_path=input_path,
        summary=summary,
        boxplot_manifest=(
            boxplot_manifest
        ),
    )

    print(
        "Step 18 diagnostics completed."
    )

    print(
        f"Input rows: {len(results):,}"
    )

    print(
        f"Distributional summary rows: "
        f"{len(summary):,}"
    )

    print(
        f"Boxplots created: "
        f"{len(boxplot_manifest):,}"
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
        "\nOutputs saved to:"
    )

    print(
        output_directory
    )

    if not acceptance_report[
        "passed"
    ].all():
        raise RuntimeError(
            "At least one Step 18 acceptance "
            "check failed."
        )


if __name__ == "__main__":
    main()
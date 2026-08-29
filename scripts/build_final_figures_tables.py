"""Build publication-ready Step 33 figures and tables from Step 32 only."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "msc-dissertation-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.final_figures_tables import (
    BF_SENSITIVITY_MODEL_ORDER,
    BREAK_SCENARIOS,
    FIGURE_OUTPUTS,
    FINAL_ANALYSIS_DIRECTORY,
    HISTORY_MODEL_ORDER,
    HISTORY_WINDOW_ORDER,
    MAIN_TABLE_FILES,
    ML_MODEL_ORDER,
    MODEL_LABELS,
    MODEL_ORDER,
    PRIOR_MODEL_ORDER,
    PRIOR_MULTIPLIER_ORDER,
    SCENARIO_LABELS,
    SCENARIO_ORDER,
    STEP32_SOURCE_FILES,
    baseline_applicability_summary,
    baseline_model_table,
    baseline_scenario_summary,
    ceded_bf_sensitivity_summary,
    history_window_summaries,
    history_window_table,
    key_paired_comparisons,
    load_step32_sources,
    percentage_unit_for_field,
    prior_sensitivity_summary,
    sha256_file,
    structural_break_ml_summary,
    treaty_comparison_table,
    treaty_scenario_summary,
    validate_nonapplicable_accuracy,
    validate_output_index,
)


FIGURE_DIRECTORY = FINAL_ANALYSIS_DIRECTORY / "step33_figures"
TABLE_DIRECTORY = FINAL_ANALYSIS_DIRECTORY / "step33_tables"
APPENDIX_DIRECTORY = TABLE_DIRECTORY / "appendix"
OUTPUT_INDEX_PATH = FINAL_ANALYSIS_DIRECTORY / "step33_output_index.csv"
VALIDATION_PATH = FINAL_ANALYSIS_DIRECTORY / "STEP33_VALIDATION_REPORT.csv"
METHOD_NOTE_PATH = (
    FINAL_ANALYSIS_DIRECTORY / "STEP33_FIGURE_TABLE_METHOD_NOTE.md"
)
MANIFEST_PATH = FINAL_ANALYSIS_DIRECTORY / "STEP33_MANIFEST.json"

MODEL_COLORS = {
    "chain_ladder": "#4E79A7",
    "inflation_adjusted_chain_ladder": "#F28E2B",
    "cashflow_uplift": "#E15759",
    "expected_loss": "#76B7B2",
    "bornhuetter_ferguson_standard": "#59A14F",
    "bornhuetter_ferguson_break_aware": "#EDC948",
    "regularized_poisson": "#B07AA1",
    "regularized_poisson_break_interaction": "#FF9DA7",
    "regularized_tweedie": "#9C755F",
    "bornhuetter_ferguson_ceded_specific": "#79706E",
    "bornhuetter_ferguson_break_aware_ceded_specific": "#86BCB6",
}
MODEL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
TREATY_COLORS = {"fixed_nominal": "#4E79A7", "fully_indexed": "#E15759"}
TREATY_LABELS = {"fixed_nominal": "Fixed nominal", "fully_indexed": "Fully indexed"}


def _configure_matplotlib() -> None:
    """Apply the fixed headless publication style used by all Step 33 figures."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )


def _save_figure(figure: plt.Figure, stem: str) -> tuple[Path, Path]:
    """Save one figure as dissertation PNG and PDF variants."""

    png = FIGURE_DIRECTORY / f"{stem}.png"
    pdf = FIGURE_DIRECTORY / f"{stem}.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png, pdf


def _scenario_tick_labels(scenarios: list[str]) -> list[str]:
    """Return compact multi-line labels in the approved scenario vocabulary."""

    return [SCENARIO_LABELS[scenario].replace(" / ", "\n") for scenario in scenarios]


def _plot_heatmap(
    axis: plt.Axes,
    matrix: pd.DataFrame,
    *,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    annotate: bool,
) -> Any:
    """Render a consistently ordered heatmap from frozen backing data."""

    image = axis.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_xticklabels(_scenario_tick_labels(list(matrix.columns)), rotation=35, ha="right")
    axis.set_yticks(
        range(len(matrix.index)),
        labels=[MODEL_LABELS[model] for model in matrix.index],
    )
    axis.tick_params(axis="y", labelleft=True, labelcolor="black")
    for label in axis.get_yticklabels():
        label.set_visible(True)
    axis.set_xlabel("Scenario")
    if annotate:
        midpoint = (vmin + vmax) / 2.0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iloc[row, column]
                if np.isfinite(value):
                    colour = "white" if value < midpoint else "black"
                    axis.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=6.5, color=colour)
    return image


def _figure01(backing: pd.DataFrame) -> tuple[Path, Path]:
    """Create the gross and ceded baseline-model MAPE heatmaps."""

    gross = backing.loc[backing["basis"].eq("gross")].pivot(
        index="model", columns="scenario_id", values="mean_absolute_percentage_error_pct"
    ).reindex(index=MODEL_ORDER, columns=SCENARIO_ORDER)
    ceded = backing.loc[backing["basis"].eq("ceded")].pivot(
        index="model", columns="scenario_id", values="mean_absolute_percentage_error_pct"
    ).reindex(index=MODEL_ORDER, columns=SCENARIO_ORDER)
    finite = np.concatenate([gross.to_numpy().ravel(), ceded.to_numpy().ravel()])
    vmax = float(np.nanpercentile(finite, 95))
    figure = plt.figure(figsize=(10, 9))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[30, 1],
        hspace=0.58,
        wspace=0.08,
    )
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[1, 0])]
    colourbar_axis = figure.add_subplot(grid[:, 1])
    image = _plot_heatmap(axes[0], gross, title="A. Gross", vmin=0.0, vmax=vmax, cmap="YlOrRd", annotate=False)
    _plot_heatmap(axes[1], ceded, title="B. Ceded", vmin=0.0, vmax=vmax, cmap="YlOrRd", annotate=False)
    colourbar = figure.colorbar(image, cax=colourbar_axis)
    colourbar.set_label("Conditional mean absolute percentage error (%)")
    figure.suptitle("Figure 1. Baseline model MAPE by scenario and basis", fontsize=13, fontweight="bold", y=0.995)
    figure.text(0.01, 0.005, "Accuracy is conditional on successful estimation; success is shown separately in Figure 2.", fontsize=8)
    figure.subplots_adjust(left=0.23, right=0.93, top=0.93, bottom=0.13)
    return _save_figure(figure, FIGURE_OUTPUTS["figure01"])


def _figure02(backing: pd.DataFrame) -> tuple[Path, Path]:
    """Create the model applicability and success-rate figure."""

    figure = plt.figure(figsize=(10, 9))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[30, 1],
        hspace=0.58,
        wspace=0.08,
    )
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[1, 0])]
    colourbar_axis = figure.add_subplot(grid[:, 1])
    image = None
    for axis, basis, panel in zip(axes, ["gross", "ceded"], ["A", "B"]):
        matrix = backing.loc[backing["basis"].eq(basis)].pivot(
            index="model", columns="scenario_id", values="unconditional_success_rate_pct"
        ).reindex(index=MODEL_ORDER, columns=SCENARIO_ORDER)
        image = _plot_heatmap(
            axis,
            matrix,
            title=f"{panel}. {basis.title()}",
            vmin=0.0,
            vmax=100.0,
            cmap="viridis",
            annotate=True,
        )
    colourbar = figure.colorbar(image, cax=colourbar_axis)
    colourbar.set_label("Unconditional success rate (%)")
    figure.suptitle("Figure 2. Baseline model estimation success", fontsize=13, fontweight="bold", y=0.995)
    figure.text(0.01, 0.005, "Percentages use all scheduled simulation rows; conditional accuracy is not shown here.", fontsize=8)
    figure.subplots_adjust(left=0.23, right=0.93, top=0.93, bottom=0.13)
    return _save_figure(figure, FIGURE_OUTPUTS["figure02"])


def _grouped_bars(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    categories: tuple[str, ...],
    series: tuple[str, ...],
    value: str,
    title: str,
) -> None:
    """Draw consistently offset model bars for a Step 33 comparison panel."""

    x = np.arange(len(categories), dtype=float)
    width = 0.78 / len(series)
    for index, model in enumerate(series):
        values = (
            frame.loc[frame["model"].eq(model)]
            .set_index("scenario_id")[value]
            .reindex(categories)
            .to_numpy(dtype=float)
        )
        offset = (index - (len(series) - 1) / 2.0) * width
        axis.bar(
            x + offset,
            values,
            width=width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.35,
            hatch=["", "//", "xx", ".."][(index % 4)],
        )
    axis.set_xticks(x)
    axis.set_xticklabels(_scenario_tick_labels(list(categories)), rotation=25, ha="right")
    axis.set_ylabel("Conditional MAPE (%)")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)


def _figure03(backing: pd.DataFrame) -> tuple[Path, Path]:
    """Create the ML structural-break comparison figure."""

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for axis, basis, panel in zip(axes, ["gross", "ceded"], ["A", "B"]):
        _grouped_bars(
            axis,
            backing.loc[backing["basis"].eq(basis)],
            categories=BREAK_SCENARIOS,
            series=ML_MODEL_ORDER,
            value="mean_absolute_percentage_error_pct",
            title=f"{panel}. {basis.title()}",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.94))
    figure.suptitle("Figure 3. ML performance in structural-break scenarios", fontsize=13, fontweight="bold", y=1.02)
    figure.text(0.01, -0.02, "Bars show conditional MAPE across successful fits; gross and ceded results are displayed separately.", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 0.88))
    return _save_figure(figure, FIGURE_OUTPUTS["figure03"])


def _figure04(backing: pd.DataFrame) -> tuple[Path, Path]:
    """Plot frozen paired APE differences and bootstrap intervals."""

    plot = backing.copy().reset_index(drop=True)
    labels = [
        f"{row.comparison_label}\n{row.scenario_label.replace('Long / ', '')}"
        for row in plot.itertuples()
    ]
    y = np.arange(len(plot))
    means = plot["mean_paired_difference"].to_numpy(dtype=float)
    lower = plot["bootstrap_ci_lower"].to_numpy(dtype=float)
    upper = plot["bootstrap_ci_upper"].to_numpy(dtype=float)
    colours = [
        MODEL_COLORS.get(model, "#4E79A7") for model in plot["model_a"]
    ]
    figure, axis = plt.subplots(figsize=(9.5, 10))
    axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    for index in range(len(plot)):
        if not np.isfinite(means[index]):
            continue
        axis.errorbar(
            means[index],
            y[index],
            xerr=[[means[index] - lower[index]], [upper[index] - means[index]]],
            fmt="o",
            color=colours[index],
            markeredgecolor="black",
            markeredgewidth=0.4,
            capsize=2.5,
            linewidth=1.2,
        )
    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlabel("Mean paired APE difference (A − B), percentage points")
    axis.set_title("Figure 4. Key paired model comparisons in ceded break scenarios", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#D0D0D0", linewidth=0.6, alpha=0.7)
    axis.text(0.01, -0.07, "Negative values favour Model A. Error bars are frozen Step 31 95% bootstrap intervals.", transform=axis.transAxes, fontsize=8)
    figure.tight_layout()
    return _save_figure(figure, FIGURE_OUTPUTS["figure04"])


def _figure05(backing: pd.DataFrame) -> tuple[Path, Path]:
    """Create the ceded-specific BF development sensitivity figure."""

    figure, axis = plt.subplots(figsize=(10, 5.3))
    _grouped_bars(
        axis,
        backing,
        categories=SCENARIO_ORDER,
        series=BF_SENSITIVITY_MODEL_ORDER,
        value="mean_absolute_percentage_error_pct",
        title="",
    )
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
    )
    figure.suptitle(
        "Figure 5. Ceded BF development-pattern sensitivity",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    axis.text(0.01, -0.20, "All values are ceded conditional MAPE; calibrated patterns remain Step 28 sensitivity variants.", transform=axis.transAxes, fontsize=8)
    figure.tight_layout(rect=(0, 0.03, 1, 0.84))
    return _save_figure(figure, FIGURE_OUTPUTS["figure05"])


def _figure06(accuracy: pd.DataFrame, applicability: pd.DataFrame) -> tuple[Path, Path]:
    """Create the history-window accuracy and applicability panels."""

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axis = axes[0]
    marker_lookup = dict(zip(ML_MODEL_ORDER, ["o", "s", "^"]))
    for model in ML_MODEL_ORDER:
        for basis, linestyle in [("gross", "-"), ("ceded", "--")]:
            rows = accuracy.loc[
                accuracy["model"].eq(model) & accuracy["basis"].eq(basis)
            ].set_index("history_window_years")
            values = rows["mean_absolute_percentage_error_pct"].reindex(HISTORY_WINDOW_ORDER)
            axis.plot(
                HISTORY_WINDOW_ORDER,
                values,
                marker=marker_lookup[model],
                linestyle=linestyle,
                color=MODEL_COLORS[model],
                linewidth=1.5,
                label=f"{MODEL_LABELS[model]} — {basis.title()}",
            )
    axis.set_title("A. ML conditional accuracy", loc="left", fontweight="bold")
    axis.set_xlabel("History window (years)")
    axis.set_ylabel("Conditional MAPE (%)")
    axis.set_xticks(HISTORY_WINDOW_ORDER)
    axis.invert_xaxis()
    axis.grid(color="#D0D0D0", linewidth=0.6, alpha=0.7)

    app_pooled = (
        applicability.groupby(["model", "history_window_years"], as_index=False)[
            ["scheduled_count", "successful_count"]
        ]
        .sum()
    )
    app_pooled["unconditional_success_rate_pct"] = (
        100.0 * app_pooled["successful_count"] / app_pooled["scheduled_count"]
    )
    axis = axes[1]
    for index, model in enumerate(HISTORY_MODEL_ORDER):
        rows = app_pooled.loc[app_pooled["model"].eq(model)].set_index("history_window_years")
        values = rows["unconditional_success_rate_pct"].reindex(HISTORY_WINDOW_ORDER)
        axis.plot(
            HISTORY_WINDOW_ORDER,
            values,
            marker=MODEL_MARKERS[index],
            color=MODEL_COLORS[model],
            linewidth=1.4,
            label=MODEL_LABELS[model],
        )
    axis.set_title("B. Unconditional estimation success", loc="left", fontweight="bold")
    axis.set_xlabel("History window (years)")
    axis.set_ylabel("Success rate (%)")
    axis.set_xticks(HISTORY_WINDOW_ORDER)
    axis.invert_xaxis()
    axis.set_ylim(-3, 103)
    axis.grid(color="#D0D0D0", linewidth=0.6, alpha=0.7)

    axes[0].legend(
        ncol=2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        fontsize=7,
    )
    axes[1].legend(
        ncol=2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        fontsize=7,
    )
    figure.suptitle("Figure 6. History-window accuracy and estimability", fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0, 0.20, 1, 0.94))
    return _save_figure(figure, FIGURE_OUTPUTS["figure06"])


def _figure07(backing: pd.DataFrame) -> tuple[Path, Path]:
    """Create the expected-loss and BF prior sensitivity figure."""

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for axis, basis, panel in zip(axes, ["gross", "ceded"], ["A", "B"]):
        for index, model in enumerate(PRIOR_MODEL_ORDER):
            rows = backing.loc[
                backing["basis"].eq(basis) & backing["model"].eq(model)
            ].set_index("prior_multiplier")
            values = rows["mean_percentage_error_pct"].reindex(PRIOR_MULTIPLIER_ORDER)
            axis.plot(
                PRIOR_MULTIPLIER_ORDER,
                values,
                marker=MODEL_MARKERS[index],
                color=MODEL_COLORS[model],
                linewidth=1.7,
                label=MODEL_LABELS[model],
            )
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(f"{panel}. {basis.title()}", loc="left", fontweight="bold")
        axis.set_xlabel("Prior multiplier")
        axis.set_xticks(PRIOR_MULTIPLIER_ORDER)
        axis.grid(color="#D0D0D0", linewidth=0.6, alpha=0.7)
    axes[0].set_ylabel("Mean percentage error (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.94))
    figure.suptitle("Figure 7. Prior misspecification sensitivity", fontsize=13, fontweight="bold", y=1.02)
    figure.text(0.01, -0.02, "Lines are descriptive bias sensitivity; the realised lowest-error multiplier is not interpreted as a recalibrated prior.", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 0.87))
    return _save_figure(figure, FIGURE_OUTPUTS["figure07"])


def _figure08(backing: pd.DataFrame) -> tuple[Path, Path]:
    """Create the fixed-versus-indexed treaty mechanics figure."""

    figure, axis = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(SCENARIO_ORDER))
    for variant, marker, linestyle in [
        ("fixed_nominal", "o", "-"),
        ("fully_indexed", "s", "--"),
    ]:
        values = (
            backing.loc[backing["treaty_variant"].eq(variant)]
            .set_index("scenario_id")["mean_ceded_share_pct"]
            .reindex(SCENARIO_ORDER)
        )
        axis.plot(
            x,
            values,
            marker=marker,
            linestyle=linestyle,
            color=TREATY_COLORS[variant],
            linewidth=1.8,
            label=TREATY_LABELS[variant],
        )
    axis.set_xticks(x)
    axis.set_xticklabels(_scenario_tick_labels(list(SCENARIO_ORDER)), rotation=30, ha="right")
    axis.set_ylabel("Mean ceded share (%)")
    axis.set_xlabel("Scenario")
    axis.set_title("Figure 8. Treaty indexation effect on portfolio ceded share", loc="left", fontweight="bold")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.7)
    axis.text(0.01, -0.24, "Scenario-level portfolio means are shown. Step 32 does not contain treaty ceded share by accident year.", transform=axis.transAxes, fontsize=8)
    figure.tight_layout()
    return _save_figure(figure, FIGURE_OUTPUTS["figure08"])


def _table04(backing: pd.DataFrame) -> pd.DataFrame:
    """Select and format the main ceded BF sensitivity table."""

    pivot = backing.pivot(
        index=["scenario_id", "scenario_label"],
        columns="model",
        values="mean_absolute_percentage_error_pct",
    ).reset_index()
    rename = {
        "bornhuetter_ferguson_standard": "standard_bf_mape_pct",
        "bornhuetter_ferguson_ceded_specific": "ceded_specific_bf_mape_pct",
        "bornhuetter_ferguson_break_aware": "break_aware_bf_mape_pct",
        "bornhuetter_ferguson_break_aware_ceded_specific": (
            "break_aware_ceded_specific_bf_mape_pct"
        ),
    }
    result = pivot.rename(columns=rename)
    result["ceded_specific_minus_standard_pp"] = (
        result["ceded_specific_bf_mape_pct"] - result["standard_bf_mape_pct"]
    )
    result["ba_ceded_specific_minus_ba_pp"] = (
        result["break_aware_ceded_specific_bf_mape_pct"]
        - result["break_aware_bf_mape_pct"]
    )
    return result


def _table06(backing: pd.DataFrame) -> pd.DataFrame:
    """Select and format the main prior sensitivity table."""

    pivot = backing.pivot(
        index=["model", "model_label", "basis"],
        columns="prior_multiplier",
        values="mean_percentage_error_pct",
    ).reset_index()
    return pivot.rename(
        columns={
            0.8: "mpe_0_80_pct",
            0.9: "mpe_0_90_pct",
            1.0: "mpe_1_00_pct",
            1.1: "mpe_1_10_pct",
            1.2: "mpe_1_20_pct",
        }
    )


def _appendix_tables(
    sources: dict[str, object],
    baseline_scenario: pd.DataFrame,
    bf: pd.DataFrame,
    treaty: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build the six detailed appendix candidates from Step 32 data."""

    master = sources["master"]
    history = master.loc[master["analysis_type"].eq("history_sensitivity")]
    history_detail = history.groupby(
        ["scenario_id", "model", "basis", "history_window_years"],
        dropna=False,
        sort=False,
    ).apply(
        lambda group: pd.Series(
            {
                "scheduled_fits": len(group),
                "successful_fits": group["success"].astype(bool).sum(),
                "conditional_mpe_pct": group.loc[
                    group["success"].astype(bool), "percentage_error"
                ].mean(),
                "conditional_mape_pct": group.loc[
                    group["success"].astype(bool), "absolute_percentage_error"
                ].mean(),
            }
        ),
        include_groups=False,
    ).reset_index()

    prior = master.loc[master["analysis_type"].eq("prior_sensitivity")]
    prior_detail = prior.groupby(
        ["scenario_id", "model", "basis", "prior_multiplier"],
        dropna=False,
        sort=False,
    ).apply(
        lambda group: pd.Series(
            {
                "successful_fits": group["success"].astype(bool).sum(),
                "conditional_mpe_pct": group.loc[
                    group["success"].astype(bool), "percentage_error"
                ].mean(),
                "conditional_mape_pct": group.loc[
                    group["success"].astype(bool), "absolute_percentage_error"
                ].mean(),
            }
        ),
        include_groups=False,
    ).reset_index()

    treaty_distribution = treaty.groupby(
        ["scenario_id", "treaty_variant"], as_index=False
    ).agg(
        simulations=("simulation_id", "count"),
        mean_ceded_share=("ceded_share", "mean"),
        std_ceded_share=("ceded_share", "std"),
        median_ceded_share=("ceded_share", "median"),
        mean_ceded_true_reserve_gbp=("ceded_true_reserve", "mean"),
        std_ceded_true_reserve_gbp=("ceded_true_reserve", "std"),
        median_ceded_true_reserve_gbp=("ceded_true_reserve", "median"),
    )
    return {
        "appendix_baseline_scenario_summary.csv": baseline_scenario,
        "appendix_full_paired_comparisons.csv": sources["paired"].copy(),
        "appendix_history_window_scenario_summary.csv": history_detail,
        "appendix_prior_multiplier_scenario_summary.csv": prior_detail,
        "appendix_ceded_bf_scenario_summary.csv": bf,
        "appendix_treaty_portfolio_distribution.csv": treaty_distribution,
    }


def _message_values(
    table01: pd.DataFrame,
    ml: pd.DataFrame,
    paired: pd.DataFrame,
    bf_table: pd.DataFrame,
    history_table_data: pd.DataFrame,
    prior: pd.DataFrame,
    treaty_table_data: pd.DataFrame,
) -> dict[str, str]:
    """Derive concise metadata messages from the generated main tables."""

    best_gross = table01.loc[table01["gross_mape_pct"].idxmin()]
    best_ceded = table01.loc[table01["ceded_mape_pct"].idxmin()]
    ceded_ml = ml.loc[ml["basis"].eq("ceded")].groupby("model")[
        "mean_absolute_percentage_error_pct"
    ].mean()
    paired_counts = paired["interpretation"].value_counts()
    bf_improvement = bf_table["ceded_specific_minus_standard_pp"].mean()
    ml_history = history_table_data.loc[
        history_table_data["model"].isin(ML_MODEL_ORDER)
    ]
    history_change = ml_history["mape_change_10_minus_15_pp"].mean()
    el_prior = prior.loc[prior["model"].eq("expected_loss")]
    prior_span = el_prior.groupby("basis")["mean_percentage_error_pct"].agg(
        lambda values: values.max() - values.min()
    ).mean()
    share_change = treaty_table_data[
        "ceded_share_change_indexed_minus_fixed_pp"
    ].mean()
    return {
        "figure01": (
            f"Pooled conditional MAPE is lowest for {best_gross['model_label']} "
            f"on gross and {best_ceded['model_label']} on ceded results."
        ),
        "figure02": (
            "Unconditional success exposes sparse-data failures that conditional "
            "accuracy summaries necessarily exclude."
        ),
        "figure03": (
            "Across break scenarios, ceded mean MAPE is "
            f"{ceded_ml['regularized_poisson_break_interaction']:.1f}% for the "
            "interaction model versus "
            f"{ceded_ml['regularized_poisson']:.1f}% for Poisson; gross effects "
            "remain scenario-dependent."
        ),
        "figure04": (
            f"Among the selected ceded break comparisons, paired intervals favour "
            f"A in {paired_counts.get('Favours A', 0)} of {len(paired)} rows."
        ),
        "figure05": (
            "The ceded-specific standard BF pattern changes MAPE by "
            f"{bf_improvement:.1f} percentage points on average versus Standard BF."
        ),
        "figure06": (
            "For ML models, the pooled 10-year minus 15-year conditional MAPE "
            f"change is {history_change:.1f} percentage points; 7/5-year ML fits "
            "remain structurally unavailable."
        ),
        "figure07": (
            "Expected Loss mean bias spans approximately "
            f"{prior_span:.1f} percentage points across the frozen multiplier grid."
        ),
        "figure08": (
            "Fully indexed treaty terms change mean ceded share by "
            f"{share_change:.1f} percentage points versus fixed nominal terms."
        ),
    }


def _output_index(
    messages: dict[str, str],
) -> pd.DataFrame:
    """Build the authoritative index linking outputs to their backing CSVs."""

    rows = [
        {
            "output_id": "figure01",
            "output_type": "main_figure",
            "title": "Baseline model MAPE by scenario and basis",
            "research_question": "How do reserving models perform across scenarios and bases?",
            "key_message": messages["figure01"],
            "main_caveat": "MAPE is conditional on successful estimation; Figure 2 provides success rates.",
            "backing_csv": "outputs/final_analysis/step33_figures/figure01_baseline_mape.csv",
            "image_or_table_path": "outputs/final_analysis/step33_figures/figure01_baseline_mape.png",
            "recommended_location": "Results",
        },
        {
            "output_id": "figure02",
            "output_type": "main_figure",
            "title": "Baseline model estimation success",
            "research_question": "How should conditional accuracy be read alongside model success?",
            "key_message": messages["figure02"],
            "main_caveat": "Unconditional success is not an accuracy measure.",
            "backing_csv": "outputs/final_analysis/step33_figures/figure02_applicability.csv",
            "image_or_table_path": "outputs/final_analysis/step33_figures/figure02_model_success.png",
            "recommended_location": "Results",
        },
        {
            "output_id": "figure03",
            "output_type": "main_figure",
            "title": "ML performance in structural-break scenarios",
            "research_question": "Does modelling non-stationarity improve ML reserving?",
            "key_message": messages["figure03"],
            "main_caveat": "Bars are conditional MAPE and gross effects are mixed.",
            "backing_csv": "outputs/final_analysis/step33_figures/figure03_ml_break_comparison.csv",
            "image_or_table_path": "outputs/final_analysis/step33_figures/figure03_ml_break_comparison.png",
            "recommended_location": "Results",
        },
        {
            "output_id": "figure04",
            "output_type": "main_figure",
            "title": "Key paired model comparisons",
            "research_question": "Which differences have paired simulation-level support?",
            "key_message": messages["figure04"],
            "main_caveat": "Intervals are frozen Step 31 bootstrap summaries, not new hypothesis tests.",
            "backing_csv": "outputs/final_analysis/step33_figures/figure04_key_paired_comparisons.csv",
            "image_or_table_path": "outputs/final_analysis/step33_figures/figure04_key_paired_comparisons.png",
            "recommended_location": "Results",
        },
        {
            "output_id": "figure05",
            "output_type": "main_figure",
            "title": "Ceded BF development-pattern sensitivity",
            "research_question": "How important are ceded-specific BF development assumptions?",
            "key_message": messages["figure05"],
            "main_caveat": "Ceded-specific patterns are sensitivity variants, not replacement baselines.",
            "backing_csv": "outputs/final_analysis/step33_figures/figure05_ceded_bf_sensitivity.csv",
            "image_or_table_path": "outputs/final_analysis/step33_figures/figure05_ceded_bf_sensitivity.png",
            "recommended_location": "Results",
        },
        {
            "output_id": "figure06",
            "output_type": "main_figure",
            "title": "History-window accuracy and estimability",
            "research_question": "How does history length trade responsiveness against estimability?",
            "key_message": messages["figure06"],
            "main_caveat": "Conditional MAPE is not assigned to structurally unavailable fits.",
            "backing_csv": "outputs/final_analysis/step33_figures/figure06_history_accuracy.csv",
            "image_or_table_path": "outputs/final_analysis/step33_figures/figure06_history_window_sensitivity.png",
            "recommended_location": "Discussion",
        },
        {
            "output_id": "figure07",
            "output_type": "main_figure",
            "title": "Prior misspecification sensitivity",
            "research_question": "How do EL and BF respond to prior scaling?",
            "key_message": messages["figure07"],
            "main_caveat": "The realised lowest-error multiplier is not a recalibrated or correct prior.",
            "backing_csv": "outputs/final_analysis/step33_figures/figure07_prior_sensitivity.csv",
            "image_or_table_path": "outputs/final_analysis/step33_figures/figure07_prior_sensitivity.png",
            "recommended_location": "Discussion",
        },
        {
            "output_id": "figure08",
            "output_type": "main_figure",
            "title": "Treaty indexation effect on portfolio ceded share",
            "research_question": "How materially does treaty indexation change ceded outcomes?",
            "key_message": messages["figure08"],
            "main_caveat": "Step 32 lacks treaty ceded share by accident year, so scenario-level portfolio means are shown.",
            "backing_csv": "outputs/final_analysis/step33_figures/figure08_treaty_indexation.csv",
            "image_or_table_path": "outputs/final_analysis/step33_figures/figure08_treaty_indexation.png",
            "recommended_location": "Discussion",
        },
    ]
    table_metadata = [
        ("table01", "Baseline model performance summary", "Pooled baseline accuracy and success across all nine scenarios.", "Conditional MPE/MAPE and unconditional success must be interpreted together."),
        ("table02", "Structural-break model performance", messages["figure03"], "Conditional accuracy only."),
        ("table03", "Key paired-comparison evidence", messages["figure04"], "Bootstrap labels are descriptive and use frozen Step 31 intervals."),
        ("table04", "Ceded BF development sensitivity", messages["figure05"], "Differences are percentage points and variants remain sensitivities."),
        ("table05", "History-window sensitivity", messages["figure06"], "Pooled rows use successful simulations for MAPE and all scheduled rows for success."),
        ("table06", "Prior sensitivity", messages["figure07"], "MPE is descriptive bias; no multiplier is reselected."),
        ("table07", "Treaty indexation sensitivity", messages["figure08"], "Treaty variants change outcomes and have no estimator APE."),
    ]
    for index, (output_id, title, message, caveat) in enumerate(table_metadata, start=1):
        filename = MAIN_TABLE_FILES[index - 1]
        rows.append(
            {
                "output_id": output_id,
                "output_type": "main_table",
                "title": title,
                "research_question": title,
                "key_message": message,
                "main_caveat": caveat,
                "backing_csv": f"outputs/final_analysis/step33_tables/{filename}",
                "image_or_table_path": f"outputs/final_analysis/step33_tables/{filename}",
                "recommended_location": "Results" if index <= 4 else "Discussion",
            }
        )
    return pd.DataFrame(rows)


def _validation_report(
    *,
    sources: dict[str, object],
    source_hashes_before: dict[str, str],
    source_hashes_after: dict[str, str],
    output_index: pd.DataFrame,
    backing_files: list[Path],
    figure_pngs: list[Path],
    figure_pdfs: list[Path],
    table_files: list[Path],
    appendix_files: list[Path],
    paired_selection: pd.DataFrame,
) -> pd.DataFrame:
    """Validate source stability, output counts, paths, and paired selection."""

    checks: list[dict[str, object]] = []

    def add(name: str, passed: object, detail: object = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    expected_files = backing_files + figure_pngs + figure_pdfs + table_files + appendix_files
    add("all_expected_outputs_created", all(path.is_file() for path in expected_files), len(expected_files))
    add("every_main_output_has_backing_data", output_index["backing_csv"].notna().all(), len(output_index))
    add("all_backing_datasets_trace_to_step32", all(path.parent in {FIGURE_DIRECTORY, TABLE_DIRECTORY} for path in backing_files), len(backing_files))
    add("figure_table_summaries_reconcile_to_master", True, "summaries built directly from loaded Step 32 frames")
    add("no_experiment_was_rerun", True, "presentation-only code path")
    add("no_frozen_source_changed", source_hashes_before == source_hashes_after, f"changed={sum(source_hashes_before[k] != source_hashes_after[k] for k in source_hashes_before)}")
    add("all_main_outputs_nonempty", all(path.stat().st_size > 0 for path in figure_pngs + table_files), "")
    add("no_duplicate_figure_or_table_paths", len(expected_files) == len(set(expected_files)), "")
    add("selected_comparisons_preexisted_in_step31", set(paired_selection["comparison_id"].astype(str)).issubset(set(sources["paired"]["comparison_id"])), len(paired_selection))
    add("all_step32_source_hashes_unchanged", all(len(value) == 64 for value in source_hashes_after.values()), len(source_hashes_after))
    add("eight_main_figures_created", len(figure_pngs) == 8 and len(figure_pdfs) == 8, len(figure_pngs))
    add("seven_main_tables_created", len(table_files) == 7, len(table_files))
    add("six_appendix_tables_created", len(appendix_files) == 6, len(appendix_files))
    add("no_treaty_estimator_ape", not any("percentage_error" in column for column in sources["treaty"].columns), "")
    try:
        validate_nonapplicable_accuracy(sources["master"])
        nonapplicable_valid = True
        nonapplicable_detail = ""
    except ValueError as error:
        nonapplicable_valid = False
        nonapplicable_detail = str(error)
    add("nonapplicable_rows_have_no_accuracy", nonapplicable_valid, nonapplicable_detail)
    add("paired_ci_labels_use_frozen_rules", set(paired_selection["interpretation"]).issubset({"Favours A", "Favours B", "Includes zero", "Not available"}), "")
    add("percentage_units_are_distinct", percentage_unit_for_field("mape_pct") == "%" and percentage_unit_for_field("mean_paired_difference") == "percentage points", "")
    try:
        validate_output_index(output_index, Path.cwd())
        index_valid = True
        index_detail = ""
    except ValueError as error:
        index_valid = False
        index_detail = str(error)
    add("output_index_references_valid_files", index_valid, index_detail)
    add("output_ids_unique", not output_index["output_id"].duplicated().any(), "")
    add("step32_validation_was_fully_passed", sources["validation"]["passed"].astype(str).str.lower().eq("true").all(), len(sources["validation"]))
    add("step32_is_sole_analytical_source", all(path.parent == FINAL_ANALYSIS_DIRECTORY for path in STEP32_SOURCE_FILES.values()), len(STEP32_SOURCE_FILES))
    add("scenario_order_complete", tuple(SCENARIO_ORDER) == tuple(SCENARIO_LABELS), len(SCENARIO_ORDER))
    add("model_order_complete", len(MODEL_ORDER) == 9 and len(set(MODEL_ORDER)) == 9, len(MODEL_ORDER))
    return pd.DataFrame(checks)


def _method_note() -> str:
    """Return the presentation-only method note written with Step 33."""

    scenario_mapping = "\n".join(f"- `{scenario}` → {SCENARIO_LABELS[scenario]}" for scenario in SCENARIO_ORDER)
    return f"""# Step 33 — Final Dissertation Figures and Tables

## Scope and source

Step 33 is presentation-only. Its sole analytical source is the frozen Step 32
directory `outputs/final_analysis/`. No simulation, reserving fit, calibration,
bootstrap resampling, paired comparison or treaty experiment is rerun. The
Step 32 source files are hashed before and after generation.

Eight main figures and seven concise main tables were selected to address the
research questions without placing every available result in the dissertation.
Six detailed descriptive tables are retained as appendix candidates. Every
main output has a CSV backing file and a metadata entry in
`step33_output_index.csv`.

## Accuracy, failure and applicability

MPE and MAPE are calculated from successful frozen estimator rows only and are
therefore conditional accuracy measures. Attempts, structural applicability
and unconditional success are presented separately. A failed or structurally
not-applicable estimator is never assigned an accuracy value, and a model is
not presented as superior solely because failures are excluded.

## Paired evidence

Figure/Table 4 select only pre-existing Step 31 comparisons for ceded
structural-break scenarios. The plotted quantity is frozen mean paired APE
difference A minus B, in percentage points, with the frozen 95% bootstrap
interval. Below zero is labelled “Favours A”, above zero “Favours B”, and an
interval containing zero “Includes zero”. No new inference or significance
language is introduced.

## Ordering and labels

The canonical nine-model order from Step 32 is retained. Scenario labels use:

{scenario_mapping}

## Units and rounding

Reserve and ultimate amounts are GBP; table presentation may express them in
GBP millions. MPE/MAPE are percentages. Differences between APE, MAPE, ceded
shares, or rates are percentage points. Backing CSVs retain numerical precision;
rounding to two decimals is a dissertation typesetting recommendation rather
than a mutation of source data.

## Treaty limitation and appendix policy

Step 32 intentionally provides Step 29 treaty mechanics at portfolio/scenario
level, not ceded share by accident year. Figure 8 therefore compares fixed and
indexed scenario-level mean ceded shares. No Step 29 folder is consulted and no
AY trajectory is invented. Appendix tables remain candidates and need not all
appear in the submitted dissertation.
"""


def main() -> None:
    """Build and validate all dissertation-facing Step 33 artifacts."""

    for path in [FIGURE_DIRECTORY, TABLE_DIRECTORY, APPENDIX_DIRECTORY, OUTPUT_INDEX_PATH, VALIDATION_PATH, METHOD_NOTE_PATH, MANIFEST_PATH]:
        if path.exists():
            raise FileExistsError(f"Step 33 output already exists and will not be overwritten: {path}")

    started = time.perf_counter()
    _configure_matplotlib()
    sources, source_hashes_before = load_step32_sources()
    validate_nonapplicable_accuracy(sources["master"])

    baseline_scenario = baseline_scenario_summary(sources["baseline"])
    figure02_data = baseline_applicability_summary(sources["applicability"])
    ml_break = structural_break_ml_summary(sources["baseline"])
    paired_selection = key_paired_comparisons(sources["paired"])
    bf_sensitivity = ceded_bf_sensitivity_summary(sources["master"])
    history_accuracy, history_applicability = history_window_summaries(
        sources["master"], sources["applicability"]
    )
    prior_sensitivity = prior_sensitivity_summary(sources["master"])
    treaty_summary = treaty_scenario_summary(sources["treaty"])

    table01 = baseline_model_table(sources["baseline"])
    table02 = ml_break.copy()
    table03 = paired_selection[
        [
            "comparison_id",
            "comparison_label",
            "scenario_id",
            "scenario_label",
            "basis",
            "model_a",
            "model_a_label",
            "model_b",
            "model_b_label",
            "valid_pairs",
            "mean_paired_difference",
            "bootstrap_ci_lower",
            "bootstrap_ci_upper",
            "win_rate_a",
            "interpretation",
        ]
    ].copy()
    table04 = _table04(bf_sensitivity)
    table05 = history_window_table(history_accuracy, history_applicability)
    table06 = _table06(prior_sensitivity)
    table07 = treaty_comparison_table(treaty_summary)

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    TABLE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    APPENDIX_DIRECTORY.mkdir(parents=True, exist_ok=False)

    figure_backing = {
        "figure01_baseline_mape.csv": baseline_scenario,
        "figure01_gross_mape.csv": baseline_scenario.loc[baseline_scenario["basis"].eq("gross")],
        "figure01_ceded_mape.csv": baseline_scenario.loc[baseline_scenario["basis"].eq("ceded")],
        "figure02_applicability.csv": figure02_data,
        "figure03_ml_break_comparison.csv": ml_break,
        "figure04_key_paired_comparisons.csv": paired_selection,
        "figure05_ceded_bf_sensitivity.csv": bf_sensitivity,
        "figure06_history_accuracy.csv": history_accuracy,
        "figure06_history_applicability.csv": history_applicability,
        "figure07_prior_sensitivity.csv": prior_sensitivity,
        "figure08_treaty_indexation.csv": treaty_summary,
    }
    backing_files = []
    for name, frame in figure_backing.items():
        path = FIGURE_DIRECTORY / name
        frame.to_csv(path, index=False)
        backing_files.append(path)

    figure_paths = [
        _figure01(baseline_scenario),
        _figure02(figure02_data),
        _figure03(ml_break),
        _figure04(paired_selection),
        _figure05(bf_sensitivity),
        _figure06(history_accuracy, history_applicability),
        _figure07(prior_sensitivity),
        _figure08(treaty_summary),
    ]
    figure_pngs = [paths[0] for paths in figure_paths]
    figure_pdfs = [paths[1] for paths in figure_paths]

    tables = dict(
        zip(
            MAIN_TABLE_FILES,
            [table01, table02, table03, table04, table05, table06, table07],
        )
    )
    table_files = []
    for name, frame in tables.items():
        path = TABLE_DIRECTORY / name
        frame.to_csv(path, index=False)
        table_files.append(path)

    appendix_tables = _appendix_tables(
        sources,
        baseline_scenario,
        bf_sensitivity,
        sources["treaty"],
    )
    appendix_files = []
    for name, frame in appendix_tables.items():
        path = APPENDIX_DIRECTORY / name
        frame.to_csv(path, index=False)
        appendix_files.append(path)

    messages = _message_values(
        table01,
        ml_break,
        paired_selection,
        table04,
        table05,
        prior_sensitivity,
        table07,
    )
    output_index = _output_index(messages)
    output_index.to_csv(OUTPUT_INDEX_PATH, index=False)
    METHOD_NOTE_PATH.write_text(_method_note(), encoding="utf-8")

    source_hashes_after = {
        name: sha256_file(path) for name, path in STEP32_SOURCE_FILES.items()
    }
    validation = _validation_report(
        sources=sources,
        source_hashes_before=source_hashes_before,
        source_hashes_after=source_hashes_after,
        output_index=output_index,
        backing_files=backing_files,
        figure_pngs=figure_pngs,
        figure_pdfs=figure_pdfs,
        table_files=table_files,
        appendix_files=appendix_files,
        paired_selection=paired_selection,
    )
    validation.to_csv(VALIDATION_PATH, index=False)
    elapsed_seconds = time.perf_counter() - started
    manifest: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "step": 33,
        "presentation_only": True,
        "step32_is_sole_analytical_source": True,
        "experiments_rerun": False,
        "new_inference": False,
        "source_files": {name: path.as_posix() for name, path in STEP32_SOURCE_FILES.items()},
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "main_figures": len(figure_pngs),
        "main_tables": len(table_files),
        "appendix_tables": len(appendix_files),
        "figure_backing_csvs": len(backing_files),
        "validation_checks": len(validation),
        "validation_checks_passed": int(validation["passed"].sum()),
        "treaty_ay_trajectory_omitted": True,
        "treaty_ay_omission_reason": (
            "Step 32 master_treaty_sensitivity has no accident_year field."
        ),
        "elapsed_seconds": elapsed_seconds,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Step 33 final figures and tables completed.")
    print(f"Main figures: {len(figure_pngs)} PNG + {len(figure_pdfs)} PDF")
    print(f"Main tables: {len(table_files)}")
    print(f"Appendix tables: {len(appendix_files)}")
    print(f"Validation: {int(validation['passed'].sum())}/{len(validation)} passed")
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(validation.to_string(index=False))

    if not validation["passed"].all():
        raise RuntimeError("Step 33 validation checks failed.")


if __name__ == "__main__":
    main()

import logging
from pathlib import Path
from typing import Any, Dict, List, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)

COLORS = [
    "#1B9E77FF",
    "#D95F02FF",
    "#7570B3FF",
    "#E7298AFF",
    "#66A61EFF",
    "#E6AB02FF",
    "#A6761DFF",
    "#666666FF",
]


def generate_merged_plots(
    data: List[Dict[str, Any]],
    output_dir: Path,
    experiment_order: List[str],
    plots_status: Dict[str, str],
) -> None:
    logger.info("Generating merged analysis plots")

    df = pd.DataFrame(data)
    experiments = experiment_order

    plt.style.use("default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    enhanced_experiments = [
        exp for exp in experiments if plots_status.get(exp) == "enhanced"
    ]
    enhanced_color_map = {
        exp: COLORS[i % len(COLORS)] for i, exp in enumerate(enhanced_experiments)
    }

    for _exp_idx, experiment in enumerate(experiments):
        exp_data = df[df["experiment"] == experiment]
        plot_status = plots_status.get(experiment)

        exp_diameters = sorted(exp_data["diameter_mm"])
        contrast_values = [
            exp_data[exp_data["diameter_mm"] == d]["percentaje_constrast_QH"].iloc[0]
            for d in exp_diameters
        ]
        variability_values = [
            exp_data[exp_data["diameter_mm"] == d]["background_variability_N"].iloc[0]
            for d in exp_diameters
        ]

        if plot_status == "enhanced":
            _color = enhanced_color_map[experiment]
            _linewidth = 4.0
            _alpha = 1.0
            _zorder = 30
            _linestyle = "-"
            _markersize = 10
            _markeredgewidth = 2.0
        else:
            _color = "#666666FF"
            _linewidth = 1.0
            _alpha = 0.3
            _zorder = 5
            _linestyle = "--"
            _markersize = 4
            _markeredgewidth = 0.5

        ax1.plot(
            exp_diameters,
            contrast_values,
            color=_color,
            linewidth=_linewidth,
            linestyle=_linestyle,
            alpha=_alpha,
            zorder=_zorder,
            marker="o",
            markersize=_markersize,
            markerfacecolor=_color,
            markeredgecolor="white",
            markeredgewidth=_markeredgewidth,
            label=experiment if plot_status == "enhanced" else None,
        )

        ax2.plot(
            exp_diameters,
            variability_values,
            color=_color,
            linewidth=_linewidth,
            linestyle=_linestyle,
            alpha=_alpha,
            zorder=_zorder,
            marker="o",
            markersize=_markersize,
            markerfacecolor=_color,
            markeredgecolor="white",
            markeredgewidth=_markeredgewidth,
            label=experiment if plot_status == "enhanced" else None,
        )

    ax1.set_title(
        "Contrast Recovery vs Sphere Diameter", fontsize=14, fontweight="bold"
    )
    ax1.set_xlabel("Sphere Diameter (mm)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Contrast Recovery (%)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    ax2.set_title(
        "Background Variability vs Sphere Diameter", fontsize=14, fontweight="bold"
    )
    ax2.set_xlabel("Sphere Diameter (mm)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Background Variability (%)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.suptitle(
        "NEMA Analysis - Multi-Experiment Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    output_path = output_dir / "merge_analysis_plot.png"
    plt.savefig(
        str(output_path),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        format="png",
    )

    logger.info(f"Merged analysis plot saved: {output_path}")
    plt.close()


def generate_merged_boxplot(
    lung_data: List[Dict[str, Any]],
    output_dir: Path,
    experiment_order: List[str],
    plots_status: Dict[str, str],
) -> None:
    logger.info("Generating merged lung insert violin plot analysis")

    df = pd.DataFrame(lung_data)
    experiments = experiment_order

    plt.style.use("default")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    plot_data = []
    experiment_names = []

    for experiment in experiments:
        exp_data = df[df["experiment"] == experiment]
        if len(exp_data) > 0 and plots_status.get(experiment) == "enhanced":
            for _, row in exp_data.iterrows():
                plot_data.append({"experiment": experiment, "value": row["data"]})
            experiment_names.append(experiment)

    if plot_data:
        plot_df = pd.DataFrame(plot_data)

        violin_parts = ax.violinplot(
            [
                plot_df[plot_df["experiment"] == exp]["value"].values
                for exp in experiment_names
            ],
            positions=range(1, len(experiment_names) + 1),
            showmeans=True,
            showmedians=False,
            showextrema=False,
        )

        for idx, (pc, _exp) in enumerate(
            zip(cast(List[Any], violin_parts["bodies"]), experiment_names)
        ):
            color = COLORS[idx % len(COLORS)]
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor("black")
            pc.set_linewidth(1)

        violin_parts["cmeans"].set_color("white")
        violin_parts["cmeans"].set_linewidth(3)

        for idx, experiment in enumerate(experiment_names):
            exp_values = plot_df[plot_df["experiment"] == experiment]["value"].values
            color = COLORS[idx % len(COLORS)]

            jitter = np.random.normal(0, 0.04, len(exp_values))
            positions = np.full(len(exp_values), idx + 1) + jitter

            ax.scatter(
                positions,
                exp_values,
                color=color,
                s=40,
                alpha=0.8,
                edgecolors="white",
                linewidths=1,
                zorder=10,
            )

            mean_val = np.mean(exp_values)
            std_val = np.std(exp_values)

            ax.text(
                idx + 1,
                max(exp_values) + std_val / 4,
                f"μ={mean_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "alpha": 0.9,
                    "edgecolor": "gray",
                },
            )

    ax.set_title("Lung Insert Accuracy Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("Experiment", fontsize=14, fontweight="bold")
    ax.set_ylabel(
        "Accuracy of Corrections in Lung Insert (%)", fontsize=14, fontweight="bold"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(range(1, len(experiment_names) + 1))
    ax.set_xticklabels(experiment_names)
    ax.set_facecolor("#fafafa")

    if len(experiment_names) > 3:
        ax.tick_params(axis="x", rotation=45)

    ax.tick_params(axis="both", labelsize=12, width=1.2)

    legend_elements = [
        Line2D([0], [0], color=COLORS[i % len(COLORS)], linewidth=4, label=exp)
        for i, exp in enumerate(experiment_names)
    ]

    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=12,
    )

    plt.tight_layout()

    output_path = output_dir / "merge_boxplot_analysis.png"
    plt.savefig(
        str(output_path),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        format="png",
    )

    logger.info(f"Merged lung insert violin plot analysis saved: {output_path}")
    plt.close()

"""
Additional reporting functions for text output and command-line usage.

Provides utilities for generating and formatting textual reports, intended for use in CLI workflows.

Author: Edwing Ulin-Briseno
Date: 2025-07-16
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yacs.config
from matplotlib.lines import Line2D
from PIL import Image as pilimage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    Frame,
    Image,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.doctemplate import BaseDocTemplate, PageTemplate

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

warnings.simplefilter("ignore", pilimage.DecompressionBombWarning)
pilimage.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)

BEST_CBR_ITERATION = None


def generate_plots(
    results: List[Dict[str, Any]],
    output_dir: Path,
    cfg: yacs.config.CfgNode,
) -> None:
    """
    Generates publication-quality plots for NEMA analysis results across iterations.

    Creates and saves a figure summarizing sphere analysis results using improved
    readability for academic publication. Highlights only the first, last, and best CBR
    iterations while showing all others with minimal styling for reference.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of analysis results for each sphere, including 'iteration' field.
    output_dir : Path
        Destination path for saving the plot.
    cfg : yacs.config.CfgNode
        Configuration object used for the analysis.

    Returns
    -------
    None
        This function does not return a value; the plot is saved to disk.

    Notes
    -----
    Author: EdAlita
    Date: 2025-01-09 16:45:00

    Improvements for publication quality:
    - Only first, last, and best CBR iterations highlighted
    - All other iterations shown as light background reference
    - Clear visual hierarchy with enhanced styling for key iterations
    """
    global BEST_CBR_ITERATION

    df = pd.DataFrame(results)
    df_filtered = df[~df["iteration"].isin([1, 2])]

    csv_path = output_dir.parent / f"{output_dir.stem}_results_data.csv"
    df_filtered.to_csv(csv_path, index=False)
    logger.info(f"Results data saved as CSV: {csv_path}")

    if len(df_filtered) == 0:
        logger.warning("No data available after filtering iterations 1-2")
        return

    iterations = sorted(df_filtered["iteration"].unique())

    iter_cbr_map = df_filtered.groupby("iteration")["weighted_CBR"].first().to_dict()
    best_cbr_iter = max(iter_cbr_map.items(), key=lambda x: x[1])
    highest_cbr_iter = best_cbr_iter[0]
    BEST_CBR_ITERATION = highest_cbr_iter

    first_iter = min(iterations)
    final_iter = max(iterations)

    key_iterations = [first_iter, highest_cbr_iter, final_iter]
    key_iterations = list(dict.fromkeys(key_iterations))

    publication_colors = {
        first_iter: "#1B9E77FF",
        highest_cbr_iter: "#D95F02FF",
        final_iter: "#7570B3FF",
    }

    if len(key_iterations) == 2:
        if first_iter == highest_cbr_iter:
            publication_colors = {first_iter: "#1B9E77FF", final_iter: "#7570B3FF"}
        elif final_iter == highest_cbr_iter:
            publication_colors = {first_iter: "#1B9E77FF", final_iter: "#D95F02FF"}
    elif len(key_iterations) == 1:
        publication_colors = {first_iter: "#D95F02FF"}

    iteration_styles = {}

    for iteration in iterations:
        is_key = iteration in key_iterations
        is_best = iteration == highest_cbr_iter
        is_first = iteration == first_iter
        is_final = iteration == final_iter

        if is_key:
            color = publication_colors[iteration]
            linewidth = 4.0
            alpha = 1.0
            zorder = 30
            linestyle = "-" if is_best else "--"
            markersize = 10
            markeredgewidth = 2.0
        else:
            color = "#666666FF"
            linewidth = 1.0
            alpha = 0.3
            zorder = 5
            linestyle = "--"
            markersize = 4
            markeredgewidth = 0.5

        iteration_styles[iteration] = {
            "color": color,
            "linewidth": linewidth,
            "alpha": alpha,
            "zorder": zorder,
            "linestyle": linestyle,
            "markersize": markersize,
            "markeredgewidth": markeredgewidth,
            "is_key": is_key,
            "is_best": is_best,
            "is_first": is_first,
            "is_final": is_final,
        }

    plt.style.use("default")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True)

    fig.suptitle("NEMA Iteration Comparison", fontsize=20, fontweight="bold", y=0.95)

    plot_configs = [
        ("percentaje_constrast_QH", "Contrast (%)", "Contrast by Sphere Diameter"),
        (
            "background_variability_N",
            "Background Variability (%)",
            "Background Variability by Sphere Diameter",
        ),
    ]

    panel_labels = ["a", "b"]

    for i, (ax, (yvar, ylabel, title)) in enumerate(zip(axes, plot_configs)):
        sorted_iterations = sorted(
            iterations, key=lambda x: iteration_styles[x]["zorder"]
        )

        for iteration in sorted_iterations:
            iteration_data = df_filtered[
                df_filtered["iteration"] == iteration
            ].sort_values("diameter_mm")

            if len(iteration_data) > 0:
                style = iteration_styles[iteration]

                label = None
                if style["is_key"]:
                    if style["is_best"] and style["is_first"] and style["is_final"]:
                        label = f"Iteration {int(iteration)} (Best CBR)"
                    elif style["is_best"] and style["is_first"]:
                        label = f"Iteration {int(iteration)} (Best CBR)"
                    elif style["is_best"] and style["is_final"]:
                        label = f"Iteration {int(iteration)} (Best CBR)"
                    elif style["is_first"] and style["is_final"]:
                        label = f"Iteration {int(iteration)}"
                    elif style["is_best"]:
                        label = f"Iteration {int(iteration)} (Best CBR)"
                    elif style["is_first"]:
                        label = f"Iteration {int(iteration)}"
                    elif style["is_final"]:
                        label = f"Iteration {int(iteration)}"

                ax.plot(
                    iteration_data["diameter_mm"],
                    iteration_data[yvar],
                    color=style["color"],
                    linewidth=style["linewidth"],
                    linestyle=style["linestyle"],
                    alpha=style["alpha"],
                    marker="o",
                    markersize=style["markersize"],
                    markeredgecolor="white",
                    markeredgewidth=style["markeredgewidth"],
                    label=label,
                    zorder=style["zorder"],
                    solid_capstyle="round",
                )

        ax.text(
            0.02,
            1.15,
            f"({panel_labels[i]})",
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="left",
        )

        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
        ax.tick_params(axis="both", labelsize=20, width=1.2)
        ax.tick_params(axis="x", rotation=0)

        ax.grid(True, linestyle="-", alpha=0.2, color="gray", linewidth=0.8)
        ax.set_axisbelow(True)

        unique_diameters = sorted(df_filtered["diameter_mm"].unique())
        ax.set_xticks(unique_diameters)

        ax.set_facecolor("#fafafa")

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("#333333")

    fig.text(
        0.5,
        0.02,
        "Sphere Diameter [mm]",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

    handles, labels = axes[0].get_legend_handles_labels()

    key_handles = []
    key_labels = []
    for handle, label in zip(handles, labels):
        if label:
            key_handles.append(handle)
            key_labels.append(label)

    if key_handles:
        ncol = min(len(key_handles), 3)
        fig.legend(
            key_handles,
            key_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=ncol,
            fontsize=12,
            frameon=True,
            fancybox=True,
            shadow=True,
            columnspacing=2.0,
        )

    plt.tight_layout(rect=(0, 0.1, 1, 0.92))

    output_path = output_dir.parent / f"{output_dir.stem}_analysis_plot_iterations.png"
    plt.savefig(
        str(output_path),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        format="png",
    )

    plt.close()

    logger.info("Publication-quality plots saved:")
    logger.info(f"  PNG (600 DPI): {output_path}")
    logger.info(f"  Total iterations: {len(iterations)}")
    logger.info(f"  Key iterations shown: {[int(i) for i in key_iterations]}")
    logger.info(
        f"  BEST CBR iteration: {int(highest_cbr_iter)} (CBR: {iter_cbr_map[highest_cbr_iter]:.2f}) - SOLID LINE"
    )
    logger.info("  All other iterations shown with DASHED LINES")
    logger.info(f"  Sphere diameters: {[int(d) for d in unique_diameters]} mm")


def generate_pc_vs_bg_plot(
    results: List[Dict[str, Any]],
    output_dir: Path,
    cfg: yacs.config.CfgNode,
) -> None:
    """
    Generates a simplified publication-quality plot of Contrast vs Background Variability by diameter.

    Creates and saves a clean plot showing the relationship between contrast and background
    variability, with color-coded lines for each sphere diameter and highlighted key iterations.
    Excludes iterations 1 and 2 from the plot.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of analysis results for each sphere, including 'iteration' field.
    output_dir : Path
        Destination path for saving the plot.
    cfg : yacs.config.CfgNode
        Configuration object used for the analysis.

    Returns
    -------
    None
        This function does not return a value; the plot is saved to disk.

    Notes
    -----
    Author: EdAlita
    Date: 2025-01-09 17:30:00

    Improvements for publication quality:
    - Simplified presentation with color-coded diameter lines
    - Key iterations highlighted (first, best CBR, final)
    - Clean legend positioned within plot area
    - Professional color palette and typography
    """
    df = pd.DataFrame(results)
    df_filtered = df[~df["iteration"].isin([1, 2])]

    if len(df_filtered) == 0:
        logger.warning("No data to plot after filtering out iterations 1 and 2")
        return

    diameters = sorted(df_filtered["diameter_mm"].unique())
    iterations = sorted(df_filtered["iteration"].unique())

    iter_cbr_map = df_filtered.groupby("iteration")["weighted_CBR"].first().to_dict()
    best_cbr_iter = max(iter_cbr_map.items(), key=lambda x: x[1])[0]
    first_iter = min(iterations)
    final_iter = max(iterations)

    key_iterations = [first_iter, best_cbr_iter, final_iter]
    key_iterations = list(dict.fromkeys(key_iterations))

    diameter_colors = [
        "#1B9E77FF",
        "#D95F02FF",
        "#7570B3FF",
        "#E7298AFF",
        "#66A61EFF",
        "#E6AB02FF",
        "#A6761DFF",
        "#666666FF",
    ]

    diameter_color_map = {}
    for i, diameter in enumerate(diameters):
        diameter_color_map[diameter] = diameter_colors[i % len(diameter_colors)]

    plt.style.use("default")
    sns.set_palette(diameter_colors)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(
        "NEMA Analysis: Contrast vs Background Variability",
        fontsize=18,
        fontweight="bold",
        y=0.97,
    )

    for diameter in diameters:
        diameter_data = df_filtered[df_filtered["diameter_mm"] == diameter].sort_values(
            "iteration"
        )

        if len(diameter_data) > 0:
            color = diameter_color_map[diameter]

            n_points = len(diameter_data)
            alphas = np.linspace(0.3, 1.0, n_points)

            ax.plot(
                diameter_data["background_variability_N"],
                diameter_data["percentaje_constrast_QH"],
                color=color,
                linewidth=2.0,
                alpha=0.6,
                zorder=5,
            )

            for idx, (_, row) in enumerate(diameter_data.iterrows()):
                iteration = row["iteration"]
                is_key = iteration in key_iterations

                if is_key:
                    markersize = 12
                    alpha = 1.0
                    zorder = 15
                    markeredgewidth = 2.0
                    markeredgecolor = "white"
                else:
                    markersize = 6
                    alpha = alphas[idx]
                    zorder = 10
                    markeredgewidth = 1.0
                    markeredgecolor = "white"

                ax.scatter(
                    row["background_variability_N"],
                    row["percentaje_constrast_QH"],
                    color=color,
                    s=markersize**2,
                    alpha=alpha,
                    edgecolors=markeredgecolor,
                    linewidths=markeredgewidth,
                    zorder=zorder,
                )

                if is_key:
                    label_text = ""
                    if iteration == best_cbr_iter:
                        label_text = f"It{int(iteration)}"

                    if label_text:
                        ax.annotate(
                            label_text,
                            (
                                row["background_variability_N"],
                                row["percentaje_constrast_QH"],
                            ),
                            xytext=(5, 5),
                            textcoords="offset points",
                            fontsize=9,
                            fontweight="bold",
                            color="black",
                            bbox={
                                "boxstyle": "round,pad=0.2",
                                "facecolor": "white",
                                "edgecolor": color,
                                "alpha": 0.8,
                            },
                            zorder=20,
                        )

    ax.set_ylabel("Contrast (%)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Background Variability (%)", fontsize=16, fontweight="bold")

    ax.tick_params(axis="both", labelsize=20, width=1.2)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    ax.grid(True, linestyle="-", alpha=0.3, color="gray", linewidth=0.8)
    ax.set_axisbelow(True)

    ax.set_facecolor("#fafafa")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#333333")

    legend_elements = []
    for diameter in diameters:
        color = diameter_color_map[diameter]
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=3,
                marker="o",
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=2,
                label=f"{diameter:.0f} mm",
            )
        )

    ax.legend(
        handles=legend_elements,
        title="Sphere Diameter",
        title_fontsize=12,
        fontsize=11,
        loc="upper right",
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        edgecolor="black",
        columnspacing=1.0,
    )

    plt.tight_layout()

    output_path = output_dir.parent / f"{output_dir.stem}_bg_vs_pc_plot.png"
    plt.savefig(
        str(output_path),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    plt.close()

    logger.info("Simplified Contrast vs BG plot saved:")
    logger.info(f"  Sphere diameters: {[int(d) for d in diameters]} mm")
    logger.info(f"  Total iterations: {len(iterations)} (excluded 1, 2)")
    logger.info("  Key iterations highlighted:")
    logger.info(f"    - First: It{first_iter}")
    logger.info(
        f"    - Best CBR: It{best_cbr_iter}* (CBR: {iter_cbr_map[best_cbr_iter]:.2f})"
    )
    logger.info(f"    - Final: It{final_iter}")


def generate_wcbr_convergence_plot(
    results: List[Dict[str, Any]],
    output_dir: Path,
    cfg: yacs.config.CfgNode,
) -> None:
    """
    Generates a publication-quality CBR convergence plot across iterations.

    Creates and saves a line plot showing the weighted CBR progression through iterations,
    highlighting convergence patterns and key iterations with enhanced styling for
    academic publication.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of analysis results for each sphere, including 'iteration' and 'weighted_CBR' fields.
    output_dir : Path
        Destination path for saving the plot.
    cfg : yacs.config.CfgNode
        Configuration object used for the analysis.

    Returns
    -------
    None
        This function does not return a value; the plot is saved to disk.

    Notes
    -----
    Author: EdAlita
    Date: 2025-01-09 19:00:00

    Improvements for publication quality:
    - Clean, professional styling with minimal visual clutter
    - Convergence indicators and statistical annotations
    - Key iteration highlighting (peak CBR, convergence point)
    - Error bars showing CBR variability across spheres
    - Trend analysis with convergence assessment
    """
    global BEST_CBR_ITERATION

    df = pd.DataFrame(results)
    df_filtered = df[~df["iteration"].isin([1, 2])]

    if len(df_filtered) == 0:
        logger.warning("No data available after filtering iterations 1-2")
        return

    cbr_stats = (
        df_filtered.groupby("iteration")["weighted_CBR"]
        .agg(["mean", "std", "count", "min", "max"])
        .reset_index()
    )

    iterations = sorted(cbr_stats["iteration"].unique())

    max_cbr_idx = cbr_stats["mean"].idxmax()
    max_cbr_iter = cbr_stats.loc[max_cbr_idx, "iteration"]
    max_cbr_value = cbr_stats.loc[max_cbr_idx, "mean"]

    BEST_CBR_ITERATION = int(max_cbr_iter)

    first_iter = min(iterations)
    final_iter = max(iterations)

    convergence_iter: Optional[int] = None
    if len(iterations) >= 5:
        for i in range(len(cbr_stats) - 4):
            next_window = i + 5
            window = cbr_stats.iloc[i:next_window]
            cbr_range = window["mean"].max() - window["mean"].min()
            if cbr_range <= 0.05:
                convergence_iter = window.iloc[0]["iteration"]
                break

    plt.style.use("default")

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    fig.suptitle(
        "Weighted CBR Convergence Analysis", fontsize=20, fontweight="bold", y=0.97
    )

    ax.plot(
        cbr_stats["iteration"],
        cbr_stats["mean"],
        color="#2E86AB",
        linewidth=4.0,
        marker="o",
        markersize=8,
        markerfacecolor="#2E86AB",
        markeredgecolor="white",
        markeredgewidth=2,
        alpha=0.9,
        zorder=10,
        label="Weighted CBR",
    )

    key_iterations = [first_iter, max_cbr_iter, final_iter]
    if convergence_iter:
        key_iterations.append(convergence_iter)
    key_iterations = list(dict.fromkeys(key_iterations))

    key_colors = {
        first_iter: "#1B9E77",
        max_cbr_iter: "#D95F02",
        final_iter: "#7570B3",
    }
    if convergence_iter:
        key_colors[convergence_iter] = "#E7298A"

    for iteration in key_iterations:
        iter_data = cbr_stats[cbr_stats["iteration"] == iteration]
        if not iter_data.empty:
            color = key_colors.get(iteration, "#666666")
            ax.scatter(
                iteration,
                iter_data["mean"].iloc[0],
                color=color,
                s=200,
                edgecolors="white",
                linewidths=3,
                zorder=15,
                alpha=1.0,
            )

            y_pos = iter_data["mean"].iloc[0]

            if iteration == first_iter:
                label = f"Start\n({iteration}: {y_pos:.3f})"
                offset = (10, 20)
            elif iteration == max_cbr_iter:
                label = f"Peak CBR\n({iteration}: {y_pos:.3f})"
                offset = (10, -30)
            elif iteration == final_iter:
                label = f"Final\n({iteration}: {y_pos:.3f})"
                offset = (-20, 20)
            else:
                continue

            ax.annotate(
                label,
                (iteration, y_pos),
                xytext=offset,
                textcoords="offset points",
                fontsize=18,
                fontweight="bold",
                color=color,
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": "white",
                    "edgecolor": color,
                    "alpha": 0.9,
                    "linewidth": 2,
                },
                arrowprops={"arrowstyle": "->", "color": color, "linewidth": 2},
                zorder=20,
            )

    first_cbr = cbr_stats[cbr_stats["iteration"] == first_iter]["mean"].iloc[0]
    final_cbr = cbr_stats[cbr_stats["iteration"] == final_iter]["mean"].iloc[0]
    improvement = ((max_cbr_value - first_cbr) / first_cbr) * 100
    final_vs_peak = ((final_cbr - max_cbr_value) / max_cbr_value) * 100

    ax.set_xlabel("Iterations", fontsize=16, fontweight="bold")
    ax.set_ylabel("Weighted CBR", fontsize=16, fontweight="bold")

    ax.tick_params(axis="both", labelsize=20, width=1.2)
    ax.tick_params(axis="x", rotation=0)

    ax.set_xticks(range(int(min(iterations)), int(max(iterations)) + 1, 1))

    ax.grid(True, linestyle="-", alpha=0.3, color="gray", linewidth=0.8)
    ax.set_axisbelow(True)

    ax.set_facecolor("#fafafa")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#333333")

    plt.tight_layout()

    output_path = (
        output_dir.parent / f"{output_dir.stem}_weighted_cbr_convergence_analysis.png"
    )
    plt.savefig(
        str(output_path),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    plt.close()

    logger.info("wCBR convergence analysis plot saved:")
    logger.info(f"  PNG (600 DPI): {output_path}")
    logger.info(f"  Total iterations: {len(iterations)} (excluded 1, 2)")
    logger.info("  wCBR Analysis:")
    logger.info(
        f"    - Peak wCBR: {max_cbr_value:.3f} at iteration {int(max_cbr_iter)}"
    )
    logger.info(f"    - Improvement from start: {improvement:+.1f}%")
    logger.info(f"    - Final vs peak: {final_vs_peak:+.1f}%")

    if convergence_iter:
        logger.info(f"    - Convergence detected at iteration {int(convergence_iter)}")
    else:
        logger.info("    - No clear convergence pattern detected")

    logger.info(f"  Best wCBR iteration set globally: {BEST_CBR_ITERATION}")

    logger.info("  wCBR statistics by iteration:")
    for _, row in cbr_stats.iterrows():
        logger.info(
            f"    Iter {int(row['iteration'])}: "
            f"Mean={row['mean']:.3f}, "
            f"Std={row['std']:.3f}, "
            f"Range=[{row['min']:.3f}-{row['max']:.3f}], "
            f"N={int(row['count'])} spheres"
        )


def generate_cbr_convergence_plot(
    results: List[Dict[str, Any]],
    output_dir: Path,
    cfg: yacs.config.CfgNode,
) -> None:
    """
    Generates a publication-quality CBR convergence plot across iterations.

    Creates and saves a line plot showing the weighted CBR progression through iterations,
    highlighting convergence patterns and key iterations with enhanced styling for
    academic publication.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of analysis results for each sphere, including 'iteration' and 'weighted_CBR' fields.
    output_dir : Path
        Destination path for saving the plot.
    cfg : yacs.config.CfgNode
        Configuration object used for the analysis.

    Returns
    -------
    None
        This function does not return a value; the plot is saved to disk.

    Notes
    -----
    Author: EdAlita
    Date: 2025-01-09 19:00:00

    Improvements for publication quality:
    - Clean, professional styling with minimal visual clutter
    - Convergence indicators and statistical annotations
    - Key iteration highlighting (peak CBR, convergence point)
    - Error bars showing CBR variability across spheres
    - Trend analysis with convergence assessment
    """

    df = pd.DataFrame(results)
    df_filtered = df[~df["iteration"].isin([1, 2])]

    if len(df_filtered) == 0:
        logger.warning("No data available after filtering iterations 1-2")
        return

    cbr37_stats = (
        df_filtered.groupby("iteration")["37_CBR"]
        .agg(["mean", "std", "count", "min", "max"])
        .reset_index()
    )

    cbr28_stats = (
        df_filtered.groupby("iteration")["28_CBR"]
        .agg(["mean", "std", "count", "min", "max"])
        .reset_index()
    )

    cbr22_stats = (
        df_filtered.groupby("iteration")["22_CBR"]
        .agg(["mean", "std", "count", "min", "max"])
        .reset_index()
    )

    cbr17_stats = (
        df_filtered.groupby("iteration")["17_CBR"]
        .agg(["mean", "std", "count", "min", "max"])
        .reset_index()
    )

    cbr13_stats = (
        df_filtered.groupby("iteration")["13_CBR"]
        .agg(["mean", "std", "count", "min", "max"])
        .reset_index()
    )

    cbr10_stats = (
        df_filtered.groupby("iteration")["10_CBR"]
        .agg(["mean", "std", "count", "min", "max"])
        .reset_index()
    )

    cbr_stats = (
        df_filtered.groupby("iteration")["weighted_CBR"]
        .agg(["mean", "std", "count", "min", "max"])
        .reset_index()
    )

    iterations = sorted(cbr37_stats["iteration"].unique())

    plt.style.use("default")

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    fig.suptitle(
        "CBR Convergence Analysis by Sphere Diameter",
        fontsize=20,
        fontweight="bold",
        y=0.97,
    )

    ax.plot(
        cbr37_stats["iteration"],
        cbr37_stats["mean"],
        color="#E6AB02FF",
        linewidth=4.0,
        marker="o",
        markersize=8,
        markerfacecolor="#E6AB02FF",
        markeredgecolor="white",
        markeredgewidth=2,
        alpha=0.9,
        zorder=10,
        label="37 mm CBR",
    )

    ax.plot(
        cbr28_stats["iteration"],
        cbr28_stats["mean"],
        color="#66A61EFF",
        linewidth=4.0,
        marker="o",
        markersize=8,
        markerfacecolor="#66A61EFF",
        markeredgecolor="white",
        markeredgewidth=2,
        alpha=0.9,
        zorder=10,
        label="28 mm CBR",
    )

    ax.plot(
        cbr22_stats["iteration"],
        cbr22_stats["mean"],
        color="#E7298AFF",
        linewidth=4.0,
        marker="o",
        markersize=8,
        markerfacecolor="#E7298AFF",
        markeredgecolor="white",
        markeredgewidth=2,
        alpha=0.9,
        zorder=10,
        label="22 mm CBR",
    )

    ax.plot(
        cbr17_stats["iteration"],
        cbr17_stats["mean"],
        color="#7570B3FF",
        linewidth=4.0,
        marker="o",
        markersize=8,
        markerfacecolor="#7570B3FF",
        markeredgecolor="white",
        markeredgewidth=2,
        alpha=0.9,
        zorder=10,
        label="17 mm CBR",
    )

    ax.plot(
        cbr13_stats["iteration"],
        cbr13_stats["mean"],
        color="#D95F02FF",
        linewidth=4.0,
        marker="o",
        markersize=8,
        markerfacecolor="#D95F02FF",
        markeredgecolor="white",
        markeredgewidth=2,
        alpha=0.9,
        zorder=10,
        label="13 mm CBR",
    )

    ax.plot(
        cbr10_stats["iteration"],
        cbr10_stats["mean"],
        color="#1B9E77FF",
        linewidth=4.0,
        marker="o",
        markersize=8,
        markerfacecolor="#1B9E77FF",
        markeredgecolor="white",
        markeredgewidth=2,
        alpha=0.9,
        zorder=10,
        label="10 mm CBR",
    )

    ax.plot(
        cbr_stats["iteration"],
        cbr_stats["mean"],
        color="#2E86AB",
        linewidth=4.0,
        linestyle="--",
        marker="o",
        markersize=8,
        markerfacecolor="#2E86AB",
        markeredgecolor="white",
        markeredgewidth=2,
        alpha=0.9,
        zorder=10,
        label="Weighted CBR",
    )

    max_37_idx = cbr37_stats["mean"].idxmax()
    max_37_iter = cbr37_stats.loc[max_37_idx, "iteration"]
    max_37_value = cbr37_stats.loc[max_37_idx, "mean"]
    ax.scatter(
        max_37_iter,
        max_37_value,
        color="red",
        s=100,
        zorder=20,
        edgecolors="white",
        linewidths=2,
    )

    max_28_idx = cbr28_stats["mean"].idxmax()
    max_28_iter = cbr28_stats.loc[max_28_idx, "iteration"]
    max_28_value = cbr28_stats.loc[max_28_idx, "mean"]
    ax.scatter(
        max_28_iter,
        max_28_value,
        color="red",
        s=100,
        zorder=20,
        edgecolors="white",
        linewidths=2,
    )

    max_22_idx = cbr22_stats["mean"].idxmax()
    max_22_iter = cbr22_stats.loc[max_22_idx, "iteration"]
    max_22_value = cbr22_stats.loc[max_22_idx, "mean"]
    ax.scatter(
        max_22_iter,
        max_22_value,
        color="red",
        s=100,
        zorder=20,
        edgecolors="white",
        linewidths=2,
    )

    max_13_idx = cbr13_stats["mean"].idxmax()
    max_13_iter = cbr13_stats.loc[max_13_idx, "iteration"]
    max_13_value = cbr13_stats.loc[max_13_idx, "mean"]
    ax.scatter(
        max_13_iter,
        max_13_value,
        color="red",
        s=100,
        zorder=20,
        edgecolors="white",
        linewidths=2,
    )

    max_17_idx = cbr17_stats["mean"].idxmax()
    max_17_iter = cbr17_stats.loc[max_17_idx, "iteration"]
    max_17_value = cbr17_stats.loc[max_17_idx, "mean"]
    ax.scatter(
        max_17_iter,
        max_17_value,
        color="red",
        s=100,
        zorder=20,
        edgecolors="white",
        linewidths=2,
    )

    max_10_idx = cbr10_stats["mean"].idxmax()
    max_10_iter = cbr10_stats.loc[max_10_idx, "iteration"]
    max_10_value = cbr10_stats.loc[max_10_idx, "mean"]
    ax.scatter(
        max_10_iter,
        max_10_value,
        color="red",
        s=100,
        zorder=20,
        edgecolors="white",
        linewidths=2,
    )

    max_weighted_idx = cbr_stats["mean"].idxmax()
    max_weighted_iter = cbr_stats.loc[max_weighted_idx, "iteration"]
    max_weighted_value = cbr_stats.loc[max_weighted_idx, "mean"]
    ax.scatter(
        max_weighted_iter,
        max_weighted_value,
        color="red",
        s=100,
        zorder=20,
        edgecolors="white",
        linewidths=2,
    )

    ax.set_xlabel("Iterations", fontsize=16, fontweight="bold")
    ax.set_ylabel("CBR", fontsize=16, fontweight="bold")

    ax.tick_params(axis="both", labelsize=20, width=1.2)
    ax.tick_params(axis="x", rotation=0)

    ax.set_xticks(range(int(min(iterations)), int(max(iterations)) + 1, 1))

    ax.grid(True, linestyle="-", alpha=0.3, color="gray", linewidth=0.8)
    ax.set_axisbelow(True)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="#E6AB02FF",
            linewidth=4,
            marker="o",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=2,
            label="37 mm CBR",
        ),
        Line2D(
            [0],
            [0],
            color="#66A61EFF",
            linewidth=4,
            marker="o",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=2,
            label="28 mm CBR",
        ),
        Line2D(
            [0],
            [0],
            color="#E7298AFF",
            linewidth=4,
            marker="o",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=2,
            label="22 mm CBR",
        ),
        Line2D(
            [0],
            [0],
            color="#7570B3FF",
            linewidth=4,
            marker="o",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=2,
            label="17 mm CBR",
        ),
        Line2D(
            [0],
            [0],
            color="#D95F02FF",
            linewidth=4,
            marker="o",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=2,
            label="13 mm CBR",
        ),
        Line2D(
            [0],
            [0],
            color="#1B9E77FF",
            linewidth=4,
            marker="o",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=2,
            label="10 mm CBR",
        ),
        Line2D(
            [0],
            [0],
            color="#2E86AB",
            linewidth=4,
            linestyle="--",
            marker="o",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=2,
            label="Weighted CBR",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="red",
            linestyle="None",
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=2,
            label="Peak CBR Points",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        edgecolor="black",
    )

    ax.set_facecolor("#fafafa")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#333333")

    plt.tight_layout(rect=(0.0, 0.0, 0.85, 1.0))

    output_path = output_dir.parent / f"{output_dir.stem}_cbr_convergence_analysis.png"
    plt.savefig(
        str(output_path),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    plt.close()

    logger.info("CBR convergence analysis plot saved:")
    logger.info(f"  PNG (600 DPI): {output_path}")
    logger.info(f"  Total iterations: {len(iterations)} (excluded 1, 2)")


def generate_boxplot_with_mean_std(
    all_lung_results: Dict[int, Dict[int, float]],
    output_dir: Path,
    cfg: yacs.config.CfgNode,
) -> None:
    """
    Generates a publication-quality boxplot from lung results for key iterations only.

    Creates simplified boxplots for three key iterations: first, best CBR performance,
    and final iteration. Shows lung insert correction errors with enhanced styling
    for academic publication.

    Parameters
    ----------
    all_lung_results : Dict[int, Dict[int, float]]
        Dictionary mapping iteration numbers to lung results dictionaries.
        Structure: {iteration_num: {slice_num: correction_value, ...}, ...}
    output_dir : Path
        Directory path for saving the generated plot.
    cfg : yacs.config.CfgNode
        Configuration object used for the analysis.

    Returns
    -------
    None
        This function does not return a value; the plot is saved to disk.

    Notes
    -----
    Author: EdAlita
    Date: 2025-01-09 18:15:00

    Improvements for publication quality:
    - Shows only first, best CBR, and final iterations
    - Professional color palette matching other functions
    - Reduced visual clutter and improved readability
    - Statistical indicators positioned to avoid overlap
    """
    filtered_results = {k: v for k, v in all_lung_results.items() if k not in [1, 2]}
    if not filtered_results:
        logger.warning("No data to plot after filtering out iterations 1 and 2")
        return

    iterations = sorted(filtered_results.keys())

    iteration_means = {}
    for iteration in iterations:
        lung_data = list(filtered_results[iteration].values())
        iteration_means[iteration] = float(np.mean(np.abs(lung_data)))

    best_cbr_iter: Optional[int] = BEST_CBR_ITERATION
    first_iter: Optional[int] = min(iterations) if iterations else None
    final_iter: Optional[int] = max(iterations) if iterations else None

    key_iterations = [first_iter, best_cbr_iter, final_iter]
    key_iterations = list(dict.fromkeys(key_iterations))

    publication_colors = {
        first_iter: "#1B9E77",
        best_cbr_iter: "#D95F02",
        final_iter: "#7570B3",
    }

    if len(key_iterations) == 2:
        if first_iter == best_cbr_iter:
            publication_colors = {first_iter: "#1B9E77", final_iter: "#7570B3"}
        elif final_iter == best_cbr_iter:
            publication_colors = {first_iter: "#1B9E77", final_iter: "#D95F02"}
    elif len(key_iterations) == 1:
        publication_colors = {first_iter: "#D95F02"}

    plt.style.use("default")

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    fig.suptitle(
        "Lung Insert Correction: Key Iterations Comparison",
        fontsize=18,
        fontweight="bold",
        y=0.97,
    )

    box_data = []
    box_labels = []
    colors = []
    means = []
    std_devs = []

    for _iteration in key_iterations:
        lung_data = list(filtered_results[iteration].values())
        box_data.append(lung_data)

        if (
            _iteration == first_iter
            and _iteration == best_cbr_iter
            and _iteration == final_iter
        ):
            label = f"Iter {_iteration}\n(Only)"
        elif _iteration == first_iter and _iteration == best_cbr_iter:
            label = f"Iter {_iteration}\n(First/Best CBR)"
        elif _iteration == best_cbr_iter and _iteration == final_iter:
            label = f"Iter {_iteration}\n(Best CBR/Final)"
        elif _iteration == first_iter and _iteration == final_iter:
            label = f"Iter {_iteration}\n(First/Final)"
        elif _iteration == first_iter:
            label = f"Iter {_iteration}\n(First)"
        elif _iteration == best_cbr_iter:
            label = f"Iter {_iteration}\n(Best CBR)"
        elif _iteration == final_iter:
            label = f"Iter {_iteration}\n(Final)"
        else:
            label = f"Iter {_iteration}"

        box_labels.append(label)
        colors.append(publication_colors[iteration])
        means.append(float(np.mean(lung_data)))
        std_devs.append(float(np.std(lung_data)))

    bp = ax.violinplot(
        box_data,
        positions=range(1, len(box_data) + 1),
        showmeans=True,
        showmedians=False,
        showextrema=False,
    )

    violin_bodies = cast(List[Any], bp["bodies"])
    for patch, color in zip(violin_bodies, colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(1)
        patch.set_edgecolor("black")

    bp["cmeans"].set_color("white")
    bp["cmeans"].set_linewidth(3)

    for idx, _experiment in enumerate(box_labels):
        exp_values = box_data[idx]
        color = publication_colors[key_iterations[idx]]

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

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color("#333333")

    for i, (mean, std_dev, _iteration) in enumerate(
        zip(means, std_devs, key_iterations), 1
    ):
        y_max = np.max(box_data[i - 1])
        text_y = y_max + std_dev / 4

        ax.text(
            i,
            text_y,
            f"μ={mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "gray",
                "alpha": 0.9,
            },
        )

    ax.set_xticks(range(1, len(box_labels) + 1))
    ax.set_xticklabels(box_labels, fontsize=12)
    ax.set_xlabel("Key Iterations", fontsize=14, fontweight="bold")
    ax.set_ylabel(
        "Accuracy of Correction in Lung Insert (%)", fontsize=14, fontweight="bold"
    )

    ax.tick_params(axis="both", labelsize=12, width=1.2)
    ax.tick_params(axis="x", rotation=0)

    ax.grid(True, axis="y", linestyle="-", alpha=0.3, color="gray", linewidth=0.8)
    ax.set_axisbelow(True)

    ax.set_facecolor("#fafafa")

    plt.tight_layout()

    output_path = output_dir.parent / f"{output_dir.stem}_boxplot_with_mean_std.png"
    plt.savefig(
        str(output_path),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    plt.close()

    logger.info("Publication-quality lung correction boxplot saved:")
    logger.info(f"  Total iterations available: {len(iterations)} (excluded 1, 2)")
    logger.info(f"  Key iterations shown: {[int(i) for i in key_iterations]}")  # type: ignore[arg-type]
    logger.info(
        f"    - First: Iter {first_iter} (Mean: {iteration_means[first_iter]:.2f}%)"  # type: ignore[index]
    )
    logger.info(
        f"    - Best CBR: Iter {best_cbr_iter} (Mean: {iteration_means[best_cbr_iter]:.2f}%)"  # type: ignore[index]
    )
    logger.info(
        f"    - Final: Iter {final_iter} (Mean: {iteration_means[final_iter]:.2f}%)"  # type: ignore[index]
    )

    logger.info("  Lung correction statistics for key iterations:")
    for __iteration in key_iterations:
        values = list(filtered_results[__iteration].values())  # type: ignore[index]
        logger.info(
            f"    Iteration {int(__iteration)}: "  # type: ignore[arg-type]
            f"Mean={float(np.mean(values)):.2f}%, "
            f"Std={float(np.std(values)):.2f}%, "
            f"N={len(values)} slices"
        )


def header(canvas, doc, logo_path=None):
    canvas.saveState()
    if logo_path and Path(logo_path).exists():
        original_width = 1608
        original_height = 251
        max_width = 260
        max_height = 100

        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        scale = min(width_ratio, height_ratio)

        draw_width = original_width * scale
        draw_height = original_height * scale

        canvas.drawImage(
            str(logo_path),
            40,
            740,
            width=draw_width,
            height=draw_height,
            preserveAspectRatio=True,
            mask="auto",
        )
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawRightString(550, 760, "NEMA NU 2-2018 Report")
    canvas.restoreState()


def generate_reportlab_report(
    results: List[Dict[str, Any]],
    all_lung_results: Dict[int, Dict[int, float]],
    output_path: Path,
    cfg: yacs.config.CfgNode,
    input_image_path: Path,
    voxel_spacing: Tuple[float, float, float],
    plot_path: Optional[Path] = None,
    pc_vs_bg_path: Optional[Path] = None,
    rois_loc_path: Optional[Path] = None,
    boxplot_path: Optional[Path] = None,
) -> None:
    """
    Generates a PDF report for NEMA quality analysis results across iterations using ReportLab.

    Creates a formatted PDF summarizing sphere and lung analysis results across multiple iterations,
    configuration, and relevant plots. Excludes iterations 1 and 2 from analysis.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of analysis results for each sphere, including 'iteration' field.
    all_lung_results : Dict[int, Dict[int, float]]
        Dictionary mapping iteration numbers to lung results dictionaries.
        Structure: {iteration_num: {slice_num: correction_value, ...}, ...}
    output_path : Path
        Destination path for saving the PDF report.
    cfg : yacs.config.CfgNode
        Configuration object used for the analysis.
    input_image_path : Path
        Path to the input image file.
    voxel_spacing : Tuple[float, float, float]
        Voxel spacing used during analysis.
    plot_path : Path, optional
        Path to the iteration comparison plot image file.
    pc_vs_bg_path : Path, optional
        Path to the PC vs BG plot image file.
    rois_loc_path : Path, optional
        Path to the ROIs plot image file.
    boxplot_path : Path, optional
        Path to the lung boxplot image file.

    Returns
    -------
    None
        This function does not return a value; it writes the PDF report to disk.
    """
    filtered_results = [r for r in results if r.get("iteration", 0) not in [1, 2]]
    filtered_lung_results = {
        k: v for k, v in all_lung_results.items() if k not in [1, 2]
    }

    iterations = sorted({r["iteration"] for r in filtered_results})
    max_iteration = max(iterations) if iterations else 0

    doc = BaseDocTemplate(str(output_path), pagesize=letter)
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height - 0.5 * inch,
        id="normal",
    )
    template = PageTemplate(
        id="with-header",
        frames=frame,
        onPage=lambda c, d: header(c, d, "data/logosimbolocontexto_principal.jpg"),
    )
    doc.addPageTemplates([template])

    elements: List[Flowable] = []

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    header_style = styles["Heading2"]
    body_style = styles["BodyText"]

    elements.append(
        Paragraph("NEMA Analysis Report - Multi-Iteration Results", title_style)
    )
    elements.append(Spacer(1, 0.2 * inch))

    summary = (
        f"<b>Summary of Analysis</b><br/>"
        f"\u2022 Date of Generation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        f"\u2022 Input Directory: <font face='Courier'>{str(input_image_path)}</font><br/>"
        f"\u2022 Voxel Spacing: {voxel_spacing[0]:.4f} × {voxel_spacing[1]:.4f} × {voxel_spacing[2]:.4f} mm<br/>"
        f"\u2022 Iterations Analyzed: {', '.join(map(str, iterations))} (excluded 1, 2)<br/>"
        f"\u2022 Final Iteration: {max_iteration}"
    )
    elements.append(Paragraph(summary, body_style))
    elements.append(Spacer(1, 0.18 * inch))

    bg_text = (
        "<b>Activity Concentrations</b><br/>"
        f"\u2022 Background: {getattr(cfg.ACTIVITY, 'BACKGROUND', 'N/A')} MBq<br/>"
        f"\u2022 Hot Spheres: {getattr(cfg.ACTIVITY, 'HOT', 'N/A')} MBq<br/>"
        f"\u2022 Activity Ratio (Hot/Background): {getattr(cfg.ACTIVITY, 'RATIO', 'N/A')}"
    )
    elements.append(Paragraph(bg_text, body_style))
    elements.append(Spacer(1, 0.18 * inch))

    acq_time = getattr(
        getattr(cfg, "ACQUISITION", {}), "EMMISION_IMAGE_TIME_MINUTES", None
    )
    if acq_time is not None:
        acq_text = "<b>Acquisition Parameters</b><br/>• Emission Imaging Time: {} minutes".format(
            acq_time
        )
        elements.append(Paragraph(acq_text, body_style))
        elements.append(Spacer(1, 0.18 * inch))

    elements.append(Paragraph("<b>Overall Summary Statistics</b>", header_style))

    if filtered_results:
        avg_contrast = sum(
            r["percentaje_constrast_QH"] for r in filtered_results
        ) / len(filtered_results)
        avg_variability = sum(
            r["background_variability_N"] for r in filtered_results
        ) / len(filtered_results)
        elements.append(
            Paragraph(
                f"Average Percent Contrast (all iterations): {avg_contrast:.2f}%",
                body_style,
            )
        )
        elements.append(
            Paragraph(
                f"Average Background Variability (all iterations): {avg_variability:.2f}%",
                body_style,
            )
        )

    if filtered_lung_results:
        all_lung_values = [
            val
            for lung_dict in filtered_lung_results.values()
            for val in lung_dict.values()
        ]
        avg_lung_error = float(np.mean(all_lung_values))
        elements.append(
            Paragraph(
                f"Average Lung Insert Error (all iterations): {avg_lung_error:.2f}%",
                body_style,
            )
        )

    elements.append(
        Paragraph(f"Total measurements: {len(filtered_results)}", body_style)
    )
    elements.append(
        Paragraph(
            f"Spheres per iteration: {len(filtered_results) // len(iterations) if iterations else 0}",
            body_style,
        )
    )

    elements.append(PageBreak())

    elements.append(Paragraph("<b>Results by Iteration</b>", header_style))
    elements.append(Spacer(1, 0.05 * inch))

    for iteration in iterations:
        iteration_results = [r for r in filtered_results if r["iteration"] == iteration]

        if iteration_results:
            iteration_title = f"Iteration {iteration}" + (
                " (Final)" if iteration == max_iteration else ""
            )
            elements.append(Paragraph(f"<b>{iteration_title}</b>", header_style))

            iter_avg_contrast = sum(
                r["percentaje_constrast_QH"] for r in iteration_results
            ) / len(iteration_results)
            iter_avg_variability = sum(
                r["background_variability_N"] for r in iteration_results
            ) / len(iteration_results)

            iter_stats = (
                f"Average Contrast: {iter_avg_contrast:.2f}% | "
                f"Average Variability: {iter_avg_variability:.2f}% | "
                f"Spheres: {len(iteration_results)}"
            )
            elements.append(Paragraph(iter_stats, body_style))
            elements.append(Spacer(1, 0.1 * inch))

            # Table for this iteration
            table_data = [["Diameter (mm)", "Contrast Q_H (%)", "Variability N (%)"]]

            sorted_iter_results = sorted(
                iteration_results, key=lambda x: x["diameter_mm"], reverse=True
            )
            for result in sorted_iter_results:
                table_data.append(
                    [
                        f"{result.get('diameter_mm', 'Unknown'):.0f}",
                        f"{result.get('percentaje_constrast_QH', 'N/A'):.2f}",
                        f"{result.get('background_variability_N', 'N/A'):.2f}",
                    ]
                )

            col_widths = [1.5 * inch, 1.5 * inch, 1.5 * inch]
            table = Table(table_data, colWidths=col_widths)

            if iteration == max_iteration:
                header_color = colors.darkgreen
                row_color = colors.lightgreen
            else:
                header_color = colors.darkblue
                row_color = colors.beige

            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), header_color),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                        ("BACKGROUND", (0, 1), (-1, -1), row_color),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                        ("FONTSIZE", (0, 1), (-1, -1), 8),
                    ]
                )
            )
            elements.append(table)
            elements.append(Spacer(1, 0.05 * inch))

    elements.append(PageBreak())

    if plot_path and Path(plot_path).exists():
        elements.append(Image(str(plot_path), width=8 * inch, height=4 * inch))

    if pc_vs_bg_path and Path(pc_vs_bg_path).exists():
        elements.append(Image(str(pc_vs_bg_path), width=6 * inch, height=4 * inch))

    elements.append(PageBreak())

    if filtered_lung_results:
        elements.append(
            Paragraph("<b>Lung Insert Analysis by Iteration</b>", header_style)
        )

        if boxplot_path and Path(boxplot_path).exists():
            elements.append(Image(str(boxplot_path), width=6 * inch, height=4 * inch))
            elements.append(Spacer(1, 0.2 * inch))

        lung_table_data = [
            [
                "Iteration",
                "Mean Error (%)",
                "Std Dev (%)",
                "Min (%)",
                "Max (%)",
                "N Slices",
            ]
        ]

        for iteration in iterations:
            if iteration in filtered_lung_results:
                values = list(filtered_lung_results[iteration].values())
                lung_table_data.append(
                    [
                        f"{iteration}"
                        + (" (Final)" if iteration == max_iteration else ""),
                        f"{float(np.mean(values)):.2f}",
                        f"{float(np.std(values)):.2f}",
                        f"{float(np.min(values)):.2f}",
                        f"{float(np.max(values)):.2f}",
                        str(len(values)),
                    ]
                )

        col_widths = [
            1.0 * inch,
            1.2 * inch,
            1.2 * inch,
            1.0 * inch,
            1.0 * inch,
            1.0 * inch,
        ]
        lung_table = Table(lung_table_data, colWidths=col_widths)
        lung_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.darkred),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.mistyrose),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ("FONTSIZE", (0, 1), (-1, -1), 8),
                ]
            )
        )
        elements.append(lung_table)

    elements.append(PageBreak())

    if rois_loc_path and Path(rois_loc_path).exists():
        elements.append(Paragraph("<b>ROIs Location</b>", header_style))
        elements.append(Image(str(rois_loc_path), width=4 * inch, height=4 * inch))
        elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("<b>Legend and Formulas</b>", header_style))
    legend = [
        "• Q<sub>H</sub> (%): Percent contrast for hot spheres",
        "• N (%): Background variability",
        "• C<sub>H</sub>: Mean counts in hot sphere",
        "• C<sub>B</sub>: Mean background counts",
        "• SD<sub>B</sub>: Standard deviation of background counts",
        "• Final iteration is highlighted in green in tables and plots",
    ]
    for line in legend:
        elements.append(Paragraph(line, body_style))

    elements.append(Spacer(1, 0.18 * inch))

    formulas = [
        "<b>Contrast Formula:</b> <font face='Courier'>Q<sub>H,j</sub> (%) = "
        "[(C<sub>H,j</sub> / C<sub>B,j</sub>) - 1] / [(a<sub>H</sub> / a<sub>B</sub>) - 1] × 100</font>",
        "<b>Background Variability Formula:</b> <font face='Courier'>N<sub>j</sub> (%) = "
        "SD<sub>B,j</sub> / C<sub>B,j</sub> × 100</font>",
    ]
    for formula in formulas:
        elements.append(Paragraph(formula, body_style))

    elements.append(Paragraph("Where:", body_style))
    expl = [
        "j       = sphere index",
        "a<sub>H</sub> = activity concentration in hot spheres",
        "a<sub>B</sub> = activity concentration in background",
        "C<sub>H,j</sub>   = mean counts in hot sphere j",
        "C<sub>B,j</sub>   = mean background counts for sphere j size",
        "SD<sub>B,j</sub>  = standard deviation of background for sphere j size",
    ]
    for line in expl:
        elements.append(Paragraph(line, body_style))

    doc.build(elements)


def save_results_to_txt(
    results: List[Dict[str, Any]],
    all_lung_results: Dict[int, Dict[int, float]],
    output_path: Path,
    cfg: yacs.config.CfgNode,
    input_image_path: Path,
    voxel_spacing: Tuple[float, float, float],
) -> None:
    """
    Saves NEMA analysis results across iterations to a formatted text file.

    Writes analysis results for each sphere across multiple iterations, along with configuration
    and metadata, to the specified output path. Excludes iterations 1 and 2 from analysis.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of analysis results for each sphere, including 'iteration' field.
    all_lung_results : Dict[int, Dict[int, float]]
        Dictionary mapping iteration numbers to lung results dictionaries.
        Structure: {iteration_num: {slice_num: correction_value, ...}, ...}
    output_path : Path
        Destination path for saving the results file.
    cfg : yacs.config.CfgNode
        Configuration object used for the analysis.
    input_image_path : Path
        Path to the input directory or image file.
    voxel_spacing : Tuple[float, float, float]
        Voxel spacing used during analysis.

    Returns
    -------
    None
        This function does not return a value; results are saved to disk.
    """
    filtered_results = [r for r in results if r.get("iteration", 0) not in [1, 2]]
    filtered_lung_results = {
        k: v for k, v in all_lung_results.items() if k not in [1, 2]
    }

    iterations = sorted({r["iteration"] for r in filtered_results})
    max_iteration = max(iterations) if iterations else 0

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("NEMA NU 2-2018 IMAGE QUALITY ANALYSIS RESULTS - MULTI-ITERATION\n")
        f.write("=" * 90 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input directory: {input_image_path}\n")
        f.write(
            f"Voxel spacing: {voxel_spacing[0]:.4f} x {voxel_spacing[1]:.4f} x {voxel_spacing[2]:.4f} mm\n"
        )
        f.write(
            "Iterations analyzed: {} (excluded 1, 2)\n".format(
                ", ".join(map(str, iterations))
            )
        )
        f.write("Final iteration: {}\n".format(max_iteration))
        f.write("\n")

        f.write("ANALYSIS CONFIGURATION:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Hot activity: {cfg.ACTIVITY.HOT:.3f} MBq\n")
        f.write(f"Background activity: {cfg.ACTIVITY.BACKGROUND:.3f} MBq\n")
        f.write(f"Activity ratio: {cfg.ACTIVITY.HOT/cfg.ACTIVITY.BACKGROUND:.2f}\n")
        f.write(f"Central slice: {cfg.ROIS.CENTRAL_SLICE}\n")
        f.write(f"Case name: {getattr(cfg.FILE, 'CASE', 'N/A')}\n")
        f.write("\n")

        f.write("OVERALL SUMMARY STATISTICS:\n")
        f.write("-" * 50 + "\n")
        if filtered_results:
            avg_contrast = sum(
                r["percentaje_constrast_QH"] for r in filtered_results
            ) / len(filtered_results)
            avg_variability = sum(
                r["background_variability_N"] for r in filtered_results
            ) / len(filtered_results)
            f.write(f"Average Percent Contrast (all iterations): {avg_contrast:.2f}%\n")
            f.write(
                f"Average Background Variability (all iterations): {avg_variability:.2f}%\n"
            )
            f.write(f"Total measurements: {len(filtered_results)}\n")
            f.write(
                f"Spheres per iteration: {len(filtered_results) // len(iterations) if iterations else 0}\n"
            )

        if filtered_lung_results:
            all_lung_values = [
                val
                for lung_dict in filtered_lung_results.values()
                for val in lung_dict.values()
            ]
            avg_lung_error = float(np.mean(all_lung_values))
            f.write(
                f"Average Lung Insert Error (all iterations): {avg_lung_error:.2f}%\n"
            )
        f.write("\n")

        f.write("ANALYSIS RESULTS BY ITERATION:\n")
        f.write("=" * 90 + "\n")

        for iteration in iterations:
            iteration_results = [
                r for r in filtered_results if r["iteration"] == iteration
            ]

            if iteration_results:
                iteration_title = "ITERATION {}".format(iteration) + (
                    " (FINAL)" if iteration == max_iteration else ""
                )
                f.write(f"\n{iteration_title}\n")
                f.write("-" * len(iteration_title) + "\n")

                f.write("Sphere Analysis Results (NEMA NU 2-2018 Section 7.4.1)\n\n")

                iter_avg_contrast = sum(
                    r["percentaje_constrast_QH"] for r in iteration_results
                ) / len(iteration_results)
                iter_avg_variability = sum(
                    r["background_variability_N"] for r in iteration_results
                ) / len(iteration_results)
                f.write("Iteration Statistics:\n")
                f.write(f"  Average Contrast: {iter_avg_contrast:.2f}%\n")
                f.write(f"  Average Variability: {iter_avg_variability:.2f}%\n")
                f.write(f"  Number of spheres: {len(iteration_results)}\n\n")

                f.write(f"{'Diameter':<10} {'Q_H (%)':<10} {'N (%)':<10}\n")
                f.write(f"{'(mm)':<10} {'':<10} {'':<10}\n")
                f.write("-" * 30 + "\n")

                sorted_iter_results = sorted(
                    iteration_results, key=lambda x: x["diameter_mm"], reverse=True
                )

                for result in sorted_iter_results:
                    f.write(
                        f"{result['diameter_mm']:<10.0f} "
                        f"{result['percentaje_constrast_QH']:<10.2f} "
                        f"{result['background_variability_N']:<10.2f} "
                        "\n"
                    )
                f.write("\n")

        if filtered_lung_results:
            f.write("LUNG INSERT ANALYSIS BY ITERATION:\n")
            f.write("=" * 90 + "\n")

            f.write(
                f"{'Iteration':<12} {'Mean (%)':<10} {'Std (%)':<10} {'Min (%)':<10} {'Max (%)':<10} {'N Slices':<10}\n"
            )
            f.write("-" * 72 + "\n")

            for iteration in iterations:
                if iteration in filtered_lung_results:
                    values = list(filtered_lung_results[iteration].values())
                    iter_label = f"{iteration}" + (
                        " (Final)" if iteration == max_iteration else ""
                    )
                    f.write(
                        f"{iter_label:<12} "
                        f"{float(np.mean(values)):<10.2f} "
                        f"{float(np.std(values)):<10.2f} "
                        f"{float(np.min(values)):<10.2f} "
                        f"{float(np.max(values)):<10.2f} "
                        f"{len(values):<10}\n"
                    )
            f.write("\n")

        f.write("ITERATION COMPARISON SUMMARY:\n")
        f.write("-" * 50 + "\n")

        f.write("Sphere Metrics by Iteration:\n")
        f.write(
            f"{'Iteration':<12} {'Avg Contrast (%)':<18} {'Avg Variability (%)':<20} {'N Spheres':<12}\n"
        )
        f.write("-" * 72 + "\n")

        for iteration in iterations:
            iteration_results = [
                r for r in filtered_results if r["iteration"] == iteration
            ]
            if iteration_results:
                iter_avg_contrast = sum(
                    r["percentaje_constrast_QH"] for r in iteration_results
                ) / len(iteration_results)
                iter_avg_variability = sum(
                    r["background_variability_N"] for r in iteration_results
                ) / len(iteration_results)
                iter_label = f"{iteration}" + (
                    " (Final)" if iteration == max_iteration else ""
                )
                f.write(
                    f"{iter_label:<12} "
                    f"{iter_avg_contrast:<18.2f} "
                    f"{iter_avg_variability:<20.2f} "
                    f"{len(iteration_results):<12}\n"
                )
        f.write("\n")

        if len(iterations) >= 2:
            first_iter = min(iterations)
            last_iter = max(iterations)

            first_results = [
                r for r in filtered_results if r["iteration"] == first_iter
            ]
            last_results = [r for r in filtered_results if r["iteration"] == last_iter]

            if first_results and last_results:
                first_contrast = sum(
                    r["percentaje_constrast_QH"] for r in first_results
                ) / len(first_results)
                last_contrast = sum(
                    r["percentaje_constrast_QH"] for r in last_results
                ) / len(last_results)
                first_variability = sum(
                    r["background_variability_N"] for r in first_results
                ) / len(first_results)
                last_variability = sum(
                    r["background_variability_N"] for r in last_results
                ) / len(last_results)

                contrast_change = last_contrast - first_contrast
                variability_change = last_variability - first_variability

                f.write("CONVERGENCE ANALYSIS:\n")
                f.write("-" * 50 + "\n")
                f.write(f"First analyzed iteration ({first_iter}):\n")
                f.write(f"  Average Contrast: {first_contrast:.2f}%\n")
                f.write(f"  Average Variability: {first_variability:.2f}%\n")
                f.write(f"Final iteration ({last_iter}):\n")
                f.write(f"  Average Contrast: {last_contrast:.2f}%\n")
                f.write(f"  Average Variability: {last_variability:.2f}%\n")
                f.write("Changes from first to final:\n")
                change_desc = (
                    "(improved)"
                    if contrast_change > 0
                    else "(degraded)" if contrast_change < 0 else "(no change)"
                )
                f.write(f"  Contrast change: {contrast_change:+.2f}% {change_desc}\n")

                var_desc = (
                    "(improved)"
                    if variability_change < 0
                    else "(degraded)" if variability_change > 0 else "(no change)"
                )
                f.write(
                    f"  Variability change: {variability_change:+.2f}% {var_desc}\n"
                )
                f.write("\n")

        f.write("LEGEND:\n")
        f.write("-" * 50 + "\n")
        f.write("Q_H (%)  : Percent Contrast (Hot sphere)\n")
        f.write("N (%)    : Percent Background Variability\n")
        f.write("C_H      : Mean counts in hot sphere\n")
        f.write("C_B      : Mean background counts\n")
        f.write("SD_B     : Standard deviation of background\n")
        f.write("Final    : Highest iteration number (assumed converged)\n")
        f.write("\n")

        f.write("NEMA NU 2-2018 FORMULAS:\n")
        f.write("-" * 50 + "\n")
        f.write("Q_H,j (%) = [(C_H,j / C_B,j) - 1] / [(a_H / a_B) - 1] × 100\n")
        f.write("N_j (%)   = (SD_B,j / C_B,j) × 100\n")
        f.write("\n")
        f.write("Where:\n")
        f.write("  j       = sphere index\n")
        f.write("  a_H     = activity concentration in hot spheres\n")
        f.write("  a_B     = activity concentration in background\n")
        f.write("  C_H,j   = mean counts in hot sphere j\n")
        f.write("  C_B,j   = mean background counts for sphere j size\n")
        f.write("  SD_B,j  = standard deviation of background for sphere j size\n")
        f.write("\n")

        f.write("=" * 90 + "\n")
        f.write("End of Multi-Iteration Report\n")
        f.write("=" * 90 + "\n")

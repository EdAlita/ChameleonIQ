import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

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

    plt.style.use("seaborn-v0_8-talk")
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "legend.title_fontsize": 24,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "axes.linewidth": 1.2,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
        }
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))

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
            _alpha = 0.6
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

    ax1.set_title("Contrast Recovery vs Sphere Diameter", fontweight="bold")
    ax1.set_xlabel("Sphere Diameter (mm)", fontweight="bold")
    ax1.set_ylabel("Contrast Recovery (%)", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.set_title("Background Variability vs Sphere Diameter", fontweight="bold")
    ax2.set_xlabel("Sphere Diameter (mm)", fontweight="bold")
    ax2.set_ylabel("Background Variability (%)", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(
        "NEMA Analysis - Multi-Experiment Comparison",
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

    plt.style.use("seaborn-v0_8-talk")
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "legend.title_fontsize": 24,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "axes.linewidth": 1.2,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
        }
    )
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

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
                max(exp_values) + std_val / 6,
                f"μ={mean_val:.2f}",
                ha="center",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "alpha": 0.9,
                    "edgecolor": "gray",
                },
            )

    ax.set_title("Lung Insert Accuracy Distribution", fontweight="bold", pad=40)
    ax.set_xlabel("Experiment", fontweight="bold")
    ax.set_ylabel("Accuracy of Corrections in Lung Insert (%)", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(range(1, len(experiment_names) + 1))
    ax.set_xticklabels(experiment_names)
    ax.set_facecolor("#fafafa")

    if len(experiment_names) > 3:
        ax.tick_params(axis="x", rotation=45)

    ax.tick_params(axis="both", width=1.2)

    legend_elements = [
        Line2D([0], [0], color=COLORS[i % len(COLORS)], linewidth=4, label=exp)
        for i, exp in enumerate(experiment_names)
    ]

    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
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


def _find_value_for_row_candidates(
    row: Dict[str, Any], candidates: List[str]
) -> Optional[Any]:
    """Returns the first matching candidate key value from the row(case-insensitive)."""
    for k in candidates:
        if k in row:
            return row[k]

    low_map = {kk.lower(): kk for kk in row.keys()}
    for k in candidates:
        if k.lower() in low_map:
            return row[low_map[k.lower()]]
    return None


def generate_dose_merged_plot(
    data: List[Dict[str, Any]],
    output_dir: Path,
    dosis_map: Optional[Dict[str, float]] = None,
) -> None:
    logger.info("Generating dose merged plot")

    if dosis_map is None:
        logger.warning("No dosis_map provided, skipping dose merged plot generation.")
        return None

    diam_candidates = ["diameter_mm", "diameter", "diam", "d"]
    pc_candidates = [
        "percentaje_constrast_QH",
        "percentaje_contrast_QH",
        "pc",
        "pc_value",
        "percentaje_contrast",
    ]
    bv_candidates = [
        "background_variability_N",
        "background_variability",
        "bv",
        "bv_value",
    ]

    normalized_rows = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(
                f"Expected dict rows (as produced by load_experiments_data). Row {i} is {type(row)}"
            )

        diameter = _find_value_for_row_candidates(row, diam_candidates)
        pc = _find_value_for_row_candidates(row, pc_candidates)
        bv = _find_value_for_row_candidates(row, bv_candidates)
        experiment = row.get("experiment") or row.get("Experiment") or row.get("exp")

        if diameter is None:
            raise ValueError(
                f"Row {i} missing diameter field. Candidates were: {diam_candidates}"
            )
        if pc is None:
            raise ValueError(
                f"Row {i} missing percentaje_constrast_QH field. Candidates were: {pc_candidates}"
            )
        if bv is None:
            raise ValueError(
                f"Row {i} missing background_variability_N field. Candidates were: {bv_candidates}"
            )
        if experiment is None:
            raise ValueError(
                f"Row {i} missing experiment field. Tried 'experiment', 'Experiment', 'exp'."
            )

        normalized_rows.append(
            {
                "diameter_mm": float(diameter),
                "percentaje_constrast_QH": float(pc),
                "background_variability_N": float(bv),
                "experiment": str(experiment),
            }
        )

    df = pd.DataFrame(normalized_rows)

    def map_exp_to_numeric(exp: str) -> float:
        return float(dosis_map[exp])

    df["dose"] = df["experiment"].map(map_exp_to_numeric)
    df_agg = df.groupby(["diameter_mm", "dose"], as_index=False).agg(
        {"percentaje_constrast_QH": "mean", "background_variability_N": "mean"}
    )

    cmap = plt.cm.get_cmap("Oranges")
    dose_min, dose_max = float(df_agg["dose"].min()), float(df_agg["dose"].max())
    norm = Normalize(vmin=dose_min, vmax=dose_max)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    diameters = np.array(sorted(df_agg["diameter_mm"].unique())).astype(float)
    diam_min, diam_max = float(diameters.min()), float(diameters.max())

    min_marker_area = 80
    max_marker_area = 620

    def diameter_to_marker_area(d: float) -> float:
        if diam_max == diam_min:
            return (min_marker_area + max_marker_area) / 2.0
        frac = (d - diam_min) / (diam_max - diam_min)
        return float(min_marker_area + frac * (max_marker_area - min_marker_area))

    plt.style.use("seaborn-v0_8-talk")
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "legend.title_fontsize": 24,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "axes.linewidth": 1.2,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
        }
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    for d in diameters:
        sub = df_agg[df_agg["diameter_mm"] == d].sort_values("dose")
        doses = sub["dose"].to_numpy()
        pcs = sub["percentaje_constrast_QH"].to_numpy()
        bvs = sub["background_variability_N"].to_numpy()

        marker_area = diameter_to_marker_area(float(d))

        ax1.scatter(
            doses,
            pcs,
            s=marker_area,
            c=doses,
            cmap=cmap,
            norm=norm,
            edgecolor="k",
            linewidth=0.7,
            zorder=3,
            label=None,
        )

        ax2.scatter(
            doses,
            bvs,
            s=marker_area,
            c=doses,
            cmap=cmap,
            norm=norm,
            edgecolor="k",
            linewidth=0.7,
            zorder=3,
            label=None,
        )

    cbar_ax = fig.add_axes((0.78, 0.15, 0.02, 0.7))
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("DLP", rotation=270, labelpad=15)

    legend_handles = []
    for d in diameters:
        ma = diameter_to_marker_area(d)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"{int(d)} mm" if float(d).is_integer() else f"{d}",
                markerfacecolor="#777777",
                markeredgecolor="k",
                markersize=np.sqrt(ma / 15),
            )
        )
    ax2.legend(
        handles=legend_handles,
        title="Sphere diameter",
        bbox_to_anchor=(1.35, 1.0),
        loc="upper left",
        framealpha=0.95,
    )

    ax1.set_title("DLP vs Contrast Recovery (PC)", fontweight="bold")
    ax1.set_xlabel("PC (%)", fontweight="bold")
    ax1.set_ylabel("DLP", fontweight="bold")
    ax1.grid(alpha=0.25)
    ax1.yaxis.set_major_locator(MaxNLocator(6))

    ax2.set_title("DLP vs Background Variability (BV)", fontweight="bold")
    ax2.set_xlabel("BV (%)", fontweight="bold")
    ax2.set_ylabel("DLP", fontweight="bold")
    ax2.grid(alpha=0.25)
    ax2.yaxis.set_major_locator(MaxNLocator(6))

    plt.suptitle(
        "Dose Merged Analysis — DLP vs PC and BV",
        fontweight="bold",
        y=0.95,
    )
    plt.subplots_adjust(left=0.08, right=0.75, top=0.88, bottom=0.12, wspace=0.25)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dose_merged_plot.png"
    plt.savefig(
        str(output_path),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    logger.info("Dose merged plot saved: %s", output_path)
    plt.close(fig)


def generate_dose_merged_plot_any_sphere(
    data: List[Dict[str, Any]],
    output_dir: Path,
    dosis_map: Optional[Dict[str, float]] = None,
    sphere_diameter: float = 10.0,
) -> None:
    logger.info("Generating dose merged plot for sphere (%.1f mm)", sphere_diameter)

    if dosis_map is None:
        logger.warning("No dosis_map provided, skipping dose merged plot generation.")
        return None

    diam_candidates = ["diameter_mm", "diameter", "diam", "d"]
    pc_candidates = [
        "percentaje_constrast_QH",
        "percentaje_contrast_QH",
        "pc",
        "pc_value",
        "percentaje_contrast",
    ]
    bv_candidates = [
        "background_variability_N",
        "background_variability",
        "bv",
        "bv_value",
    ]

    normalized_rows = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(
                f"Expected dict rows (as produced by load_experiments_data). Row {i} is {type(row)}"
            )

        diameter = _find_value_for_row_candidates(row, diam_candidates)
        pc = _find_value_for_row_candidates(row, pc_candidates)
        bv = _find_value_for_row_candidates(row, bv_candidates)
        experiment = row.get("experiment") or row.get("Experiment") or row.get("exp")

        if diameter is None:
            raise ValueError(
                f"Row {i} missing diameter field. Candidates were: {diam_candidates}"
            )
        if pc is None:
            raise ValueError(
                f"Row {i} missing percentaje_constrast_QH field. Candidates were: {pc_candidates}"
            )
        if bv is None:
            raise ValueError(
                f"Row {i} missing background_variability_N field. Candidates were: {bv_candidates}"
            )
        if experiment is None:
            raise ValueError(
                f"Row {i} missing experiment field. Tried 'experiment', 'Experiment', 'exp'."
            )

        if float(diameter) != sphere_diameter:
            continue

        normalized_rows.append(
            {
                "diameter_mm": float(diameter),
                "percentaje_constrast_QH": float(pc),
                "background_variability_N": float(bv),
                "experiment": str(experiment),
            }
        )

    if not normalized_rows:
        logger.warning("No data found for %.1f mm sphere", sphere_diameter)
        return None

    df = pd.DataFrame(normalized_rows)

    def map_exp_to_numeric(exp: str) -> float:
        return float(dosis_map[exp])

    df["dose"] = df["experiment"].map(map_exp_to_numeric)

    df_agg = df.groupby("dose", as_index=False).agg(
        {"percentaje_constrast_QH": "mean", "background_variability_N": "mean"}
    )

    cmap = plt.cm.get_cmap("Oranges")
    dose_min, dose_max = float(df_agg["dose"].min()), float(df_agg["dose"].max())
    norm = Normalize(vmin=dose_min, vmax=dose_max)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    plt.style.use("seaborn-v0_8-talk")
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "legend.title_fontsize": 24,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "axes.linewidth": 1.2,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    df_sorted = df_agg.sort_values("dose")
    doses = df_sorted["dose"].to_numpy()
    pcs = df_sorted["percentaje_constrast_QH"].to_numpy()
    bvs = df_sorted["background_variability_N"].to_numpy()

    marker_size = 120

    ax1.scatter(
        doses,
        pcs,
        s=marker_size,
        c=doses,
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=1.5,
        zorder=3,
    )

    ax2.scatter(
        doses,
        bvs,
        s=marker_size,
        c=doses,
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=1.5,
        zorder=3,
    )

    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("DLP", rotation=270, labelpad=15)

    ax1.set_title(
        "DLP vs Contrast Recovery (PC)",
        fontweight="bold",
    )
    ax1.set_xlabel("DLP")
    ax1.set_ylabel("PC (%)")
    ax1.grid(alpha=0.3)

    ax2.set_title(
        "DLP vs Background Variability (BV)",
        fontweight="bold",
    )
    ax2.set_xlabel("DLP")
    ax2.set_ylabel("BV (%)")
    ax2.grid(alpha=0.3)

    plt.suptitle(
        f"Dosage Merged Analysis - Sphere {sphere_diameter:.1f}mm",
        fontweight="bold",
        y=0.95,
    )
    plt.tight_layout(rect=(0.0, 0.0, 0.9, 0.93))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"dose_{sphere_diameter:.1f}mm_sphere_plot.png"
    plt.savefig(
        str(output_path),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    logger.info("Small sphere dose plot saved: %s", output_path)
    plt.close(fig)


def generate_global_metrics_boxplot(
    advanced_metric_data: List[Dict[str, Any]],
    output_dir: Path,
    metrics_to_plot: List[str],
    name: str = "global_metrics_violinplot",
) -> None:
    """
    Generates violin plots for each metric across all experiments,
    ignoring experiment grouping, with styled violin bodies.
    """

    logger.info("Generating global violinplot for advanced metrics")

    df = pd.DataFrame(advanced_metric_data)

    plt.style.use("seaborn-v0_8-talk")
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "legend.title_fontsize": 24,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "axes.linewidth": 1.2,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
        }
    )
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    plot_data = []
    for _, row in df.iterrows():
        for metric in metrics_to_plot:
            plot_data.append({"metric": metric, "value": row[metric]})

    if not plot_data:
        logger.warning("No data available for global metrics violinplot")
        return

    plot_df = pd.DataFrame(plot_data)

    violin_parts = ax.violinplot(
        [
            plot_df[plot_df["metric"] == metric]["value"].values
            for metric in metrics_to_plot
        ],
        positions=range(1, len(metrics_to_plot) + 1),
        showmeans=True,
        showmedians=False,
        showextrema=False,
    )

    for idx, patch in enumerate(cast(List[Any], violin_parts["bodies"])):
        color = COLORS[idx % len(COLORS)]
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")
        patch.set_linewidth(1)

    violin_parts["cmeans"].set_color("white")
    violin_parts["cmeans"].set_linewidth(3)
    logger.info("Metrics resume:")
    for idx, metric in enumerate(metrics_to_plot):
        values = plot_df[plot_df["metric"] == metric]["value"].values
        color = COLORS[idx % len(COLORS)]
        jitter = np.random.normal(0, 0.04, len(values))
        ax.scatter(
            np.full(len(values), idx + 1) + jitter,
            values,
            color=color,
            s=40,
            alpha=0.8,
            edgecolors="white",
            linewidths=1,
            zorder=10,
        )

        mean_val = np.mean(values)
        std_dev = np.std(values)
        logger.info(
            f"  {metric}: mean={mean_val:.3f}, std={std_dev:.3f}, n={len(values)}"
        )
        ax.text(
            idx + 1,
            mean_val + std_dev * 1.5,
            f"μ={mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=20,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "gray",
            },
        )

    ax.set_title("Global distribution of advanced metrics", fontweight="bold", pad=40)
    ax.set_xlabel("Metric", fontweight="bold")
    ax.set_ylabel("Value", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(range(1, len(metrics_to_plot) + 1))
    ax.set_xticklabels(metrics_to_plot)
    ax.set_facecolor("#fafafa")

    if len(metrics_to_plot) > 5:
        ax.tick_params(axis="x", rotation=45)

    ax.tick_params(axis="both", width=1.2)

    legend_elements = [
        Line2D([0], [0], color=COLORS[i % len(COLORS)], linewidth=4, label=metric)
        for i, metric in enumerate(metrics_to_plot)
    ]
    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
    )

    plt.tight_layout()

    output_path = output_dir / name
    plt.savefig(
        str(output_path),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        format="png",
    )

    logger.info(f"Global metrics violinplot saved: {output_path}")
    plt.close()


def generate_unified_statistical_heatmaps(
    statistical_results: Dict[str, Any],
    experiment_order: List[str],
    output_dir: Path,
    metrics_list: List[str],
    test_name: str,
) -> None:
    """Generate unified statistical heatmaps for all metrics in one figure"""

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    available_metrics = [m for m in metrics_list if m in statistical_results]
    if not available_metrics:
        logging.warning("No metrics available for unified statistical heatmaps")
        return

    n_metrics = len(available_metrics)

    if n_metrics <= 2:
        rows, cols = 1, n_metrics
        figsize = (12 * n_metrics, 10)
    elif n_metrics <= 4:
        rows, cols = 2, 2
        figsize = (24, 20)
    elif n_metrics <= 6:
        rows, cols = 2, 3
        figsize = (36, 20)
    else:
        rows, cols = 3, 3
        figsize = (36, 30)

    plt.style.use("seaborn-v0_8-talk")
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "legend.title_fontsize": 24,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "axes.linewidth": 1.2,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
        }
    )

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, metric_name in enumerate(available_metrics):
        ax = axes[idx]

        if metric_name not in statistical_results:
            ax.text(
                0.5,
                0.5,
                f"No data for\n{metric_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgray"},
            )
            ax.set_title(metric_name)
            continue

        p_corrected = statistical_results[metric_name]["p_values"]
        print("Unique p-values corrected for", metric_name, ":", np.unique(p_corrected))
        effect_sizes = statistical_results[metric_name]["effect_sizes"]
        print("Effect sizes for", metric_name, ":", np.unique(effect_sizes))

        mask = np.triu(np.ones_like(p_corrected), k=1)

        sns.heatmap(
            effect_sizes,
            mask=mask,
            cmap="coolwarm",
            center=0,
            vmin=-2,
            vmax=2,
            ax=ax,
            cbar_kws={"label": "Cohen's d", "shrink": 0.8},
            xticklabels=False,
            yticklabels=False,
        )

        mask_bool = np.triu(np.ones_like(p_corrected, dtype=bool), k=1)
        sig_mask_bool = (p_corrected < 0.05).astype(bool)

        y, x = np.where(sig_mask_bool & (~mask_bool))

        if len(x) > 0:
            ax.scatter(x + 0.5, y + 0.5, color="black", s=15, marker="*", alpha=0.8)

        ax.set_title(f"{metric_name}", fontweight="bold", pad=10)

        n_significant = len(x)
        total_comparisons = int(np.sum(~mask_bool))
        ax.text(
            0.02,
            0.98,
            f"{n_significant}/{total_comparisons} sig.",
            transform=ax.transAxes,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8},
            verticalalignment="top",
        )

    for idx in range(len(available_metrics), len(axes)):
        axes[idx].axis("off")

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="black",
            markersize=8,
            label="p < 0.05",
            linestyle="None",
        )
    ]

    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.95))

    plt.suptitle(
        "Statistical Analysis: Effect Sizes & Significance", fontweight="bold", y=0.98
    )

    plt.tight_layout(rect=(0, 0, 0.95, 0.95))

    output_path = output_dir / f"unified_statistical_heatmaps_{test_name}.png"
    plt.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()

    logging.info(f"Unified statistical heatmaps saved: {output_path}")

    generate_statistical_summary_matrix(
        statistical_results, available_metrics, output_dir
    )


def generate_statistical_summary_matrix(
    statistical_results: Dict[str, Any], metrics_list: List[str], output_dir: Path
) -> None:
    """Generate a summary matrix showing significant pairs count for each metric"""

    import matplotlib.pyplot as plt

    summary_data = []
    for metric_name in metrics_list:
        if metric_name not in statistical_results:
            continue

        p_corrected = statistical_results[metric_name]["p_corrected"]
        effect_sizes = statistical_results[metric_name]["effect_sizes"]

        mask = np.triu(np.ones_like(p_corrected, dtype=bool), k=1)
        sig_mask = (p_corrected < 0.05) & (~mask)
        n_significant = np.sum(sig_mask)
        total_comparisons = np.sum(~mask)

        significant_effects = effect_sizes[sig_mask]
        if len(significant_effects) > 0:
            mean_effect = np.mean(np.abs(significant_effects))
            max_effect = np.max(np.abs(significant_effects))
        else:
            mean_effect = 0
            max_effect = 0

        summary_data.append(
            {
                "Metric": metric_name,
                "Significant Pairs": n_significant,
                "Total Comparisons": total_comparisons,
                "Percentage (%)": (
                    (n_significant / total_comparisons * 100)
                    if total_comparisons > 0
                    else 0
                ),
                "Mean Effect Size": mean_effect,
                "Max Effect Size": max_effect,
            }
        )

    if not summary_data:
        return

    df_summary = pd.DataFrame(summary_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    bars = ax1.bar(
        df_summary["Metric"],
        df_summary["Percentage (%)"],
        color=[
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
        ][: len(df_summary)],
    )

    ax1.set_title("Percentage of Significant Pairs by Metric", fontweight="bold")
    ax1.set_ylabel("Significant Pairs (%)")
    ax1.set_xlabel("Metric")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    for bar, value in zip(bars, df_summary["Percentage (%)"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax2.scatter(
        df_summary["Mean Effect Size"],
        df_summary["Max Effect Size"],
        s=df_summary["Significant Pairs"] * 10,
        alpha=0.7,
        c=range(len(df_summary)),
        cmap="viridis",
    )

    for _, row in df_summary.iterrows():
        ax2.annotate(
            row["Metric"],
            (row["Mean Effect Size"], row["Max Effect Size"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8,
        )

    ax2.set_title("Effect Size Analysis", fontweight="bold")
    ax2.set_xlabel("Mean Effect Size (|Cohen's d|)")
    ax2.set_ylabel("Max Effect Size (|Cohen's d|)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "statistical_summary_matrix.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    df_summary.to_csv(output_dir / "statistical_summary_table.csv", index=False)

    logging.info(
        f"Statistical summary matrix saved: {output_dir / 'statistical_summary_matrix.png'}"
    )

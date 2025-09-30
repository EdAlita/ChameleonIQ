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

    # Plotting setup
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

    plt.style.use("default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # For each diameter, plot PC and BV against numeric dose - EJES VOLTEADOS
    for d in diameters:
        sub = df_agg[df_agg["diameter_mm"] == d].sort_values("dose")
        doses = sub["dose"].to_numpy()
        pcs = sub["percentaje_constrast_QH"].to_numpy()
        bvs = sub["background_variability_N"].to_numpy()

        marker_area = diameter_to_marker_area(float(d))

        ax1.plot(doses, pcs, color=COLORS[-1], lw=1.6, zorder=1, solid_capstyle="round")
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

        ax2.plot(doses, bvs, color=COLORS[-1], lw=1.6, zorder=1, solid_capstyle="round")
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

    # Colorbar shared across both axes
    cbar_ax = fig.add_axes((0.78, 0.15, 0.02, 0.7))
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("DLP", rotation=270, labelpad=15)

    # Legend for diameters (marker sizes)
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

    # Titles, labels, layout - ETIQUETAS VOLTEADAS
    ax1.set_title("DLP vs Contrast Recovery (PC)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("PC (%)", fontsize=12)
    ax1.set_ylabel("DLP", fontsize=12)
    ax1.grid(alpha=0.25)
    ax1.yaxis.set_major_locator(MaxNLocator(6))

    ax2.set_title("DLP vs Background Variability (BV)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("BV (%)", fontsize=12)
    ax2.set_ylabel("DLP", fontsize=12)
    ax2.grid(alpha=0.25)
    ax2.yaxis.set_major_locator(MaxNLocator(6))

    plt.suptitle(
        "Dose Merged Analysis — DLP vs PC and BV",
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )
    plt.subplots_adjust(left=0.08, right=0.75, top=0.88, bottom=0.12, wspace=0.25)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dose_merged_plot.png"
    plt.savefig(
        str(output_path),
        dpi=300,
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

        # FILTRAR SOLO ESFERA DE 10mm
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

    # No necesitamos agrupar por diámetro ya que solo tenemos 10mm
    df_agg = df.groupby("dose", as_index=False).agg(
        {"percentaje_constrast_QH": "mean", "background_variability_N": "mean"}
    )

    # Plotting setup
    cmap = plt.cm.get_cmap("Oranges")
    dose_min, dose_max = float(df_agg["dose"].min()), float(df_agg["dose"].max())
    norm = Normalize(vmin=dose_min, vmax=dose_max)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    plt.style.use("default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Datos ordenados por dosis
    df_sorted = df_agg.sort_values("dose")
    doses = df_sorted["dose"].to_numpy()
    pcs = df_sorted["percentaje_constrast_QH"].to_numpy()
    bvs = df_sorted["background_variability_N"].to_numpy()

    marker_size = 120

    # PC vs DLP
    ax1.plot(
        doses,
        pcs,
        color=COLORS[-1],
        lw=3,
        zorder=2,
        solid_capstyle="round",
        label=f"{sphere_diameter:.1f}mm sphere",
    )
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

    # BV vs DLP
    ax2.plot(
        doses,
        bvs,
        color=COLORS[-1],
        lw=3,
        zorder=2,
        solid_capstyle="round",
        label=f"{sphere_diameter:.1f}mm sphere",
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

    # Colorbar
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("DLP", rotation=270, labelpad=15)

    # Títulos y etiquetas
    ax1.set_title(
        f"Contrast Recovery (PC) vs DLP - {sphere_diameter:.1f}mm Sphere",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("DLP", fontsize=12)
    ax1.set_ylabel("PC (%)", fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.set_title(
        f"Background Variability (BV) vs DLP - {sphere_diameter:.1f}mm Sphere",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlabel("DLP", fontsize=12)
    ax2.set_ylabel("BV (%)", fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.suptitle(
        f"Dose Analysis - {sphere_diameter:.1f}mm Sphere Only",
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )
    plt.tight_layout(rect=(0.0, 0.0, 0.9, 0.93))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"dose_{sphere_diameter:.1f}mm_sphere_plot.png"
    plt.savefig(
        str(output_path),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    logger.info("Small sphere dose plot saved: %s", output_path)
    plt.close(fig)

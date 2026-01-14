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
import numpy.typing as npt
import pandas as pd
import yacs.config
from matplotlib.patches import Circle
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


def save_results_to_txt(
    results: List[Dict[str, Any]],
    output_path: Path,
    cfg: yacs.config.CfgNode,
    input_image_path: Path,
    voxel_spacing: Tuple[float, float, float],
) -> None:
    """
    Saves NEMA analysis results to a formatted text file.

    Writes analysis results for each sphere, along with configuration and metadata, to the specified output path.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of analysis results for each sphere.
    output_path : Path
        Destination path for saving the results file.
    cfg : yacs.config.CfgNode
        Configuration object used for the analysis.
    input_image_path : Path
        Path to the input image file.
    voxel_spacing : Tuple[float, float, float]
        Voxel spacing used during analysis.

    Returns
    -------
    None
        This function does not return a value; results are saved to disk.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("NEMA NU 2-2018 IMAGE QUALITY ANALYSIS RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input image: {input_image_path}\n")
        f.write(
            f"Voxel spacing: {voxel_spacing[0]:.4f} x {voxel_spacing[1]:.4f} x {voxel_spacing[2]:.4f} mm\n"
        )
        f.write("\n")

        f.write("ANALYSIS CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        if cfg.ACTIVITY.HOT == 0.0:
            f.write("Hot activity: not reported\n")
        else:
            f.write(f"Hot activity: {cfg.ACTIVITY.HOT:.7f} {cfg.ACTIVITY.UNITS}\n")
        if cfg.ACTIVITY.BACKGROUND == 0.0:
            f.write("Background activity: not reported\n")
        else:
            f.write(
                f"Background activity: {cfg.ACTIVITY.BACKGROUND:.7f} {cfg.ACTIVITY.UNITS}\n"
            )
        f.write(f"Activity ratio: {cfg.ACTIVITY.RATIO:.3f}\n")
        if (
            cfg.ACTIVITY.ACTIVITY_TOTAL == "0.0 mCi"
            or cfg.ACTIVITY.ACTIVITY_TOTAL == "0.0 MBq"
        ):
            f.write("Total activity: not reported\n")
        else:
            f.write(f"Total activity: {cfg.ACTIVITY.ACTIVITY_TOTAL}\n")
        f.write(f"Central slice: {cfg.ROIS.CENTRAL_SLICE}\n")
        f.write("\n")

        f.write("ANALYSIS RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write("Sphere Analysis Results (NEMA NU 2-2018 Section 7.4.1)\n\n")

        f.write(
            f"{'Diameter':<10} {'Q_H (%)':<10} {'N (%)':<10} {'C_H':<12} {'C_B':<12} {'SD_B':<12}\n"
        )
        f.write(
            f"{'(mm)':<10} {'':<10} {'':<10} {'(counts)':<12} {'(counts)':<12} {'(counts)':<12}\n"
        )
        f.write("-" * 76 + "\n")

        sorted_results = sorted(results, key=lambda x: x["diameter_mm"], reverse=True)

        for result in sorted_results:
            f.write(
                f"{result['diameter_mm']:<10.0f} "
                f"{result['percentaje_constrast_QH']:<10.2f} "
                f"{result['background_variability_N']:<10.2f} "
                f"{result['avg_hot_counts_CH']:<12.6f} "
                f"{result['avg_bkg_counts_CB']:<12.6f} "
                f"{result['bkg_std_dev_SD']:<12.6f}\n"
            )

        f.write("\n")

        f.write("LEGEND:\n")
        f.write("-" * 40 + "\n")
        f.write("Q_H (%)  : Percent Contrast (Hot sphere)\n")
        f.write("N (%)    : Percent Background Variability\n")
        f.write("C_H      : Mean counts in hot sphere\n")
        f.write("C_B      : Mean background counts\n")
        f.write("SD_B     : Standard deviation of background\n")
        f.write("\n")

        f.write("NEMA NU 2-2018 FORMULAS:\n")
        f.write("-" * 40 + "\n")
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

        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 40 + "\n")
        avg_contrast = sum(r["percentaje_constrast_QH"] for r in results) / len(results)
        avg_variability = sum(r["background_variability_N"] for r in results) / len(
            results
        )
        f.write(f"Average Percent Contrast: {avg_contrast:.2f}%\n")
        f.write(f"Average Background Variability: {avg_variability:.2f}%\n")
        f.write(f"Number of spheres analyzed: {len(results)}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("End of Report\n")
        f.write("=" * 80 + "\n")


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
    output_path: Path,
    cfg: yacs.config.CfgNode,
    input_image_path: Path,
    voxel_spacing: Tuple[float, float, float],
    lung_results: Dict[str, Any],
    plot_path: Optional[Path] = None,
    rois_loc_path: Optional[Path] = None,
    boxplot_path: Optional[Path] = None,
) -> None:
    """
    Generates a PDF report for NEMA quality analysis results using ReportLab.

    Creates a formatted PDF summarizing sphere and lung analysis results, configuration, and relevant plots.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of analysis results for each sphere.
    output_path : Path
        Destination path for saving the PDF report.
    cfg : yacs.config.CfgNode
        Configuration object used for the analysis.
    input_image_path : Path
        Path to the input image file.
    voxel_spacing : Tuple[float, float, float]
        Voxel spacing used during analysis.
    lung_results : Dict[str, Any]
        Dictionary of results from the lung analysis.
    plot_path : Path
        Path to the summary plot image file.
    rois_loc_path : Path
        Path to the ROIs plot image file.
    boxplot_path : Path
        Path to the boxplot image file.

    Returns
    -------
    None
        This function does not return a value; it writes the PDF report to disk.
    """
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

    elements.append(Paragraph("Image Quality Analysis Report", title_style))
    elements.append(Spacer(1, 0.2 * inch))

    summary = (
        f"<b>Summary of Analysis</b><br/>"
        f"\u2022 Date of Generation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        f"\u2022 Input Image: <font face='Courier'>{str(input_image_path)}</font><br/>"
        f"\u2022 Voxel Spacing: {voxel_spacing[0]:.4f} × {voxel_spacing[1]:.4f} × {voxel_spacing[2]:.4f} mm"
    )
    elements.append(Paragraph(summary, body_style))
    elements.append(Spacer(1, 0.18 * inch))

    background_val = getattr(cfg.ACTIVITY, "BACKGROUND", None)
    hot_val = getattr(cfg.ACTIVITY, "HOT", None)
    total_val = getattr(cfg.ACTIVITY, "ACTIVITY_TOTAL", None)
    units_val = getattr(cfg.ACTIVITY, "UNITS", "N/A")

    if background_val is None or background_val == 0.0:
        background_text = "not reported"
    else:
        background_text = f"{background_val:.7f} {units_val}"

    if hot_val is None or hot_val == 0.0:
        hot_text = "not reported"
    else:
        hot_text = f"{hot_val:.7f} {units_val}"

    if total_val is None or total_val == "0.0 mCi" or total_val == "0.0 MBq":
        total_text = "not reported"
    else:
        total_text = f"{total_val}"

    bg_text = (
        "<b>Activity Concentrations</b><br/>"
        f"\u2022 Background: {background_text}<br/>"
        f"\u2022 Hot Spheres: {hot_text}<br/>"
        f"\u2022 Activity Ratio (Hot/Background): {getattr(cfg.ACTIVITY, 'RATIO', 'N/A')}"
        f"<br/>\u2022 Total Activity: {total_text}"
    )
    elements.append(Paragraph(bg_text, body_style))
    elements.append(Spacer(1, 0.18 * inch))

    acq_time = getattr(
        getattr(cfg, "ACQUISITION", {}), "EMMISION_IMAGE_TIME_MINUTES", None
    )
    if acq_time is not None:
        acq_text = f"<b>Acquisition Parameters</b><br/>• Emission Imaging Time: {acq_time} minutes"
        elements.append(Paragraph(acq_text, body_style))
        elements.append(Spacer(1, 0.18 * inch))

    values = list(lung_results.values())
    average = float(np.mean(values))
    elements.append(Paragraph("<b>Summary Statistics</b>", header_style))
    avg_contrast = (
        sum(r["percentaje_constrast_QH"] for r in results) / len(results)
        if results
        else 0.0
    )
    avg_variability = (
        sum(r["background_variability_N"] for r in results) / len(results)
        if results
        else 0.0
    )
    elements.append(
        Paragraph(f"Average Percent Contrast: {avg_contrast:.2f}%", body_style)
    )
    elements.append(
        Paragraph(f"Average Background Variability: {avg_variability:.2f}%", body_style)
    )
    elements.append(
        Paragraph(f"Average error in Lung Insert: {average:.2f} %", body_style)
    )
    elements.append(
        Paragraph(f"Number of spheres analyzed: {len(results)}", body_style)
    )

    elements.append(PageBreak())

    elements.append(Paragraph("<b>Analysis Results</b>", header_style))
    elements.append(Spacer(1, 0.1 * inch))

    table_data = [
        [
            "Sphere Diameter (mm)",
            "Percent Contrast Q_H (%)",
            "Background Variability N (%)",
        ]
    ]
    for result in results:
        table_data.append(
            [
                str(result.get("diameter_mm", "Unknown")),
                f"{result.get('percentaje_constrast_QH', 'N/A'):.2f}",
                f"{result.get('background_variability_N', 'N/A'):.2f}",
            ]
        )
    col_widths = [2.0 * inch, 2.5 * inch, 2.5 * inch]
    table = Table(table_data, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ]
        )
    )
    elements.append(table)

    if plot_path and Path(plot_path).exists():
        elements.append(Paragraph("<b>Hot Sphere Plot</b>", header_style))
        elements.append(Image(str(plot_path), width=8 * inch, height=4 * inch))
        elements.append(Spacer(1, 0.2 * inch))

    elements.append(PageBreak())

    if boxplot_path and Path(boxplot_path).exists():
        elements.append(Paragraph("<b>Lung Insert Plot</b>", header_style))
        elements.append(Image(str(boxplot_path), width=4 * inch, height=4 * inch))
        elements.append(Spacer(1, 0.2 * inch))

    table_data_2 = [
        ["Statistic", "Value"],
        ["Total slices", f"{len(values)}"],
        ["Min", f"{float(np.min(values)):.2f}"],
        ["Max", f"{float(np.max(values)):.2f}"],
        ["Mean", f"{float(np.mean(values)):.2f}"],
        ["Median", f"{float(np.median(values)):.2f}"],
        ["Standard deviation", f"{float(np.std(values)):.2f}"],
        ["IQR", f"{float(np.percentile(values, 75) - np.percentile(values, 25)):.2f}"],
    ]

    col_widths = [2.5 * inch, 1.5 * inch]

    table2 = Table(table_data_2, colWidths=col_widths)

    table2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ]
        )
    )

    elements.append(table2)

    elements.append(PageBreak())

    if rois_loc_path and Path(rois_loc_path).exists():
        elements.append(Paragraph("<b>ROIs Location</b>", header_style))
        elements.append(Image(str(rois_loc_path), width=3.5 * inch, height=3.5 * inch))
        elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("<b>Legend and Formulas</b>", header_style))
    legend = [
        "• Q<sub>H</sub> (%): Percent contrast for hot spheres",
        "• N (%): Background variability",
        "• C<sub>H</sub>: Mean counts in hot sphere",
        "• C<sub>B</sub>: Mean background counts",
        "• SD<sub>B</sub>: Standard deviation of background counts",
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


def generate_plots(
    results: List[Dict[str, Any]],
    output_dir: Path,
    cfg: yacs.config.CfgNode,
) -> None:
    """
    Generates the plot for the results of the NEMA analysis.

    Creates and saves a figure summarizing sphere analysis results using the given configuration.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of analysis results for each sphere.
    output_path : Path
        Destination path for saving the plot.
    cfg : yacs.config.CfgNode
        Configuration object used for the analysis.

    Returns
    -------
    None
        This function does not return a value; the plot is saved to disk.
    """
    df = pd.DataFrame(results)

    csv_path = output_dir.parent / "csv" / "analysis_results.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"Results saved to CSV at: {csv_path}")

    plt.style.use(cfg.STYLE.PLT_STYLE)
    plt.rcParams.update(cfg.STYLE.RCPARAMS)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharex=True)

    for ax, yvar, title, ylabel in zip(
        axes,
        ["percentaje_constrast_QH", "background_variability_N"],
        ["Percent Contrast by Diameter", "Percent Background Variability by Diameter"],
        ["Percent Contrast (Hot sphere) [%]", "Percent Background Variability [%]"],
    ):
        ax.plot(
            df["diameter_mm"],
            df[yvar],
            marker="o",
            color="#023743FF",
            linestyle="-",
            alpha=1.0,
            linewidth=4.0,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=2.0,
        )

        ax.set_title(title, fontsize=16, weight=cfg.STYLE.LEGEND.FONTWEIGHT)
        ax.set_ylabel(
            ylabel,
            fontsize=14,
            weight=cfg.STYLE.LEGEND.FONTWEIGHT,
            labelpad=cfg.STYLE.LEGEND.LABELPAD,
        )
        ax.set_xlabel(
            "Sphere Diameter [mm]",
            fontsize=14,
            weight=cfg.STYLE.LEGEND.FONTWEIGHT,
            labelpad=cfg.STYLE.LEGEND.LABELPAD,
        )

        ax.grid(
            True,
            linestyle=cfg.STYLE.GRID.LINESTYLE,
            alpha=cfg.STYLE.GRID.ALPHA,
            color=cfg.STYLE.GRID.COLOR,
            linewidth=cfg.STYLE.GRID.LINEWIDTH,
        )

        ax.tick_params(axis="both", labelsize=20, width=1.2)
        ax.tick_params(axis="x", rotation=0)

        ax.set_axisbelow(True)

        ax.set_xticks(sorted(df["diameter_mm"].unique()))

        ax.set_facecolor("#fafafa")

    plt.tight_layout(rect=(0, 0.1, 1, 0.92))

    output_path = output_dir / "analysis_plot.png"
    plt.savefig(
        str(output_path),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        format="png",
    )
    plt.close()


def generate_boxplot_with_mean_std(
    data_dict: Dict[int, float],
    output_dir: Path,
    cfg: yacs.config.CfgNode,
) -> None:
    """
    Generates a high-contrast boxplot from a dataset dictionary with integer keys and float values.

    Displays means as red dots and standard deviations as error bars, and saves the plot to the specified directory.

    Parameters
    ----------
    data_dict : Dict[int, float]
        Dictionary containing the dataset (e.g., for different spheres or samples).
    output_dir : Path
        Directory path for saving the generated plot.
    cfg : yacs.config.CfgNode
        Configuration object used for the analysis.

    Returns
    -------
    None
        This function does not return a value; the plot is saved to disk.
    """
    plt.style.use(cfg.STYLE.PLT_STYLE)
    plt.rcParams.update(cfg.STYLE.RCPARAMS)

    data = list(data_dict.values())
    std_dev = float(np.std(data))

    csv_path = output_dir.parent / "csv" / "lung_results.csv"
    df = pd.DataFrame({"data": data})
    df.to_csv(csv_path, index=False)
    logging.info(f"Lung Results saved to CSV at: {csv_path}")

    label = f"{(output_dir.stem).capitalize()}"

    if label == "Png":
        label = ""

    plt.figure(figsize=(10, 8))
    bp = plt.violinplot(
        data,
        positions=[1],
        showmeans=True,
        showmedians=False,
        showextrema=False,
    )

    violin_bodies = cast(List[Any], bp["bodies"])
    for patch in violin_bodies:
        patch.set_facecolor("#023743FF")
        patch.set_alpha(0.7)
        patch.set_linewidth(1)
        patch.set_edgecolor("black")

    bp["cmeans"].set_color("white")
    bp["cmeans"].set_linewidth(3)

    exp_values = df["data"].values
    color = "#023743FF"

    jitter = np.random.normal(0, 0.04, size=len(exp_values))
    positions = np.full(len(exp_values), 1) + jitter

    plt.scatter(
        positions,
        exp_values,
        color=color,
        s=40,
        alpha=0.8,
        edgecolor="white",
        linewidth=1,
        zorder=10,
    )

    mean_val = np.mean(exp_values)

    plt.gca().text(
        1,
        max(exp_values) + std_dev / 4,
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

    plt.xticks([1], [label])
    plt.xlabel(
        "Lung Insert Accuracy Distribution",
        fontsize=14,
        fontweight=cfg.STYLE.LEGEND.FONTWEIGHT,
        labelpad=cfg.STYLE.LEGEND.LABELPAD,
    )
    plt.ylabel(
        "Accuracy of Correction in Lung Insert (%)",
        fontsize=14,
        fontweight=cfg.STYLE.LEGEND.FONTWEIGHT,
        labelpad=cfg.STYLE.LEGEND.LABELPAD,
    )

    plt.tick_params(axis="both", labelsize=12, width=1.2)
    plt.tick_params(axis="x", rotation=0)

    plt.grid(
        True,
        linestyle=cfg.STYLE.GRID.LINESTYLE,
        alpha=cfg.STYLE.GRID.ALPHA,
        color=cfg.STYLE.GRID.COLOR,
        linewidth=cfg.STYLE.GRID.LINEWIDTH,
    )
    plt.gca().set_axisbelow(True)

    plt.gca().set_facecolor("#fafafa")

    plt.tight_layout()

    output_path = output_dir / "boxplot_with_mean_std.png"
    plt.savefig(
        str(output_path),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()


def generate_rois_plots(
    image: npt.NDArray[Any], output_dir: Path, cfg: yacs.config.CfgNode
) -> None:
    """
    Generates a plot of the ROIs for the input image.

    Creates and saves a visualization highlighting the regions of interest (ROIs) defined for the analysis.

    Parameters
    ----------
    image : numpy.ndarray
        The image loaded for analysis.
    output_dir : Path
        Directory path for saving the resulting plot.
    cfg : yacs.config.CfgNode
        Configuration object used for the analysis.

    Returns
    -------
    None
        This function does not return a value; the plot is saved to disk.
    """
    rois = cfg.PHANTHOM.ROI_DEFINITIONS_MM
    background_offset = [
        (y * cfg.ROIS.ORIENTATION_YX[0], x * cfg.ROIS.ORIENTATION_YX[1])
        for y, x in cfg.ROIS.BACKGROUND_OFFSET_YX
    ]
    pixel_spacing = cfg.ROIS.SPACING

    fig, ax2 = plt.subplots(figsize=(10, 10))
    ax2.imshow(image[cfg.ROIS.CENTRAL_SLICE], cmap="binary", origin="lower")

    for roi in rois:
        y, x = roi["center_yx"]
        radius_pix = (roi["diameter_mm"] / 2) / pixel_spacing
        circle = Circle(
            (x, y),
            radius_pix,
            edgecolor=roi["color"],
            alpha=roi["alpha"],
            lw=2,
            label=roi["name"],
        )
        if roi["name"] == "hot_sphere_37mm":
            centro_37 = roi["center_yx"]
        ax2.add_patch(circle)
        ax2.plot(x, y, "+", color=roi["color"], markersize=12)

    background_radius = (37 / 2) / pixel_spacing
    for dy, dx in background_offset:
        background_y, background_x = centro_37[0] + dy, centro_37[1] + dx
        circle = Circle(
            (background_x, background_y),
            background_radius,
            edgecolor="orange",
            facecolor="none",
            lw=2,
            linestyle="--",
            label="Background" if (dy, dx) == background_offset[0] else "",
        )
        ax2.add_patch(circle)
        ax2.plot(background_x, background_y, "o", color="orange", markersize=7)

    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower right",
        fontsize=12,
        framealpha=0.7,
    )
    ax2.set_aspect("equal")
    ax2.set_xlabel("X (pixels)")
    ax2.set_ylabel("Y (pixels)")
    ax2.grid(False)
    plt.tight_layout()

    output_path = output_dir / "rois_location.png"
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close()


def generate_transverse_sphere_plots(
    image: npt.NDArray[Any], output_dir: Path, cfg: yacs.config.CfgNode
) -> None:
    """
    Generates a plot of the transverse view of all the spheres.

    Parameters
    ----------
    image : numpy.ndarray
        The image loaded for analysis.
    output_dir : Path
        Directory path for saving the resulting plot.
    cfg : yacs.config.CfgNode
        Configuration object used for the analysis.

    Returns
    -------
    None
        This function does not return a value; the plot is saved to disk.
    """
    rois = cfg.PHANTHOM.ROI_DEFINITIONS_MM

    _, axs = plt.subplots(1, len(rois), figsize=(3 * len(rois), 3))
    if len(rois) == 1:
        axs = [axs]

    for ax, roi in zip(axs, rois):
        y, x = roi["center_yx"]

        crop = image[
            cfg.ROIS.CENTRAL_SLICE, y - 10 : y + 10, x - 10 : x + 10  # noqa: E203
        ]
        # crop = crop >= 0.41 * np.max(crop)
        # logging.debug(f"Unique values in crop for {roi['name']}: {np.unique(crop)}")
        ax.imshow(crop, cmap="binary", origin="lower")
        ax.axis("off")
        ax.set_title(roi["diameter_mm"], y=-0.15)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.tight_layout(pad=0)

    output_path = output_dir / "transverse_sphere.png"
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close()


def generate_coronal_sphere_plots(
    image: npt.NDArray[Any], output_dir: Path, cfg: yacs.config.CfgNode
) -> None:
    """
    Generates a plot of the coronal view of all the spheres.

    Parameters
    ----------
    image : numpy.ndarray
        The image loaded for analysis.
    output_dir : Path
        Directory path for saving the resulting plot.
    cfg : yacs.config.CfgNode
        Configuration object used for the analysis.

    Returns
    -------
    None
        This function does not return a value; the plot is saved to disk.
    """
    rois = cfg.PHANTHOM.ROI_DEFINITIONS_MM

    fig, axs = plt.subplots(1, len(rois), figsize=(3 * len(rois), 3))
    if len(rois) == 1:
        axs = [axs]

    for ax, roi in zip(axs, rois):
        y, x = roi["center_yx"]

        crop = image[
            cfg.ROIS.CENTRAL_SLICE - 10 : cfg.ROIS.CENTRAL_SLICE + 10,  # noqa: E203
            y,
            x - 10 : x + 10,  # noqa: E203
        ]

        ax.imshow(crop, cmap="binary", origin="lower")
        ax.axis("off")
        ax.set_title(roi["diameter_mm"], y=-0.15)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.tight_layout(pad=0)

    output_path = output_dir / "coronal_sphere.png"
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close()


def generate_torso_plot(
    image: npt.NDArray[Any], output_dir: Path, cfg: yacs.config.CfgNode
) -> None:
    """
    Generates a plot of the input image.

    Parameters
    ----------
    image : numpy.ndarray
        The image loaded for analysis.
    output_dir : Path
        Directory path for saving the resulting plot.
    cfg : yacs.config.CfgNode
        Configuration object used for the analysis.

    Returns
    -------
    None
        This function does not return a value; the plot is saved to disk.
    """
    fig, ax2 = plt.subplots(figsize=(10, 10))
    ax2.imshow(image[cfg.ROIS.CENTRAL_SLICE], cmap="binary", origin="lower")

    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.grid(False)
    plt.tight_layout()

    output_path = output_dir / "torso.png"
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close()

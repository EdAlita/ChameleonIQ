"""
Command Line Interface for NEMA Analysis Tool

This module provides the command-line interface functionality for the NEMA NU 2-2018 image quality analysis tool.

Author: Edwing Ulin-Briseno
Date: 2025-07-16
"""

import argparse
import datetime
import logging
import os
import re
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any, Optional, Tuple
from venv import logger

import matplotlib
import numpy as np
import numpy.typing as npt
import yacs.config
from rich.highlighter import Highlighter
from rich.logging import RichHandler

from config.defaults import get_cfg_defaults

from .analysis import calculate_nema_metrics, calculate_nema_metrics_nu4_2008
from .io import load_nii_image
from .phantom import NemaPhantom
from .reporting import (
    generate_boxplot_with_mean_std,
    generate_coronal_sphere_plots,
    generate_crc_plots_nu4,
    generate_iq_plot,
    generate_plots,
    generate_reportlab_report,
    generate_reportlab_report_nu4,
    generate_rois_plots,
    generate_rois_plots_coronal,
    generate_spillover_barplot_nu4,
    generate_torso_plot,
    generate_transverse_sphere_plots,
    save_results_to_txt,
    save_results_to_txt_nu4,
)

# Set environment variables for headless operation only when no display available
is_headless = not os.environ.get("DISPLAY") and sys.platform != "win32"
if is_headless:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

matplotlib.use("Agg")  # Set non-interactive backend before importing pyplot


class NumberHighlighter(Highlighter):
    """Highlight numeric values in log messages without highlighting paths."""

    _number_pattern = re.compile(
        r"(?<![A-Za-z0-9_/.-])([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?%?)(?![A-Za-z0-9_/.-])"
    )

    def highlight(self, text: Any) -> None:
        for match in self._number_pattern.finditer(text.plain):
            text.stylize("bold cyan", match.start(1), match.end(1))


def _log_section(title: str) -> None:
    logging.info(f"── {title} ──")


def _log_kv(key: str, value: Any, key_width: int = 20) -> None:
    logging.info(f"{key + ':':<{key_width}} {value}")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="ChameleonIQ Quant Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Basic analysis
    chameleoniq_quant input.nii --config custom_config.yaml --output results.txt

    """,
    )

    # Required arguments
    parser.add_argument(
        "input_image", type=str, help="Path to input NIfTI image file (.nii or .nii.gz)"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output file for results",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to custom YAML configuration file. Check defaults/config.yaml for reference or in HOW IT WORKS section from Documentation.",
    )

    parser.add_argument(
        "--standard",
        choices=["NU_2_2018", "NU_4_2008", "DEDICATED_IQ"],
        default="NU_2_2018",
        help="NEMA standard to use for phantom definitions (default: NU_2_2018)",
    )

    # Optional arguments
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save visualization images of ROI masks and analysis regions",
    )

    parser.add_argument(
        "--spacing", nargs=3, type=float, help="Voxel spacing in mm (x, y, z)"
    )

    parser.add_argument(
        "--visualizations-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualization images (default: visualizations)",
    )

    parser.add_argument(
        "--advanced-metrics",
        "-a",
        action="store_true",
        help="Calculate advanced segmentation metrics",
    )

    parser.add_argument(
        "--gt-image",
        type=str,
        help="Path to ground truth NIfTI image file for advanced metrics",
    )

    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    parser.add_argument(
        "--outliner",
        action="store_true",
        help="Remove outliers from the image data by clipping the maximum value to the second highest value",
        default=False,
    )

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('ChameleonIQ')}"
    )

    return parser


def setup_logging(log_level: int = 20, output_path: Optional[str] = None) -> None:
    """Configuration logging for the application."""

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_path:
        output_path_obj = Path(output_path)
        run_name = output_path_obj.stem
        log_filename = f"{run_name}_{timestamp}.log"
        log_dir = output_path_obj.parent / "logs"
    else:
        log_filename = f"{timestamp}.log"
        log_dir = Path("logs")

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / log_filename

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[
            logging.FileHandler(log_file_path, mode="w", encoding="utf-8"),
            RichHandler(
                rich_tracebacks=True,
                markup=True,
                highlighter=NumberHighlighter(),
                keywords=[
                    "Input image:",
                    "Output file:",
                    "Configuration:",
                    "NEMA standard:",
                    "Loading configuration:",
                    "Saving results to:",
                    "Results saved to:",
                    "RC Results:",
                    "Spillover Ratios:",
                    "Tool:",
                    "Uniformity Results:",
                    "RC Results:" "Loading NIfTI image:",
                    "Phantom initialized with:",
                    "Using Ratio from config:",
                    "Average of Accuracy Corrections:",
                    "Elapsed:",
                ],
            ),
        ],
    )

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("report_lab").setLevel(logging.WARNING)

    logging.info(f"Logging initialized. Log file: {log_file_path}")


def load_configuration(
    config_path: Optional[str], standard: str
) -> yacs.config.CfgNode:
    """Load configuration from file or use defaults."""
    cfg = get_cfg_defaults()

    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        cfg.merge_from_file(config_path)

    return cfg


def get_image_properties(
    image_data: npt.NDArray[Any],
    affine: Optional[npt.NDArray[Any]],
    spacing_override: Optional[Tuple[float, float, float]],
) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
    """Extract image dimensions and voxel spacing."""
    image_dims = (image_data.shape[0], image_data.shape[1], image_data.shape[2])

    if spacing_override:
        voxel_spacing = spacing_override
        logging.info(f"Using provided voxel spacing: {voxel_spacing} mm")
    elif affine is not None:
        voxel_spacing = (
            float(np.abs(affine[0, 0])),
            float(np.abs(affine[1, 1])),
            float(np.abs(affine[2, 2])),
        )
        logging.debug(f"Extracted voxel spacing from image: {voxel_spacing} mm")
    else:
        # Default spacing
        voxel_spacing = (1.0, 1.0, 1.0)
        logging.warning(
            "No voxel spacing information available. Using default: (1.0, 1.0, 1.0) mm"
        )

    logging.debug(f"Image dimensions: {image_dims}")

    return image_dims, voxel_spacing


def run_analysis(args: argparse.Namespace) -> int:
    """Run the NEMA analysis with the provided arguments."""
    try:
        start_time = datetime.datetime.now()

        numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
        setup_logging(log_level=numeric_level)

        _log_section("Run")
        _log_kv("Tool", "ChameleonIQ")
        _log_kv("Input image", args.input_image)
        _log_kv("Output file", args.output)
        _log_kv("Configuration", args.config)
        _log_kv("NEMA standard", args.standard)

        input_path = Path(args.input_image)
        if not input_path.exists():
            error_msg = f"Input image file not found: {args.input_image}"
            logging.error(error_msg)
            print(f"ERROR: {error_msg}")
            return 1

        if not input_path.suffix.lower() in [".nii", ".gz"]:
            error_msg = (
                f"Input file must be a NIfTI file (.nii or .nii.gz): {args.input_image}"
            )
            logging.error(error_msg)
            print(f"ERROR: {error_msg}")
            return 1

        try:
            cfg = load_configuration(args.config, args.standard)
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            if args.log_level == "DEBUG":
                import traceback

                logging.error(traceback.format_exc())
            print(f"ERROR: Failed to load configuration: {e}")
            return 1

        _log_section("Load")
        _log_kv("Loading NIfTI image", input_path.name)
        try:
            image_data, affine = load_nii_image(
                input_path,
                return_affine=True,
                inverse_axes=cfg.ROIS.INVERSE_AXES,
                outliner=args.outliner,
            )

            logging.debug("Image loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load image: {e}")
            if args.log_level == "DEBUG":
                import traceback

                logging.error(traceback.format_exc())
            print(f"ERROR: Failed to load image: {e}")
            return 1

        try:
            image_dims, voxel_spacing = get_image_properties(
                image_data, affine, args.spacing
            )
        except Exception as e:
            logging.error(f"Failed to extract image properties: {e}")
            if args.log_level == "DEBUG":
                import traceback

                logging.error(traceback.format_exc())
            print(f"ERROR: Failed to extract image properties: {e}")
            return 1

        _log_section("Initialize")
        try:
            phantom = NemaPhantom(cfg, image_dims, voxel_spacing)
            _log_kv("Phantom initialized with", f"{len(phantom.rois)} ROIs")
        except Exception as e:
            logging.error(f"Failed to initialize phantom: {e}")
            if args.log_level == "DEBUG":
                import traceback

                logging.error(traceback.format_exc())
            print(f"ERROR: Failed to initialize phantom: {e}")
            return 1

        # Perform NEMA analysis
        _log_section("Analysis")
        if args.standard != "NU_4_2008":
            _log_kv("Using Ratio from config", cfg.ACTIVITY.RATIO)
        try:
            if args.standard == "NU_4_2008":
                crc_results, spillover_results, uniformity_results = (
                    calculate_nema_metrics_nu4_2008(
                        image_data,
                        phantom,
                        cfg,
                        save_visualizations=args.save_visualizations,
                        visualizations_dir=args.visualizations_dir,
                    )
                )
            else:
                results, lung_results = calculate_nema_metrics(
                    image_data,
                    phantom,
                    cfg,
                    save_visualizations=args.save_visualizations,
                    visualizations_dir=args.visualizations_dir,
                    protocol=args.standard,
                )
                if args.standard == "NU_2_2018":
                    values = list(lung_results.values())
                    average = float(np.mean(values))
                    _log_kv("Average of Accuracy Corrections", f"{average:.3f} %")
        except Exception as e:
            logging.error(f"Failed to perform analysis: {e}")
            if args.log_level == "DEBUG":
                import traceback

                logging.error(traceback.format_exc())
            print(f"ERROR: Failed to perform analysis: {e}")
            return 1

        output_path = Path(args.output)
        png_dir = output_path.parent / "png"
        png_dir.mkdir(parents=True, exist_ok=True)
        csv_dir = output_path.parent / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)

        if args.standard == "NU_4_2008":
            try:
                generate_crc_plots_nu4(crc_results=crc_results, output_dir=png_dir, cfg=cfg)  # type: ignore[arg-type]
                generate_iq_plot(image=image_data, output_dir=png_dir, cfg=cfg)
                generate_spillover_barplot_nu4(spillover_ratio=spillover_results, output_dir=png_dir, cfg=cfg)  # type: ignore[arg-type]
                logging.debug("Plots saved successfully")
            except Exception as e:
                logging.error(f"Failed to generate plots: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    import traceback

                    logging.error(traceback.format_exc())
                print(f"ERROR: Failed to generate plots: {e}")
                return 1
        else:
            try:
                generate_plots(
                    results=results, output_dir=png_dir, cfg=cfg, protocol=args.standard
                )

                generate_rois_plots(
                    image=image_data,
                    output_dir=png_dir,
                    cfg=cfg,
                    protocol=args.standard,
                )

                generate_rois_plots_coronal(
                    image=image_data,
                    output_dir=png_dir,
                    cfg=cfg,
                    protocol=args.standard,
                )

                if args.standard == "NU_2_2018":
                    generate_boxplot_with_mean_std(
                        data_dict=lung_results, output_dir=png_dir, cfg=cfg
                    )
                    generate_coronal_sphere_plots(
                        image=image_data, output_dir=png_dir, cfg=cfg
                    )
                    generate_transverse_sphere_plots(
                        image=image_data, output_dir=png_dir, cfg=cfg
                    )
                generate_torso_plot(image=image_data, output_dir=png_dir, cfg=cfg)
                logging.debug("Plots saved successfully")
            except Exception as e:
                logging.error(f"Failed to generate plots: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    import traceback

                    logging.error(traceback.format_exc())
                print(f"ERROR: Failed to generate plots: {e}")
                return 1
        if args.standard == "NU_4_2008":
            try:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plot_path = output_path.parent / "png" / "pc_plot.png"
                iq_rois_path = output_path.parent / "png" / "iq_rois.png"
                spillover_path = output_path.parent / "png" / "spillover_ratio.png"

                save_results_to_txt_nu4(
                    crc_results=crc_results,  # type: ignore[arg-type]
                    spillover_results=spillover_results,  # type: ignore[arg-type]
                    uniformity_results=uniformity_results,
                    output_path=output_path,
                    cfg=cfg,
                    input_image_path=input_path,
                    voxel_spacing=voxel_spacing,
                )

                pdf_output_path = output_path.with_suffix(".pdf")
                generate_reportlab_report_nu4(
                    crc_results=crc_results,  # type: ignore[arg-type]
                    spillover_results=spillover_results,  # type: ignore[arg-type]
                    uniformity_results=uniformity_results,
                    output_path=pdf_output_path,
                    cfg=cfg,
                    input_image_path=input_path,
                    voxel_spacing=voxel_spacing,
                    plot_path=plot_path,
                    rois_loc_path=iq_rois_path,
                    spillover_ratio_path=spillover_path,
                )
                logging.debug("Results saved successfully")

            except Exception as e:
                logging.error(f"Failed to save Results: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    import traceback

                    logging.error(traceback.format_exc())
                print(f"ERROR: Failed to save Results: {e}")
                return 1
        else:
            try:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plot_path = output_path.parent / "png" / "analysis_plot.png"
                rois_loc_path = output_path.parent / "png" / "rois_location.png"
                boxplot_path = output_path.parent / "png" / "boxplot_with_mean_std.png"

                save_results_to_txt(
                    results,
                    output_path,
                    cfg,
                    input_path,
                    voxel_spacing,
                    protocol=args.standard,
                )

                lung_results_any = {str(k): v for k, v in lung_results.items()}

                pdf_output_path = output_path.with_suffix(".pdf")
                generate_reportlab_report(
                    results,
                    pdf_output_path,
                    cfg,
                    input_path,
                    voxel_spacing,
                    lung_results_any,
                    plot_path,
                    rois_loc_path,
                    boxplot_path,
                    protocol=args.standard,
                )
                logging.debug("Results saved successfully")
            except Exception as e:
                logging.error(f"Failed to save results: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    import traceback

                    logging.error(traceback.format_exc())
                print(f"ERROR: Failed to save results: {e}")
                return 1

        if args.advanced_metrics:
            if not args.gt_image:
                error_msg = "Ground truth image path must be provided with --gt-image for advanced metrics"
                logging.error(error_msg)
                print(f"ERROR: {error_msg}")
                return 1

            gt_path = Path(args.gt_image)
            if not gt_path.exists():
                error_msg = f"Ground truth image file not found: {args.gt_image}"
                logging.error(error_msg)
                print(f"ERROR: {error_msg}")
                return 1

            try:
                logging.info("Loading ground truth image...")
                gt_data, _ = load_nii_image(gt_path, return_affine=True)
                logging.info("Ground truth image loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load ground truth image: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    import traceback

                    logging.error(traceback.format_exc())
                print(f"ERROR: Failed to load ground truth image: {e}")
                return 1

            try:
                logging.info("Calculating advanced segmentation metrics...")
                from .analysis import calculate_advanced_metrics

                mask_data = image_data > 0.41 * np.max(image_data)
                mask_gt = gt_data > 0.41 * np.max(gt_data)
                advanced_metrics = calculate_advanced_metrics(
                    mask_data,
                    mask_gt,
                    (
                        "Dice",
                        "Jaccard",
                        "VS",
                        "1-VOI",
                        "HD",
                        "ASSD",
                        "1-GCE",
                        "Kappa",
                        "MI",
                        "RI",
                        "ASSD",
                        "Recall",
                        "F1",
                    ),
                    cfg,
                )
                import pandas as pd

                advanced_metrics_df = pd.DataFrame([advanced_metrics])
                advanced_metrics_path = csv_dir / "advanced_metrics.csv"
                advanced_metrics_df.to_csv(advanced_metrics_path, index=False)
                logging.info(f"Saving advanced metrics in: {advanced_metrics_path}")
                logging.info("Advanced metrics calculated successfully")
            except Exception as e:
                logging.error(f"Failed to calculate advanced metrics: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    import traceback

                    logging.error(traceback.format_exc())
                print(f"ERROR: Failed to calculate advanced metrics: {e}")
                return 1
        elapsed_seconds = (datetime.datetime.now() - start_time).total_seconds()

        _log_section("Summary")
        _log_kv("Results saved to", Path(args.output).parent)
        _log_kv("Elapsed", f"{elapsed_seconds:.2f} s")

        if args.save_visualizations:
            print(f"  Visualizations saved to: {args.visualizations_dir}/")

        return 0

    except KeyboardInterrupt:
        logging.info("Analysis interrupted by user")
        print("Analysis interrupted by user")
        return 130
    except Exception as e:
        logging.error("Unexpected error:")
        logging.exception(e)
        return 1


def main(cli_args: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(cli_args)
    return run_analysis(args)


if __name__ == "__main__":
    sys.exit(main())

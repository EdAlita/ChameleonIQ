"""
Command Line Interface for NEMA Analysis Tool

This module provides the command-line interface functionality for the NEMA NU 2-2018 image quality analysis tool.

Author: Edwing Yair Ulin Briseño
Date: 2025-07-28
"""

import argparse
import datetime
import logging
import os
import re
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import yacs.config
from rich.logging import RichHandler

from config.defaults import get_cfg_defaults
from nema_quant.analysis import (
    calculate_nema_metrics,
    calculate_nema_metrics_nu4_2008,
    calculate_weighted_cbr_from,
)
from nema_quant.io import load_nii_image
from nema_quant.phantom import NemaPhantom

from .reporting import (
    generate_boxplot_with_mean_std,
    generate_cbr_convergence_plot,
    generate_crc_convergence_plot_nu4_iter,
    generate_pc_vs_bg_plot,
    generate_plots,
    generate_reportlab_report,
    generate_reportlab_report_nu4_iter,
    generate_spillover_convergence_plot_nu4_iter,
    generate_uniformity_convergence_plot_nu4_iter,
    generate_wcbr_convergence_plot,
    save_results_to_txt,
    save_results_to_txt_nu4_iter,
)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="ChameleonIQ Quant Iterations Domain Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Basic analysis
    chameleoniq_quant_iter input_dir/ --config custom_config.yaml --output results.txt
    """,
    )

    parser.add_argument(
        "input_path", type=str, help="Path for the input of the files iterations"
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
        choices=["NU_2_2018", "NU_4_2008"],
        default="NU_2_2018",
        help="NEMA standard to use for phantom definitions (default: NU_2_2018)",
    )

    # Optional arguments
    parser.add_argument(
        "--spacing", nargs=3, type=float, help="Voxel spacing in mm (x, y, z)"
    )

    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save visualization images of ROI masks and analysis regions",
    )

    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
        choices=[10, 20, 30, 40, 50],
        help="Set logging level: 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL (default: 20)",
    )

    parser.add_argument(
        "--visualizations-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualization images (default: visualizations)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('ChameleonIQ')}"
    )

    return parser


def setup_logging(log_level: int = 20, output_path: Optional[str] = None) -> None:
    """Configure logging for the application."""

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
            RichHandler(rich_tracebacks=True),
        ],
    )

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("report_lab").setLevel(logging.WARNING)

    logging.info(f"Log file created: {log_file_path}")


def load_configuration(
    config_path: Optional[str], standard: str = "NU_2_2018"
) -> yacs.config.CfgNode:
    """Load configuration from file or use defaults."""
    cfg = get_cfg_defaults()

    standard_config_map = {
        "NU_2_2018": "nema_phantom_config.yaml",
        "NU_4_2008": "nema_phantom_config_nu4_2008.yaml",
    }
    standard_config_name = standard_config_map.get(standard)
    if standard_config_name:
        standard_config_path = (
            Path(__file__).resolve().parents[2] / "config" / standard_config_name
        )
        if standard_config_path.exists():
            logging.info(
                f"Loading base config for standard {standard}: {standard_config_path}"
            )
            cfg.merge_from_file(str(standard_config_path))
        else:
            logging.warning(
                f"Base config for standard {standard} not found at {standard_config_path}"
            )

    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logging.info(f"Loading configuration: {config_path}")
        cfg.merge_from_file(config_path)
    else:
        logging.info("Using default configuration")

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
        logging.debug(f" Using provided voxel spacing: {voxel_spacing} mm")
    elif affine is not None:
        voxel_spacing = (
            float(np.abs(affine[0, 0])),
            float(np.abs(affine[1, 1])),
            float(np.abs(affine[2, 2])),
        )
        logging.debug(f" Extracted voxel spacing from image: {voxel_spacing} mm")
    else:
        # Default spacing
        voxel_spacing = (1.0, 1.0, 1.0)
        logging.warning(
            "No voxel spacing information available. Using default: (1.0, 1.0, 1.0) mm"
        )

    logging.debug(f" Image dimensions: {image_dims}")

    return image_dims, voxel_spacing


def get_nii_iterations(folder_path: Path, pattern: str) -> List[Dict[str, Any]]:
    r"""
    Finds .nii files in a directory and extracts iteration indices from filenames using a user-specified pattern.

    Scans the given folder for all files ending in '.nii' and applies a regular expression to their filenames,
    extracting the numeric iteration label according to the pattern (e.g., 'frame01', 'iter05'). Returns a list
    of dictionaries containing the file path and the extracted iteration number.

    Parameters
    ----------
    folder_path : str
        Path to the directory containing .nii files.
    pattern : str, optional
        Regular expression pattern containing a capturing group for the iteration number
        (default is r'frame(\d+)').

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries for each .nii file matching the pattern, with:
            'path' : str
                Absolute path to the .nii file.
            'iteration' : int
                The extracted iteration number from the filename
    """
    iter_pattern = re.compile(pattern, re.IGNORECASE)
    nii_files = [f for f in os.listdir(folder_path) if f.endswith(".nii")]
    results: List[Dict[str, Any]] = []
    for fname in nii_files:
        match = iter_pattern.search(fname)
        if match:
            iteration = int(match.group(1))
            results.append(
                {"path": os.path.join(folder_path, fname), "iteration": iteration}
            )
    return sorted(results, key=lambda x: x["iteration"])


def process_single_iteration(
    iteration_info: Dict[str, Any], cfg, args: argparse.Namespace
) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[str]]:
    """Process a single NIfTI iteration and return results based on standard."""
    iteration_num = iteration_info["iteration"]
    file_path = Path(iteration_info["path"])

    logging.info(f" Processing iteration {iteration_num}: {file_path.name}")

    try:
        logging.info(" Loading NIfTI image")
        image_data, affine = load_nii_image(file_path, return_affine=True)
        logging.info(" Image loaded successfully")

        image_dims, voxel_spacing = get_image_properties(
            image_data, affine, args.spacing
        )

        logging.info(" Initializing NEMA phantom...")
        phantom = NemaPhantom(cfg, image_dims, voxel_spacing)
        logging.info(f" Phantom initialized with {len(phantom.rois)} ROIs")
        logging.info(" Performing NEMA analysis...")

        if args.standard == "NU_4_2008":
            logging.info("Running NU_4_2008 analysis path")
            crc_results, spillover_results, uniformity_results = (
                calculate_nema_metrics_nu4_2008(
                    image_data,
                    phantom,
                    cfg,
                    save_visualizations=args.save_visualizations,
                    visualizations_dir=args.visualizations_dir,
                )
            )
            logging.info(f" Iteration {iteration_num}: NU_4_2008 analysis completed")
            return crc_results, spillover_results, uniformity_results, None
        else:
            logging.info("Running NU_2_2018 analysis path")
            results, lung_results = calculate_nema_metrics(
                image_data,
                phantom,
                cfg,
                save_visualizations=args.save_visualizations,
                visualizations_dir=args.visualizations_dir,
            )

            values = list(lung_results.values())
            average = float(np.mean(values)) if values else 0.0
            logging.info(
                f" Iteration {iteration_num}: Average Accuracy Correction: {average:.3f}%, Spheres: {len(results)}"
            )

            return results, lung_results, None, None

    except Exception as e:
        error_msg = f" Failed to process iteration {iteration_num}: {e}"
        logging.error(error_msg)
        if args.verbose:
            import traceback

            logging.error(traceback.format_exc())
        return None, None, None, error_msg


def run_analysis(args: argparse.Namespace) -> int:
    """Run the NEMA analysis with the provided arguments"""
    try:
        setup_logging(args.log_level, args.output)

        logging.info("Starting ChameleonIQ for Multiple Iterations")
        logging.info(f" Input folder: {args.input_path}")
        logging.info(f" Output file: {args.output}")
        logging.info(f" NEMA standard: {args.standard}")

        try:
            cfg = load_configuration(args.config, args.standard)
            logging.info(
                f"Number of ROIs defined: {len(cfg.PHANTHOM.ROI_DEFINITIONS_MM)}"
            )
            logging.info(f"Background offsets: {len(cfg.ROIS.BACKGROUND_OFFSET_YX)}")
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            if args.verbose:
                import traceback

                logging.error(traceback.format_exc())
            print(f"ERROR: Failed to load configuration: {e}")
            return 1

        input_path = Path(args.input_path)

        if not input_path.exists():
            error_msg = f" Input folder does not exist: {args.input_path}"
            logging.error(error_msg)
            print(f"ERROR: {error_msg}")
            return 1

        logging.info(" Finding NIfTI iterations...")
        pattern = cfg.FILE.USER_PATTERN
        logging.info(f" Looking for pattern: {pattern}")
        nii_iterations = get_nii_iterations(input_path, pattern)

        if not nii_iterations:
            error_msg = (
                f" No .nii files found matching pattern '{pattern}' in {input_path}"
            )
            logging.error(error_msg)
            print(f"ERROR: {error_msg}")
            return 1

        logging.info(f" Found {len(nii_iterations)} .nii files matching pattern")

        all_results = []
        all_lung_results: Dict[int, Dict[int, float]] = {}
        all_crc_results: Dict[int, Any] = {}
        all_spillover_results: Dict[int, Any] = {}
        all_uniformity_results: Dict[int, Any] = {}
        failed_iterations: List[Tuple[int, str]] = []

        total_iterations = len(nii_iterations)
        for i, iteration_info in enumerate(nii_iterations, 1):
            iteration_num = iteration_info["iteration"]

            logging.info(
                f"Processing iteration {i}/{total_iterations}: {iteration_num}"
            )

            result1, result2, result3, error = process_single_iteration(
                iteration_info, cfg, args
            )

            if args.standard == "NU_4_2008":
                # Handle NU_4_2008 results
                crc_results, spillover_results, uniformity_results = (
                    result1,
                    result2,
                    result3,
                )
                if error is None and crc_results is not None:
                    all_crc_results[iteration_num] = crc_results
                    all_spillover_results[iteration_num] = spillover_results
                    all_uniformity_results[iteration_num] = uniformity_results
                    logging.info(
                        f" Iteration {iteration_num}: NU_4_2008 analysis stored"
                    )
                else:
                    failed_iterations.append((iteration_num, error or "Unknown error"))
            else:
                # Handle NU_2_2018 results (default)
                results, lung_results = result1, result2
                metrics = calculate_weighted_cbr_from(results)
                _cbrs = [float(x) for x in metrics["CBRs"]] if metrics["CBRs"] else []
                logging.info(f"With the diameters {metrics['diameters']}")
                logging.info(f"With the CBRs {_cbrs}")
                logging.info(
                    f"Iteration {iteration_num} Weighted CBR: {metrics['weighted_CBR']:.3f}"
                )
                logging.info(
                    f"Iteration {iteration_num} Weighted FOM: {metrics['weighted_FOM']:.3f}"
                )

                if error is None and results is not None and lung_results is not None:
                    for result in results:
                        all_results.append(
                            {
                                "diameter_mm": result["diameter_mm"],
                                "percentaje_constrast_QH": result[
                                    "percentaje_constrast_QH"
                                ],
                                "background_variability_N": result[
                                    "background_variability_N"
                                ],
                                "avg_hot_counts_CH": result["avg_hot_counts_CH"],
                                "avg_bkg_counts_CB": result["avg_bkg_counts_CB"],
                                "bkg_std_dev_SD": result["bkg_std_dev_SD"],
                                "iteration": iteration_num,
                                "weighted_CBR": metrics["weighted_CBR"],
                                "weighted_FOM": metrics["weighted_FOM"],
                                "37_CBR": _cbrs[0] if len(_cbrs) > 0 else 0.0,
                                "28_CBR": _cbrs[1] if len(_cbrs) > 1 else 0.0,
                                "22_CBR": _cbrs[2] if len(_cbrs) > 2 else 0.0,
                                "17_CBR": _cbrs[3] if len(_cbrs) > 3 else 0.0,
                                "13_CBR": _cbrs[4] if len(_cbrs) > 4 else 0.0,
                                "10_CBR": _cbrs[5] if len(_cbrs) > 5 else 0.0,
                            }
                        )
                    all_lung_results[iteration_num] = lung_results
                else:
                    failed_iterations.append((iteration_num, error or "Unknown error"))

        successful_iterations = (
            len(all_results) if args.standard == "NU_2_2018" else len(all_crc_results)
        )
        failed_count = len(failed_iterations)

        if args.standard == "NU_2_2018":
            iteration_metrics = {}
            for result in all_results:
                iter_num = result["iteration"]
                if iter_num not in iteration_metrics:
                    iteration_metrics[iter_num] = {
                        "weighted_CBR": result["weighted_CBR"],
                        "weighted_FOM": result["weighted_FOM"],
                    }

            best_cbr_iter_num, best_cbr_metrics = max(
                iteration_metrics.items(), key=lambda x: x[1]["weighted_CBR"]
            )
            best_fom_iter_num, best_fom_metrics = max(
                iteration_metrics.items(), key=lambda x: x[1]["weighted_FOM"]
            )
            logging.info(
                f"Iteration with highest weighted CBR: {best_cbr_iter_num} (CBR={best_cbr_metrics['weighted_CBR']:.3f})"
            )
            logging.info(
                f"Iteration with highest weighted FOM: {best_fom_iter_num} (FOM={best_fom_metrics['weighted_FOM']:.3f})"
            )

            logging.info(
                f"Iteration with highest weighted CBR: {best_cbr_iter_num} (CBR={best_cbr_metrics['weighted_CBR']:.3f})"
            )
            logging.info(
                f"Iteration with highest weighted FOM: {best_fom_iter_num} (FOM={best_fom_metrics['weighted_FOM']:.3f})"
            )

            logging.info(
                f"Processing complete: {successful_iterations // 6} successful, {failed_count // 6} failed"
            )

            if failed_iterations:
                logging.warning("Failed iterations:")
                for iteration_num, error in failed_iterations:
                    logging.warning(f"  Iteration {iteration_num}: {error}")

            if successful_iterations == 0:
                error_msg = "No iterations were processed successfully"
                logging.error(error_msg)
                print(f"ERROR: {error_msg}")
                return 1

            logging.info("Summary of results:")
            for iteration_num in sorted(all_lung_results.keys()):
                sphere_count = len(
                    [r for r in all_results if r["iteration"] == iteration_num]
                )
                lung_results = all_lung_results[iteration_num]

                lung_values = list(lung_results.values())
                avg_lung = float(np.mean(lung_values)) if lung_values else 0.0

                logging.info(
                    f"  Iteration {iteration_num}: {sphere_count} spheres, avg lung correction: {avg_lung:.3f}%"
                )

            logging.info(
                f"Total sphere measurements across all iterations: {len(all_results)}"
            )
            logging.info(f"Results stored for {successful_iterations // 6} iterations")

            from collections import defaultdict

            results_by_iteration = defaultdict(list)
            for result in all_results:
                results_by_iteration[result["iteration"]].append(result)

            for iteration_num in sorted(results_by_iteration.keys()):
                sphere_results = results_by_iteration[iteration_num]
                lung_results = all_lung_results[iteration_num]

                contrasts = [r["percentaje_constrast_QH"] for r in sphere_results]
                background = [r["background_variability_N"] for r in sphere_results]
                avg_contrast = np.mean(contrasts) if contrasts else 0.0
                avg_background = np.mean(background) if background else 0.0

                lung_values = list(lung_results.values())
                avg_lung = float(np.mean(lung_values)) if lung_values else 0.0

                logging.info(
                    f"  Iteration {iteration_num}: {len(sphere_results)} spheres, "
                    f"avg contrast: {avg_contrast:.1f}%, avg background varibility: {avg_background:.1f}% "
                    f"avg lung correction: {avg_lung:.3f}%"
                )

            logging.info("Data ready for plotting and analysis")
        else:
            logging.info(
                f"NU_4_2008 Analysis Iterations: {len(all_crc_results)} successful"
            )
            if failed_iterations:
                logging.warning("Failed iterations:")
                for iteration_num, error in failed_iterations:
                    logging.warning(f"  Iteration {iteration_num}: {error}")

            if len(all_crc_results) == 0:
                error_msg = "No iterations were processed successfully"
                logging.error(error_msg)
                print(f"ERROR: {error_msg}")
                return 1

            logging.info("Data ready for plotting and analysis")

        try:
            output_path = Path(args.output)
            png_dir = output_path.parent / "png"
            png_dir.mkdir(parents=True, exist_ok=True)
            csv_dir = output_path.parent / "csv"
            csv_dir.mkdir(parents=True, exist_ok=True)

            if args.standard == "NU_4_2008":
                logging.info("Generating NU_4_2008 iteration analysis plots...")
                try:
                    crc_plot = generate_crc_convergence_plot_nu4_iter(
                        all_crc_results, png_dir, cfg
                    )
                    spillover_plot = generate_spillover_convergence_plot_nu4_iter(
                        all_spillover_results, png_dir, cfg
                    )
                    uniformity_plot = generate_uniformity_convergence_plot_nu4_iter(
                        all_uniformity_results, png_dir, cfg
                    )
                    logging.info("NU_4_2008 iteration plots generated successfully")
                except Exception as e:
                    logging.error(f"Failed to generate NU_4_2008 plots: {e}")
                    crc_plot = None
                    spillover_plot = None
                    uniformity_plot = None
                    if args.verbose:
                        import traceback

                        logging.error(traceback.format_exc())
            else:
                logging.info("Generating NU_2_2018 plots...")
                generate_plots(all_results, png_dir, cfg)
                generate_pc_vs_bg_plot(all_results, png_dir, cfg)
                generate_boxplot_with_mean_std(all_lung_results, png_dir, cfg)
                generate_wcbr_convergence_plot(all_results, png_dir, cfg)
                generate_cbr_convergence_plot(all_results, png_dir, cfg)
            logging.info("Plots generated successfully")
        except Exception as e:
            logging.error(f"Failed to generate plots: {e}")
            if args.verbose:
                import traceback

                logging.error(traceback.format_exc())
            print(f"WARNING: Failed to generate plots: {e}")

        logging.info(f"Saving results to: {args.output}")
        try:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if args.standard == "NU_4_2008":
                logging.info("Saving NU_4_2008 iteration results...")
                try:
                    # Save text results
                    save_results_to_txt_nu4_iter(
                        all_crc_results,
                        all_spillover_results,
                        all_uniformity_results,
                        output_path,
                        cfg,
                        input_path,
                        (cfg.ROIS.SPACING, cfg.ROIS.SPACING, cfg.ROIS.SPACING),
                    )
                    logging.info("NU_4_2008 text results saved successfully")

                    # Generate PDF report
                    pdf_output_path = output_path.with_suffix(".pdf")
                    logging.info(f"Generating NU_4_2008 PDF report: {pdf_output_path}")

                    crc_plot_path = None
                    spillover_plot_path = None
                    uniformity_plot_path = None

                    if "crc_plot" in locals() and crc_plot:
                        crc_plot_path = png_dir / "crc_convergence_nu4_iter.png"
                    if "spillover_plot" in locals() and spillover_plot:
                        spillover_plot_path = (
                            png_dir / "spillover_convergence_nu4_iter.png"
                        )
                    if "uniformity_plot" in locals() and uniformity_plot:
                        uniformity_plot_path = (
                            png_dir / "uniformity_convergence_nu4_iter.png"
                        )

                    generate_reportlab_report_nu4_iter(
                        all_crc_results,
                        all_spillover_results,
                        all_uniformity_results,
                        pdf_output_path,
                        cfg,
                        input_path,
                        (cfg.ROIS.SPACING, cfg.ROIS.SPACING, cfg.ROIS.SPACING),
                        crc_plot_path,
                        spillover_plot_path,
                        uniformity_plot_path,
                    )
                    logging.info("NU_4_2008 PDF report saved successfully")
                except Exception as e:
                    logging.error(f"Failed to save NU_4_2008 results: {e}")
                    if args.verbose:
                        import traceback

                        logging.error(traceback.format_exc())
                    print(f"WARNING: Failed to save NU_4_2008 results: {e}")
            else:
                plot_path = output_path.parent / "png" / "analysis_plot_iterations.png"
                rois_loc_path = output_path.parent / "png" / "rois_location.png"
                pc_vs_bg_path = output_path.parent / "png" / "bg_vs_pc_plot.png"
                boxplot_path = (
                    output_path.parent / "png" / "lung_boxplot_iterations.png"
                )
                wcbr_conv_path = (
                    output_path.parent / "png" / "weighted_cbr_convergence_analysis.png"
                )
                cbr_conv_path = (
                    output_path.parent / "png" / "cbr_convergence_analysis.png"
                )

                save_results_to_txt(
                    all_results,
                    all_lung_results,
                    output_path,
                    cfg,
                    input_path,
                    (cfg.ROIS.SPACING, cfg.ROIS.SPACING, cfg.ROIS.SPACING),
                )
                logging.info("Text results saved successfully")

                try:
                    pdf_output_path = output_path.with_suffix(".pdf")
                    logging.info(f"Generating PDF report: {pdf_output_path}")
                    generate_reportlab_report(
                        all_results,
                        all_lung_results,
                        pdf_output_path,
                        cfg,
                        input_path,
                        (cfg.ROIS.SPACING, cfg.ROIS.SPACING, cfg.ROIS.SPACING),
                        plot_path,
                        pc_vs_bg_path,
                        rois_loc_path,
                        boxplot_path,
                        cbr_conv_path,
                        wcbr_conv_path,
                    )
                    logging.info("PDF report saved successfully")
                except Exception as e:
                    logging.error(f"Failed to generate PDF report: {e}")
                    if args.verbose:
                        import traceback

                        logging.error(traceback.format_exc())
                    print(f"WARNING: Failed to generate PDF report: {e}")

        except Exception as e:
            logging.error(f"Failed to save results: {e}")
            if args.verbose:
                import traceback

                logging.error(traceback.format_exc())
            print(f"ERROR: Failed to save results: {e}")
            return 1

        logging.info("Analysis completed successfully")
        print("SUCCESS: Analysis completed successfully")
        return 0

    except KeyboardInterrupt:
        logging.info("Analysis interrupted by user")
        print("\nINFO: Analysis interrupted by user")
        return 130
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        print(f"ERROR: File not found: {e}")
        return 2
    except PermissionError as e:
        logging.error(f"Permission denied: {e}")
        print(f"ERROR: Permission denied: {e}")
        return 13
    except MemoryError as e:
        logging.error(f"Out of memory: {e}")
        print(
            "ERROR: Out of memory. Try processing fewer iterations or reducing image size."
        )
        return 3
    except ValueError as e:
        logging.error(f"Invalid value: {e}")
        print(f"ERROR: Invalid value: {e}")
        return 4
    except Exception as e:
        logging.error("Unexpected error:")
        logging.exception(e)
        return 1


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    return run_analysis(args)


if __name__ == "__main__":
    sys.exit(main())

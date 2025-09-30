"""
Command Line Interface for NEMA Analysis Tool

This module provides the command-line interface functionality for the NEMA NU 2-2018 image quality analysis tool.

Author: Edwing Ulin-Briseno
Date: 2025-07-16
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import numpy.typing as npt
import yacs.config

from config.defaults import get_cfg_defaults

from .analysis import calculate_nema_metrics
from .io import load_nii_image
from .phantom import NemaPhantom
from .reporting import (
    generate_boxplot_with_mean_std,
    generate_coronal_sphere_plots,
    generate_plots,
    generate_reportlab_report,
    generate_rois_plots,
    generate_torso_plot,
    generate_transverse_sphere_plots,
    save_results_to_txt,
)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="NEMA NU 2-2018 Image Quality Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Basic analysis
    nema_quant input.nii --config custom_config.yaml --output results.txt

    # Verbose output
    nema_quant input.nii --config custom_config.yaml --output results.txt --verbose
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
        help="Path to custom YAML configuration file",
    )

    parser.add_argument(
        "--spacing", nargs=3, type=float, help="Voxel spacing in mm (x, y, z)"
    )

    # Optional arguments
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save visualization images of ROI masks and analysis regions",
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
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--version", action="version", version="NEMA Analysis Tool v0.2.0"
    )

    return parser


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("reportlab").setLevel(logging.WARNING)


def load_configuration(config_path: Optional[str]) -> yacs.config.CfgNode:
    """Load configuration from file or use defaults."""
    cfg = get_cfg_defaults()

    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logging.info(f"Loading configuration from: {config_path}")
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
        logging.info(f"Using provided voxel spacing: {voxel_spacing} mm")
    elif affine is not None:
        voxel_spacing = (
            float(np.abs(affine[0, 0])),
            float(np.abs(affine[1, 1])),
            float(np.abs(affine[2, 2])),
        )
        logging.info(f"Extracted voxel spacing from image: {voxel_spacing} mm")
    else:
        # Default spacing
        voxel_spacing = (1.0, 1.0, 1.0)
        logging.warning(
            "No voxel spacing information available. Using default: (1.0, 1.0, 1.0) mm"
        )

    logging.info(f"Image dimensions: {image_dims}")

    return image_dims, voxel_spacing


def run_analysis(args: argparse.Namespace) -> int:
    """Run the NEMA analysis with the provided arguments."""
    try:
        # Setup logging
        setup_logging(args.verbose)

        logging.info("Starting NEMA NU 2-2018 Image Quality Analysis")
        logging.info(f"Input image: {args.input_image}")
        logging.info(f"Output file: {args.output}")

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
            cfg = load_configuration(args.config)
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            if args.verbose:
                import traceback

                logging.error(traceback.format_exc())
            print(f"ERROR: Failed to load configuration: {e}")
            return 1

        logging.info("Loading NIfTI image...")
        try:
            image_data, affine = load_nii_image(input_path, return_affine=True)
            logging.info("Image loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load image: {e}")
            if args.verbose:
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
            if args.verbose:
                import traceback

                logging.error(traceback.format_exc())
            print(f"ERROR: Failed to extract image properties: {e}")
            return 1

        logging.info("Initializing NEMA phantom...")
        try:
            phantom = NemaPhantom(cfg, image_dims, voxel_spacing)
            logging.info(f"Phantom initialized with {len(phantom.rois)} ROIs")
        except Exception as e:
            logging.error(f"Failed to initialize phantom: {e}")
            if args.verbose:
                import traceback

                logging.error(traceback.format_exc())
            print(f"ERROR: Failed to initialize phantom: {e}")
            return 1

        # Perform NEMA analysis
        logging.info("Performing NEMA analysis...")
        try:
            results, lung_results = calculate_nema_metrics(
                image_data,
                phantom,
                cfg,
                save_visualizations=args.save_visualizations,
                visualizations_dir=args.visualizations_dir,
            )
            values = list(lung_results.values())
            average = float(np.mean(values))
            logging.info(f"Average of Accuracy Corrections: {average:.3f} %")
            logging.info(
                f"Analysis completed. Found {len(results)} sphere measurements"
            )

        except Exception as e:
            logging.error(f"Failed to perform analysis: {e}")
            if args.verbose:
                import traceback

                logging.error(traceback.format_exc())
            print(f"ERROR: Failed to perform analysis: {e}")
            return 1

        output_path = Path(args.output)
        png_dir = output_path.parent / "png"
        png_dir.mkdir(parents=True, exist_ok=True)
        csv_dir = output_path.parent / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)

        logging.info("Saving analysis plots...")
        try:
            generate_plots(results=results, output_dir=png_dir, cfg=cfg)
            generate_rois_plots(image=image_data, output_dir=png_dir, cfg=cfg)
            generate_transverse_sphere_plots(
                image=image_data, output_dir=png_dir, cfg=cfg
            )
            generate_boxplot_with_mean_std(
                data_dict=lung_results, output_dir=png_dir, cfg=cfg
            )
            generate_coronal_sphere_plots(image=image_data, output_dir=png_dir, cfg=cfg)
            generate_torso_plot(image=image_data, output_dir=png_dir, cfg=cfg)
            logging.info("Plots generated successfully")
        except Exception as e:
            logging.error(f"Failed to generated plots: {e}")
            if args.verbose:
                import traceback

                logging.error(traceback.format_exc())
            print(f"ERROR: Failed to generate plots: {e}")
            return 1

        logging.info(f"Saving results to: {args.output}")
        try:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plot_path = output_path.parent / "png" / "analysis_plot.png"
            rois_loc_path = output_path.parent / "png" / "rois_location.png"
            boxplot_path = output_path.parent / "png" / "boxplot_with_mean_std.png"

            save_results_to_txt(results, output_path, cfg, input_path, voxel_spacing)

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
            )
            logging.info("Results saved successfully")
        except Exception as e:
            logging.error(f"Failed to save results: {e}")
            if args.verbose:
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
                if args.verbose:
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
                if args.verbose:
                    import traceback

                    logging.error(traceback.format_exc())
                print(f"ERROR: Failed to calculate advanced metrics: {e}")
                return 1
        # Print summary
        logging.info("Analysis completed successfully!")
        logging.info("Summary:")
        logging.info(f"  Input image: {args.input_image}")
        logging.info(f"  Image dimensions: {image_dims}")
        logging.info(f"  Voxel spacing: {voxel_spacing} mm")
        logging.info(f"  Number of spheres analyzed: {len(results)}")
        logging.info(f"  Results saved to: {args.output}")

        if args.save_visualizations:
            print(f"  Visualizations saved to: {args.visualizations_dir}/")

        return 0

    except KeyboardInterrupt:
        logging.info("Analysis interrupted by user")
        print("Analysis interrupted by user")
        return 130
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"ERROR: Unexpected error: {e}")
        if hasattr(args, "verbose") and args.verbose:
            import traceback

            logging.error(traceback.format_exc())
            print("\nFull traceback:")
            traceback.print_exc()
        return 1


def main(cli_args: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(cli_args)
    return run_analysis(args)


if __name__ == "__main__":
    sys.exit(main())

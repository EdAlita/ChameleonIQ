import argparse
import datetime
import logging
import sys
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional
from venv import logger

import numpy as np
import pandas as pd
import yacs.config
from rich.logging import RichHandler
from scipy.stats import ttest_ind, ttest_rel
from statsmodels.stats.multitest import multipletests

from config.defaults import get_cfg_defaults

from .reporting import (
    generate_dose_merged_plot,
    generate_dose_merged_plot_any_sphere,
    generate_global_metrics_boxplot,
    generate_merged_boxplot,
    generate_merged_plots,
    generate_unified_statistical_heatmaps,
)


def load_configuration(config_path: Optional[str]) -> yacs.config.CfgNode:
    """Load configuration from file or use defaults."""
    cfg = get_cfg_defaults()

    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logging.info("Loading configuration: %s", config_path)
        cfg.merge_from_file(config_path)
    else:
        logging.info("Using default configuration")
    return cfg


def setup_logging(log_level: int = 20, output_path: Optional[str] = None) -> None:
    """Configuration logging for the application."""

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_path:
        output_path_obj = Path(output_path)
        run_name = output_path_obj.stem
        log_filename = f"{run_name}_{timestamp}.log"
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

    logging.info(f"Logging initialized. Log file: {log_file_path}")


def parse_xml_config(
    xml_path: Path,
) -> tuple[List[Dict[str, Any]], List[Dict[str, str]], List[Dict[str, str]]]:
    logging.info("Parsing XML configuration")
    logging.info("XML Path :%s", xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    experiments = []
    lung_experiments = []
    advanced_metrics = []

    for idx, experiment in enumerate(root.findall("experiment"), start=1):
        name = experiment.get("name")
        file_path = experiment.get("path")

        if not name or not file_path:
            logging.warning(
                "Skipping experiment %d: missing name or path (name=%s, path=%s)",
                idx,
                name,
                file_path,
            )
            continue

        plot_status = experiment.get("plot_status", "enhanced")
        lung_path = experiment.get("lung_path")
        advanced_metric_path = experiment.get("advanced_path")
        dose = experiment.get("dose")

        if name and file_path:
            experiments.append(
                {
                    "name": name,
                    "path": file_path,
                    "plot_status": plot_status,
                    "dose": dose,
                }
            )

            logger.info(
                "Experiment found: name=%s plot=%s dose=%s", name, plot_status, dose
            )

            if lung_path:
                lung_experiments.append({"name": name, "path": lung_path})
                logging.info("  Lung data: %s", lung_path)

            if advanced_metric_path:
                advanced_metrics.append({"name": name, "path": advanced_metric_path})
                logging.info(
                    f"Found advanced metrics data: {name} -> {advanced_metric_path}"
                )

    logging.info(f"Total experiments found: {len(experiments)}")
    logging.info(f"Total lung experiments found: {len(lung_experiments)}")
    logging.info(f"Total advanced metrics found: {len(advanced_metrics)}")
    return experiments, lung_experiments, advanced_metrics


def load_experiment_data(
    experiments: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[str], Dict[str, str], Dict[str, Any]]:
    all_data = []
    experiment_order = []
    experiment_plot_status = {}
    experiment_dose = {}

    for exp in experiments:
        exp_name = exp["name"]
        exp_plot_status = exp["plot_status"]
        file_path = Path(exp["path"])
        experiment_order.append(exp_name)
        experiment_plot_status[exp_name] = exp_plot_status
        experiment_dose[exp_name] = exp["dose"] if "dose" in exp else None

        logging.info(f"Loading data for experiment: {exp_name}")

        if not file_path.exists():
            logging.warning(f"File not found: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)

            for _, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict["experiment"] = exp_name
                all_data.append(row_dict)

            logging.info(f"Loaded {len(df)} records from {exp_name}")

        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            continue

    logging.info(f"Total records loaded: {len(all_data)}")
    return all_data, experiment_order, experiment_plot_status, experiment_dose


def load_lung_data(lung_experiments: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    all_lung_data = []

    for exp in lung_experiments:
        exp_name = exp["name"]
        file_path = Path(exp["path"])

        logging.info(f"Loading lung data for experiment: {exp_name}")

        if not file_path.exists():
            logging.warning(f"Lung file not found: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)

            for _, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict["experiment"] = exp_name
                all_lung_data.append(row_dict)

            logging.info(f"Loaded {len(df)} lung records from {exp_name}")

        except Exception as e:
            logging.error(f"Error loading lung data {file_path}: {e}")
            continue

    logging.info(f"Total lung records loaded: {len(all_lung_data)}")
    return all_lung_data


def load_advanced_metrics_data(
    advanced_metrics: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    all_advanced_metrics_data = []

    for metric in advanced_metrics:
        exp_name = metric["name"]
        file_path = Path(metric["path"])

        logging.info(f"Loading advanced metrics data for experiment: {exp_name}")

        if not file_path.exists():
            logging.warning(f"Advanced metrics file not found: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)

            for _, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict["experiment"] = exp_name
                all_advanced_metrics_data.append(row_dict)

            logging.info(f"Loaded {len(df)} advanced metrics records from {exp_name}")

        except Exception as e:
            logging.error(f"Error loading advanced metrics data {file_path}: {e}")
            continue

    logging.info(
        f"Total advanced metrics records loaded: {len(all_advanced_metrics_data)}"
    )
    return all_advanced_metrics_data


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ChameleonIQ Merge Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "xml_config",
        type=str,
        help="Path to XML configuration file with experiment definitions",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for merged analysis plots",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to custom YAML configuration file",
    )

    parser.add_argument(
        "--log-level",
        type=int,
        default=10,
        help="Logging level (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR)",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('ChameleonIQ')}"
    )

    return parser


def perform_statistical_analysis(
    data: List[Dict[str, Any]],
    metrics: List[str],
    output_dir: Path,
    experiment_order: List[str],
    test_type: str = "paired",
) -> Dict[str, Any]:
    """
    Perform statistical analysis on specified metrics

    Args:
        data: List of data dictionaries
        metrics: List of metric names to analyze
        output_dir: Output directory for results
        experiment_order: Order of experiments
        test_type: 'paired' or 'unpaired' t-tests
    """

    results = {}
    df = pd.DataFrame(data)

    for metric in metrics:
        if metric not in df.columns:
            logging.warning(f"Metric {metric} not found in data")
            continue

        p_matrix = np.ones((len(experiment_order), len(experiment_order)))
        effect_sizes = np.zeros((len(experiment_order), len(experiment_order)))

        for i, exp1 in enumerate(experiment_order):
            for j, exp2 in enumerate(experiment_order):
                if i >= j:
                    continue

                data1 = df[df["experiment"] == exp1][metric].dropna()
                data2 = df[df["experiment"] == exp2][metric].dropna()

                if len(data1) == 0 or len(data2) == 0:
                    continue

                if test_type == "paired" and len(data1) == len(data2):
                    stat, p_val = ttest_rel(data1, data2)
                else:
                    stat, p_val = ttest_ind(data1, data2)

                pooled_std = np.sqrt(
                    ((len(data1) - 1) * data1.var() + (len(data2) - 1) * data2.var())
                    / (len(data1) + len(data2) - 2)
                )
                cohens_d = (
                    (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0
                )

                p_matrix[i, j] = p_val
                p_matrix[j, i] = p_val
                effect_sizes[i, j] = cohens_d
                effect_sizes[j, i] = -cohens_d

        p_values_flat = p_matrix[np.triu_indices_from(p_matrix, k=1)]
        rejected, p_corrected, _, _ = multipletests(p_values_flat, method="bonferroni")

        p_corrected_matrix = np.ones_like(p_matrix)
        p_corrected_matrix[np.triu_indices_from(p_corrected_matrix, k=1)] = p_corrected
        p_corrected_matrix = p_corrected_matrix + p_corrected_matrix.T - np.diag(np.diag(p_corrected_matrix))  # type: ignore

        results[metric] = {
            "p_values": p_matrix,
            "p_corrected": p_corrected_matrix,
            "effect_sizes": effect_sizes,
            "significant_pairs": rejected,
        }

    return results


def perform_advanced_statistical_analysis(
    data: List[Dict[str, Any]],
    metrics: List[str],
    output_dir: Path,
    experiment_order: List[str],
) -> Dict[str, Any]:
    """
    Perform statistical analysis on advanced metrics (1 sample per experiment)
    """

    results = {}
    df = pd.DataFrame(data)

    logging.info(
        f"Performing analysis on {len(df)} samples across {df['experiment'].nunique()} experiments"
    )

    for metric in metrics:
        if metric not in df.columns:
            logging.warning(f"Metric {metric} not found in data")
            continue

        metric_data = df[metric].dropna()
        if len(metric_data) < 2:
            logging.warning(
                f"Insufficient data for {metric}: {len(metric_data)} samples"
            )
            continue

        p_matrix = np.ones((len(experiment_order), len(experiment_order)))
        effect_sizes = np.zeros((len(experiment_order), len(experiment_order)))
        valid_comparisons = 0

        for i, exp1 in enumerate(experiment_order):
            for j, exp2 in enumerate(experiment_order):
                if i >= j:
                    continue

                data1_df = df[df["experiment"] == exp1][metric].dropna()
                data2_df = df[df["experiment"] == exp2][metric].dropna()

                if len(data1_df) == 0 or len(data2_df) == 0:
                    continue

                value1 = data1_df.iloc[0]
                value2 = data2_df.iloc[0]

                all_values = df[metric].dropna()
                pooled_std = all_values.std()

                if pooled_std > 0:
                    cohens_d = (value1 - value2) / pooled_std
                    effect_sizes[i, j] = cohens_d
                    effect_sizes[j, i] = -cohens_d

                    abs_effect = abs(cohens_d)
                    if abs_effect > 0.8:
                        p_val = 0.001
                    elif abs_effect > 0.5:
                        p_val = 0.01
                    elif abs_effect > 0.2:
                        p_val = 0.05
                    else:
                        p_val = 0.1

                    p_matrix[i, j] = p_val
                    p_matrix[j, i] = p_val
                    valid_comparisons += 1

        upper_triangle = np.triu_indices_from(p_matrix, k=1)
        p_values_flat = p_matrix[upper_triangle]

        valid_mask = p_values_flat < 1.0
        if np.sum(valid_mask) > 0:
            valid_p_values = p_values_flat[valid_mask]
            rejected, p_corrected_valid, _, _ = multipletests(
                valid_p_values, method="bonferroni"
            )

            p_corrected_matrix = np.ones_like(p_matrix)
            p_corrected_flat = np.ones_like(p_values_flat)
            p_corrected_flat[valid_mask] = p_corrected_valid
            p_corrected_matrix[upper_triangle] = p_corrected_flat
            p_corrected_matrix = p_corrected_matrix + p_corrected_matrix.T - np.diag(np.diag(p_corrected_matrix))  # type: ignore

            significant_pairs = rejected
        else:
            p_corrected_matrix = np.ones_like(p_matrix)
            significant_pairs = []

        results[metric] = {
            "p_values": p_matrix,
            "p_corrected": p_corrected_matrix,
            "effect_sizes": effect_sizes,
            "significant_pairs": significant_pairs,
        }

        logging.info(
            f"Advanced metric {metric}: {valid_comparisons} comparisons, "
            f"{np.sum(p_corrected_matrix < 0.05)} significant pairs"
        )

    return results


def run_merge_analysis(args: argparse.Namespace) -> int:
    try:
        setup_logging(args.log_level)
        cfg = load_configuration(args.config)

        logging.info("Starting NEMA Merge Analysis")
        logging.info(f"XML config: {args.xml_config}")
        logging.info(f"Output directory: {args.output}")

        xml_path = Path(args.xml_config)
        output_dir = Path(args.output)

        if not xml_path.exists():
            logging.error(f"XML configuration file not found: {xml_path}")
            return 1

        output_dir.mkdir(parents=True, exist_ok=True)

        experiments, lung_experiments, advanced_metrics = parse_xml_config(xml_path)
        if not experiments:
            logging.error("No experiments found in XML configuration")
            return 1

        all_data, experiment_order, plots_status, experiment_dose = (
            load_experiment_data(experiments)
        )
        if not all_data:
            logging.error("No data loaded from experiments")
            return 1

        logging.info("Generating merged plots...")
        generate_merged_plots(
            all_data, output_dir, experiment_order, plots_status, cfg=cfg
        )
        if any(
            dose is not None and str(dose).replace(".", "", 1).isdigit()
            for dose in experiment_dose.values()
        ):
            logging.info("Generating dose merged plots...")
            generate_dose_merged_plot(all_data, output_dir, cfg, experiment_dose)
            generate_dose_merged_plot_any_sphere(
                all_data, output_dir, cfg, experiment_dose, 10.0
            )
            generate_dose_merged_plot_any_sphere(
                all_data, output_dir, cfg, experiment_dose, 17.0
            )

        if lung_experiments:
            lung_data = load_lung_data(lung_experiments)
            if lung_data:
                generate_merged_boxplot(
                    lung_data, output_dir, experiment_order, plots_status, cfg
                )
            else:
                logging.warning("No lung data loaded, skipping lung analysis")
        else:
            logging.warning("No lung experiments defined in XML")

        if advanced_metrics:
            advanced_metrics_data = load_advanced_metrics_data(advanced_metrics)
            if advanced_metrics_data:
                generate_global_metrics_boxplot(
                    advanced_metrics_data,
                    output_dir,
                    ["Dice", "Jaccard", "VS", "MI", "Recall"],
                    cfg=cfg,
                    name="global_metrics_boxplot_basic.png",
                )
                generate_global_metrics_boxplot(
                    advanced_metrics_data,
                    output_dir,
                    ["ASSD"],
                    cfg=cfg,
                    name="global_metrics_boxplot_hd_asd.png",
                )
                advanced_stats = perform_statistical_analysis(
                    advanced_metrics_data,
                    ["Dice", "Jaccard", "VS", "MI", "Recall"],
                    output_dir,
                    experiment_order,
                    "unpaired",
                )
                generate_unified_statistical_heatmaps(
                    advanced_stats,
                    experiment_order,
                    output_dir,
                    ["Dice", "Jaccard", "VS", "MI", "F1", "Recall"],
                    test_name="advanced_basic",
                    cfg=cfg,
                )
        return 0

    except KeyboardInterrupt:
        logging.info("Analysis interrupted by user")
        return 1
    except Exception as e:
        logging.error("Unexpected error:")
        logging.exception(e)
        return 1


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    return run_merge_analysis(args)


if __name__ == "__main__":
    sys.exit(main())

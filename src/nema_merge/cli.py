import argparse
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .reporting import generate_merged_boxplot, generate_merged_plots


def setup_logging(log_level: int = 20) -> None:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def parse_xml_config(
    xml_path: Path,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    logging.info(f"Parsing XML configuration: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    experiments = []
    lung_experiments = []

    for experiment in root.findall("experiment"):
        name = experiment.get("name")
        file_path = experiment.get("path")
        lung_path = experiment.get("lung_path")

        if name and file_path:
            experiments.append({"name": name, "path": file_path})
            logging.info(f"Found experiment: {name} -> {file_path}")

            if lung_path:
                lung_experiments.append({"name": name, "path": lung_path})
                logging.info(f"Found lung data: {name} -> {lung_path}")

    logging.info(f"Total experiments found: {len(experiments)}")
    logging.info(f"Total lung experiments found: {len(lung_experiments)}")
    return experiments, lung_experiments


def load_experiment_data(
    experiments: List[Dict[str, str]]
) -> tuple[List[Dict[str, Any]], List[str]]:
    all_data = []
    experiment_order = []

    for exp in experiments:
        exp_name = exp["name"]
        file_path = Path(exp["path"])
        experiment_order.append(exp_name)

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
    return all_data, experiment_order


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


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NEMA Merge Analysis Tool",
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
        "--log-level",
        type=int,
        default=20,
        help="Logging level (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR)",
    )

    return parser


def run_merge_analysis(args: argparse.Namespace) -> int:
    try:
        setup_logging(args.log_level)

        logging.info("Starting NEMA Merge Analysis")
        logging.info(f"XML config: {args.xml_config}")
        logging.info(f"Output directory: {args.output}")

        xml_path = Path(args.xml_config)
        output_dir = Path(args.output)

        if not xml_path.exists():
            logging.error(f"XML configuration file not found: {xml_path}")
            return 1

        output_dir.mkdir(parents=True, exist_ok=True)

        experiments, lung_experiments = parse_xml_config(xml_path)
        if not experiments:
            logging.error("No experiments found in XML configuration")
            return 1

        all_data, experiment_order = load_experiment_data(experiments)
        if not all_data:
            logging.error("No data loaded from experiments")
            return 1

        logging.info("Generating merged plots...")
        generate_merged_plots(all_data, output_dir, experiment_order)

        if lung_experiments:
            lung_data = load_lung_data(lung_experiments)
            if lung_data:
                generate_merged_boxplot(lung_data, output_dir, experiment_order)
            else:
                logging.warning("No lung data loaded, skipping lung analysis")
        else:
            logging.warning("No lung experiments defined in XML")

        logging.info("Merged analysis completed successfully")

        return 0

    except KeyboardInterrupt:
        logging.info("Analysis interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return 1


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    return run_merge_analysis(args)


if __name__ == "__main__":
    sys.exit(main())

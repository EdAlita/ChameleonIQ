import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.nema_quant.cli import create_parser, get_image_properties, main, setup_logging


def _create_cli_files(tmp_path: Path, input_suffix: str = ".nii"):
    input_file = tmp_path / f"input{input_suffix}"
    config_file = tmp_path / "config.yaml"
    output_file = tmp_path / "output.txt"

    input_file.touch()
    config_file.touch()

    return input_file, config_file, output_file


def test_cli_help():
    """Test CLI help functionality."""
    parser = create_parser()

    # Test that help can be displayed without errors
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])

    # Help should exit with code 0
    assert exc_info.value.code == 0


def test_cli_version():
    """Test CLI version display."""
    parser = create_parser()

    # Test that version can be displayed
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])

    # Version should exit with code 0
    assert exc_info.value.code == 0


def test_cli_missing_required_args():
    """Test CLI fails with missing required arguments."""
    parser = create_parser()

    # Missing all required args
    with pytest.raises(SystemExit):
        parser.parse_args([])

    # Missing --output
    with pytest.raises(SystemExit):
        parser.parse_args(["input.nii", "--config", "config.yaml"])

    # Missing --config
    with pytest.raises(SystemExit):
        parser.parse_args(["input.nii", "--output", "output.txt"])


def test_cli_parser_basic_args():
    """Test CLI parser with basic arguments."""
    parser = create_parser()

    args = parser.parse_args(
        ["input.nii", "--config", "config.yaml", "--output", "output.txt"]
    )

    assert args.input_image == "input.nii"
    assert args.config == "config.yaml"
    assert args.output == "output.txt"


def test_cli_parser_all_args():
    """Test CLI parser with all arguments."""
    parser = create_parser()

    args = parser.parse_args(
        [
            "input.nii",
            "--config",
            "config.yaml",
            "--output",
            "output.txt",
            "--standard",
            "NU_2_2018",
            "--spacing",
            "2.0",
            "2.0",
            "2.0",
            "--visualizations-dir",
            "viz",
            "--save-visualizations",
            "--advanced-metrics",
            "--gt-image",
            "ground_truth.nii",
            "--log_level",
            "DEBUG",
        ]
    )

    assert args.input_image == "input.nii"
    assert args.config == "config.yaml"
    assert args.output == "output.txt"
    assert args.standard == "NU_2_2018"
    assert args.spacing == [2.0, 2.0, 2.0]  # Parser converts to floats
    assert args.visualizations_dir == "viz"
    assert args.save_visualizations is True
    assert args.advanced_metrics is True
    assert args.gt_image == "ground_truth.nii"
    assert args.log_level == "DEBUG"


def test_cli_parser_nema_standards():
    """Test CLI parser accepts both NEMA standards."""
    parser = create_parser()

    # Test NU_2_2018
    args = parser.parse_args(
        [
            "input.nii",
            "--config",
            "config.yaml",
            "--output",
            "output.txt",
            "--standard",
            "NU_2_2018",
        ]
    )
    assert args.standard == "NU_2_2018"

    # Test NU_4_2008
    args = parser.parse_args(
        [
            "input.nii",
            "--config",
            "config.yaml",
            "--output",
            "output.txt",
            "--standard",
            "NU_4_2008",
        ]
    )
    assert args.standard == "NU_4_2008"

    # Test invalid standard
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "input.nii",
                "--config",
                "config.yaml",
                "--output",
                "output.txt",
                "--standard",
                "INVALID",
            ]
        )


def test_cli_parser_spacing_values():
    """Test CLI parser correctly handles spacing values."""
    parser = create_parser()

    args = parser.parse_args(
        [
            "input.nii",
            "--config",
            "config.yaml",
            "--output",
            "output.txt",
            "--spacing",
            "2.0644",
            "2.0644",
            "2.0644",
        ]
    )

    assert len(args.spacing) == 3
    assert args.spacing == [2.0644, 2.0644, 2.0644]  # Parser converts to floats

    # Test invalid spacing (wrong number of values)
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "input.nii",
                "--config",
                "config.yaml",
                "--output",
                "output.txt",
                "--spacing",
                "2.0",
                "2.0",  # Only 2 values instead of 3
            ]
        )


@patch("src.nema_quant.cli.load_nii_image")
@patch("src.nema_quant.cli.NemaPhantom")
@patch("src.nema_quant.cli.calculate_nema_metrics")
@patch("src.nema_quant.cli.save_results_to_txt")
@patch("src.nema_quant.cli.get_cfg_defaults")
def test_cli_main_execution(
    mock_get_cfg,
    mock_save_results,
    mock_calculate_metrics,
    mock_phantom_class,
    mock_load_image,
):
    """Test main CLI execution with mocked dependencies."""
    # Setup mocks
    mock_cfg = MagicMock()
    mock_get_cfg.return_value = mock_cfg

    test_image = np.ones((50, 100, 100), dtype=np.float32)
    mock_load_image.return_value = (test_image, np.eye(4))

    mock_phantom = MagicMock()
    mock_phantom_class.return_value = mock_phantom

    mock_calculate_metrics.return_value = ([], {})

    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.nii"
        config_file = Path(tmpdir) / "config.yaml"
        output_file = Path(tmpdir) / "output.txt"

        # Create dummy files
        input_file.touch()
        config_file.touch()

        # Call main with arguments
        try:
            main(
                [
                    str(input_file),
                    "--config",
                    str(config_file),
                    "--output",
                    str(output_file),
                ]
            )
        except SystemExit:
            pass  # CLI may exit normally

        # Verify key functions were called
        assert mock_load_image.called
        assert mock_phantom_class.called
        assert mock_calculate_metrics.called


@patch("src.nema_quant.cli.get_cfg_defaults")
def test_cli_argument_parser_creation(mock_get_cfg):
    """Test that CLI argument parser is created correctly."""
    parser = create_parser()

    # Verify parser has expected arguments
    arg_names = [action.dest for action in parser._actions]

    assert "input_image" in arg_names
    assert "output" in arg_names
    assert "config" in arg_names
    assert "standard" in arg_names
    assert "spacing" in arg_names
    assert "save_visualizations" in arg_names
    assert "advanced_metrics" in arg_names


def test_cli_get_image_properties_variants():
    """Test voxel spacing extraction logic."""
    image_data = np.zeros((5, 6, 7), dtype=np.float32)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])

    dims, spacing = get_image_properties(image_data, affine, (1.5, 1.6, 1.7))
    assert dims == (5, 6, 7)
    assert spacing == (1.5, 1.6, 1.7)

    dims, spacing = get_image_properties(image_data, affine, None)
    assert dims == (5, 6, 7)
    assert spacing == (2.0, 3.0, 4.0)

    dims, spacing = get_image_properties(image_data, None, None)
    assert dims == (5, 6, 7)
    assert spacing == (1.0, 1.0, 1.0)


def test_setup_logging_creates_log_dir(tmp_path: Path):
    """Test setup_logging creates a logs directory near output path."""
    output_path = tmp_path / "results.txt"
    setup_logging(output_path=str(output_path))

    assert (tmp_path / "logs").exists()


def test_cli_input_missing_file_returns_error():
    """Test missing input image path returns error code."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "config.yaml"
        output_file = Path(tmpdir) / "output.txt"
        missing_input = Path(tmpdir) / "missing.nii"

        config_file.touch()

        result = main(
            [
                str(missing_input),
                "--config",
                str(config_file),
                "--output",
                str(output_file),
            ]
        )

        assert result == 1


def test_cli_input_bad_suffix_returns_error():
    """Test invalid input suffix returns error code."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "config.yaml"
        output_file = Path(tmpdir) / "output.txt"
        bad_input = Path(tmpdir) / "input.txt"

        config_file.touch()
        bad_input.touch()

        result = main(
            [
                str(bad_input),
                "--config",
                str(config_file),
                "--output",
                str(output_file),
            ]
        )

        assert result == 1


@patch("src.nema_quant.cli.load_configuration", side_effect=Exception("boom"))
def test_cli_load_configuration_error_returns_1(mock_load_config, tmp_path: Path):
    """Test config load failure returns error code."""
    input_file, config_file, output_file = _create_cli_files(tmp_path)

    result = main(
        [
            str(input_file),
            "--config",
            str(config_file),
            "--output",
            str(output_file),
        ]
    )

    assert result == 1


@patch("src.nema_quant.cli.load_nii_image", side_effect=RuntimeError("bad image"))
@patch("src.nema_quant.cli.load_configuration")
def test_cli_load_image_error_returns_1(
    mock_load_config, mock_load_image, tmp_path: Path
):
    """Test image load failure returns error code."""
    mock_load_config.return_value = MagicMock()
    input_file, config_file, output_file = _create_cli_files(tmp_path)

    result = main(
        [
            str(input_file),
            "--config",
            str(config_file),
            "--output",
            str(output_file),
        ]
    )

    assert result == 1


@patch("src.nema_quant.cli.get_image_properties", side_effect=RuntimeError("bad props"))
@patch("src.nema_quant.cli.load_nii_image")
@patch("src.nema_quant.cli.load_configuration")
def test_cli_image_properties_error_returns_1(
    mock_load_config, mock_load_image, mock_get_props, tmp_path: Path
):
    """Test image properties failure returns error code."""
    mock_load_config.return_value = MagicMock()
    mock_load_image.return_value = (np.ones((5, 5, 5), dtype=np.float32), np.eye(4))
    input_file, config_file, output_file = _create_cli_files(tmp_path)

    result = main(
        [
            str(input_file),
            "--config",
            str(config_file),
            "--output",
            str(output_file),
        ]
    )

    assert result == 1


@patch("src.nema_quant.cli.NemaPhantom", side_effect=RuntimeError("bad phantom"))
@patch("src.nema_quant.cli.load_nii_image")
@patch("src.nema_quant.cli.load_configuration")
def test_cli_phantom_init_error_returns_1(
    mock_load_config, mock_load_image, mock_phantom, tmp_path: Path
):
    """Test phantom init failure returns error code."""
    mock_load_config.return_value = MagicMock()
    mock_load_image.return_value = (np.ones((5, 5, 5), dtype=np.float32), np.eye(4))
    input_file, config_file, output_file = _create_cli_files(tmp_path)

    result = main(
        [
            str(input_file),
            "--config",
            str(config_file),
            "--output",
            str(output_file),
        ]
    )

    assert result == 1


@patch(
    "src.nema_quant.cli.calculate_nema_metrics", side_effect=RuntimeError("bad metrics")
)
@patch("src.nema_quant.cli.NemaPhantom")
@patch("src.nema_quant.cli.load_nii_image")
@patch("src.nema_quant.cli.load_configuration")
def test_cli_analysis_error_returns_1(
    mock_load_config, mock_load_image, mock_phantom, mock_metrics, tmp_path: Path
):
    """Test analysis failure returns error code."""
    mock_load_config.return_value = MagicMock()
    mock_load_image.return_value = (np.ones((5, 5, 5), dtype=np.float32), np.eye(4))
    mock_phantom.return_value = MagicMock(rois=[1, 2, 3])
    input_file, config_file, output_file = _create_cli_files(tmp_path)

    result = main(
        [
            str(input_file),
            "--config",
            str(config_file),
            "--output",
            str(output_file),
        ]
    )

    assert result == 1


@patch("src.nema_quant.cli.generate_plots", side_effect=RuntimeError("plot fail"))
@patch("src.nema_quant.cli.calculate_nema_metrics")
@patch("src.nema_quant.cli.NemaPhantom")
@patch("src.nema_quant.cli.load_nii_image")
@patch("src.nema_quant.cli.load_configuration")
def test_cli_plot_error_nu2_returns_1(
    mock_load_config,
    mock_load_image,
    mock_phantom,
    mock_metrics,
    mock_generate_plots,
    tmp_path: Path,
):
    """Test plot failure in NU_2_2018 path returns error code."""
    mock_load_config.return_value = MagicMock()
    mock_load_image.return_value = (np.ones((5, 5, 5), dtype=np.float32), np.eye(4))
    mock_phantom.return_value = MagicMock(rois=[1, 2, 3])
    mock_metrics.return_value = ([], {1: 1.0})
    input_file, config_file, output_file = _create_cli_files(tmp_path)

    result = main(
        [
            str(input_file),
            "--config",
            str(config_file),
            "--output",
            str(output_file),
        ]
    )

    assert result == 1


@patch("src.nema_quant.cli.save_results_to_txt", side_effect=RuntimeError("save fail"))
@patch("src.nema_quant.cli.generate_torso_plot")
@patch("src.nema_quant.cli.generate_coronal_sphere_plots")
@patch("src.nema_quant.cli.generate_boxplot_with_mean_std")
@patch("src.nema_quant.cli.generate_transverse_sphere_plots")
@patch("src.nema_quant.cli.generate_rois_plots")
@patch("src.nema_quant.cli.generate_plots")
@patch("src.nema_quant.cli.calculate_nema_metrics")
@patch("src.nema_quant.cli.NemaPhantom")
@patch("src.nema_quant.cli.load_nii_image")
@patch("src.nema_quant.cli.load_configuration")
def test_cli_save_results_error_nu2_returns_1(
    mock_load_config,
    mock_load_image,
    mock_phantom,
    mock_metrics,
    mock_generate_plots,
    mock_generate_rois,
    mock_generate_transverse,
    mock_generate_boxplot,
    mock_generate_coronal,
    mock_generate_torso,
    mock_save_results,
    tmp_path: Path,
):
    """Test save results failure in NU_2_2018 path returns error code."""
    mock_load_config.return_value = MagicMock()
    mock_load_image.return_value = (np.ones((5, 5, 5), dtype=np.float32), np.eye(4))
    mock_phantom.return_value = MagicMock(rois=[1, 2, 3])
    mock_metrics.return_value = ([], {1: 1.0})
    input_file, config_file, output_file = _create_cli_files(tmp_path)

    result = main(
        [
            str(input_file),
            "--config",
            str(config_file),
            "--output",
            str(output_file),
        ]
    )

    assert result == 1


@patch(
    "src.nema_quant.cli.generate_crc_plots_nu4", side_effect=RuntimeError("plot fail")
)
@patch("src.nema_quant.cli.calculate_nema_metrics_nu4_2008")
@patch("src.nema_quant.cli.NemaPhantom")
@patch("src.nema_quant.cli.load_nii_image")
@patch("src.nema_quant.cli.load_configuration")
def test_cli_plot_error_nu4_returns_1(
    mock_load_config,
    mock_load_image,
    mock_phantom,
    mock_metrics,
    mock_generate_crc,
    tmp_path: Path,
):
    """Test plot failure in NU_4_2008 path returns error code."""
    mock_load_config.return_value = MagicMock()
    mock_load_image.return_value = (np.ones((5, 5, 5), dtype=np.float32), np.eye(4))
    mock_phantom.return_value = MagicMock(rois=[1, 2, 3])
    mock_metrics.return_value = ([], {}, {})
    input_file, config_file, output_file = _create_cli_files(tmp_path)

    result = main(
        [
            str(input_file),
            "--config",
            str(config_file),
            "--output",
            str(output_file),
            "--standard",
            "NU_4_2008",
        ]
    )

    assert result == 1


@patch(
    "src.nema_quant.cli.save_results_to_txt_nu4", side_effect=RuntimeError("save fail")
)
@patch("src.nema_quant.cli.generate_spillover_barplot_nu4")
@patch("src.nema_quant.cli.generate_iq_plot")
@patch("src.nema_quant.cli.generate_crc_plots_nu4")
@patch("src.nema_quant.cli.calculate_nema_metrics_nu4_2008")
@patch("src.nema_quant.cli.NemaPhantom")
@patch("src.nema_quant.cli.load_nii_image")
@patch("src.nema_quant.cli.load_configuration")
def test_cli_save_results_error_nu4_returns_1(
    mock_load_config,
    mock_load_image,
    mock_phantom,
    mock_metrics,
    mock_generate_crc,
    mock_generate_iq,
    mock_generate_spillover,
    mock_save_results,
    tmp_path: Path,
):
    """Test save results failure in NU_4_2008 path returns error code."""
    mock_load_config.return_value = MagicMock()
    mock_load_image.return_value = (np.ones((5, 5, 5), dtype=np.float32), np.eye(4))
    mock_phantom.return_value = MagicMock(rois=[1, 2, 3])
    mock_metrics.return_value = ([], {}, {})
    input_file, config_file, output_file = _create_cli_files(tmp_path)

    result = main(
        [
            str(input_file),
            "--config",
            str(config_file),
            "--output",
            str(output_file),
            "--standard",
            "NU_4_2008",
        ]
    )

    assert result == 1


@patch("src.nema_quant.cli.generate_reportlab_report")
@patch("src.nema_quant.cli.save_results_to_txt")
@patch("src.nema_quant.cli.generate_torso_plot")
@patch("src.nema_quant.cli.generate_coronal_sphere_plots")
@patch("src.nema_quant.cli.generate_boxplot_with_mean_std")
@patch("src.nema_quant.cli.generate_transverse_sphere_plots")
@patch("src.nema_quant.cli.generate_rois_plots")
@patch("src.nema_quant.cli.generate_plots")
@patch("src.nema_quant.cli.calculate_nema_metrics")
@patch("src.nema_quant.cli.NemaPhantom")
@patch("src.nema_quant.cli.load_nii_image")
@patch("src.nema_quant.cli.load_configuration")
def test_cli_advanced_metrics_missing_gt_file_returns_1(
    mock_load_config,
    mock_load_image,
    mock_phantom,
    mock_metrics,
    mock_generate_plots,
    mock_generate_rois,
    mock_generate_transverse,
    mock_generate_boxplot,
    mock_generate_coronal,
    mock_generate_torso,
    mock_save_results,
    mock_generate_report,
    tmp_path: Path,
):
    """Test --advanced-metrics with missing gt file returns error code."""
    mock_load_config.return_value = MagicMock()
    mock_load_image.return_value = (np.ones((5, 5, 5), dtype=np.float32), np.eye(4))
    mock_phantom.return_value = MagicMock(rois=[1, 2, 3])
    mock_metrics.return_value = ([], {1: 1.0})
    input_file, config_file, output_file = _create_cli_files(tmp_path)

    result = main(
        [
            str(input_file),
            "--config",
            str(config_file),
            "--output",
            str(output_file),
            "--advanced-metrics",
            "--gt-image",
            str(tmp_path / "missing_gt.nii"),
        ]
    )

    assert result == 1


@patch("src.nema_quant.cli.generate_reportlab_report")
@patch("src.nema_quant.cli.save_results_to_txt")
@patch("src.nema_quant.cli.generate_torso_plot")
@patch("src.nema_quant.cli.generate_coronal_sphere_plots")
@patch("src.nema_quant.cli.generate_boxplot_with_mean_std")
@patch("src.nema_quant.cli.generate_transverse_sphere_plots")
@patch("src.nema_quant.cli.generate_rois_plots")
@patch("src.nema_quant.cli.generate_plots")
@patch("src.nema_quant.cli.calculate_nema_metrics")
@patch("src.nema_quant.cli.NemaPhantom")
@patch("src.nema_quant.cli.load_nii_image")
@patch("src.nema_quant.cli.load_configuration")
def test_cli_advanced_metrics_gt_load_error_returns_1(
    mock_load_config,
    mock_load_image,
    mock_phantom,
    mock_metrics,
    mock_generate_plots,
    mock_generate_rois,
    mock_generate_transverse,
    mock_generate_boxplot,
    mock_generate_coronal,
    mock_generate_torso,
    mock_save_results,
    mock_generate_report,
    tmp_path: Path,
):
    """Test ground truth load failure returns error code."""
    mock_load_config.return_value = MagicMock()
    mock_load_image.side_effect = [
        (np.ones((5, 5, 5), dtype=np.float32), np.eye(4)),
        RuntimeError("gt load fail"),
    ]
    mock_phantom.return_value = MagicMock(rois=[1, 2, 3])
    mock_metrics.return_value = ([], {1: 1.0})
    input_file, config_file, output_file = _create_cli_files(tmp_path)
    gt_file = tmp_path / "gt.nii"
    gt_file.touch()

    result = main(
        [
            str(input_file),
            "--config",
            str(config_file),
            "--output",
            str(output_file),
            "--advanced-metrics",
            "--gt-image",
            str(gt_file),
        ]
    )

    assert result == 1


@patch(
    "src.nema_quant.analysis.calculate_advanced_metrics",
    side_effect=RuntimeError("adv fail"),
)
@patch("src.nema_quant.cli.generate_reportlab_report")
@patch("src.nema_quant.cli.save_results_to_txt")
@patch("src.nema_quant.cli.generate_torso_plot")
@patch("src.nema_quant.cli.generate_coronal_sphere_plots")
@patch("src.nema_quant.cli.generate_boxplot_with_mean_std")
@patch("src.nema_quant.cli.generate_transverse_sphere_plots")
@patch("src.nema_quant.cli.generate_rois_plots")
@patch("src.nema_quant.cli.generate_plots")
@patch("src.nema_quant.cli.calculate_nema_metrics")
@patch("src.nema_quant.cli.NemaPhantom")
@patch("src.nema_quant.cli.load_nii_image")
@patch("src.nema_quant.cli.load_configuration")
def test_cli_advanced_metrics_calculation_error_returns_1(
    mock_load_config,
    mock_load_image,
    mock_phantom,
    mock_metrics,
    mock_generate_plots,
    mock_generate_rois,
    mock_generate_transverse,
    mock_generate_boxplot,
    mock_generate_coronal,
    mock_generate_torso,
    mock_save_results,
    mock_generate_report,
    mock_calculate_adv,
    tmp_path: Path,
):
    """Test advanced metrics calculation failure returns error code."""
    mock_load_config.return_value = MagicMock()
    mock_load_image.side_effect = [
        (np.ones((5, 5, 5), dtype=np.float32), np.eye(4)),
        (np.ones((5, 5, 5), dtype=np.float32), np.eye(4)),
    ]
    mock_phantom.return_value = MagicMock(rois=[1, 2, 3])
    mock_metrics.return_value = ([], {1: 1.0})
    input_file, config_file, output_file = _create_cli_files(tmp_path)
    gt_file = tmp_path / "gt.nii"
    gt_file.touch()

    result = main(
        [
            str(input_file),
            "--config",
            str(config_file),
            "--output",
            str(output_file),
            "--advanced-metrics",
            "--gt-image",
            str(gt_file),
        ]
    )

    assert result == 1


@patch("src.nema_quant.cli.generate_reportlab_report")
@patch("src.nema_quant.cli.save_results_to_txt")
@patch("src.nema_quant.cli.generate_torso_plot")
@patch("src.nema_quant.cli.generate_coronal_sphere_plots")
@patch("src.nema_quant.cli.generate_boxplot_with_mean_std")
@patch("src.nema_quant.cli.generate_transverse_sphere_plots")
@patch("src.nema_quant.cli.generate_rois_plots")
@patch("src.nema_quant.cli.generate_plots")
@patch("src.nema_quant.cli.calculate_nema_metrics")
def test_cli_integration_with_temp_nifti(
    mock_calculate_metrics,
    mock_generate_plots,
    mock_generate_rois,
    mock_generate_transverse,
    mock_generate_boxplot,
    mock_generate_coronal,
    mock_generate_torso,
    mock_save_results,
    mock_generate_report,
    tmp_path,
):
    """Run CLI with a temp NIfTI and minimal config file."""
    from nibabel.nifti1 import Nifti1Image
    from nibabel.nifti1 import save as nib_save

    image_data = np.zeros((10, 10, 10), dtype=np.float32)
    nifti = Nifti1Image(image_data, affine=np.eye(4))

    input_path = tmp_path / "input.nii"
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "results.txt"

    nib_save(nifti, str(input_path))
    config_path.write_text(
        "ACTIVITY:\n  HOT: 8.0\n  BACKGROUND: 1.0\n", encoding="utf-8"
    )

    mock_calculate_metrics.return_value = ([], {1: 1.0})

    result = main(
        [
            str(input_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    assert mock_calculate_metrics.called
    assert mock_save_results.called


@patch("src.nema_quant.cli.generate_reportlab_report_nu4")
@patch("src.nema_quant.cli.save_results_to_txt_nu4")
@patch("src.nema_quant.cli.generate_spillover_barplot_nu4")
@patch("src.nema_quant.cli.generate_iq_plot")
@patch("src.nema_quant.cli.generate_crc_plots_nu4")
@patch("src.nema_quant.cli.calculate_nema_metrics_nu4_2008")
@patch("src.nema_quant.cli.NemaPhantom")
@patch("src.nema_quant.cli.load_nii_image")
@patch("src.nema_quant.cli.load_configuration")
def test_cli_run_analysis_nu4_2008_path(
    mock_load_config,
    mock_load_image,
    mock_phantom_class,
    mock_calculate_metrics,
    mock_generate_crc,
    mock_generate_iq,
    mock_generate_spillover,
    mock_save_results,
    mock_generate_report,
):
    """Test NU_4_2008 execution path wiring."""
    mock_load_config.return_value = MagicMock()
    mock_load_image.return_value = (np.ones((10, 10, 10), dtype=np.float32), np.eye(4))
    mock_phantom_class.return_value = MagicMock(rois=[1, 2, 3])
    mock_calculate_metrics.return_value = ([], {}, {})

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.nii"
        config_file = Path(tmpdir) / "config.yaml"
        output_file = Path(tmpdir) / "output.txt"

        input_file.touch()
        config_file.touch()

        result = main(
            [
                str(input_file),
                "--config",
                str(config_file),
                "--output",
                str(output_file),
                "--standard",
                "NU_4_2008",
            ]
        )

        assert result == 0
        assert mock_calculate_metrics.called
        assert mock_save_results.called
        assert mock_generate_report.called


@patch("src.nema_quant.cli.generate_reportlab_report")
@patch("src.nema_quant.cli.save_results_to_txt")
@patch("src.nema_quant.cli.generate_torso_plot")
@patch("src.nema_quant.cli.generate_coronal_sphere_plots")
@patch("src.nema_quant.cli.generate_boxplot_with_mean_std")
@patch("src.nema_quant.cli.generate_transverse_sphere_plots")
@patch("src.nema_quant.cli.generate_rois_plots")
@patch("src.nema_quant.cli.generate_plots")
@patch("src.nema_quant.cli.calculate_nema_metrics")
@patch("src.nema_quant.cli.NemaPhantom")
@patch("src.nema_quant.cli.load_nii_image")
@patch("src.nema_quant.cli.load_configuration")
def test_cli_advanced_metrics_requires_gt_image(
    mock_load_config,
    mock_load_image,
    mock_phantom_class,
    mock_calculate_metrics,
    mock_generate_plots,
    mock_generate_rois,
    mock_generate_transverse,
    mock_generate_boxplot,
    mock_generate_coronal,
    mock_generate_torso,
    mock_save_results,
    mock_generate_report,
):
    """Test --advanced-metrics enforces --gt-image."""
    mock_load_config.return_value = MagicMock()
    mock_load_image.return_value = (np.ones((10, 10, 10), dtype=np.float32), np.eye(4))
    mock_phantom_class.return_value = MagicMock(rois=[1, 2, 3])
    mock_calculate_metrics.return_value = ([], {1: 1.0})

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.nii"
        config_file = Path(tmpdir) / "config.yaml"
        output_file = Path(tmpdir) / "output.txt"

        input_file.touch()
        config_file.touch()

        result = main(
            [
                str(input_file),
                "--config",
                str(config_file),
                "--output",
                str(output_file),
                "--advanced-metrics",
            ]
        )

        assert result == 1

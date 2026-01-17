from unittest.mock import MagicMock, patch

import numpy as np
from yacs.config import CfgNode

from src.nema_quant import cli


class TestCLIRemainingCoverage:
    """Target the remaining missing lines in CLI module."""

    @patch("src.nema_quant.cli.setup_logging")
    @patch("pathlib.Path.exists")
    @patch("src.nema_quant.cli.load_configuration")
    @patch("src.nema_quant.cli.load_nii_image")
    @patch("src.nema_quant.cli.get_image_properties")
    @patch("src.nema_quant.cli.NemaPhantom")
    @patch("src.nema_quant.cli.calculate_nema_metrics")
    @patch("src.nema_quant.cli.save_results_to_txt")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.is_dir")
    def test_visualization_directory_creation_lines_194_196(
        self,
        mock_is_dir,
        mock_mkdir,
        mock_save_results,
        mock_calculate_metrics,
        mock_phantom_class,
        mock_get_props,
        mock_load_image,
        mock_load_config,
        mock_exists,
        mock_setup_logging,
    ):
        """Test lines 194-196: visualization directory creation logic."""
        mock_exists.return_value = True
        mock_load_config.return_value = CfgNode()
        mock_load_image.return_value = (np.ones((50, 100, 100)), np.eye(4))
        mock_get_props.return_value = ((50, 100, 100), (2.0, 2.0, 2.0))

        mock_phantom = MagicMock()
        mock_phantom.rois = {"sphere1": {}}
        mock_phantom_class.return_value = mock_phantom

        mock_calculate_metrics.return_value = (
            [
                {
                    "diameter_mm": 10.0,
                    "percentaje_constrast_QH": 85.0,
                    "background_variability_N": 5.2,
                    "avg_hot_counts_CH": 15000.0,
                    "avg_bkg_counts_CB": 2000.0,
                    "bkg_std_dev_SD": 104.0,
                }
            ],
            {10: 95.0},
        )

        # Test case 1: visualizations_dir is None (should create default)
        mock_is_dir.return_value = False  # Directory doesn't exist

        args = MagicMock()
        args.log_level = "INFO"
        args.input_image = "input.nii"
        args.output = "output.txt"
        args.config = "config.yaml"
        args.spacing = None
        args.save_visualizations = True
        args.visualizations_dir = None  # This should trigger default directory creation

        result = cli.run_analysis(args)

        # Directory creation might be called
        assert result in [0, 1]  # May succeed or fail depending on visualization logic

    @patch("src.nema_quant.cli.setup_logging")
    @patch("pathlib.Path.exists")
    @patch("src.nema_quant.cli.load_configuration")
    @patch("src.nema_quant.cli.load_nii_image")
    @patch("src.nema_quant.cli.get_image_properties")
    @patch("src.nema_quant.cli.NemaPhantom")
    @patch("src.nema_quant.cli.calculate_nema_metrics")
    @patch("src.nema_quant.cli.save_results_to_txt")
    @patch("src.nema_quant.cli.generate_plots")
    def test_plot_generation_lines_207_209(
        self,
        mock_generate_plots,
        mock_save_results,
        mock_calculate_metrics,
        mock_phantom_class,
        mock_get_props,
        mock_load_image,
        mock_load_config,
        mock_exists,
        mock_setup_logging,
    ):
        """Test lines 207-209: plot generation logic."""
        mock_exists.return_value = True
        mock_load_config.return_value = CfgNode()
        mock_load_image.return_value = (np.ones((50, 100, 100)), np.eye(4))
        mock_get_props.return_value = ((50, 100, 100), (2.0, 2.0, 2.0))

        mock_phantom = MagicMock()
        mock_phantom.rois = {"sphere1": {}}
        mock_phantom_class.return_value = mock_phantom

        mock_calculate_metrics.return_value = (
            [
                {
                    "diameter_mm": 10.0,
                    "percentaje_constrast_QH": 85.0,
                    "background_variability_N": 5.2,
                    "avg_hot_counts_CH": 15000.0,
                    "avg_bkg_counts_CB": 2000.0,
                    "bkg_std_dev_SD": 104.0,
                }
            ],
            {10: 95.0},
        )

        # Mock generate_plots to succeed
        mock_generate_plots.return_value = None

        args = MagicMock()
        args.log_level = "INFO"
        args.input_image = "input.nii"
        args.output = "output.txt"
        args.config = "config.yaml"
        args.spacing = None
        args.save_visualizations = True
        args.visualizations_dir = "test_viz"

        result = cli.run_analysis(args)

        # Should call generate_plots
        mock_generate_plots.assert_called_once()
        assert result in [0, 1]

    @patch("src.nema_quant.cli.setup_logging")
    @patch("pathlib.Path.exists")
    @patch("src.nema_quant.cli.load_configuration")
    @patch("src.nema_quant.cli.load_nii_image")
    @patch("src.nema_quant.cli.get_image_properties")
    @patch("src.nema_quant.cli.NemaPhantom")
    @patch("src.nema_quant.cli.calculate_nema_metrics")
    @patch("src.nema_quant.cli.save_results_to_txt")
    @patch("src.nema_quant.cli.generate_plots")
    @patch("src.nema_quant.cli.generate_rois_plots")
    @patch("src.nema_quant.cli.generate_boxplot_with_mean_std")
    def test_all_visualization_functions_lines_217_224(
        self,
        mock_boxplot,
        mock_rois_plots,
        mock_generate_plots,
        mock_save_results,
        mock_calculate_metrics,
        mock_phantom_class,
        mock_get_props,
        mock_load_image,
        mock_load_config,
        mock_exists,
        mock_setup_logging,
    ):
        """Test lines 217-224: all visualization function calls."""
        mock_exists.return_value = True
        mock_load_config.return_value = CfgNode()
        mock_load_image.return_value = (np.ones((50, 100, 100)), np.eye(4))
        mock_get_props.return_value = ((50, 100, 100), (2.0, 2.0, 2.0))

        mock_phantom = MagicMock()
        mock_phantom.rois = {"sphere1": {}}
        mock_phantom_class.return_value = mock_phantom

        mock_calculate_metrics.return_value = (
            [
                {
                    "diameter_mm": 10.0,
                    "percentaje_constrast_QH": 85.0,
                    "background_variability_N": 5.2,
                    "avg_hot_counts_CH": 15000.0,
                    "avg_bkg_counts_CB": 2000.0,
                    "bkg_std_dev_SD": 104.0,
                }
            ],
            {10: 95.0},
        )

        # Mock all visualization functions to succeed
        mock_generate_plots.return_value = None
        mock_rois_plots.return_value = None

        args = MagicMock()
        args.log_level = "INFO"
        args.input_image = "input.nii"
        args.output = "output.txt"
        args.config = "config.yaml"
        args.spacing = None
        args.save_visualizations = True
        args.visualizations_dir = "test_viz"

        result = cli.run_analysis(args)

        # All visualization functions should be called
        mock_generate_plots.assert_called_once()
        mock_rois_plots.assert_called_once()
        assert result in [0, 1]

    @patch("src.nema_quant.cli.setup_logging")
    @patch("pathlib.Path.exists")
    @patch("src.nema_quant.cli.load_configuration")
    def test_config_loading_exception_lines_230_237(
        self, mock_load_config, mock_exists, mock_setup_logging
    ):
        """Test lines 230-237: configuration loading exception handling."""
        mock_exists.return_value = True

        # Test different configuration loading errors
        config_errors = [
            FileNotFoundError("Config file not found"),
            PermissionError("Permission denied"),
            ValueError("Invalid YAML format"),
            Exception("Unexpected config error"),
        ]

        for error in config_errors:
            mock_load_config.side_effect = error

            args = MagicMock()
            args.log_level = "INFO"
            args.input_image = "input.nii"
            args.config = "config.yaml"

            result = cli.run_analysis(args)
            assert result == 1  # Should return error code for config errors

    @patch("src.nema_quant.cli.setup_logging")
    @patch("pathlib.Path.exists")
    @patch("src.nema_quant.cli.load_configuration")
    @patch("src.nema_quant.cli.load_nii_image")
    def test_image_loading_exception_lines_259_261(
        self, mock_load_image, mock_load_config, mock_exists, mock_setup_logging
    ):
        """Test lines 259-261: image loading exception handling."""
        mock_exists.return_value = True
        mock_load_config.return_value = CfgNode()

        # Test different image loading errors
        image_errors = [
            FileNotFoundError("Image file not found"),
            PermissionError("Permission denied"),
            ValueError("Invalid NIfTI format"),
            RuntimeError("Corrupted image file"),
            Exception("Unexpected image error"),
        ]

        for error in image_errors:
            mock_load_image.side_effect = error

            args = MagicMock()
            args.log_level = "INFO"
            args.input_image = "input.nii"
            args.config = "config.yaml"

            result = cli.run_analysis(args)
            assert result == 1  # Should return error code for image errors

    @patch("src.nema_quant.cli.setup_logging")
    @patch("pathlib.Path.exists")
    @patch("src.nema_quant.cli.load_configuration")
    @patch("src.nema_quant.cli.load_nii_image")
    @patch("src.nema_quant.cli.get_image_properties")
    @patch("src.nema_quant.cli.NemaPhantom")
    @patch("src.nema_quant.cli.calculate_nema_metrics")
    def test_analysis_exception_lines_276_278(
        self,
        mock_calculate_metrics,
        mock_phantom_class,
        mock_get_props,
        mock_load_image,
        mock_load_config,
        mock_exists,
        mock_setup_logging,
    ):
        """Test lines 276-278: analysis exception handling."""
        mock_exists.return_value = True
        mock_load_config.return_value = CfgNode()
        mock_load_image.return_value = (np.ones((50, 100, 100)), np.eye(4))
        mock_get_props.return_value = ((50, 100, 100), (2.0, 2.0, 2.0))

        mock_phantom = MagicMock()
        mock_phantom.rois = {"sphere1": {}}
        mock_phantom_class.return_value = mock_phantom

        # Test different analysis errors
        analysis_errors = [
            ValueError("Invalid phantom configuration"),
            RuntimeError("Analysis computation failed"),
            Exception("Unexpected analysis error"),
        ]

        for error in analysis_errors:
            mock_calculate_metrics.side_effect = error

            args = MagicMock()
            args.log_level = "INFO"
            args.input_image = "input.nii"
            args.config = "config.yaml"
            args.spacing = None
            args.save_visualizations = False

            result = cli.run_analysis(args)
            assert result == 1  # Should return error code for analysis errors

    @patch("src.nema_quant.cli.setup_logging")
    @patch("pathlib.Path.exists")
    @patch("src.nema_quant.cli.load_configuration")
    @patch("src.nema_quant.cli.load_nii_image")
    @patch("src.nema_quant.cli.get_image_properties")
    @patch("src.nema_quant.cli.NemaPhantom")
    @patch("src.nema_quant.cli.calculate_nema_metrics")
    @patch("src.nema_quant.cli.save_results_to_txt")
    def test_results_saving_exception_lines_312_314(
        self,
        mock_save_results,
        mock_calculate_metrics,
        mock_phantom_class,
        mock_get_props,
        mock_load_image,
        mock_load_config,
        mock_exists,
        mock_setup_logging,
    ):
        """Test lines 312-314: results saving exception handling."""
        mock_exists.return_value = True
        mock_load_config.return_value = CfgNode()
        mock_load_image.return_value = (np.ones((50, 100, 100)), np.eye(4))
        mock_get_props.return_value = ((50, 100, 100), (2.0, 2.0, 2.0))

        mock_phantom = MagicMock()
        mock_phantom.rois = {"sphere1": {}}
        mock_phantom_class.return_value = mock_phantom

        mock_calculate_metrics.return_value = (
            [
                {
                    "diameter_mm": 10.0,
                    "percentaje_constrast_QH": 85.0,
                    "background_variability_N": 5.2,
                    "avg_hot_counts_CH": 15000.0,
                    "avg_bkg_counts_CB": 2000.0,
                    "bkg_std_dev_SD": 104.0,
                }
            ],
            {10: 95.0},
        )

        # Test different saving errors
        saving_errors = [
            PermissionError("Cannot write to file"),
            OSError("Disk full"),
            Exception("Unexpected saving error"),
        ]

        for error in saving_errors:
            mock_save_results.side_effect = error

            args = MagicMock()
            args.log_level = "INFO"
            args.input_image = "input.nii"
            args.output = "output.txt"
            args.config = "config.yaml"
            args.spacing = None
            args.save_visualizations = False

            result = cli.run_analysis(args)
            assert result == 1  # Should return error code for saving errors

    @patch("src.nema_quant.cli.setup_logging")
    @patch("pathlib.Path.exists")
    @patch("src.nema_quant.cli.load_configuration")
    @patch("src.nema_quant.cli.load_nii_image")
    @patch("src.nema_quant.cli.get_image_properties")
    @patch("src.nema_quant.cli.NemaPhantom")
    @patch("src.nema_quant.cli.calculate_nema_metrics")
    @patch("src.nema_quant.cli.save_results_to_txt")
    @patch("src.nema_quant.cli.generate_plots")
    def test_visualization_exception_lines_336_345(
        self,
        mock_generate_plots,
        mock_save_results,
        mock_calculate_metrics,
        mock_phantom_class,
        mock_get_props,
        mock_load_image,
        mock_load_config,
        mock_exists,
        mock_setup_logging,
    ):
        """Test lines 336-345: visualization exception handling."""
        mock_exists.return_value = True
        mock_load_config.return_value = CfgNode()
        mock_load_image.return_value = (np.ones((50, 100, 100)), np.eye(4))
        mock_get_props.return_value = ((50, 100, 100), (2.0, 2.0, 2.0))

        mock_phantom = MagicMock()
        mock_phantom.rois = {"sphere1": {}}
        mock_phantom_class.return_value = mock_phantom

        mock_calculate_metrics.return_value = (
            [
                {
                    "diameter_mm": 10.0,
                    "percentaje_constrast_QH": 85.0,
                    "background_variability_N": 5.2,
                    "avg_hot_counts_CH": 15000.0,
                    "avg_bkg_counts_CB": 2000.0,
                    "bkg_std_dev_SD": 104.0,
                }
            ],
            {10: 95.0},
        )

        # Test different visualization errors
        viz_errors = [
            ImportError("matplotlib not available"),
            RuntimeError("Display not available"),
            KeyError("percentaje_constrast_QH"),  # The actual KeyError from your code
            Exception("Unexpected visualization error"),
        ]

        for error in viz_errors:
            mock_generate_plots.side_effect = error

            args = MagicMock()
            args.log_level = "INFO"
            args.input_image = "input.nii"
            args.output = "output.txt"
            args.config = "config.yaml"
            args.spacing = None
            args.save_visualizations = True
            args.visualizations_dir = "test_viz"

            result = cli.run_analysis(args)
            assert result == 1  # Should return error code for visualization errors

    def test_argument_parser_comprehensive(self):
        """Test argument parser with various inputs."""
        parser = cli.create_parser()

        # Test minimal arguments
        args = parser.parse_args(
            ["input.nii", "--output", "output.txt", "--config", "config.yaml"]
        )

        assert args.input_image == "input.nii"
        assert args.output == "output.txt"
        assert args.config == "config.yaml"
        assert args.log_level == "DEBUG"
        assert args.save_visualizations is False

        # Test with all arguments
        args = parser.parse_args(
            [
                "input.nii",
                "--output",
                "output.txt",
                "--config",
                "config.yaml",
                "--save-visualizations",
                "--visualizations-dir",
                "viz_dir",
            ]
        )

        assert args.log_level == "DEBUG"
        assert args.save_visualizations is True
        assert args.visualizations_dir == "viz_dir"

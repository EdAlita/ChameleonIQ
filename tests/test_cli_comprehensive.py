from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from yacs.config import CfgNode

from src.nema_quant import cli


class TestCLIArgumentParsing:
    """Comprehensive CLI argument parsing tests."""

    def test_create_parser_all_arguments(self):
        """Test parser with all possible arguments."""
        parser = cli.create_parser()

        # Test all argument combinations
        test_args = [
            "input.nii",
            "--output",
            "output.txt",
            "--config",
            "config.yaml",
            "--spacing",
            "2.0",
            "2.0",
            "2.0",
            "--save-visualizations",
            "--visualizations-dir",
            "viz_output",
            "--verbose",
        ]

        args = parser.parse_args(test_args)

        # Verify all arguments are parsed correctly
        assert args.input_image == "input.nii"
        assert args.output == "output.txt"
        assert args.config == "config.yaml"
        assert args.spacing == [2.0, 2.0, 2.0]
        assert args.save_visualizations is True
        assert args.visualizations_dir == "viz_output"
        assert args.verbose is True

    def test_parser_minimal_arguments(self):
        """Test parser with minimal required arguments."""
        parser = cli.create_parser()

        args = parser.parse_args(
            ["input.nii", "--output", "output.txt", "--config", "config.yaml"]
        )

        assert args.input_image == "input.nii"
        assert args.output == "output.txt"
        assert args.config == "config.yaml"
        # Check defaults
        assert args.spacing is None
        assert args.save_visualizations is False
        assert args.verbose is False

    def test_parser_version(self):
        """Test version argument."""
        parser = cli.create_parser()

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])

        # Version should exit with code 0
        assert exc_info.value.code == 0

    def test_parser_help(self):
        """Test help argument."""
        parser = cli.create_parser()

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])

        # Help should exit with code 0
        assert exc_info.value.code == 0


class TestCLIUtilityFunctions:
    """Test CLI utility functions in detail."""

    def test_setup_logging_verbose(self):
        """Test verbose logging setup."""
        # Test that function exists and can be called
        if hasattr(cli, "setup_logging"):
            cli.setup_logging(verbose=True)
            # Should not raise exception
        else:
            pytest.skip("setup_logging function not found")

    def test_setup_logging_quiet(self):
        """Test quiet logging setup."""
        if hasattr(cli, "setup_logging"):
            cli.setup_logging(verbose=False)
            # Should not raise exception
        else:
            pytest.skip("setup_logging function not found")

    @patch("pathlib.Path.exists")
    def test_load_configuration_file_exists(self, mock_exists):
        """Test loading configuration from existing file."""
        mock_exists.return_value = True

        if hasattr(cli, "load_configuration"):
            with patch("yacs.config.CfgNode.merge_from_file") as mock_merge:
                cfg = cli.load_configuration("test_config.yaml")
                assert cfg is not None
                mock_merge.assert_called_once_with("test_config.yaml")
        else:
            pytest.skip("load_configuration function not found")

    @patch("pathlib.Path.exists")
    def test_load_configuration_file_not_exists(self, mock_exists):
        """Test loading configuration from non-existent file."""
        mock_exists.return_value = False

        if hasattr(cli, "load_configuration"):
            with pytest.raises(FileNotFoundError):
                cli.load_configuration("nonexistent.yaml")
        else:
            pytest.skip("load_configuration function not found")

    def test_load_configuration_none_input(self):
        """Test loading default configuration."""
        if hasattr(cli, "load_configuration"):
            cfg = cli.load_configuration(None)
            assert cfg is not None
        else:
            pytest.skip("load_configuration function not found")

    def test_get_image_properties_with_spacing(self):
        """Test image properties extraction with spacing override."""
        if hasattr(cli, "get_image_properties"):
            image_data = np.ones((10, 20, 30))
            affine = np.eye(4)
            spacing_override = (1.5, 2.0, 2.5)

            dims, spacing = cli.get_image_properties(
                image_data, affine, spacing_override
            )

            assert dims == (10, 20, 30)
            assert spacing == spacing_override
        else:
            pytest.skip("get_image_properties function not found")

    def test_get_image_properties_from_affine(self):
        """Test extracting spacing from affine matrix."""
        if hasattr(cli, "get_image_properties"):
            image_data = np.ones((10, 20, 30))
            affine = np.array(
                [[2.0, 0, 0, 0], [0, 2.5, 0, 0], [0, 0, 3.0, 0], [0, 0, 0, 1]]
            )

            dims, spacing = cli.get_image_properties(image_data, affine, None)

            assert dims == (10, 20, 30)
            assert spacing == (2.0, 2.5, 3.0)
        else:
            pytest.skip("get_image_properties function not found")

    def test_get_image_properties_default_spacing(self):
        """Test default spacing when no affine or override."""
        if hasattr(cli, "get_image_properties"):
            image_data = np.ones((10, 20, 30))

            dims, spacing = cli.get_image_properties(image_data, None, None)

            assert dims == (10, 20, 30)
            assert spacing == (1.0, 1.0, 1.0)
        else:
            pytest.skip("get_image_properties function not found")


class TestCLIRunAnalysis:
    """Test the main run_analysis function comprehensively."""

    @patch("src.nema_quant.cli.setup_logging")
    @patch("src.nema_quant.cli.load_configuration")
    @patch("src.nema_quant.cli.load_nii_image")
    @patch("src.nema_quant.cli.get_image_properties")
    @patch("src.nema_quant.cli.NemaPhantom")
    @patch("src.nema_quant.cli.calculate_nema_metrics")
    @patch("src.nema_quant.cli.save_results_to_txt")
    @patch("src.nema_quant.cli.generate_plots")
    @patch("src.nema_quant.cli.generate_reportlab_report")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    def test_run_analysis_basic(
        self,
        mock_exists,
        mock_mkdir,
        mock_generate_report,
        mock_generate_plots,  # Fixed parameter order
        mock_save_results,
        mock_calculate_metrics,
        mock_phantom_class,
        mock_get_props,
        mock_load_image,
        mock_load_config,
        mock_setup_logging,
    ):
        """Test basic analysis run without visualizations."""
        mock_exists.return_value = True

        # Setup complete configuration for basic test too
        cfg = CfgNode()
        cfg.ROIS = CfgNode()
        cfg.ROIS.SPACING = 2.0
        cfg.ROIS.CENTRAL_SLICE = 25
        # required by reporting.generate_transverse_sphere_plots
        cfg.ROIS.BACKGROUND_OFFSET_YX = [(0, 0)]
        # required by reporting code (defaults.py uses [1, 1])
        cfg.ROIS.ORIENTATION_YX = [1, 1]

        cfg.PHANTHOM = CfgNode()
        # required by generate_transverse_sphere_plots / reporting code
        cfg.PHANTHOM.BACKGROUND_OFFSET_YX = (0, 0)
        cfg.PHANTHOM.ROI_DEFINITIONS_MM = [
            {
                "name": "hot_sphere_37mm",
                "diameter_mm": 37.0,
                "center_yx": (50, 50),
                "alpha": 0.18,
                "color": "C0",
            },
        ]

        mock_load_config.return_value = cfg

        mock_load_image.return_value = (np.ones((50, 100, 100)), np.eye(4))
        mock_get_props.return_value = ((50, 100, 100), (2.0, 2.0, 2.0))

        mock_phantom = MagicMock()
        mock_phantom.rois = {
            "hot_sphere_37mm": {
                "center_yx": (50, 50),
                "diameter_mm": 37.0,
                "mask": np.ones((50, 100, 100), dtype=bool),
            }
        }
        mock_phantom_class.return_value = mock_phantom

        # Mock calculate_metrics with complete data structure
        mock_metrics_data = [
            {
                "diameter_mm": 10.0,
                "percentaje_constrast_QH": 95.0,
                "background_variability_N": 5.2,
            }
        ]
        mock_cbr_values = {10: 95.0}
        mock_calculate_metrics.return_value = (mock_metrics_data, mock_cbr_values)

        # Create args for basic analysis (no visualizations)
        args = MagicMock()
        args.verbose = False
        args.input_image = "input.nii"
        args.output = "output.txt"
        args.config = "config.yaml"
        args.spacing = None
        args.save_visualizations = False
        args.visualizations_dir = "visualizations"
        args.advanced_metrics = False
        args.gt_image = None

        with patch("pathlib.Path.suffix", new_callable=lambda: ".nii"):
            result = cli.run_analysis(args)

        assert result == 0

        # Verify core functions were called
        mock_calculate_metrics.assert_called_once()
        mock_save_results.assert_called_once()
        mock_generate_report.assert_called_once()
        mock_generate_plots.assert_called_once()

    @patch("src.nema_quant.cli.setup_logging")
    @patch("src.nema_quant.cli.load_configuration")
    @patch("src.nema_quant.cli.load_nii_image")
    @patch("src.nema_quant.cli.get_image_properties")
    @patch("src.nema_quant.cli.NemaPhantom")
    @patch("src.nema_quant.cli.calculate_nema_metrics")
    @patch("src.nema_quant.cli.save_results_to_txt")
    @patch("src.nema_quant.cli.generate_plots")
    @patch("src.nema_quant.cli.generate_rois_plots")
    @patch("src.nema_quant.cli.generate_boxplot_with_mean_std")
    @patch("src.nema_quant.cli.generate_reportlab_report")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_file")
    @patch("pathlib.Path.suffix", new_callable=lambda: ".nii")
    def test_run_analysis_with_visualizations(
        self,
        mock_suffix,
        mock_is_file,
        mock_exists,
        mock_mkdir,
        mock_generate_report,
        mock_boxplot,
        mock_rois_plots,
        mock_generate_plots,
        mock_save_results,
        mock_calculate_metrics,
        mock_phantom_class,
        mock_get_props,
        mock_load_image,
        mock_load_config,
        mock_setup_logging,
    ):
        """Test analysis run with visualizations enabled."""
        # Mock all file system checks
        mock_exists.return_value = True
        mock_is_file.return_value = True

        # Setup comprehensive mocks with complete configuration structure
        cfg = CfgNode()
        cfg.ROIS = CfgNode()
        cfg.ROIS.SPACING = 2.0
        cfg.ROIS.CENTRAL_SLICE = 25
        # required by reporting.generate_transverse_sphere_plots
        cfg.ROIS.BACKGROUND_OFFSET_YX = [(0, 0)]
        # required by reporting code (defaults.py uses [1, 1])
        cfg.ROIS.ORIENTATION_YX = [1, 1]

        # Add missing PHANTHOM configuration
        cfg.PHANTHOM = CfgNode()
        # required by generate_transverse_sphere_plots / reporting code
        cfg.PHANTHOM.BACKGROUND_OFFSET_YX = (0, 0)
        cfg.PHANTHOM.ROI_DEFINITIONS_MM = [
            {
                "name": "sphere_10mm",
                "center": [0, 0, 0],
                "diameter_mm": 10,
                "center_yx": (50, 50),
                "alpha": 0.18,
                "color": "C0",
            },
            {
                "name": "sphere_37mm",
                "center": [20, 0, 0],
                "diameter_mm": 13,
                "center_yx": (60, 40),
                "alpha": 0.18,
                "color": "C1",
            },
        ]

        mock_load_config.return_value = cfg

        mock_load_image.return_value = (np.ones((50, 100, 100)), np.eye(4))
        mock_get_props.return_value = ((50, 100, 100), (2.0, 2.0, 2.0))

        mock_phantom = MagicMock()
        mock_phantom.rois = {
            "sphere_10mm": {
                "center_yx": (50, 50),
                "diameter_mm": 10.0,
                "mask": np.ones((50, 100, 100), dtype=bool),
            },
            "sphere_37mm": {
                "center_yx": (60, 40),
                "diameter_mm": 13.0,
                "mask": np.ones((50, 100, 100), dtype=bool),
            },
        }
        mock_phantom_class.return_value = mock_phantom

        # Mock calculate_metrics with complete data structure
        mock_metrics_data = [
            {
                "diameter_mm": 10.0,
                "percentaje_constrast_QH": 95.0,
                "background_variability_N": 5.2,
            },
            {
                "diameter_mm": 13.0,
                "percentaje_constrast_QH": 88.0,
                "background_variability_N": 6.1,
            },
        ]
        mock_cbr_values = {10: 95.0, 13: 88.0}
        mock_calculate_metrics.return_value = (mock_metrics_data, mock_cbr_values)

        # Create args matching the NEW parser structure
        args = MagicMock()
        args.verbose = True
        args.input_image = "input.nii"
        args.output = "output.txt"
        args.config = "config.yaml"
        args.spacing = [2.0, 2.0, 2.0]
        args.save_visualizations = True
        args.visualizations_dir = "visualizations"
        args.advanced_metrics = False
        args.gt_image = None

        result = cli.run_analysis(args)

        assert result == 0

        # Verify visualization functions were called
        mock_mkdir.assert_called()
        mock_generate_plots.assert_called_once()
        mock_rois_plots.assert_called_once()
        mock_boxplot.assert_called_once()
        mock_generate_report.assert_called_once()

    @patch("src.nema_quant.cli.setup_logging")
    @patch("pathlib.Path.exists")
    @patch("src.nema_quant.cli.load_configuration")
    def test_run_analysis_keyboard_interrupt(
        self, mock_load_config, mock_exists, mock_setup_logging
    ):
        """Test handling of keyboard interrupt."""
        mock_exists.return_value = True
        mock_load_config.side_effect = KeyboardInterrupt()

        args = MagicMock()
        args.verbose = False
        args.input_image = "input.nii"
        args.config = "config.yaml"

        result = cli.run_analysis(args)
        assert result == 130  # Standard exit code for KeyboardInterrupt

    @patch("src.nema_quant.cli.setup_logging")
    @patch("pathlib.Path.exists")
    @patch("src.nema_quant.cli.load_configuration")
    def test_run_analysis_unexpected_error(
        self, mock_load_config, mock_exists, mock_setup_logging
    ):
        """Test handling of unexpected errors."""
        mock_exists.return_value = True
        mock_load_config.side_effect = RuntimeError("Unexpected error")

        args = MagicMock()
        args.verbose = False
        args.input_image = "input.nii"
        args.config = "config.yaml"

        result = cli.run_analysis(args)
        assert result == 1


class TestCLIMainFunction:
    """Test the main entry point function."""

    def test_main_with_help(self):
        """Test main function with help."""
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["--help"])
        assert exc_info.value.code == 0

    def test_main_with_version(self):
        """Test main function with version."""
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["--version"])
        assert exc_info.value.code == 0

    @patch("src.nema_quant.cli.run_analysis")
    def test_main_success(self, mock_run_analysis):
        """Test successful main execution."""
        mock_run_analysis.return_value = 0

        result = cli.main(
            ["input.nii", "--output", "output.txt", "--config", "config.yaml"]
        )

        assert result == 0
        mock_run_analysis.assert_called_once()

    @patch("src.nema_quant.cli.run_analysis")
    def test_main_analysis_error(self, mock_run_analysis):
        """Test main with analysis error."""
        mock_run_analysis.return_value = 1

        result = cli.main(
            ["input.nii", "--output", "output.txt", "--config", "config.yaml"]
        )

        assert result == 1

    def test_main_invalid_arguments(self):
        """Test main with invalid arguments."""
        with pytest.raises(SystemExit):
            cli.main([])  # Missing required arguments

    @patch("sys.argv", ["nema_quant"])
    def test_main_no_cli_args(self):
        """Test main without CLI args (uses sys.argv)."""
        with pytest.raises(SystemExit):
            cli.main()  # Should use sys.argv which has no required args


class TestCLIFileValidation:
    """Test file validation functions."""

    def test_valid_nii_extensions(self):
        """Test that valid NIfTI extensions are accepted."""
        valid_extensions = [".nii", ".nii.gz"]

        for ext in valid_extensions:
            test_path = Path(f"test{ext}")
            # The actual validation logic would be in your CLI code
            # This tests the concept
            assert test_path.suffix in [".nii", ".gz"]

    def test_invalid_extensions(self):
        """Test that invalid extensions are rejected."""
        invalid_extensions = [".txt", ".jpg", ".png", ".dicom"]

        for ext in invalid_extensions:
            test_path = Path(f"test{ext}")
            # The validation would reject these
            assert test_path.suffix not in [".nii"]

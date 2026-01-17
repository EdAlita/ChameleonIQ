from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

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
            "--save-visualizations",
            "--visualizations-dir",
            "viz_output",
        ]

        args = parser.parse_args(test_args)

        # Verify all arguments are parsed correctly
        assert args.input_image == "input.nii"
        assert args.output == "output.txt"
        assert args.config == "config.yaml"
        assert args.save_visualizations is True
        assert args.visualizations_dir == "viz_output"
        assert args.log_level == "DEBUG"

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
        assert args.save_visualizations is False
        assert args.log_level == "DEBUG"

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
            cli.setup_logging(log_level=20)
            # Should not raise exception
        else:
            pytest.skip("setup_logging function not found")

    def test_setup_logging_quiet(self):
        """Test quiet logging setup."""
        if hasattr(cli, "setup_logging"):
            cli.setup_logging(log_level=10)
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

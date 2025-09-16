from unittest.mock import patch

import numpy as np
import pytest

# Import your CLI module - adjust import path as needed
from src.nema_quant.cli import main


@patch("sys.argv", ["nema_quant", "--help"])
def test_cli_help():
    """Test CLI help functionality."""
    # Test that help can be displayed without errors
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    # Help should exit with code 0
    if exc_info.value.code != 0:
        pytest.fail(f"Help should exit with code 0, got {exc_info.value.code}")


@patch("src.nema_quant.analysis.calculate_nema_metrics")
@patch("src.nema_quant.io.load_nii_image")
def test_cli_basic_analysis(mock_load_image, mock_calculate_metrics):
    """Test basic CLI analysis command."""
    # Mock the dependencies
    mock_load_image.return_value = (np.ones((50, 100, 100)), np.eye(4))
    mock_calculate_metrics.return_value = ([], {})

    # Test CLI execution - adjust based on your CLI structure
    # This is a template - you'll need to adapt it to your actual CLI
    pass

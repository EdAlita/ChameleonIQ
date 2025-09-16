from unittest.mock import patch

import pytest


def test_main_module_import():
    """Test that main module can be imported."""
    try:
        import src.nema_quant.__main__  # noqa: F401
    except ImportError as e:
        pytest.fail(f"Could not import __main__ module: {e}")


@patch("sys.argv", ["nema_quant", "--version"])
def test_main_version():
    """Test version display."""
    # Test version functionality if available

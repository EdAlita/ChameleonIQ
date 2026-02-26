"""Pytest configuration for nema-analysis-tool tests."""

import matplotlib

# Use non-interactive backend for tests to avoid Qt/display issues
matplotlib.use("Agg")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def setup_matplotlib():
    """Ensure matplotlib uses non-interactive backend for all tests."""
    matplotlib.use("Agg", force=True)
    yield

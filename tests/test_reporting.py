import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.config.defaults import get_cfg_defaults
from src.nema_quant import reporting


@pytest.fixture
def sample_results():
    """Sample results for testing."""
    return [
        {
            "diameter_mm": 10.0,
            "percentaje_constrast_QH": 85.0,
            "background_variability_N": 5.2,
            "avg_hot_counts_CH": 15000.0,
            "avg_bkg_counts_CB": 2000.0,
            "bkg_std_dev_SD": 104.0,
        },
        {
            "diameter_mm": 13.0,
            "percentaje_constrast_QH": 78.5,
            "background_variability_N": 5.8,
            "avg_hot_counts_CH": 12000.0,
            "avg_bkg_counts_CB": 1950.0,
            "bkg_std_dev_SD": 113.0,
        },
    ]


@pytest.fixture
def sample_lung_results():
    """Sample lung results for testing."""
    return {10: 1800.0, 11: 1750.0, 12: 1720.0}


@pytest.fixture
def mock_cfg():
    """Create a mock configuration with all required fields."""
    cfg = get_cfg_defaults()
    cfg.ACTIVITY.HOT = 8000.0
    cfg.ACTIVITY.BACKGROUND = 2000.0
    cfg.ACTIVITY.RATIO = 4.0
    cfg.ACTIVITY.UNITS = "mCi/mL"
    cfg.ACTIVITY.ACTIVITY_TOTAL = "29.24 MBq"
    return cfg


def test_save_results_to_txt(sample_results, mock_cfg):
    """Test saving results to text file."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".txt")
    try:
        import os

        os.close(tmp_fd)

        reporting.save_results_to_txt(
            sample_results,
            Path(tmp_path),
            mock_cfg,
            input_image_path=Path("test_input.nii"),
            voxel_spacing=(2.0, 2.0, 2.0),
        )

        # Verify file was created and has content
        with open(tmp_path, "r") as f:
            content = f.read()
            assert len(content) > 0

    finally:
        try:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
        except OSError:
            pass


@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.close")
def test_generate_plots(mock_close, mock_savefig, sample_results, mock_cfg):
    """Test plot generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Need to create csv directory
        csv_dir = Path(tmpdir).parent / "csv"
        csv_dir.mkdir(exist_ok=True, parents=True)

        reporting.generate_plots(sample_results, Path(tmpdir), mock_cfg)
        # Should call savefig
        assert mock_savefig.called


@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.close")
def test_generate_rois_plots(mock_close, mock_savefig, mock_cfg):
    """Test ROI plots generation."""
    test_image = np.random.rand(100, 250, 250)

    with tempfile.TemporaryDirectory() as tmpdir:
        reporting.generate_rois_plots(test_image, Path(tmpdir), mock_cfg)
        # Function should complete without error


@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.close")
def test_generate_boxplot_with_mean_std(mock_close, mock_savefig, mock_cfg):
    """Test boxplot generation with mean and std."""
    test_data = {1: 150.0, 2: 160.0, 3: 155.0, 4: 158.0, 5: 162.0}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Need to create csv directory
        csv_dir = Path(tmpdir).parent / "csv"
        csv_dir.mkdir(exist_ok=True, parents=True)

        reporting.generate_boxplot_with_mean_std(test_data, Path(tmpdir), mock_cfg)
        # Function should complete without error


@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.close")
def test_generate_transverse_sphere_plots(mock_close, mock_savefig, mock_cfg):
    """Test transverse sphere plot generation."""
    test_image = np.random.rand(100, 250, 250)

    with tempfile.TemporaryDirectory() as tmpdir:
        reporting.generate_transverse_sphere_plots(test_image, Path(tmpdir), mock_cfg)
        # Function should complete without error


@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.close")
def test_generate_coronal_sphere_plots(mock_close, mock_savefig, mock_cfg):
    """Test coronal sphere plot generation."""
    test_image = np.random.rand(100, 250, 250)

    with tempfile.TemporaryDirectory() as tmpdir:
        reporting.generate_coronal_sphere_plots(test_image, Path(tmpdir), mock_cfg)
        # Function should complete without error


def test_save_results_to_txt_nu4(mock_cfg):
    """Test saving NU4 results to text file."""
    crc_results = {
        "rod_1": {"recovery_coeff": 0.9, "cError": 0.1, "percentage_STD_rc": 2.0}
    }
    spillover_results = {"lung": 0.05}
    uniformity_results = {"uniformity": 0.02}

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".txt")
    try:
        import os

        os.close(tmp_fd)

        reporting.save_results_to_txt_nu4(
            crc_results=crc_results,
            spillover_results=spillover_results,
            uniformity_results=uniformity_results,
            output_path=Path(tmp_path),
            cfg=mock_cfg,
            input_image_path=Path("test_input.nii"),
            voxel_spacing=(2.0, 2.0, 2.0),
        )

        with open(tmp_path, "r") as f:
            content = f.read()
            assert "NU 4-2008" in content

    finally:
        try:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
        except OSError:
            pass


def test_generate_reportlab_report(tmp_path: Path, mock_cfg):
    """Test reportlab report generation for NU 2-2018."""
    results = [
        {
            "diameter_mm": 10.0,
            "percentaje_constrast_QH": 85.0,
            "background_variability_N": 5.2,
            "avg_hot_counts_CH": 15000.0,
            "avg_bkg_counts_CB": 2000.0,
            "bkg_std_dev_SD": 104.0,
        }
    ]
    lung_results = {"10": 1800.0}

    output_path = tmp_path / "report.pdf"

    reporting.generate_reportlab_report(
        results=results,
        output_path=output_path,
        cfg=mock_cfg,
        input_image_path=Path("test_input.nii"),
        voxel_spacing=(2.0, 2.0, 2.0),
        lung_results=lung_results,
    )

    assert output_path.exists()


def test_generate_reportlab_report_nu4(tmp_path: Path, mock_cfg):
    """Test reportlab report generation for NU 4-2008."""
    crc_results = {
        "rod_1": {"recovery_coeff": 0.9, "cError": 0.1, "percentage_STD_rc": 2.0}
    }
    spillover_results = {"lung": {"SOR": 0.05, "SOR_error": 0.01, "%STD": 1.0}}
    uniformity_results = {"mean": 1.0, "maximum": 1.1, "minimum": 0.9, "%STD": 2.0}

    output_path = tmp_path / "report_nu4.pdf"

    reporting.generate_reportlab_report_nu4(
        crc_results=crc_results,
        spillover_results=spillover_results,
        uniformity_results=uniformity_results,
        output_path=output_path,
        cfg=mock_cfg,
        input_image_path=Path("test_input.nii"),
        voxel_spacing=(2.0, 2.0, 2.0),
    )

    assert output_path.exists()

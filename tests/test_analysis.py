import pytest
import numpy as np
import numpy.typing as npt
from typing import Any
from yacs.config import CfgNode
from unittest.mock import MagicMock
from src.nema_quant import analysis


@pytest.fixture
def mock_cfg() -> CfgNode:
    """Create a fake YACS config for testing"""
    cfg = CfgNode()
    cfg.ACTIVITY = CfgNode()
    cfg.ACTIVITY.HOT = 8.0
    cfg.ACTIVITY.BACKGROUND = 1.0

    cfg.ROIS = CfgNode()
    cfg.ROIS.CENTRAL_SLICE = 10
    cfg.ROIS.BACKGROUND_OFFSET_YX = [(-20, -20), (20, 20)]

    return cfg


@pytest.fixture
def mock_phantom() -> MagicMock:
    """Create a mock NemaPhantom object for testing."""
    phantom = MagicMock()

    phantom.rois = {
        'hot_sphere_10mm': {
            'name': 'hot_sphere_10mm',
            'diameter': 10.0,
            'center_vox': (50, 50),
            'radius_vox': 5.0
        }
    }

    def get_roi_side_effect(name):
        roi_data = phantom.rois.get(name)
        if roi_data:
            return {
                'diameter': roi_data['diameter'],
                'center_vox': roi_data['center_vox'],
                'radius_vox': roi_data['radius_vox']
            }
        return None

    phantom.get_roi.side_effect = get_roi_side_effect

    phantom._mm_to_voxels.return_value = 5.0

    return phantom


@pytest.fixture
def test_image_data() -> npt.NDArray[Any]:
    """Creates a 3D test image with predictable values."""
    image = np.full((20, 100, 100), 100.0, dtype=np.float32)

    center_y, center_x = 50, 50
    radius = 5
    y, x = np.ogrid[:100, :100]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

    image[10, mask] = 800.0

    for offset_y, offset_x in [(-20, -20), (20, 20)]:
        bg_y, bg_x = center_y + offset_y, center_x + offset_x
        bg_mask = (x - bg_x) ** 2 + (y - bg_y) ** 2 <= radius ** 2
        for z in range(8, 13):
            image[z, bg_mask] = 100.0

    return image


def test_extract_circular_mask_2d():
    """Test that the creation of the 2D mask is correct."""
    mask = analysis.extract_circular_mask_2d(
        slice_dims=(10, 10),
        roi_center_vox=(5.0, 5.0),
        roi_radius_vox=2.0
    )
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == (10, 10)
    assert mask[5, 5] is True
    assert 10 <= np.sum(mask) <= 15


def test_calculate_nema_metrics(mock_cfg, mock_phantom, test_image_data):
    """Tests the calculation of NEMA metrics with controlled data."""
    results, lung_results = analysis.calculate_nema_metrics(
        test_image_data, mock_phantom, mock_cfg
    )

    assert isinstance(results, list)
    assert len(results) >= 1

    result = results[0]

    expected_keys = [
        'diameter_mm', 'percentaje_constrast_QH', 'background_variability_N',
        'avg_hot_counts_CH', 'avg_bkg_counts_CB', 'bkg_std_dev_SD'
    ]
    for key in expected_keys:
        assert key in result

    assert result['avg_hot_counts_CH'] > result['avg_bkg_counts_CB']
    assert result['diameter_mm'] == 10.0
    assert result['percentaje_constrast_QH'] > 0
    assert result['background_variability_N'] >= 0


def test_calculate_nema_metrics_bad_activity_ratio(mock_cfg, mock_phantom, test_image_data):
    """Tests that the function fails if the activity ratio is not valid."""
    mock_cfg.ACTIVITY.HOT = 1.0
    mock_cfg.ACTIVITY.BACKGROUND = 1.0

    with pytest.raises(ValueError, match="Activity ratio"):
        analysis.calculate_nema_metrics(
            test_image_data, mock_phantom, mock_cfg
        )


def test_calculate_nema_metrics_zero_background_activity(mock_cfg, mock_phantom, test_image_data):
    """Tests that the function fails if background activity is zero or negative."""
    mock_cfg.ACTIVITY.BACKGROUND = 0.0

    with pytest.raises(ValueError, match="background activity must be positive"):
        analysis.calculate_nema_metrics(
            test_image_data, mock_phantom, mock_cfg
        )


def test_background_stats_calculation(mock_cfg, mock_phantom, test_image_data):
    """Test the background statistics calculation separately."""
    results, lung_results = analysis.calculate_nema_metrics(
        test_image_data, mock_phantom, mock_cfg
    )

    result = results[0]

    assert 90 <= result['avg_bkg_counts_CB'] <= 110
    assert result['bkg_std_dev_SD'] >= 0


def test_hot_sphere_counts_calculation(mock_cfg, mock_phantom, test_image_data):
    """Test the hot sphere counts calculation."""
    results, lung_results = analysis.calculate_nema_metrics(
        test_image_data, mock_phantom, mock_cfg
    )

    result = results[0]

    assert 700 <= result['avg_hot_counts_CH'] <= 900

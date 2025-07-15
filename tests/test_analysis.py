import pytest
import numpy as np
from yacs.config import CfgNode
from unittest.mock import MagicMock

from nema_quant import analysis


@pytest.fixture
def mock_cfg() -> CfgNode:
    """Create a fake YACS config for testing"""
    cfg = CfgNode()
    cfg.ACTIVITY = CfgNode()
    cfg.ACTIVITY.HOT = 8.0
    cfg.ACTIVITY.BACKGROUND = 1.0

    cfg.ROIS = CfgNode()
    cfg.ROIS.CENTRAL_SLICE = 10
    cfg.ROIS.BACKGROUND_CENTER_YX = [(30, 30)]

    return cfg


@pytest.fixture
def mock_phantom() -> MagicMock:
    """Crea un objeto NemaPhantom falso (mock) para las pruebas."""
    phantom = MagicMock()

    phantom.rois = {
        'sphere_10mm': {
            'name': 'sphere_10mm',
            'diameter': 10.0,
            'center_vox': (50, 50, 10),
            'radius_vox': 5.0
        },
        'bkg_10mm': {
            'name': 'bkg_10mm',
            'diameter': 10.0,
            'radius_vox': 5.0
        }
    }

    def get_roi_side_effect(name):
        return phantom.rois.get(name)

    phantom.get_roi.side_effect = get_roi_side_effect

    phantom._mm_to_voxels.return_value = 5
    return phantom


@pytest.fixture
def test_image_data() -> np.ndarray:
    """Creates a 3D test image with predictable values."""
    image = np.full((20, 100, 100), 100, dtype=np.float32)

    center_y, center_x = 50, 50
    radius = 5
    y, x = np.ogrid[:100, :100]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    image[10, mask] = 800

    center_y_bkg, center_x_bkg = 30, 30
    mask_bkg = (x - center_y_bkg) ** 2 + (y - center_x_bkg) ** 2 <= radius ** 2
    image[10, mask_bkg] = 100

    return image


def test_extract_circular_mask_2d():
    """Test that the creation of the 2D mask is correct."""
    mask = analysis.extract_circular_mask_2d(
        slice_dims=(10, 10),
        roi_center_vox=(5, 5),
        roi_radius_vox=2
    )
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == (10, 10)
    assert np.sum(mask) == 13


def test_calculate_nema_metrics(mock_cfg, mock_phantom, test_image_data):
    """Tests the calculation of NEMA
    metrics with controlled data."""
    results = analysis.calculate_nema_metrics(test_image_data,
                                              mock_phantom,
                                              mock_cfg)

    assert isinstance(results, list)
    assert len(results) == 1

    result = results[0]

    assert result['avg_hot_counts_CH'] == pytest.approx(800.0)
    assert result['avg_bkg_counts_CB'] == pytest.approx(100.0)
    assert result['bkg_std_dev_SD'] == pytest.approx(0.0)
    assert result['percentaje_constrast_QH'] == pytest.approx(100.0)
    assert result['background_variability_N'] == pytest.approx(0.0)


def test_calculate_nema_metrics_bad_activity_ratio(mock_cfg,
                                                   mock_phantom,
                                                   test_image_data):
    """Tests that the function fails if the activity ratio is not valid."""
    mock_cfg.ACTIVITY.HOT = 1.0

    with pytest.raises(ValueError, match="Activity ratio"):
        analysis.calculate_nema_metrics(test_image_data,
                                        mock_phantom,
                                        mock_cfg)

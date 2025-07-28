import pytest
import numpy as np
import numpy.typing as npt
from typing import Any
from yacs.config import CfgNode
from unittest.mock import MagicMock, patch
from src.nema_quant import analysis


@pytest.fixture
def mock_cfg() -> CfgNode:
    """Create a fake YACS config for testing"""
    cfg = CfgNode()
    cfg.ACTIVITY = CfgNode()
    cfg.ACTIVITY.HOT = 8.0
    cfg.ACTIVITY.BACKGROUND = 1.0

    cfg.ROIS = CfgNode()
    cfg.ROIS.CENTRAL_SLICE = 10  # Central slice for 40-slice image
    cfg.ROIS.BACKGROUND_OFFSET_YX = [(-10, -10), (10, 10)]  # Smaller offsets
    cfg.ROIS.SPACING = 2.0644  # Add required SPACING

    return cfg


@pytest.fixture
def mock_phantom() -> MagicMock:
    """Create a mock NemaPhantom object for testing."""
    phantom = MagicMock()

    # Mock the rois dictionary with all expected spheres
    phantom.rois = {
        'hot_sphere_10mm': {
            'name': 'hot_sphere_10mm',
            'diameter': 10.0,
            'center_vox': (50, 50),  # (y, x) coordinates for 2D
            'radius_vox': 2.42  # 10mm / 2 / 2.0644
        },
        'hot_sphere_37mm': {
            'name': 'hot_sphere_37mm',
            'diameter': 37.0,
            'center_vox': (50, 50),
            'radius_vox': 8.96  # 37mm / 2 / 2.0644
        }
    }

    # Mock the get_roi method
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
    phantom._mm_to_voxels.return_value = 4.84  # 10mm / 2.0644
    phantom.list_hot_spheres.return_value = list(phantom.rois.keys())

    return phantom


@pytest.fixture
def test_image_data() -> npt.NDArray[Any]:
    """Creates a 3D test image with predictable values."""
    # Create larger image to avoid index errors
    image = np.full((40, 100, 100), 100.0, dtype=np.float32)

    center_y, center_x = 50, 50
    radius = 5
    y, x = np.ogrid[:100, :100]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

    # Set hot sphere values on central slice and nearby slices
    for z in range(8, 13):  # Around slice 10
        image[z, mask] = 800.0

    # Add background regions with known values
    for offset_y, offset_x in [(-10, -10), (10, 10)]:
        bg_y, bg_x = center_y + offset_y, center_x + offset_x
        if 0 <= bg_y < 100 and 0 <= bg_x < 100:
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
    # Fix the boolean comparison
    assert mask[5, 5] == True
    assert 10 <= np.sum(mask) <= 15


@patch('src.nema_quant.analysis.find_phantom_center')
@patch('src.nema_quant.analysis.extract_canny_mask')
def test_calculate_nema_metrics(mock_extract_canny, mock_find_center, mock_cfg, mock_phantom, test_image_data):
    """Tests the calculation of NEMA metrics with controlled data."""
    # Mock the functions that would cause issues
    mock_find_center.return_value = (20, 50, 50)  # (z, y, x)
    mock_extract_canny.return_value = np.array([[20, 50, 50], [21, 50, 50]])  # Mock lung centers
    
    results, lung_results = analysis.calculate_nema_metrics(
        test_image_data, mock_phantom, mock_cfg
    )

    assert isinstance(results, list)
    assert len(results) >= 1

    # Check the first result
    result = results[0]

    # Verify all expected keys are present
    expected_keys = [
        'diameter_mm', 'percentaje_constrast_QH', 'background_variability_N',
        'avg_hot_counts_CH', 'avg_bkg_counts_CB', 'bkg_std_dev_SD'
    ]
    for key in expected_keys:
        assert key in result

    # Check that values are reasonable
    assert result['avg_hot_counts_CH'] > result['avg_bkg_counts_CB']
    assert result['diameter_mm'] in [10.0, 37.0]  # Should be one of our test spheres
    assert result['percentaje_constrast_QH'] > 0
    assert result['background_variability_N'] >= 0

    # Check lung results
    assert isinstance(lung_results, dict)
    assert len(lung_results) > 0


def test_calculate_nema_metrics_bad_activity_ratio(mock_cfg, mock_phantom, test_image_data):
    """Tests that the function fails if the activity ratio is not valid."""
    # Set invalid activity ratio (ratio <= 1)
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


@patch('src.nema_quant.analysis.find_phantom_center')
@patch('src.nema_quant.analysis.extract_canny_mask')
def test_background_stats_calculation(mock_extract_canny, mock_find_center, mock_cfg, mock_phantom, test_image_data):
    """Test the background statistics calculation separately."""
    # Mock the functions that would cause issues
    mock_find_center.return_value = (20, 50, 50)
    mock_extract_canny.return_value = np.array([[20, 50, 50]])
    
    results, lung_results = analysis.calculate_nema_metrics(
        test_image_data, mock_phantom, mock_cfg
    )

    result = results[0]

    # Background should be around 100 (our test data value)
    assert 90 <= result['avg_bkg_counts_CB'] <= 110
    # Standard deviation should be low for our uniform background
    assert result['bkg_std_dev_SD'] >= 0


@patch('src.nema_quant.analysis.find_phantom_center')
@patch('src.nema_quant.analysis.extract_canny_mask')
def test_hot_sphere_counts_calculation(mock_extract_canny, mock_find_center, mock_cfg, mock_phantom, test_image_data):
    """Test the hot sphere counts calculation."""
    # Mock the functions that would cause issues
    mock_find_center.return_value = (20, 50, 50)
    mock_extract_canny.return_value = np.array([[20, 50, 50]])
    
    results, lung_results = analysis.calculate_nema_metrics(
        test_image_data, mock_phantom, mock_cfg
    )

    result = results[0]

    # Hot sphere should be around 800 (our test data value)
    assert 700 <= result['avg_hot_counts_CH'] <= 900


def test_calculate_background_stats():
    """Test the internal background stats calculation function."""
    # Create a simple test image
    image = np.full((20, 50, 50), 100.0, dtype=np.float32)
    
    # Create a simple mock phantom
    phantom = MagicMock()
    phantom.rois = {
        'hot_sphere_10mm': {
            'name': 'hot_sphere_10mm',
            'diameter': 10.0,
            'center_vox': (25, 25),
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
    
    # Test the background stats calculation
    slices_indices = [8, 9, 10, 11, 12]
    centers_offset = [(-5, -5), (5, 5)]
    
    stats = analysis._calculate_background_stats(
        image, phantom, slices_indices, centers_offset
    )
    
    assert isinstance(stats, dict)
    assert 10 in stats  # Should have stats for 10mm sphere
    assert 'C_B' in stats[10]
    assert 'SD_B' in stats[10]
    assert stats[10]['C_B'] > 0


def test_calculate_hot_sphere_counts():
    """Test the internal hot sphere counts calculation function."""
    # Create a test image with a hot spot
    image = np.full((20, 50, 50), 100.0, dtype=np.float32)
    
    # Add a hot sphere at the center of slice 10
    center_y, center_x = 25, 25
    radius = 3
    y, x = np.ogrid[:50, :50]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    image[10, mask] = 800.0
    
    # Create a mock phantom
    phantom = MagicMock()
    phantom.rois = {
        'hot_sphere_10mm': {
            'name': 'hot_sphere_10mm',
            'diameter': 10.0,
            'center_vox': (25, 25),
            'radius_vox': 3.0
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
    
    # Test the hot sphere counts calculation
    counts = analysis._calculate_hot_sphere_counts(image, phantom, 10)
    
    assert isinstance(counts, dict)
    assert 'hot_sphere_10mm' in counts
    assert counts['hot_sphere_10mm'] > 700  # Should be close to 800


def test_calculate_lung_insert_counts():
    """Test the lung insert counts calculation."""
    # Create test image
    image = np.full((20, 50, 50), 100.0, dtype=np.float32)
    
    # Create lung centers array
    lung_centers = np.array([[10, 25, 25], [11, 25, 25]])
    
    # Test the lung insert calculation
    CB_37 = 100.0  # Background count
    voxel_size = 2.0644
    
    lung_counts = analysis._calculate_lung_insert_counts(
        image, lung_centers, CB_37, voxel_size
    )
    
    assert isinstance(lung_counts, dict)
    assert len(lung_counts) == 2  # Should have 2 slices
    assert 10 in lung_counts
    assert 11 in lung_counts
    
    # Values should be numeric types
    for count in lung_counts.values():
        # Accept both Python float and NumPy float types (float32, float64, etc.)
        assert isinstance(count, (float, np.float32, np.float64))
        assert float(count) > 0  # Convert to Python float for comparison

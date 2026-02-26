from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest
from yacs.config import CfgNode

from src.config.defaults import get_cfg_defaults
from src.nema_quant import analysis


@pytest.fixture
def mock_cfg() -> CfgNode:
    """Create a fake YACS config for testing"""
    cfg = CfgNode()
    cfg.ACTIVITY = CfgNode()
    cfg.ACTIVITY.HOT = 8.0
    cfg.ACTIVITY.BACKGROUND = 1.0
    cfg.ACTIVITY.RATIO = cfg.ACTIVITY.HOT / cfg.ACTIVITY.BACKGROUND
    cfg.ACTIVITY.UNITS = "mCi/mL"

    cfg.ROIS = CfgNode()
    cfg.ROIS.CENTRAL_SLICE = 10  # Central slice for 40-slice image
    cfg.ROIS.BACKGROUND_OFFSET_YX = [(-10, -10), (10, 10)]  # Smaller offsets
    cfg.ROIS.ORIENTATION_YX = [1, 1]  # Add required ORIENTATION_YX
    cfg.ROIS.SPACING = 2.0644  # Add required SPACING

    cfg.STATISTICS = CfgNode()
    cfg.STATISTICS.MODE = "empirical"
    cfg.STATISTICS.ESTIMATE_COVARIANCE = False
    cfg.STATISTICS.SD_VARIANCE_MODEL = "gaussian"
    cfg.STATISTICS.EPSILON = 1e-12
    cfg.STATISTICS.FAST_ROI_SHIFT = True

    return cfg


@pytest.fixture
def mock_phantom() -> MagicMock:
    """Create a mock NemaPhantom object for testing."""
    phantom = MagicMock()

    # Use a real dict for rois, not a MagicMock
    rois_dict = {
        "hot_sphere_10mm": {
            "name": "hot_sphere_10mm",
            "diameter": 10.0,
            "center_vox": (50, 50),  # (y, x) coordinates for 2D
            "radius_vox": 2.42,  # 10mm / 2 / 2.0644
        },
        "hot_sphere_37mm": {
            "name": "hot_sphere_37mm",
            "diameter": 37.0,
            "center_vox": (50, 50),
            "radius_vox": 8.96,  # 37mm / 2 / 2.0644
        },
    }
    phantom.rois = rois_dict

    # Mock the get_roi method
    def get_roi_side_effect(name):
        roi_data = rois_dict.get(name)
        if roi_data:
            return {
                "diameter": roi_data["diameter"],
                "center_vox": roi_data["center_vox"],
                "radius_vox": roi_data["radius_vox"],
            }
        return None

    phantom.get_roi.side_effect = get_roi_side_effect
    phantom._mm_to_voxels.return_value = 4.84  # 10mm / 2.0644

    return phantom


@pytest.fixture
def test_image_data() -> npt.NDArray[Any]:
    """Creates a 3D test image with predictable values."""
    # Create larger image to avoid index errors
    image = np.full((100, 100, 100), 100.0, dtype=np.float32)

    center_y, center_x = 50, 50
    radius = 5
    y, x = np.ogrid[:100, :100]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2

    # Set hot sphere values on central slice and nearby slices
    for z in range(8, 13):  # Around slice 10
        image[z, mask] = 800.0

    # Add background regions with known values
    for offset_y, offset_x in [(-10, -10), (10, 10)]:
        bg_y, bg_x = center_y + offset_y, center_x + offset_x
        if 0 <= bg_y < 100 and 0 <= bg_x < 100:
            bg_mask = (x - bg_x) ** 2 + (y - bg_y) ** 2 <= radius**2
            for z in range(8, 13):
                image[z, bg_mask] = 100.0

    return image


def test_extract_circular_mask_2d():
    """Test that the creation of the 2D mask is correct."""
    mask = analysis.extract_circular_mask_2d(
        slice_dims=(10, 10), roi_center_vox=(5.0, 5.0), roi_radius_vox=2.0
    )

    if not isinstance(mask, np.ndarray):
        pytest.fail(f"Expected mask to be np.ndarray, got {type(mask)}")

    if mask.dtype != bool:
        pytest.fail(f"Expected mask dtype to be bool, got {mask.dtype}")

    if mask.shape != (10, 10):
        pytest.fail(f"Expected mask shape to be (10, 10), got {mask.shape}")

    if not mask[5, 5]:
        pytest.fail("Expected center pixel (5,5) to be True")

    if not (10 <= np.sum(mask) <= 15):
        pytest.fail(f"Expected mask sum between 10-15, got {np.sum(mask)}")


@patch("src.nema_quant.analysis.find_phantom_center")
@patch("src.nema_quant.analysis.extract_canny_mask")
def test_calculate_nema_metrics(
    mock_extract_canny, mock_find_center, mock_cfg, mock_phantom, test_image_data
):
    """Tests the calculation of NEMA metrics with controlled data."""
    # Mock the functions that would cause issues
    mock_find_center.return_value = (20, 50, 50)  # (z, y, x)
    mock_extract_canny.return_value = np.array(
        [[20, 50, 50], [21, 50, 50]]
    )  # Mock lung centers

    results, lung_results = analysis.calculate_nema_metrics(
        test_image_data, mock_phantom, mock_cfg
    )

    if not isinstance(results, list):
        pytest.fail(f"Expected results to be list, got {type(results)}")

    if len(results) < 1:
        pytest.fail(f"Expected at least 1 result, got {len(results)}")

    # Check the first result
    result = results[0]

    # Verify all expected keys are present
    expected_keys = [
        "diameter_mm",
        "percentaje_constrast_QH",
        "background_variability_N",
        "avg_hot_counts_CH",
        "avg_bkg_counts_CB",
        "bkg_std_dev_SD",
    ]
    for key in expected_keys:
        if key not in result:
            pytest.fail(f"Expected key '{key}' not found in result")

    # Check that values are reasonable
    if not (result["avg_hot_counts_CH"] > result["avg_bkg_counts_CB"]):
        pytest.fail("Expected hot counts to be greater than background counts")

    if result["diameter_mm"] not in [10.0, 37.0]:
        pytest.fail(
            f"Expected diameter to be 10.0 or 37.0, got {result['diameter_mm']}"
        )

    if not (result["percentaje_constrast_QH"] > 0):
        pytest.fail(
            f"Expected positive contrast, got {result['percentaje_constrast_QH']}"
        )

    if not (result["background_variability_N"] >= 0):
        pytest.fail(
            f"Expected non-negative variability, got {result['background_variability_N']}"
        )

    # Check lung results
    if not isinstance(lung_results, dict):
        pytest.fail(f"Expected lung_results to be dict, got {type(lung_results)}")

    if len(lung_results) == 0:
        pytest.fail("Expected lung_results to have at least one entry")


def test_calculate_nema_metrics_bad_activity_ratio(
    mock_cfg, mock_phantom, test_image_data
):
    """Tests that the function fails if the activity ratio is not valid."""
    # Set invalid activity ratio (ratio <= 1)
    mock_cfg.ACTIVITY.HOT = 1.0
    mock_cfg.ACTIVITY.BACKGROUND = 1.0
    mock_cfg.ACTIVITY.RATIO = mock_cfg.ACTIVITY.HOT / mock_cfg.ACTIVITY.BACKGROUND

    with pytest.raises(ValueError, match="Activity ratio"):
        analysis.calculate_nema_metrics(test_image_data, mock_phantom, mock_cfg)


@patch("src.nema_quant.analysis.find_phantom_center")
@patch("src.nema_quant.analysis.extract_canny_mask")
def test_background_stats_calculation(
    mock_extract_canny, mock_find_center, mock_cfg, mock_phantom, test_image_data
):
    """Test the background statistics calculation separately."""
    # Mock the functions that would cause issues
    mock_find_center.return_value = (20, 50, 50)
    mock_extract_canny.return_value = np.array([[20, 50, 50]])

    results, lung_results = analysis.calculate_nema_metrics(
        test_image_data, mock_phantom, mock_cfg
    )

    result = results[0]

    # Background should be around 100 (our test data value)
    if not (90 <= result["avg_bkg_counts_CB"] <= 110):
        pytest.fail(
            f"Expected background counts 90-110, got {result['avg_bkg_counts_CB']}"
        )

    # Standard deviation should be low for our uniform background
    if not (result["bkg_std_dev_SD"] >= 0):
        pytest.fail(f"Expected non-negative std dev, got {result['bkg_std_dev_SD']}")


@patch("src.nema_quant.analysis.find_phantom_center")
@patch("src.nema_quant.analysis.extract_canny_mask")
def test_hot_sphere_counts_calculation(
    mock_extract_canny, mock_find_center, mock_cfg, mock_phantom, test_image_data
):
    """Test the hot sphere counts calculation."""
    # Mock the functions that would cause issues
    mock_find_center.return_value = (20, 50, 50)
    mock_extract_canny.return_value = np.array([[20, 50, 50]])

    results, lung_results = analysis.calculate_nema_metrics(
        test_image_data, mock_phantom, mock_cfg
    )

    result = results[0]

    # Hot sphere should be around 800 (our test data value)
    if not (700 <= result["avg_hot_counts_CH"] <= 900):
        pytest.fail(f"Expected hot counts 700-900, got {result['avg_hot_counts_CH']}")


def test_calculate_background_stats():
    """Test the internal background stats calculation function."""
    # Create a simple test image
    image = np.full((20, 50, 50), 100.0, dtype=np.float32)

    # Create a simple mock phantom
    phantom = MagicMock()
    phantom.rois = {
        "hot_sphere_10mm": {
            "name": "hot_sphere_10mm",
            "diameter": 10.0,
            "center_vox": (25, 25),
            "radius_vox": 5.0,
        }
    }

    def get_roi_side_effect(name):
        roi_data = phantom.rois.get(name)
        if roi_data:
            return {
                "diameter": roi_data["diameter"],
                "center_vox": roi_data["center_vox"],
                "radius_vox": roi_data["radius_vox"],
            }
        return None

    phantom.get_roi.side_effect = get_roi_side_effect

    # Test the background stats calculation
    slices_indices = [8, 9, 10, 11, 12]
    centers_offset = [(-5, -5), (5, 5)]

    stats = analysis._calculate_background_stats(
        image, phantom, slices_indices, centers_offset
    )

    if not isinstance(stats, dict):
        pytest.fail(f"Expected stats to be dict, got {type(stats)}")

    if 10 not in stats:
        pytest.fail("Expected stats to contain key '10'")

    if "C_B" not in stats[10]:
        pytest.fail("Expected stats[10] to contain 'C_B' key")

    if "SD_B" not in stats[10]:
        pytest.fail("Expected stats[10] to contain 'SD_B' key")

    if not (stats[10]["C_B"] > 0):
        pytest.fail(f"Expected positive C_B, got {stats[10]['C_B']}")


def test_calculate_hot_sphere_counts():
    """Test the internal hot sphere counts calculation function."""
    # Create a test image with a hot spot
    image = np.full((20, 50, 50), 100.0, dtype=np.float32)

    # Add a hot sphere at the center of slice 10
    center_y, center_x = 25, 25
    radius = 3
    y, x = np.ogrid[:50, :50]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
    image[10, mask] = 800.0

    # Create a mock phantom
    phantom = MagicMock()
    phantom.rois = {
        "hot_sphere_10mm": {
            "name": "hot_sphere_10mm",
            "diameter": 10.0,
            "center_vox": (25, 25),
            "radius_vox": 3.0,
        }
    }

    def get_roi_side_effect(name):
        roi_data = phantom.rois.get(name)
        if roi_data:
            return {
                "diameter": roi_data["diameter"],
                "center_vox": roi_data["center_vox"],
                "radius_vox": roi_data["radius_vox"],
            }
        return None

    phantom.get_roi.side_effect = get_roi_side_effect

    # Test the hot sphere counts calculation
    counts = analysis._calculate_hot_sphere_counts_offset_zxy(image, phantom, 10)

    if not isinstance(counts, dict):
        pytest.fail(f"Expected counts to be dict, got {type(counts)}")

    if "hot_sphere_10mm" not in counts:
        pytest.fail("Expected 'hot_sphere_10mm' in counts")

    # The function returns a nested dict with "mean", "std_H", "n_H"
    sphere_data = counts["hot_sphere_10mm"]
    if not isinstance(sphere_data, dict):
        pytest.fail(f"Expected sphere_data to be dict, got {type(sphere_data)}")

    if "mean" not in sphere_data:
        pytest.fail("Expected 'mean' key in sphere_data")

    sphere_mean = sphere_data["mean"]
    if not (sphere_mean > 700):
        pytest.fail(f"Expected mean > 700, got {sphere_mean}")


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

    if not isinstance(lung_counts, dict):
        pytest.fail(f"Expected lung_counts to be dict, got {type(lung_counts)}")

    if len(lung_counts) != 2:
        pytest.fail(f"Expected 2 slices, got {len(lung_counts)}")

    if 10 not in lung_counts:
        pytest.fail("Expected slice 10 in lung_counts")

    if 11 not in lung_counts:
        pytest.fail("Expected slice 11 in lung_counts")

    # Values should be numeric types
    for count in lung_counts.values():
        # Accept both Python float and NumPy float types (float32, float64, etc.)
        if not isinstance(count, (float, np.float32, np.float64)):
            pytest.fail(f"Expected count to be float type, got {type(count)}")

        if not (float(count) > 0):
            pytest.fail(f"Expected count to be positive, got {float(count)}")


def test_calculate_advanced_metrics_basic(mock_cfg):
    """Test advanced metrics with simple binary masks."""
    image_data = np.zeros((3, 3, 3), dtype=np.uint8)
    gt_data = np.zeros((3, 3, 3), dtype=np.uint8)
    image_data[1, 1, 1] = 1
    gt_data[1, 1, 1] = 1
    gt_data[1, 1, 2] = 1

    metrics = analysis.calculate_advanced_metrics(
        image_data,
        gt_data,
        measures=("Dice", "Jaccard"),
        cfg=mock_cfg,
    )

    assert "Dice" in metrics
    assert "Jaccard" in metrics
    assert metrics["Dice"] > 0


@patch("src.nema_quant.analysis.plt.savefig")
@patch("src.nema_quant.analysis.plt.close")
def test_save_sphere_visualization(mock_close, mock_savefig, tmp_path: Path):
    """Test sphere visualization output."""
    image_slice = np.zeros((10, 10), dtype=np.float32)
    roi_mask = np.zeros((10, 10), dtype=bool)
    roi_mask[4:6, 4:6] = True

    analysis.save_sphere_visualization(
        image_slice=image_slice,
        sphere_name="hot_sphere_10mm",
        center_yx=(5.0, 5.0),
        radius_vox=2.0,
        roi_mask=roi_mask,
        output_dir=tmp_path,
        slice_idx=3,
    )

    assert mock_savefig.called


@patch("src.nema_quant.analysis.plt.savefig")
@patch("src.nema_quant.analysis.plt.close")
def test_save_background_visualization(mock_close, mock_savefig, tmp_path: Path):
    """Test background visualization output."""
    image_slice = np.zeros((10, 10), dtype=np.float32)

    analysis.save_background_visualization(
        image_slice=image_slice,
        centers_offset=[(-1, -1), (1, 1)],
        pivot_point_yx=(5.0, 5.0),
        radius_vox=2.0,
        output_dir=tmp_path,
        slice_idx=2,
    )

    assert mock_savefig.called


@patch("src.nema_quant.analysis._calculate_crc_std")
@patch("src.nema_quant.analysis._calculate_spillover_ratio")
@patch("src.nema_quant.analysis.find_phantom_center_cv2_threshold")
def test_calculate_nema_metrics_nu4_2008_basic(
    mock_find_center,
    mock_spillover,
    mock_crc,
):
    """Test NU4 analysis path with patched helpers."""
    cfg = get_cfg_defaults()
    cfg.ROIS.CENTRAL_SLICE = 5
    cfg.ROIS.SPACING = 2.0
    cfg.ROIS.ORIENTATION_Z = 1
    cfg.ROIS.ORIENTATION_YX = [1, 1]

    image_data = np.ones((10, 10, 10), dtype=np.float32)
    phantom = MagicMock()

    mock_find_center.return_value = (5.0, 5.0, 5.0)
    mock_spillover.return_value = {"air": {"SOR": 0.01, "SOR_error": 0.0, "%STD": 1.0}}
    mock_crc.return_value = {
        "rod_1": {"recovery_coeff": 1.0, "cError": 0.0, "percentage_STD_rc": 0.0}
    }

    crc_results, spillover_results, uniformity_results = (
        analysis.calculate_nema_metrics_nu4_2008(
            image_data=image_data,
            phantom=phantom,
            cfg=cfg,
            save_visualizations=False,
        )
    )

    assert isinstance(crc_results, dict)
    assert isinstance(spillover_results, dict)
    assert isinstance(uniformity_results, dict)


def test_propagate_ratio_stddev():
    """Test error propagation for ratio calculation."""
    mu1, mu2 = 10.0, 5.0
    var1, var2 = 1.0, 0.25
    cov12 = 0.0

    stddev = analysis._propagate_ratio(mu1, mu2, var1, var2, cov12)

    assert stddev >= 0.0
    assert isinstance(stddev, float)

    # Test with near-zero denominator
    stddev_zero = analysis._propagate_ratio(10.0, 0.0, 1.0, 0.0, 0.0, eps=1e-6)
    assert stddev_zero == 0.0

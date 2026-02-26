import numpy as np
import pytest

from src.nema_quant import metrics, utils
from src.nema_quant.statistics import Estimator, MultiEstimator, ratio_multivariate


def test_find_phantom_center():
    """Test phantom center detection."""
    # Create a simple image with clear center activity
    image = np.zeros((50, 100, 100), dtype=np.float32)

    # Add prominent activity at center
    center_z, center_y, center_x = 25, 50, 50
    for z in range(center_z - 5, center_z + 5):
        for y in range(center_y - 10, center_y + 10):
            for x in range(center_x - 10, center_x + 10):
                if 0 <= z < 50 and 0 <= y < 100 and 0 <= x < 100:
                    image[z, y, x] = 1000.0

    center = utils.find_phantom_center(image)

    assert isinstance(center, tuple)
    assert len(center) == 3
    z_coord, y_coord, x_coord = center
    assert 0 <= z_coord < 50 and 0 <= y_coord < 100 and 0 <= x_coord < 100


def test_find_phantom_center_with_threshold():
    """Test phantom center detection with custom threshold."""
    image = np.random.rand(20, 60, 60).astype(np.float32) * 100

    # Add a clear phantom
    center_z, center_y, center_x = 10, 30, 30
    for z in range(center_z - 3, center_z + 3):
        for y in range(center_y - 8, center_y + 8):
            for x in range(center_x - 8, center_x + 8):
                if 0 <= z < 20 and 0 <= y < 60 and 0 <= x < 60:
                    image[z, y, x] = 800.0

    center = utils.find_phantom_center(image, threshold=0.003)

    assert isinstance(center, tuple)
    assert len(center) == 3


def test_find_phantom_center_wrong_dimensions():
    """Test that find_phantom_center handles wrong dimensions."""
    # 2D image should raise an error
    image_2d = np.random.rand(100, 100)

    with pytest.raises((ValueError, IndexError)):
        utils.find_phantom_center(image_2d)


def test_voxel_to_mm():
    """Test voxel to mm conversion."""
    voxel_indices = (10, 20, 30)  # z, y, x
    image_dims = (100, 100, 100)  # x, y, z
    voxel_spacing = (2.0, 2.0, 2.0)  # x, y, z spacing

    mm_coords = utils.voxel_to_mm(voxel_indices, image_dims, voxel_spacing)

    assert isinstance(mm_coords, tuple)
    assert len(mm_coords) == 3
    # Should be offset from center
    assert all(isinstance(c, float) for c in mm_coords)


def test_mm_to_voxel():
    """Test mm to voxel conversion."""
    mm_coords = (10.0, 20.0, 30.0)  # x, y, z in mm
    image_dims = (100, 100, 100)  # x, y, z dimensions
    voxel_spacing = (2.0, 2.0, 2.0)  # x, y, z spacing

    voxel_indices = utils.mm_to_voxel(mm_coords, image_dims, voxel_spacing)

    assert isinstance(voxel_indices, tuple)
    assert len(voxel_indices) == 3
    assert all(isinstance(idx, int) for idx in voxel_indices)


def test_voxel_to_mm_and_back():
    """Test round-trip conversion voxel->mm->voxel."""
    original_voxel = (25, 50, 75)
    image_dims = (150, 150, 150)
    voxel_spacing = (2.0644, 2.0644, 2.0644)

    # Convert to mm
    mm_coords = utils.voxel_to_mm(original_voxel, image_dims, voxel_spacing)

    # Convert back to voxel
    recovered_voxel = utils.mm_to_voxel(mm_coords, image_dims, voxel_spacing)

    # Should be close to original (within rounding)
    assert all(abs(o - r) <= 1 for o, r in zip(original_voxel, recovered_voxel))


def test_calculate_weighted_cbr_fom():
    """Test weighted CBR FOM calculation."""
    results = [
        {
            "diameter_mm": 10.0,
            "percentaje_constrast_QH": 85.0,
            "background_variability_N": 5.2,
        },
        {
            "diameter_mm": 13.0,
            "percentaje_constrast_QH": 78.5,
            "background_variability_N": 5.8,
        },
        {
            "diameter_mm": 17.0,
            "percentaje_constrast_QH": 70.0,
            "background_variability_N": 6.5,
        },
    ]

    result = utils.calculate_weighted_cbr_fom(results)

    # Function returns a dict with weighted_CBR and weighted_FOM
    assert isinstance(result, dict)
    assert "weighted_CBR" in result
    assert "weighted_FOM" in result
    assert isinstance(result["weighted_CBR"], (float, int))
    assert isinstance(result["weighted_FOM"], (float, int))
    assert result["weighted_CBR"] > 0
    assert result["weighted_FOM"] > 0


def test_calculate_weighted_cbr_fom_empty():
    """Test weighted CBR FOM with empty results."""
    results: list[dict[str, float]] = []

    result = utils.calculate_weighted_cbr_fom(results)

    # Function returns a dict with None values for empty input
    assert isinstance(result, dict)
    assert result["weighted_CBR"] is None
    assert result["weighted_FOM"] is None


def test_calculate_weighted_cbr_fom_single():
    """Test weighted CBR FOM with single result."""
    results = [
        {
            "diameter_mm": 10.0,
            "percentaje_constrast_QH": 85.0,
            "background_variability_N": 5.2,
        }
    ]

    result = utils.calculate_weighted_cbr_fom(results)

    # Function returns a dict
    assert isinstance(result, dict)
    assert "weighted_CBR" in result
    assert "weighted_FOM" in result
    assert isinstance(result["weighted_CBR"], (float, int))
    assert result["weighted_CBR"] > 0


def test_find_phantom_center_cv2_threshold_methods():
    """Test CV2-based center finding with both methods."""
    image = np.zeros((10, 20, 20), dtype=np.float32)
    image[5, 8:12, 8:12] = 100.0

    center_weighted = utils.find_phantom_center_cv2_threshold(
        image, threshold_fraction=0.5, method="weighted_slices"
    )
    center_max = utils.find_phantom_center_cv2_threshold(
        image, threshold_fraction=0.5, method="max_slice"
    )

    assert len(center_weighted) == 3
    assert len(center_max) == 3


def test_metrics_get_values_overlap_and_distance():
    """Test overlap and distance metrics on small masks."""
    mask_a = np.zeros((3, 3, 3), dtype=np.uint8)
    mask_b = np.zeros((3, 3, 3), dtype=np.uint8)
    mask_a[1, 1, 1] = 1
    mask_b[1, 1, 1] = 1

    values = metrics.get_values(mask_a, mask_b, ["Dice", "Jaccard", "HD", "ASSD"])

    assert np.isclose(values["Dice"], 1.0)
    assert np.isclose(values["Jaccard"], 1.0)
    assert np.isclose(values["HD"], 0.0)
    assert np.isclose(values["ASSD"], 0.0)


def test_metrics_background_nan():
    """Background metrics should be NaN when background flag is True."""
    mask_a = np.zeros((3, 3), dtype=np.uint8)
    mask_b = np.zeros((3, 3), dtype=np.uint8)

    values = metrics.get_values(
        mask_a,
        mask_b,
        ["AVD", "HD", "F1", "Recall", "ASSD"],
        background=True,
    )

    assert np.isnan(values["AVD"])
    assert np.isnan(values["HD"])
    assert np.isnan(values["F1"])
    assert np.isnan(values["Recall"])
    assert np.isnan(values["ASSD"])


def test_metrics_auc_basic():
    """Test AUC calculation on simple masks."""
    mask_a = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    mask_b = np.array([[1, 1], [0, 0]], dtype=np.uint8)

    auc = metrics.get_AUC(mask_a, mask_b)

    assert 0.0 <= auc <= 1.0


def test_estimator_from_samples_empty():
    """Empty sample set should yield zero estimator."""
    est = Estimator.from_samples(np.array([]))

    assert est.mean == 0.0
    assert est.variance == 0.0
    assert est.n == 1


def test_estimator_ratio_zero_denominator():
    """Ratio against near-zero mean should be zeroed safely."""
    est_a = Estimator(10.0, 1.0, n=5)
    est_b = Estimator(0.0, 1.0, n=5)

    ratio = est_a.ratio(est_b, eps=1e-6)

    assert ratio.mean == 0.0
    assert ratio.variance == 0.0


def test_multiestimator_from_independent():
    """Independent estimators should build a diagonal covariance."""
    est_a = Estimator(1.0, 0.25)
    est_b = Estimator(2.0, 0.04)

    multi = MultiEstimator.from_independent([est_a, est_b])

    assert multi.cov.shape == (2, 2)
    assert multi.cov[0, 1] == 0.0


def test_ratio_multivariate():
    """Ratio propagation should return a single-mean estimator."""
    means = np.array([2.0, 1.0])
    cov = np.eye(2)
    multi = MultiEstimator(means, cov)

    ratio = ratio_multivariate(multi)

    assert ratio.means.shape == (1,)
    assert np.isclose(ratio.means[0], 2.0)


def test_interactive_roi_editor_import():
    """Smoke test for interactive ROI editor module."""
    pytest.importorskip("pyqtgraph")

    from src.nema_quant import interactive_roi_editor

    assert hasattr(interactive_roi_editor, "InteractiveROIEditor")


def test_metrics_sensitivity_specificity_f1():
    """Test sensitivity, specificity, and F1 score metrics."""
    # Perfect overlap
    mask_a = np.ones((3, 3), dtype=np.uint8)
    mask_b = np.ones((3, 3), dtype=np.uint8)

    sensitivity = metrics.get_sensitivity(mask_a, mask_b)
    f1 = metrics.get_F1(mask_a, mask_b)

    assert np.isclose(sensitivity, 1.0)
    assert np.isclose(f1, 1.0)

    # Partial overlap
    mask_c = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)
    mask_d = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)

    sensitivity2 = metrics.get_sensitivity(mask_c, mask_d)
    f1_2 = metrics.get_F1(mask_c, mask_d)

    assert 0.0 <= sensitivity2 <= 1.0
    assert 0.0 <= f1_2 <= 1.0


def test_metrics_volumetric_similarity():
    """Test volumetric similarity metric."""
    mask_a = np.ones((5, 5), dtype=np.uint8)
    mask_b = np.ones((5, 5), dtype=np.uint8)

    vs = metrics.get_volumetric_similarity(mask_a, mask_b)

    assert np.isclose(vs, 1.0)


def test_metrics_global_consistency_error():
    """Test global consistency error metric."""
    mask_a = np.ones((3, 3), dtype=np.uint8)
    mask_b = np.ones((3, 3), dtype=np.uint8)

    gce = metrics.get_global_consistency_error(mask_a, mask_b)

    # GCE can be NaN for degenerate cases
    assert gce >= 0.0 or np.isnan(gce)


def test_metrics_rand_idx():
    """Test Rand index metric."""
    mask_a = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    mask_b = np.array([[1, 1], [0, 0]], dtype=np.uint8)

    # get_rand_idx returns (RI, ARI) tuple
    ri, ari = metrics.get_rand_idx(mask_a, mask_b)

    assert 0.0 <= ri <= 1.0
    assert -1.0 <= ari <= 1.0


def test_metrics_mutual_information():
    """Test mutual information metric."""
    mask_a = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    mask_b = np.array([[1, 0], [0, 1]], dtype=np.uint8)

    # get_MI returns (MI_normalized, VOI) tuple
    mi_norm, voi = metrics.get_MI(mask_a, mask_b)

    assert 0.0 <= mi_norm <= 1.0
    assert voi >= 0.0


def test_metrics_icc_pbd_kap():
    """Test ICC, PBD, and Cohen's Kappa metrics."""
    mask_a = np.ones((4, 4), dtype=np.uint8)
    mask_b = np.ones((4, 4), dtype=np.uint8)

    icc = metrics.get_ICC(mask_a, mask_b)
    pbd = metrics.get_PBD(mask_a, mask_b)
    kap = metrics.get_KAP(mask_a, mask_b)

    # ICC and Kappa can be NaN for degenerate cases (identical masks)
    assert (-1.0 <= icc <= 1.0) or np.isnan(icc)
    assert pbd >= 0.0
    assert (-1.0 <= kap <= 1.0) or np.isnan(kap)


def test_metrics_truepos_trueneg_falsepos_falseneg():
    """Test confusion matrix components."""
    mask_a = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    mask_b = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)

    tp = metrics.get_truepos(mask_a, mask_b)
    tn = metrics.get_trueneg(mask_a, mask_b)
    fp = metrics.get_falsepos(mask_a, mask_b)
    fn = metrics.get_falseneg(mask_a, mask_b)

    # All should be non-negative
    assert tp >= 0
    assert tn >= 0
    assert fp >= 0
    assert fn >= 0

    # Total should equal number of pixels
    assert tp + tn + fp + fn == mask_a.size


def test_utils_calculate_weighted_cbr_fom():
    """Test weighted CBR FOM calculation."""
    results = [
        {
            "diameter_mm": 10,
            "percentaje_constrast_QH": 2.0,
            "background_variability_N": 1.0,
        },
        {
            "diameter_mm": 13,
            "percentaje_constrast_QH": 2.5,
            "background_variability_N": 1.0,
        },
        {
            "diameter_mm": 17,
            "percentaje_constrast_QH": 3.0,
            "background_variability_N": 1.0,
        },
        {
            "diameter_mm": 22,
            "percentaje_constrast_QH": 3.5,
            "background_variability_N": 1.0,
        },
        {
            "diameter_mm": 28,
            "percentaje_constrast_QH": 4.0,
            "background_variability_N": 1.0,
        },
        {
            "diameter_mm": 37,
            "percentaje_constrast_QH": 4.5,
            "background_variability_N": 1.0,
        },
    ]

    result = utils.calculate_weighted_cbr_fom(results)

    assert isinstance(result, dict)
    assert "weighted_CBR" in result
    assert "weighted_FOM" in result
    assert result["weighted_CBR"] > 0.0
    assert result["weighted_FOM"] > 0.0

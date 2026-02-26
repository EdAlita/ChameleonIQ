import numpy as np
import pytest

from config.defaults import get_cfg_defaults
from src.nema_quant.phantom import NemaPhantom

CFG = get_cfg_defaults()


@pytest.fixture
def phantom_instance():
    """
    Pytest fixture to create a standard instance of NemaPhantom for testing.
    This avoids recreating the object in every test function.
    """
    return NemaPhantom(
        cfg=CFG, image_dims=(391, 391, 346), voxel_spacing=(2.0644, 2.0644, 2.0644)
    )


def test_phantom_initialization(phantom_instance):
    """
    Tests if the NemaPhantom class is initialized correctly with valid inputs.
    """
    # Using explicit checks instead of pytest.assume
    expected_dims = (391, 391, 346)
    expected_spacing = (2.0644, 2.0644, 2.0644)

    if phantom_instance.image_dims != expected_dims:
        pytest.fail(
            f"Expected image_dims {expected_dims}, got {phantom_instance.image_dims}"
        )

    if phantom_instance.voxel_spacing != expected_spacing:
        pytest.fail(
            f"Expected voxel_spacing {expected_spacing}, got {phantom_instance.voxel_spacing}"
        )

    if not isinstance(phantom_instance.rois, dict):
        pytest.fail(f"Expected rois to be dict, got {type(phantom_instance.rois)}")

    # Check for actual sphere names from config (hot_sphere_1mm through hot_sphere_5mm)
    expected_spheres = [f"hot_sphere_{i}mm" for i in range(1, 6)]
    found_spheres = [name for name in expected_spheres if name in phantom_instance.rois]

    if not found_spheres:
        pytest.fail(
            f"Expected at least one of {expected_spheres} to be in rois dictionary, found: {list(phantom_instance.rois.keys())}"
        )


def test_initialization_with_invalid_dims():
    """
    Tests that the class raises a ValueError when initialized with incorrect
    dimensions for image_dims or voxel_spacing.
    """
    with pytest.raises(ValueError, match="Expected 3 elements" " for 'image_dims'"):
        NemaPhantom(
            cfg=CFG,
            image_dims=(100, 100),  # type: ignore
            voxel_spacing=(2.0644, 2.0644, 2.0644),
        )  # type: ignore

    with pytest.raises(ValueError, match="Expected 3 elements" " for 'voxel_spacing'"):
        NemaPhantom(
            cfg=CFG,
            image_dims=(391, 391, 346),
            voxel_spacing=(2.0, 2.0, 2.0, 2.0),  # type: ignore
        )  # type: ignore


def test_mm_to_voxels_conversion(phantom_instance):
    """
    Tests the internal _mm_to_voxels conversion logic with a known value.
    """
    expected_voxel = 10 / 2.0644
    calculated_voxels = phantom_instance._mm_to_voxels(10, 0)

    if not np.isclose(calculated_voxels, expected_voxel):
        pytest.fail(f"Expected {expected_voxel}, got {calculated_voxels}")


def test_get_roi_success(phantom_instance):
    """
    Tests successful retrieval of a defined ROI using the get_roi method.
    """
    # Use actual sphere name from config (hot_sphere_5mm)
    roi = phantom_instance.get_roi("hot_sphere_5mm")

    if roi is None:
        pytest.fail("Expected roi to be not None for 'hot_sphere_5mm'")

    if not isinstance(roi, dict):
        pytest.fail(f"Expected roi to be dict, got {type(roi)}")

    if "center_vox" not in roi:
        pytest.fail("Expected 'center_vox' key in roi dictionary")

    if "radius_vox" not in roi:
        pytest.fail("Expected 'radius_vox' key in roi dictionary")


def test_get_roi_failure(phantom_instance):
    """
    Tests that get_roi returns None for a non-existent ROI name.
    """
    roi = phantom_instance.get_roi("non_existent_sphere")

    if roi is not None:
        pytest.fail(f"Expected None for non-existent ROI, got {roi}")


def test_roi_radius_calculation(phantom_instance):
    """
    Tests the calculated voxel radius for a specific, known ROI.
    """
    # Use actual sphere name - 4mm diameter sphere
    hot_sphere_roi = phantom_instance.get_roi("hot_sphere_4mm")
    # Diameter is 4mm, so radius is 2mm.
    expected_radius_voxels = 2 / 2.0644

    if hot_sphere_roi is None:
        pytest.fail("Expected hot_sphere_4mm_roi to be not None")

    if not np.isclose(hot_sphere_roi["radius_vox"], expected_radius_voxels):
        pytest.fail(
            f"Expected radius {expected_radius_voxels}, got {hot_sphere_roi['radius_vox']}"
        )

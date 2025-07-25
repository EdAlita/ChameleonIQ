import pytest
import numpy as np
from src.nema_quant.phantom import NemaPhantom
from config.defaults import get_cfg_defaults

CFG = get_cfg_defaults()


@pytest.fixture
def phantom_instance():
    """
    Pytest fixture to create a standard instance of NemaPhantom for testing.
    This avoids recreating the object in every test function.
    """
    return NemaPhantom(
        cfg=CFG,
        image_dims=(391, 391, 346),
        voxel_spacing=(2.0644, 2.0644, 2.0644)
    )


def test_phantom_initialization(phantom_instance):
    """
    Tests if the NemaPhantom class is initialized correctly with valid inputs.
    """
    assert phantom_instance.image_dims == (391, 391, 346)
    assert phantom_instance.voxel_spacing == (2.0644, 2.0644, 2.0644)
    assert isinstance(phantom_instance.rois, dict)
    assert 'hot_sphere_10mm' in phantom_instance.rois


def test_initialization_with_invalid_dims():
    """
    Tests that the class raises a ValueError when initialized with incorrect
    dimensions for image_dims or voxel_spacing.
    """
    with pytest.raises(ValueError, match="Expected 3 elements"
                       " for 'image_dims'"):
        NemaPhantom(
            cfg=CFG,
            image_dims=(100, 100),  # type: ignore
            voxel_spacing=(2.0644, 2.0644, 2.0644)
        )  # type: ignore

    with pytest.raises(ValueError, match="Expected 3 elements"
                       " for 'voxel_spacing'"):
        NemaPhantom(
            cfg=CFG,
            image_dims=(391, 391, 346),
            voxel_spacing=(2.0, 2.0, 2.0, 2.0)  # type: ignore
        )  # type: ignore


def test_mm_to_voxels_conversion(phantom_instance):
    """
    Tests the internal _mm_to_voxels conversion logic with a known value.
    """
    expected_voxel = 10 / 2.0644
    calculated_voxels = phantom_instance._mm_to_voxels(10, 0)
    assert np.isclose(calculated_voxels, expected_voxel)


def test_get_roi_success(phantom_instance):
    """
    Tests successful retrieval of a defined ROI using the get_roi method.
    """
    roi = phantom_instance.get_roi('hot_sphere_37mm')
    assert roi is not None
    assert isinstance(roi, dict)
    assert 'center_vox' in roi
    assert 'radius_vox' in roi


def test_get_roi_failure(phantom_instance):
    """
    Tests that get_roi returns None for a non-existent ROI name.
    """
    roi = phantom_instance.get_roi('non_existent_sphere')
    assert roi is None


def test_roi_radius_calculation(phantom_instance):
    """
    Tests the calculated voxel radius for a specific, known ROI.
    """
    hot_sphere_10mm_roi = phantom_instance.get_roi('hot_sphere_10mm')
    # Diameter is 10mm, so radius is 5mm.
    expected_radius_voxels = 5 / 2.0644

    assert hot_sphere_10mm_roi is not None
    assert np.isclose(hot_sphere_10mm_roi['radius_vox'],
                      expected_radius_voxels)

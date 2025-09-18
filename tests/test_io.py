import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nibabel.loadsave import save
from nibabel.nifti1 import Nifti1Image

from src.nema_quant.io import load_nii_image

# Define the expected properties of the test data file.
# These must match the properties used to generate the test file.
TEST_IMAGE_DIMS = (346, 391, 391)
TEST_IMAGE_DTYPE = np.float32
TEST_FILE_PATH = Path(
    "data/EARL_TORSO_CTstudy.2400s.DOI.EQZ.att_yes.frame02.subs05.nii"
)


def test_load_nii_image_success():
    """
    Tests successful loading of the predefined raw image file.
    """
    # Ensure the test file exists before running the test
    if not TEST_FILE_PATH.is_file():
        pytest.skip(
            f"Test data file not found at {TEST_FILE_PATH}. "
            "Run the data generation script."
        )

    loaded_data, _ = load_nii_image(filepath=TEST_FILE_PATH, return_affine=False)

    if not isinstance(loaded_data, np.ndarray):
        pytest.fail(f"Expected loaded_data to be np.ndarray, got {type(loaded_data)}")

    if loaded_data.shape != TEST_IMAGE_DIMS:
        pytest.fail(f"Expected shape {TEST_IMAGE_DIMS}, got {loaded_data.shape}")

    if loaded_data.dtype != TEST_IMAGE_DTYPE:
        pytest.fail(f"Expected dtype {TEST_IMAGE_DTYPE}, got {loaded_data.dtype}")


def test_load_nii_image_with_affine():
    """
    Tests loading with return_affine=True to cover both code paths.
    """
    if not TEST_FILE_PATH.is_file():
        pytest.skip(f"Test data file not found at {TEST_FILE_PATH}")

    loaded_data, affine = load_nii_image(filepath=TEST_FILE_PATH, return_affine=True)

    if not isinstance(loaded_data, np.ndarray):
        pytest.fail(f"Expected loaded_data to be np.ndarray, got {type(loaded_data)}")

    if affine is None:
        pytest.fail("Expected affine matrix to be returned, got None")

    if not isinstance(affine, np.ndarray):
        pytest.fail(f"Expected affine to be np.ndarray, got {type(affine)}")


def test_load_nii_image_file_not_found():
    """
    Tests that the function raises FileNotFoundError for a non-existent path.
    """
    non_existent_path = Path("path/that/does/not/exist/fake.nii")
    with pytest.raises(FileNotFoundError, match="The file was not found at"):
        load_nii_image(non_existent_path, False)


def test_load_nii_image_invalid_file():
    """
    Tests handling of invalid NIfTI files to cover error cases.
    """
    # Create a temporary invalid file
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp_file:
        tmp_file.write(b"invalid nifti content")
        tmp_path = Path(tmp_file.name)

    try:
        with pytest.raises(
            (nib.filebasedimages.ImageFileError, ValueError, OSError), match=".*"
        ):  # nibabel will raise various exceptions for invalid files
            load_nii_image(tmp_path, False)
    finally:
        tmp_path.unlink()  # Clean up


def test_load_nii_image_created_test_file():
    """
    Create a small test NIfTI file to ensure we can test the function independently.
    """
    # Create test data with more realistic values (avoid random for reproducibility)
    test_data = np.ones((10, 10, 10), dtype=np.float32) * 100.0
    test_data[5, 5, 5] = 1000.0  # Add a known hot spot
    test_affine = np.eye(4)

    # Create NIfTI image
    nii_img = Nifti1Image(test_data, test_affine)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        save(nii_img, str(tmp_path))

        # Test loading with return_affine=True
        loaded_data, loaded_affine = load_nii_image(tmp_path, return_affine=True)

        # Check basic properties
        if not isinstance(loaded_data, np.ndarray):
            pytest.fail(
                f"Expected loaded_data to be np.ndarray, got {type(loaded_data)}"
            )

        if loaded_data.shape != test_data.shape:
            pytest.fail(f"Expected shape {test_data.shape}, got {loaded_data.shape}")

        if not isinstance(loaded_affine, np.ndarray):
            pytest.fail(
                f"Expected loaded_affine to be np.ndarray, got {type(loaded_affine)}"
            )

        # Check that the data values are approximately correct (allowing for precision differences)
        if not np.allclose(loaded_data, test_data, rtol=1e-5, atol=1e-6):
            pytest.fail("Loaded data doesn't approximately match original test data")

        # Check that affine matrix has the expected shape and basic properties
        # Note: NIfTI save/load may modify the affine matrix, so we check shape and basic validity
        if loaded_affine.shape != (4, 4):
            pytest.fail(f"Expected affine shape (4, 4), got {loaded_affine.shape}")

        # Check that it's a valid transformation matrix (last row should be [0, 0, 0, 1])
        expected_last_row = np.array([0, 0, 0, 1])
        if not np.allclose(loaded_affine[3, :], expected_last_row, rtol=1e-5):
            pytest.fail(f"Expected last row [0, 0, 0, 1], got {loaded_affine[3, :]}")

        # Test loading without affine
        loaded_data_only, returned_affine = load_nii_image(
            tmp_path, return_affine=False
        )

        if returned_affine is not None:
            pytest.fail("Expected None for affine when return_affine=False")

        if not np.allclose(loaded_data_only, test_data, rtol=1e-5, atol=1e-6):
            pytest.fail("Loaded data (no affine) doesn't approximately match original")

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_load_nii_image_data_types():
    """
    Test loading with different data types to ensure proper conversion.
    """
    # Test with integer data
    test_data_int = np.ones((5, 5, 5), dtype=np.int16) * 500
    test_affine = np.eye(4)

    nii_img = Nifti1Image(test_data_int, test_affine)

    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        save(nii_img, str(tmp_path))

        loaded_data, _ = load_nii_image(tmp_path, return_affine=False)

        # Check that data was loaded successfully
        if not isinstance(loaded_data, np.ndarray):
            pytest.fail(
                f"Expected loaded_data to be np.ndarray, got {type(loaded_data)}"
            )

        if loaded_data.shape != test_data_int.shape:
            pytest.fail(
                f"Expected shape {test_data_int.shape}, got {loaded_data.shape}"
            )

        # Values should be approximately equal (allowing for type conversion)
        if not np.allclose(loaded_data, test_data_int.astype(np.float32), rtol=1e-5):
            pytest.fail("Loaded integer data doesn't match expected values")

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_load_nii_image_edge_cases():
    """
    Test edge cases for the load_nii_image function.
    """
    # Test with minimal data
    minimal_data = np.array([[[1.0]]], dtype=np.float32)
    minimal_affine = np.eye(4)

    nii_img = Nifti1Image(minimal_data, minimal_affine)

    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        save(nii_img, str(tmp_path))

        loaded_data, loaded_affine = load_nii_image(tmp_path, return_affine=True)

        assert loaded_data is not None
        assert loaded_affine is not None

        if loaded_data.shape != (1, 1, 1):
            pytest.fail(f"Expected shape (1, 1, 1), got {loaded_data.shape}")

        if not np.isclose(loaded_data[0, 0, 0], 1.0):
            pytest.fail(f"Expected value 1.0, got {loaded_data[0, 0, 0]}")

        # Check that affine is valid
        if loaded_affine.shape != (4, 4):
            pytest.fail(f"Expected affine shape (4, 4), got {loaded_affine.shape}")

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_load_nii_image_path_types():
    """
    Test that the function accepts both Path objects and strings.
    """
    # Create test data
    test_data = np.ones((3, 3, 3), dtype=np.float32)
    test_affine = np.eye(4)
    nii_img = Nifti1Image(test_data, test_affine)

    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        save(nii_img, str(tmp_path))

        # Test with Path object
        loaded_data_path, _ = load_nii_image(tmp_path, return_affine=False)

        # Test with string path (convert to Path first if function requires it)
        string_path = str(tmp_path)
        loaded_data_str, _ = load_nii_image(Path(string_path), return_affine=False)

        # Both should produce the same result
        if not np.array_equal(loaded_data_path, loaded_data_str):
            pytest.fail("Loading with Path vs string produced different results")

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_load_nii_image_function_signature():
    """
    Test that the function properly handles different parameter combinations.
    """
    # Create minimal test data
    test_data = np.array([[[42.0]]], dtype=np.float32)
    test_affine = np.eye(4)
    nii_img = Nifti1Image(test_data, test_affine)

    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        save(nii_img, str(tmp_path))

        # Test explicit parameter names
        loaded_data, affine = load_nii_image(filepath=tmp_path, return_affine=True)

        assert loaded_data is not None
        assert affine is not None

        if not isinstance(loaded_data, np.ndarray):
            pytest.fail(f"Expected np.ndarray, got {type(loaded_data)}")

        if affine is None:
            pytest.fail("Expected affine matrix, got None")

        # Test positional arguments
        loaded_data_pos, affine_pos = load_nii_image(tmp_path, True)

        assert loaded_data_pos is not None
        assert affine_pos is not None

        if not np.array_equal(loaded_data, loaded_data_pos):
            pytest.fail("Positional args produced different result than named args")

        if not np.array_equal(affine, affine_pos):
            pytest.fail("Positional args produced different affine than named args")

    finally:
        if tmp_path.exists():
            tmp_path.unlink()

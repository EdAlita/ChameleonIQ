from pathlib import Path

import numpy as np
import pytest

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
    This test assumes 'data/rawData.imgrec' has been created.
    """
    # Ensure the test file exists before running the test
    if not TEST_FILE_PATH.is_file():
        pytest.skip(
            f"Test data file not found at {TEST_FILE_PATH}. "
            "Run the data generation script."
        )

    # Execute the function to be tested
    loaded_data, _ = load_nii_image(filepath=TEST_FILE_PATH, return_affine=False)

    # Assert the results are as expected
    assert isinstance(loaded_data, np.ndarray)
    assert loaded_data.shape == TEST_IMAGE_DIMS
    assert loaded_data.dtype == TEST_IMAGE_DTYPE


def test_load_nii_image_file_not_found():
    """
    Tests that the function raises FileNotFoundError for a non-existent path.
    """
    non_existent_path = Path("path/that/does/not/exist/fake.nii")
    with pytest.raises(FileNotFoundError, match="The file was not found at"):
        load_nii_image(non_existent_path, False)

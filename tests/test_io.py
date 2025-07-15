import pytest
import numpy as np
from pathlib import Path
from nema_quant.io import load_raw_image

# Define the expected properties of the test data file.
# These must match the properties used to generate the test file.
TEST_IMAGE_DIMS = (391, 391, 346)
TEST_IMAGE_DTYPE = np.float32
TEST_FILE_PATH = Path("data/rawData.imgrec")


def test_load_raw_image_success():
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
    loaded_data = load_raw_image(
        filepath=TEST_FILE_PATH,
        image_dims=TEST_IMAGE_DIMS,
        dtype=np.dtype(TEST_IMAGE_DTYPE)
    ).reshape(TEST_IMAGE_DIMS)

    # Assert the results are as expected
    assert isinstance(loaded_data, np.ndarray)
    assert loaded_data.shape == TEST_IMAGE_DIMS
    assert loaded_data.dtype == TEST_IMAGE_DTYPE


def test_load_raw_image_file_not_found():
    """
    Tests that the function raises FileNotFoundError for a non-existent path.
    """
    non_existent_path = Path("path/that/does/not/exist/fake.raw")
    with pytest.raises(FileNotFoundError, match="file was not found"):
        load_raw_image(non_existent_path, TEST_IMAGE_DIMS)


def test_load_raw_image_size_mismatch():
    """
    Tests that the function raises ValueError if the file size does not
    match the expected dimensions.
    """
    if not TEST_FILE_PATH.is_file():
        pytest.skip(f"Test data file not found at {TEST_FILE_PATH}.")

    incorrect_dims = (100, 100, 100)

    with pytest.raises(ValueError, match="File size mismatch"):
        load_raw_image(
            filepath=TEST_FILE_PATH,
            image_dims=incorrect_dims,
            dtype=np.dtype(TEST_IMAGE_DTYPE)
        )

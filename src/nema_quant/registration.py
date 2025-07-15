import numpy as np
from scipy.ndimage import affine_transform
from typing import Tuple, List


def calculate_rigid_transform(
    fixed_points: np.ndarray,
    moving_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the rigid transformation (rotation and translation)
    between two sets of points.

    Use the Kabsch algorithm to calculate the rotation
    matrix and the translation vector
    that minimize the distance between the points.

    Parameters
    ----------
    fixed_points : numpy.ndarray, shape (N, 3)
        Coordinates of the points in the still image.
    moving_points : numpy.ndarray, shape (N, 3)
        Coordinates of the points in the moving image.

    Returns
    -------
    rotation_matrix : numpy.ndarray, shape (3, 3)
        Rotation matrix that aligns the points.
    translation_matrix : numpy.ndarray, shape(3,)
        Translation vector that aligns the points.
    """
    centroid_fixed = np.mean(fixed_points, axis=0)
    centroid_moving = np.mean(moving_points, axis=0)

    fixed_centered = fixed_points - centroid_fixed
    moving_centered = moving_points - centroid_moving

    print("centroid_fixed:\n", centroid_fixed)
    print("centroid_moving:\n", centroid_moving)
    print("fixed_centered:\n", fixed_centered)
    print("moving_centered:\n", moving_centered)

    covariance_matrix = np.dot(moving_centered.T, fixed_centered)

    U, _, Vt = np.linalg.svd(covariance_matrix)

    rotation_matrix = np.dot(Vt.T, U.T)

    if np.linalg.det(rotation_matrix) < 0:
        Vt[-1, :] *= -1
        rotation_matrix = np.dot(Vt.T, U.T)

    translation_vector = centroid_fixed - np.dot(
        rotation_matrix,
        centroid_moving)

    return rotation_matrix, translation_vector


def apply_rigid_trasform(
    image_data: np.ndarray,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    output_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Applies a rigid transformation (rotation and translation) to a 3D image.

    Parameters
    ----------
    image_data : numpy.ndarray
         3D image data (z, y, x).
    rotation_matrix : numpy.ndarray, shape (3, 3)
         Rotation matrix that aligns the points.
    translation_vector : numpy.ndarray, shape (3,)
         Translation vector that aligns the points.
    output_shape : tuple of int
        Output image dimensions (z, y, x).

    Returns
    -------
    numpy.ndarray
        The transformed image according to the
        specified rotation and translation.
    """
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = rotation_matrix
    affine_matrix[:3, 3] = translation_vector

    transform_matrix = affine_matrix[:3, :3]
    offset = affine_matrix[:3, 3]

    print("Rotation Matrix:\n", rotation_matrix)
    print("Translation Vector:\n", translation_vector)

    transformed_image = affine_transform(
        image_data,
        matrix=transform_matrix,
        offset=offset,
        output_shape=output_shape,
        order=0,
        mode='constant',
        cval=0.0
    )

    return np.asarray(transformed_image)


def register_images_with_landmarks(
    moving_image: np.ndarray,
    fixed_image: np.ndarray,
    moving_landmarks: List[Tuple[int, int, int]],
    fixed_landmarks: List[Tuple[int, int, int]]
) -> np.ndarray:
    """
    Performs a rigid registration between two 3D images using landmarks.

    Parameters
    ----------
    moving_image : numpy.ndarray
        Moving image to be transformed
    fixed_image: numpy.ndarray
        Fixed image defining the reference space.
    moving_landmarks : list of tuple, shape (N, 3)
        Coordinates of the landmarks in the moving image.
    fixed_landmarks : list of tuple, shape(N, 3)
        Coordinates of the landmarks in the fixed image.

    Returns
    -------
    numpy.ndarray
        The moving image registered to the fixed image space.
    """
    moving_points = np.array(moving_landmarks)
    fixed_points = np.array(fixed_landmarks)

    rotation_matrix, translation_vector = calculate_rigid_transform(
        fixed_points, moving_points
    )

    output_shape = fixed_image.shape
    registered_image = apply_rigid_trasform(
        moving_image, rotation_matrix, translation_vector, output_shape
    )

    return registered_image


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    def plot_registration_results(
        fixed_image, moving_image, registered_image,
        fixed_landmarks, moving_landmarks
    ):
        """
        Plots the comparison of fixed, moving, and registered images in axial,
        coronal, and sagittal planes, overlaid with landmarks.
        """
        axial_slice = fixed_image.shape[0] // 2
        coronal_slice = fixed_image.shape[1] // 2
        sagittal_slice = fixed_image.shape[2] // 2

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle(
            "Comparison of Fixed, Moving, and Registered Images", fontsize=16
        )

        print("Data type of registered_image:", registered_image.dtype)
        print("Min value in registered_image:", np.min(registered_image))
        print("Max value in registered_image:", np.max(registered_image))

        # Axial plane
        ax = axes[0, 0]
        ax.imshow(fixed_image[axial_slice, :, :], cmap="gray", vmin=0, vmax=1)
        ax.set_title("Fixed - Axial")
        ax.axis("off")
        for landmark in fixed_landmarks:
            ax.plot(landmark[2], landmark[1], 'ro', markersize=4)

        ax = axes[0, 1]
        ax.imshow(moving_image[axial_slice, :, :], cmap="gray", vmin=0, vmax=1)
        ax.set_title("Moving - Axial")
        ax.axis("off")
        for landmark in moving_landmarks:
            ax.plot(landmark[2], landmark[1], 'ro', markersize=4)

        ax = axes[0, 2]
        img = registered_image[axial_slice, :, :]
        print(
            "Data type of registered_image[axial_slice, :, :]:", img.dtype
        )
        print(
            "Unique values in registered_image[axial_slice, :, :]:",
            np.unique(img)
        )
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title("Registered - Axial")
        ax.axis("off")

        # Coronal plane
        ax = axes[1, 0]
        ax.imshow(
            fixed_image[:, coronal_slice, :], cmap="gray", vmin=0, vmax=1
        )
        ax.set_title("Fixed - Coronal")
        ax.axis("off")
        for landmark in fixed_landmarks:
            ax.plot(landmark[2], landmark[0], 'ro', markersize=4)

        ax = axes[1, 1]
        ax.imshow(
            moving_image[:, coronal_slice, :], cmap="gray", vmin=0, vmax=1
        )
        ax.set_title("Moving - Coronal")
        ax.axis("off")
        for landmark in moving_landmarks:
            ax.plot(landmark[2], landmark[0], 'ro', markersize=4)

        ax = axes[1, 2]
        ax.imshow(
            registered_image[:, coronal_slice, :], cmap="gray", vmin=0, vmax=1
        )
        ax.set_title("Registered - Coronal")
        ax.axis("off")

        # Sagittal plane
        ax = axes[2, 0]
        ax.imshow(
            fixed_image[:, :, sagittal_slice], cmap="gray", vmin=0, vmax=1
        )
        ax.set_title("Fixed - Sagittal")
        ax.axis("off")
        for landmark in fixed_landmarks:
            ax.plot(landmark[1], landmark[0], 'ro', markersize=4)

        ax = axes[2, 1]
        ax.imshow(
            moving_image[:, :, sagittal_slice], cmap="gray", vmin=0, vmax=1
        )
        ax.set_title("Moving - Sagittal")
        ax.axis("off")
        for landmark in moving_landmarks:
            ax.plot(landmark[1], landmark[0], 'ro', markersize=4)

        ax = axes[2, 2]
        ax.imshow(
            registered_image[:, :, sagittal_slice], cmap="gray", vmin=0, vmax=1
        )
        ax.set_title("Registered - Sagittal")
        ax.axis("off")

        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.savefig(
            "tests/registration_results.png", dpi=1200, bbox_inches="tight"
        )

    # Example usage
    fixed_image = np.zeros((100, 100, 100), dtype=np.float32)
    fixed_image[40:60, 40:60, 40:60] = 1.0

    moving_image = np.zeros((100, 100, 100), dtype=np.float32)
    moving_image[50:70, 50:70, 50:70] = 1.0

    moving_landmarks = [
        (50, 50, 50), (50, 50, 70), (50, 70, 50), (50, 70, 70),
        (70, 50, 50), (70, 50, 70), (70, 70, 50), (70, 70, 70)
    ]

    fixed_landmarks = [
        (40, 40, 40), (40, 40, 60), (40, 60, 40), (40, 60, 60),
        (60, 40, 40), (60, 40, 60), (60, 60, 40), (60, 60, 60)
    ]

    moving_landmarks = np.array(moving_landmarks) + np.random.normal(
        0, 1e-6, size=(len(moving_landmarks), 3)
    )  # type: ignore
    moving_landmarks = moving_landmarks.tolist()  # type: ignore

    registered_image = register_images_with_landmarks(
        moving_image, fixed_image, moving_landmarks, fixed_landmarks
    )

    print("Registration completed.")

    plot_registration_results(
        fixed_image, moving_image, registered_image,
        fixed_landmarks, moving_landmarks
    )

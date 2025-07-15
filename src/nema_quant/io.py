import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
from . import utils


def load_raw_image(
    filepath: Path,
    image_dims: Tuple[int, int, int],
    dtype: np.dtype = np.dtype(np.float32)
) -> np.ndarray:
    """
    Loads a raw binary image file into a NumPy array.

    The function calculates the expected file size based on the dimensions
    and data type, providing a basic validation check.

    Parameters
    ----------
    filepath : pathlib.Path
        The path to the raw image file.
    image_dims : tuple of int, shape (3,)
        The (x, y, z) dimensions of the image.
    dtype : numpy.dtype, optional
        The data type of the pixels in the raw file (default is np.float32).

    Returns
    -------
    numpy.ndarray
        A 1D NumPy array containing the raw image data. The user is
        expected to reshape this array to the correct 3D dimensions.

    Notes
    -----
    Author: EdAlita
    Date: 2025-07-08 09:21:56
    """
    if not filepath.exists():
        raise FileNotFoundError(f"The file was not found at: {filepath}")

    # Calculate expected size
    bytes_per_pixel = np.dtype(dtype).itemsize
    expected_size = (
        image_dims[0] * image_dims[1] * image_dims[2] * bytes_per_pixel
    )
    actual_size = filepath.stat().st_size

    if expected_size != actual_size:
        raise ValueError(
            f"File size mismatch. Expected {expected_size} bytes, but found {actual_size} bytes. Check image dimensions and dtype."
        )

    return np.fromfile(filepath, dtype=dtype)


if __name__ == '__main__':
    IMAGE_DIMS_XYZ = (392, 392, 346)
    FILE_PATH_EXAMPLE = Path("data/data_orig.imgrec")

    print(f"Intentando cargar imagen raw desde: {FILE_PATH_EXAMPLE}")

    try:
        dummy_path = Path(FILE_PATH_EXAMPLE)
        if not dummy_path.exists():
            print("\nNOTA: Creando archivo de prueba para la demostración.")
            dummy_data = np.zeros(
                (IMAGE_DIMS_XYZ[2], IMAGE_DIMS_XYZ[1], IMAGE_DIMS_XYZ[0]),
                dtype=np.dtype("float32")
            )
            dummy_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_data.tofile(dummy_path)

        image_array_1d = load_raw_image(
            filepath=FILE_PATH_EXAMPLE,
            image_dims=IMAGE_DIMS_XYZ,
            dtype=np.dtype("float32")
        )
        image_array_3d = image_array_1d.reshape(
            (IMAGE_DIMS_XYZ[2], IMAGE_DIMS_XYZ[1], IMAGE_DIMS_XYZ[0])
        )

        print("\nImagen cargada exitosamente.")

        # --- Calcular y mostrar centros ---
        # 1. Centro del array (geométrico, potencialmente incorrecto)
        dim_z, dim_y, dim_x = image_array_3d.shape
        array_center_z = dim_z // 2
        array_center_y = dim_y // 2
        array_center_x = dim_x // 2
        print(
            f"Centro del Array (z,y,x):"
            f"({array_center_z}, {array_center_y}, {array_center_x})"
        )

        # 2. Centro del FANTOMA (real, usando centro de masa)
        ce_z, ce_y, ce_x = utils.find_phantom_center(image_array_3d)
        phantom_center_z = int(ce_z)
        phantom_center_y = int(ce_y)
        phantom_center_x = int(ce_x)
        print(
            f"Centro del Fantoma (z,y,x):"
            f"({phantom_center_z}, {phantom_center_y}, {phantom_center_x})"
        )

        points = [
            (161, 231), (200, 250), (200, 170),
            (155, 205), (153, 193), (162, 185),
            (235, 235), (242, 225), (245, 214),
            (245, 202), (242, 190), (235, 180)
        ]
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]

        # --- Generar un gráfico de prueba ---
        plt.figure(figsize=(12, 12))
        # Mostrar la rebanada en el centro REAL del fantoma
        plt.imshow(image_array_3d[171, :, :], cmap='gray')

        centro = (171, 211)
        plt.plot(
            centro[0], centro[1], marker='o', color='blue', markersize=17.92,
            mew=3, label=f'37 mm ({centro[0]}, {centro[1]})', linestyle='none'
        )

        centro = (184, 188)
        plt.plot(
            centro[0], centro[1], marker='o', color='blue', markersize=13.56,
            mew=3, label=f'28 mm ({centro[0]}, {centro[1]})', linestyle='none'
        )

        centro = (212, 187)
        plt.plot(
            centro[0], centro[1], marker='o', color='blue', markersize=10.65,
            mew=3, label=f'22 mm ({centro[0]}, {centro[1]})', linestyle='none'
        )

        centro = (228, 212)
        plt.plot(
            centro[0], centro[1], marker='o', color='blue', markersize=8.23,
            mew=3, label=f'17 mm ({centro[0]}, {centro[1]})', linestyle='none'
        )

        centro = (214, 238)
        plt.plot(
            centro[0], centro[1], marker='o', color='blue', markersize=6.29,
            mew=3, label=f'13 mm ({centro[0]}, {centro[1]})', linestyle='none'
        )

        centro = (186, 236)
        plt.plot(
            centro[0], centro[1], marker='o', color='blue', markersize=4.84,
            mew=3, label=f'10 mm ({centro[0]}, {centro[1]})', linestyle='none'
        )

        plt.plot(
            x_vals, y_vals, 'o', color='orange', markersize=17.92, mew=3,
            label='background rois', linestyle='none'
        )  # 'o' means circular marker

        plt.title(f"Rebanada del Fantoma (z = {171})")
        plt.xlabel("Eje X")
        plt.ylabel("Eje Y")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.grid(True, linestyle='--', alpha=0.5)

        output_filename = "data/rois_positions.png"
        plt.savefig(output_filename)

    except Exception as e:
        print(f"\nUn error inesperado ocurrió: {e}")

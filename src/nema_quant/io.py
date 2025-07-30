from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from matplotlib.patches import Circle

from . import analysis, utils


def load_nii_image(
    filepath: Path, return_affine: bool = False
) -> Tuple[npt.NDArray[Any], Optional[npt.NDArray[Any]]]:
    """
    Loads a NIfTI image file into a NumPy array using SimpleITK.

    Reads a NIfTI file (.nii or .nii.gz) and returns the image data as a NumPy array. Optionally returns the affine matrix for spatial
    transformations.

    Parameters
    ----------
    filepath : pathlib.Path
        Path to the NIfTI image file (.nii or .nii.gz).
    return_affine : bool, optional
        If True, also returns the affine transformation matrix. Default is False.

    Returns
    -------
    numpy.ndarray
        3D array of image data.
    numpy.ndarray, optional
        Affine transformation matrix (if return_affine=True).

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file cannot be loaded as a NIfTI image.

    Notes
    -----
    Author: EdAlita
    Date: 2025-07-15
    """
    if not filepath.exists():
        raise FileNotFoundError(f"The file was not found at: {filepath}")

    try:
        sitk_image = sitk.ReadImage(str(filepath))
        image_data = sitk.GetArrayFromImage(sitk_image)

        image_data = image_data.astype(np.float32)

        if return_affine:
            spacing = sitk_image.GetSpacing()
            origin = sitk_image.GetOrigin()
            direction = sitk_image.GetDirection()
            affine = np.eye(4)
            direction_matrix = np.array(direction).reshape(3, 3)
            affine[:3, :3] = direction_matrix * np.array(spacing)
            affine[:3, 3] = origin

            return image_data, affine
        else:
            return image_data, None

    except Exception as e:
        raise ValueError(f"Could not load NIfTI file {filepath}: {str(e)}")


if __name__ == "__main__":
    FILE_PATH_EXAMPLE = Path(
        "data/EARL_TORSO_CTstudy.2400s.DOI.EQZ.att_yes.frame10.subs05.nii"
    )

    print(f"Intentando cargar imagen NIfTI desde: {FILE_PATH_EXAMPLE}")

    try:
        # Load the NIfTI image
        image_array_3d, affine = load_nii_image(
            filepath=FILE_PATH_EXAMPLE, return_affine=True
        )

        print("\nImagen cargada exitosamente.")
        print(f"Dimensiones de la imagen: {image_array_3d.shape}")
        print(f"Tipo de datos: {image_array_3d.dtype}")
        print(
            f"Rango de valores: [{np.min(image_array_3d):.3f}, {np.max(image_array_3d):.3f}]"
        )
        print(f"Valores Unicos: [{np.unique(image_array_3d)}]")

        if affine is not None:
            print(f"Matriz afín disponible: {affine.shape}")

        # --- Calcular y mostrar centros ---
        # Note: NIfTI images typically have (x, y, z) ordering
        dim_z, dim_y, dim_x = image_array_3d.shape
        array_center_x = dim_x // 2
        array_center_y = dim_y // 2
        array_center_z = dim_z // 2
        print(
            f"Centro del Array (z,y,x):"
            f"({array_center_z}, {array_center_y}, {array_center_x})"
        )

        # 2. Centro del FANTOMA (real, usando centro de masa)
        ce_z, ce_y, ce_x = utils.find_phantom_center(image_array_3d)
        phantom_center_x = int(ce_x)
        phantom_center_y = int(ce_y)
        phantom_center_z = int(ce_z)
        print(
            f"Centro del Fantoma (z,y,x):"
            f"({phantom_center_z}, {phantom_center_y}, {phantom_center_x})"
        )

        # Determine which slice to visualize (middle slice in z-direction)
        center_slice = phantom_center_z if phantom_center_z < dim_z else dim_z // 2

        lung_insert_centers = utils.extract_canny_mask(image_array_3d)

        print(lung_insert_centers[50])

        # --- Generar un gráfico de prueba ---
        plt.figure(figsize=(12, 12))
        # Mostrar la rebanada en el centro REAL del fantoma
        # Note: For NIfTI, we need to adapt slice indexing based on image orientation
        plt.imshow(image_array_3d[172, :, :])

        hot_37 = analysis.extract_circular_mask_2d(
            (391, 391), (211, 171), (37 / 2) / 2.0644
        )
        centro_37 = (211, 171)
        plt.imshow(hot_37, cmap="Reds", alpha=0.1)

        centro = (187, 184)
        hot_28 = analysis.extract_circular_mask_2d(
            (391, 391), centro, (28 / 2) / 2.0644
        )
        plt.imshow(hot_28, cmap="binary", alpha=0.1)

        centro = (187, 212)
        hot_22 = analysis.extract_circular_mask_2d(
            (391, 391), centro, (22 / 2) / 2.0644
        )
        plt.imshow(hot_22, cmap="binary", alpha=0.1)

        centro = (211, 226)
        hot_17 = analysis.extract_circular_mask_2d(
            (391, 391), centro, (17 / 2) / 2.0644
        )
        plt.imshow(hot_17, cmap="binary", alpha=0.1)

        centro = (235, 212)
        hot_13 = analysis.extract_circular_mask_2d(
            (391, 391), centro, (13 / 2) / 2.0644
        )
        plt.imshow(hot_13, cmap="binary", alpha=0.1)

        centro = (235, 184)
        hot_10 = analysis.extract_circular_mask_2d(
            (391, 391), centro, (10 / 2) / 2.0644
        )
        plt.imshow(hot_10, cmap="binary", alpha=0.1)

        points = [
            (centro_37[0] - 16, centro_37[1] - 28),
            (centro_37[0] - 33, centro_37[1] - 19),
            (centro_37[0] - 40, centro_37[1] - 1),
            (centro_37[0] - 35, centro_37[1] + 28),
            (centro_37[0] - 39, centro_37[1] + 50),
            (centro_37[0] - 32, centro_37[1] + 69),
            (centro_37[0] - 15, centro_37[1] + 79),
            (centro_37[0] + 3, centro_37[1] + 76),
            (centro_37[0] + 19, centro_37[1] + 65),
            (centro_37[0] + 34, centro_37[1] + 51),
            (centro_37[0] + 38, centro_37[1] + 28),
            (centro_37[0] + 25, centro_37[1] - 3),
        ]
        x_vals = [p[1] for p in points]
        y_vals = [p[0] for p in points]

        plt.plot(
            x_vals,
            y_vals,
            "o",
            color="orange",
            markersize=31,
            mew=3,
            label="background rois",
            linestyle="none",
        )  # 'o' means circular marker

        plt.title(f"Rebanada del Fantoma (z = {172})")
        plt.xlabel("Eje X")
        plt.ylabel("Eje Y")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.gca().set_aspect("equal")
        plt.grid(True, linestyle="--", alpha=0.5)

        output_filename = "tests/rois_positions.png"
        plt.savefig(output_filename)
        print(f"\nGráfico guardado en: {output_filename}")

        # --- Mostrar la imagen base ---
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image_array_3d[172, :, :], cmap="gray", origin="lower")

        # --- Definir ROIs principales ---
        rois: List[Dict[str, Any]] = [
            {
                "center": (211, 171),
                "diameter_mm": 37,
                "color": "red",
                "alpha": 0.18,
                "label": "hot_sphere_37mm",
            },
            {
                "center": (187, 184),
                "diameter_mm": 28,
                "color": "orange",
                "alpha": 0.18,
                "label": "hot_sphere_28mm",
            },
            {
                "center": (187, 212),
                "diameter_mm": 22,
                "color": "gold",
                "alpha": 0.18,
                "label": "hot_sphere_22mm",
            },
            {
                "center": (211, 226),
                "diameter_mm": 17,
                "color": "lime",
                "alpha": 0.18,
                "label": "hot_sphere_17mm",
            },
            {
                "center": (235, 213),
                "diameter_mm": 13,
                "color": "cyan",
                "alpha": 0.18,
                "label": "hot_sphere_13mm",
            },
            {
                "center": (235, 185),
                "diameter_mm": 10,
                "color": "blue",
                "alpha": 0.18,
                "label": "hot_sphere_10mm",
            },
        ]
        pixel_spacing: float = 2.0644  # mm/pixel

        for roi in rois:
            y: int
            x: int
            y, x = roi["center"]  # Note: (y, x) order
            radius_pix: float = (roi["diameter_mm"] / 2) / pixel_spacing
            circ = Circle(
                (x, y),
                radius_pix,
                edgecolor=roi["color"],
                facecolor=roi["color"],
                alpha=roi["alpha"],
                lw=2,
                label=roi["label"],
            )
            ax.add_patch(circ)
            ax.plot(x, y, "+", color=roi["color"], markersize=12)

        # --- Dibujar background ROIs como círculos (no solo puntos) ---
        bg_offsets = [
            (-16, -28),
            (-33, -19),
            (-40, -1),
            (-35, 28),
            (-39, 50),
            (-32, 69),
            (-15, 79),
            (3, 76),
            (19, 65),
            (34, 51),
            (38, 28),
            (25, -3),
        ]
        centro_37 = (211, 171)
        bg_radius_mm = 37  # example value
        bg_radius_pix = (bg_radius_mm / 2) / pixel_spacing
        for dy, dx in bg_offsets:
            bg_y, bg_x = centro_37[0] + dy, centro_37[1] + dx
            bg_circle = Circle(
                (bg_x, bg_y),
                bg_radius_pix,
                edgecolor="orange",
                facecolor="none",
                lw=2,
                linestyle="--",
                label="Background" if (dy, dx) == bg_offsets[0] else "",
            )
            ax.add_patch(bg_circle)
            ax.plot(bg_x, bg_y, "o", color="orange", markersize=7)

        lung_circle = Circle(
            (195, 209),
            15 / 2.0644,
            edgecolor="lime",
            facecolor="none",
            lw=2,
            linestyle="--",
            label="",
        )
        ax.add_patch(lung_circle)

        # --- Leyendas y detalles ---
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="lower right",
            fontsize=12,
            framealpha=0.7,
        )
        ax.set_title("Ubicación de ROIs en el fantoma", fontsize=16)
        ax.set_xlabel("X (pixeles)")
        ax.set_ylabel("Y (pixeles)")
        ax.set_aspect("equal")
        plt.tight_layout()

        output_filename = "tests/rois_positions2.png"
        plt.savefig(output_filename)
        print(f"\nGráfico guardado en: {output_filename}")

    except Exception as e:
        print(f"\nUn error inesperado ocurrió: {e}")

"""
Input/output utilities for loading images and saving analysis results.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

# Only use QtAgg backend if not running tests (pytest sets 'Agg' backend)
if "pytest" not in sys.modules:
    matplotlib.use("QtAgg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import SimpleITK as sitk  # noqa: E402
from matplotlib.patches import Circle, Patch  # noqa: E402

from nema_quant import utils  # noqa: E402


def load_nii_image(
    filepath: Path, return_affine: bool = False
) -> Tuple[npt.NDArray[Any], Optional[npt.NDArray[Any]]]:
    """Load a NIfTI image into a NumPy array.

    Reads a .nii or .nii.gz file using SimpleITK and returns the image data as a
    3D NumPy array. Optionally returns a 4x4 affine matrix derived from spacing,
    origin, and direction.

    Parameters
    ----------
    filepath : pathlib.Path
        Path to the NIfTI image file.
    return_affine : bool, optional
        If True, also return the 4x4 affine matrix. Default is False.

    Returns
    -------
    numpy.ndarray
        3D image array (z, y, x) in float32.
    numpy.ndarray or None
        4x4 affine matrix if `return_affine` is True, otherwise None.

    Raises
    ------
    FileNotFoundError
        If `filepath` does not exist.
    ValueError
        If the file cannot be loaded as a NIfTI image.
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


def create_cylindrical_mask(
    shape_zyx: Tuple[int, int, int],
    center_zyx: Tuple[float, float, float],
    radius_mm: float,
    height_mm: float,
    spacing_xyz: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Create a boolean mask for a cylinder aligned with the z-axis."""
    center_z, center_y, center_x = center_zyx
    radius_vox_x = radius_mm / spacing_xyz[0]
    radius_vox_y = radius_mm / spacing_xyz[1]
    half_height_vox_z = (height_mm / spacing_xyz[2]) / 2.0

    z_min = max(0, int(np.floor(center_z - half_height_vox_z)))
    z_max = min(shape_zyx[0] - 1, int(np.ceil(center_z + half_height_vox_z)))

    yy, xx = np.ogrid[: shape_zyx[1], : shape_zyx[2]]
    ellipse = ((xx - center_x) / radius_vox_x) ** 2 + (
        (yy - center_y) / radius_vox_y
    ) ** 2 <= 1.0

    mask = np.zeros(shape_zyx, dtype=bool)
    mask[slice(z_min, z_max + 1), :, :] = ellipse

    return mask


if __name__ == "__main__":
    FILE_PATH_EXAMPLE = Path(
        "data/IQ.05022024.DOI.petfus.3420s.att_no.frame20.imgrec.nii"  # "data/EARL_TORSO_CTstudy.2400s.DOI.EQZ.att_yes.frame10.subs05.nii"
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

        spacing_xyz = (
            np.linalg.norm(affine[:3, :3], axis=0)
            if affine is not None
            else np.array([1.0, 1.0, 1.0])
        )
        print(f"Espaciado de vóxeles (mm): {spacing_xyz}")

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
        # center_slice = center_slice - 2

        # lung_insert_centers = utils.extract_canny_mask(image_array_3d)

        # print(lung_insert_centers[50])

        # --- Generar un gráfico con las tres vistas ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Centro del Fantoma (marcado en rojo)")

        uniformity_radius_mm = 22.5 / 2.0
        uniformity_height_mm = 10.0
        uniformity_radius_vox_x = uniformity_radius_mm / spacing_xyz[0]
        uniformity_radius_vox_y = uniformity_radius_mm / spacing_xyz[1]
        uniformity_height_vox_z = uniformity_height_mm / spacing_xyz[2]

        cylinder_mask = create_cylindrical_mask(
            (image_array_3d.shape[0], image_array_3d.shape[1], image_array_3d.shape[2]),  # type: ignore[arg-type]
            (phantom_center_z + 7, phantom_center_y, phantom_center_x),
            uniformity_radius_mm,
            uniformity_height_mm,
            spacing_xyz,
        )

        air_radius_mm = 4.0 / 2.0
        air_height_mm = 7.5
        air_radius_vox_x = air_radius_mm / spacing_xyz[0]
        air_radius_vox_y = air_radius_mm / spacing_xyz[1]
        air_height_vox_z = air_height_mm / spacing_xyz[2]

        cylinder_air = create_cylindrical_mask(
            (image_array_3d.shape[0], image_array_3d.shape[1], image_array_3d.shape[2]),  # type: ignore[arg-type]
            (phantom_center_z - 25.0, phantom_center_y - 15.0, phantom_center_x),
            air_radius_mm,
            air_height_mm,
            spacing_xyz,
        )

        water_radius_mm = 4.0 / 2.0
        water_height_mm = 7.5
        water_radius_vox_x = water_radius_mm / spacing_xyz[0]
        water_radius_vox_y = water_radius_mm / spacing_xyz[1]
        water_height_vox_z = water_height_mm / spacing_xyz[2]

        cylinder_water = create_cylindrical_mask(
            (image_array_3d.shape[0], image_array_3d.shape[1], image_array_3d.shape[2]),  # type: ignore[arg-type]
            (phantom_center_z - 25.0, phantom_center_y + 15.0, phantom_center_x),
            water_radius_mm,
            water_height_mm,
            spacing_xyz,
        )

        cylinder_values = image_array_3d[cylinder_mask]
        if cylinder_values.size > 0:
            print(
                "ROI Uniformidad:"
                f" voxels={cylinder_values.size},"
                f" mean={np.mean(cylinder_values):.6f},"
                f" std={np.std(cylinder_values):.6f}"
            )

        cylinder_air_values = image_array_3d[cylinder_air]
        if cylinder_air_values.size > 0:
            sor_air = np.mean(cylinder_air_values) / np.mean(cylinder_values)
            std_term_air = np.sqrt(
                (np.std(cylinder_air_values) / np.mean(cylinder_air_values)) ** 2
                + (np.std(cylinder_values) / np.mean(cylinder_values)) ** 2
            )
            print(
                "ROI Aire:"
                f" voxels={cylinder_air_values.size},"
                f" mean={np.mean(cylinder_air_values):.6f},"
                f" std={np.std(cylinder_air_values):.6f},"
                f" SOR={sor_air:.6f}"
                f" %STD={std_term_air:.2f}%"
            )

        cylinder_water_values = image_array_3d[cylinder_water]
        if cylinder_water_values.size > 0:
            sor_water = np.mean(cylinder_water_values) / np.mean(cylinder_values)
            std_term_water = np.sqrt(
                (np.std(cylinder_water_values) / np.mean(cylinder_water_values)) ** 2
                + (np.std(cylinder_values) / np.mean(cylinder_values)) ** 2
            )
            print(
                "ROI Agua:"
                f" voxels={cylinder_water_values.size},"
                f" mean={np.mean(cylinder_water_values):.6f},"
                f" std={np.std(cylinder_water_values):.6f},"
                f" SOR={sor_water:.6f}"
                f" %STD={std_term_water:.2f}%"
            )

        # Axial (z fijo): vista (y, x)
        axes[0].imshow(
            image_array_3d[phantom_center_z + 22 + 20, :, :],
            cmap="binary",
            origin="lower",
        )

        # axes[0].imshow(
        #     cylinder_mask[phantom_center_z, :, :],
        #     cmap="Reds",
        #     alpha=0.25,
        #     origin="lower",
        # )
        # axes[0].imshow(
        #     cylinder_air[phantom_center_z, :, :],
        #     cmap="Blues",
        #     alpha=0.25,
        #     origin="lower",
        # )
        # axes[0].imshow(
        #     cylinder_water[phantom_center_z, :, :],
        #     cmap="Greens",
        #     alpha=0.25,
        #     origin="lower",
        # )
        axes[0].plot(
            phantom_center_x, phantom_center_y, "x", color="red", markersize=12
        )
        # axes[0].add_patch(
        #     Circle(
        #         (phantom_center_x, phantom_center_y),
        #         radius=uniformity_radius_vox_x,
        #         edgecolor="red",
        #         facecolor="none",
        #         lw=1.5,
        #     )
        # )
        # axes[0].add_patch(
        #     Circle(
        #         (phantom_center_x, phantom_center_y),
        #         radius=air_radius_vox_x,
        #         edgecolor="blue",
        #         facecolor="none",
        #         lw=1.5,
        #     )
        # )
        # axes[0].add_patch(
        #     Circle(
        #         (phantom_center_x, phantom_center_y),
        #         radius=water_radius_vox_x,
        #         edgecolor="green",
        #         facecolor="none",
        #         lw=1.5,
        #     )
        # )
        # --- Definir ROIs principales ---
        rois: List[Dict[str, Any]] = [
            {
                "center": (67, 87),
                "diameter_mm": 5,
                "color": "red",
                "alpha": 0.18,
                "label": "hot_sphere_5mm",
            },
            {
                "center": (83, 88),
                "diameter_mm": 4,
                "color": "orange",
                "alpha": 0.18,
                "label": "hot_sphere_4mm",
            },
            {
                "center": (89, 73),
                "diameter_mm": 3,
                "color": "gold",
                "alpha": 0.18,
                "label": "hot_sphere_3mm",
            },
            {
                "center": (77, 62),
                "diameter_mm": 2,
                "color": "lime",
                "alpha": 0.18,
                "label": "hot_sphere_2mm",
            },
            {
                "center": (63, 71),
                "diameter_mm": 1,
                "color": "cyan",
                "alpha": 0.18,
                "label": "hot_sphere_1mm",
            },
        ]
        pixel_spacing: float = 0.5  # mm/pixel

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
            axes[0].add_patch(circ)
            axes[0].plot(x, y, "+", color=roi["color"], markersize=12)

        axes[0].set_title(f"Axial z={phantom_center_z + 22 + 20}")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")

        # Coronal (y fijo): vista (z, x)
        axes[1].imshow(
            image_array_3d[:, phantom_center_y, :], cmap="binary", origin="lower"
        )
        axes[1].imshow(
            cylinder_mask[:, phantom_center_y, :],
            cmap="Reds",
            alpha=0.25,
            origin="lower",
        )
        axes[1].imshow(
            cylinder_air[:, phantom_center_y, :],
            cmap="Blues",
            alpha=0.25,
            origin="lower",
        )
        axes[1].imshow(
            cylinder_water[:, phantom_center_y, :],
            cmap="Greens",
            alpha=0.25,
            origin="lower",
        )
        axes[1].plot(
            phantom_center_x, phantom_center_z, "x", color="red", markersize=12
        )
        # axes[1].plot(phantom_center_x, phantom_center_z + 22 + 40, "x", color="blue", markersize=12)
        # axes[1].plot(phantom_center_x, phantom_center_z + 22, "x", color="blue", markersize=12)
        # axes[1].plot(phantom_center_x, phantom_center_z + 22 + 30, "x", color="green", markersize=12)
        # axes[1].plot(phantom_center_x, phantom_center_z + 22 + 10, "x", color="green", markersize=12)
        # axes[1].plot(phantom_center_x, phantom_center_z + 22 + 20, "x", color="purple", markersize=12)
        axes[1].contour(
            cylinder_mask[:, phantom_center_y, :],
            levels=[0.5],
            colors=["red"],
            linewidths=1.5,
        )
        axes[1].contour(
            cylinder_air[:, phantom_center_y, :],
            levels=[0.5],
            colors=["blue"],
            linewidths=1.5,
        )
        axes[1].contour(
            cylinder_water[:, phantom_center_y, :],
            levels=[0.5],
            colors=["green"],
            linewidths=1.5,
        )
        axes[1].set_title(f"Coronal y={phantom_center_y}")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Z")

        # Sagital (x fijo): vista (z, y)
        axes[2].imshow(
            image_array_3d[:, :, phantom_center_x], cmap="binary", origin="lower"
        )
        axes[2].imshow(
            cylinder_mask[:, :, phantom_center_x],
            cmap="Reds",
            alpha=0.25,
            origin="lower",
        )
        axes[2].imshow(
            cylinder_air[:, :, phantom_center_x],
            cmap="Blues",
            alpha=0.25,
            origin="lower",
        )
        axes[2].imshow(
            cylinder_water[:, :, phantom_center_x],
            cmap="Greens",
            alpha=0.25,
            origin="lower",
        )
        axes[2].plot(
            phantom_center_y, phantom_center_z, "x", color="red", markersize=12
        )
        axes[2].contour(
            cylinder_mask[:, :, phantom_center_x],
            levels=[0.5],
            colors=["red"],
            linewidths=1.5,
        )
        axes[2].contour(
            cylinder_air[:, :, phantom_center_x],
            levels=[0.5],
            colors=["blue"],
            linewidths=1.5,
        )
        axes[2].contour(
            cylinder_water[:, :, phantom_center_x],
            levels=[0.5],
            colors=["green"],
            linewidths=1.5,
        )
        axes[2].set_title(f"Sagital x={phantom_center_x}")
        axes[2].set_xlabel("Y")
        axes[2].set_ylabel("Z")

        uniformity_handle = Patch(
            facecolor="red",
            edgecolor="red",
            alpha=0.25,
            label="Uniformity",
        )
        air_handle = Patch(
            facecolor="blue",
            edgecolor="blue",
            alpha=0.25,
            label="Air",
        )
        water_handle = Patch(
            facecolor="green",
            edgecolor="green",
            alpha=0.25,
            label="Water",
        )
        axes[2].legend(
            handles=[uniformity_handle, air_handle, water_handle], loc="upper right"
        )

        for ax in axes:
            ax.set_aspect("equal")
            ax.grid(True, linestyle="--", alpha=0.3)

        # hot_37 = analysis.extract_circular_mask_2d(
        #     (391, 391), (211, 171), (37 / 2) / 2.0644
        # )
        # centro_37 = (211, 171)
        # plt.imshow(hot_37, cmap="Reds", alpha=0.1)

        # centro = (187, 184)
        # hot_28 = analysis.extract_circular_mask_2d(
        #     (391, 391), centro, (28 / 2) / 2.0644
        # )
        # plt.imshow(hot_28, cmap="binary", alpha=0.1)

        # centro = (187, 212)
        # hot_22 = analysis.extract_circular_mask_2d(
        #     (391, 391), centro, (22 / 2) / 2.0644
        # )
        # plt.imshow(hot_22, cmap="binary", alpha=0.1)

        # centro = (211, 226)
        # hot_17 = analysis.extract_circular_mask_2d(
        #     (391, 391), centro, (17 / 2) / 2.0644
        # )
        # plt.imshow(hot_17, cmap="binary", alpha=0.1)

        # centro = (235, 212)
        # hot_13 = analysis.extract_circular_mask_2d(
        #     (391, 391), centro, (13 / 2) / 2.0644
        # )
        # plt.imshow(hot_13, cmap="binary", alpha=0.1)

        # centro = (235, 184)
        # hot_10 = analysis.extract_circular_mask_2d(
        #     (391, 391), centro, (10 / 2) / 2.0644
        # )
        # plt.imshow(hot_10, cmap="binary", alpha=0.1)

        # points = [
        #     (centro_37[0] - 16, centro_37[1] - 28),
        #     (centro_37[0] - 33, centro_37[1] - 19),
        #     (centro_37[0] - 40, centro_37[1] - 1),
        #     (centro_37[0] - 35, centro_37[1] + 28),
        #     (centro_37[0] - 39, centro_37[1] + 50),
        #     (centro_37[0] - 32, centro_37[1] + 69),
        #     (centro_37[0] - 15, centro_37[1] + 79),
        #     (centro_37[0] + 3, centro_37[1] + 76),
        #     (centro_37[0] + 19, centro_37[1] + 65),
        #     (centro_37[0] + 34, centro_37[1] + 51),
        #     (centro_37[0] + 38, centro_37[1] + 28),
        #     (centro_37[0] + 25, centro_37[1] - 3),
        # ]
        # x_vals = [p[1] for p in points]
        # y_vals = [p[0] for p in points]

        # plt.plot(
        #     x_vals,
        #     y_vals,
        #     "o",
        #     color="orange",
        #     markersize=31,
        #     mew=3,
        #     label="background rois",
        #     linestyle="none",
        # )  # 'o' means circular marker

        output_filename = "rois_positions.png"
        plt.tight_layout()
        plt.savefig(output_filename)
        print(f"\nGráfico guardado en: {output_filename}")

        # --- Mostrar la imagen base ---
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(
            image_array_3d[phantom_center_z + 22 + 20, :, :],
            cmap="binary",
            origin="lower",
        )

        # --- Definir ROIs principales ---
        rois: List[Dict[str, Any]] = [  # type: ignore[no-redef]
            {
                "center": (67, 87),
                "diameter_mm": 5,
                "color": "red",
                "alpha": 0.18,
                "label": "hot_sphere_5mm",
            },
            {
                "center": (83, 88),
                "diameter_mm": 4,
                "color": "orange",
                "alpha": 0.18,
                "label": "hot_sphere_4mm",
            },
            {
                "center": (89, 73),
                "diameter_mm": 3,
                "color": "gold",
                "alpha": 0.18,
                "label": "hot_sphere_3mm",
            },
            {
                "center": (77, 62),
                "diameter_mm": 2,
                "color": "lime",
                "alpha": 0.18,
                "label": "hot_sphere_2mm",
            },
            {
                "center": (63, 71),
                "diameter_mm": 1,
                "color": "cyan",
                "alpha": 0.18,
                "label": "hot_sphere_1mm",
            },
        ]
        pixel_spacing_2: float = 0.5  # mm/pixel

        for roi in rois:
            y_2: int
            x_2: int
            y_2, x_2 = roi["center"]  # Note: (y, x) order
            radius_pix_2: float = (roi["diameter_mm"] / 2) / pixel_spacing_2
            circ = Circle(
                (x_2, y_2),
                radius_pix_2,
                edgecolor=roi["color"],
                facecolor=roi["color"],
                alpha=roi["alpha"],
                lw=2,
                label=roi["label"],
            )
            ax.add_patch(circ)
            ax.plot(x_2, y_2, "+", color=roi["color"], markersize=12)

        # # --- Dibujar background ROIs como círculos (no solo puntos) ---
        # bg_offsets = [
        #     (-16, -28),
        #     (-33, -19),
        #     (-40, -1),
        #     (-35, 28),
        #     (-39, 50),
        #     (-32, 69),
        #     (-15, 79),
        #     (3, 76),
        #     (19, 65),
        #     (34, 51),
        #     (38, 28),
        #     (25, -3),
        # ]
        # centro_37 = (211, 171)
        # bg_radius_mm = 37  # example value
        # bg_radius_pix = (bg_radius_mm / 2) / pixel_spacing
        # for dy, dx in bg_offsets:
        #     bg_y, bg_x = centro_37[0] + dy, centro_37[1] + dx
        #     bg_circle = Circle(
        #         (bg_x, bg_y),
        #         bg_radius_pix,
        #         edgecolor="orange",
        #         facecolor="none",
        #         lw=2,
        #         linestyle="--",
        #         label="Background" if (dy, dx) == bg_offsets[0] else "",
        #     )
        #     ax.add_patch(bg_circle)
        #     ax.plot(bg_x, bg_y, "o", color="orange", markersize=7)

        # lung_circle = Circle(
        #     (195, 209),
        #     15 / 2.0644,
        #     edgecolor="lime",
        #     facecolor="none",
        #     lw=2,
        #     linestyle="--",
        #     label="",
        # )
        # ax.add_patch(lung_circle)

        # # --- Leyendas y detalles ---
        # handles, labels = ax.get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # ax.legend(
        #     by_label.values(),
        #     by_label.keys(),
        #     loc="lower right",
        #     fontsize=12,
        #     framealpha=0.7,
        # )
        ax.set_title("Ubicación de ROIs en el fantoma", fontsize=16)
        ax.set_xlabel("X (pixeles)")
        ax.set_ylabel("Y (pixeles)")
        ax.set_aspect("equal")
        plt.tight_layout()

        output_filename = "rois_positions2.png"
        plt.savefig(output_filename)
        print(f"\nGráfico guardado en: {output_filename}")

    except Exception as e:
        print(f"\nUn error inesperado ocurrió: {e}")

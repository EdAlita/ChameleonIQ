import logging
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from scipy.ndimage import binary_fill_holes, center_of_mass, gaussian_filter
from scipy.ndimage import label as ndimage_label

logger = logging.getLogger(__name__)


def find_phantom_center(
    image_data_3d: npt.NDArray[Any], threshold: float = 0.003
) -> Tuple[float, float, float]:
    """
    Finds the center of the phantom using a robust morphological approach.

    Parameters
    ----------
    image_data_3d : np.ndarray
        The image data as a 3D NumPy array (z, y, x).
    threshold : float, optional
        Value used to determine the threshold for segmentation. Default is 0.003.

    Returns
    -------
    tuple of float
        The centroid coordinates (z, y, x) of the phantom.

    Notes
    -----
    Author: EdAlita
    Date: 2025-07-08 09:32:08
    """
    if image_data_3d.ndim != 3:
        raise ValueError("La imagen de entrada debe ser un array 3D (z,y,x).")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f" Binary th: {threshold:06f}")
    binary_mask = image_data_3d > threshold
    labeled_mask, num_features = ndimage_label(binary_mask)  # type: ignore[misc]
    if num_features == 0:
        raise RuntimeError(
            "No se pudo encontrar ningún objeto" "en la imagen con el umbral actual."
        )

    largest_label = np.argmax(np.bincount(labeled_mask.ravel())[1:]) + 1
    phantom_mask = labeled_mask == largest_label

    centroid_zyx_raw = tuple(float(x) for x in center_of_mass(phantom_mask))

    if not isinstance(centroid_zyx_raw, tuple) or len(centroid_zyx_raw) != 3:
        raise RuntimeError("El centroide calculado no tiene 3 dimensiones.")

    centroid_zyx: Tuple[float, float, float] = (
        float(centroid_zyx_raw[0]),
        float(centroid_zyx_raw[1]),
        float(centroid_zyx_raw[2]),
    )

    return centroid_zyx


def voxel_to_mm(
    voxel_indices_zyx: Tuple[int, int, int],
    image_dims_xyz: Tuple[int, int, int],
    voxel_spacing_xyz: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """
    Converts voxel indices (order z, y, x) to physical coordinates (mm, relative to the center).

    This function is considered the "ground truth" for conversion. It assumes that the voxel coordinate represents the center of that voxel.

    Parameters
    ----------
    voxel_indices_zyx : tuple of int
        The voxel indices in (z, y, x) order.
    image_dims_xyz : tuple of int
        The total image dimensions in voxels (dim_x, dim_y, dim_z).
    voxel_spacing_xyz : tuple of float
        The voxel size in millimeters (spacing_x, spacing_y, spacing_z).

    Returns
    -------
    tuple of float
        The physical coordinates (x, y, z) in millimeters from the center.
    """
    center_vox_x = (image_dims_xyz[0] - 1) / 2.0
    center_vox_y = (image_dims_xyz[1] - 1) / 2.0
    center_vox_z = (image_dims_xyz[2] - 1) / 2.0

    offset_vox_x = voxel_indices_zyx[2] - center_vox_x
    offset_vox_y = voxel_indices_zyx[1] - center_vox_y
    offset_vox_z = voxel_indices_zyx[0] - center_vox_z

    mm_x = offset_vox_x * voxel_spacing_xyz[0]
    mm_y = offset_vox_y * voxel_spacing_xyz[1]
    mm_z = offset_vox_z * voxel_spacing_xyz[2]

    return (mm_x, mm_y, mm_z)


def mm_to_voxel(
    mm_coords: Tuple[float, float, float],
    image_dims_xyz: Tuple[int, int, int],
    voxel_spacing_xyz: Tuple[float, float, float],
) -> Tuple[int, int, int]:
    """
    Converts physical coordinates (in mm, relative to the center) to the indices of the nearest voxel.

    This is the inverse function of voxel_to_mm.

    Parameters
    ----------
    mm_coords : tuple of float
        The coordinates (x, y, z) in millimeters from the center.
    image_dims_xyz : tuple of int
        The total image dimensions in voxels (dim_x, dim_y, dim_z).
    voxel_spacing_xyz : tuple of float
        The voxel size in millimeters (spacing_x, spacing_y, spacing_z).

    Returns
    -------
    tuple of int
        The corresponding voxel indices in (z, y, x) order.

    Notes
    -----
    Author: EdAlita
    Date: 2025-07-08 09:13:50
    """
    center_vox_x = (image_dims_xyz[0] - 1) / 2.0
    center_vox_y = (image_dims_xyz[1] - 1) / 2.0
    center_vox_z = (image_dims_xyz[2] - 1) / 2.0

    offset_vox_x = mm_coords[0] / voxel_spacing_xyz[0]
    offset_vox_y = mm_coords[1] / voxel_spacing_xyz[1]
    offset_vox_z = mm_coords[2] / voxel_spacing_xyz[2]

    final_vox_x = int(np.round(center_vox_x + offset_vox_x))
    final_vox_y = int(np.round(center_vox_y + offset_vox_y))
    final_vox_z = int(np.round(center_vox_z + offset_vox_z))

    return (final_vox_z, final_vox_y, final_vox_x)


def extract_canny_mask(
    image: npt.NDArray[Any],
    voxel_size: float = 2.0644,
    fantoma_z_center: int = 157,
    phantom_center_yx: Optional[Tuple[int, int]] = None,
) -> npt.NDArray[Any]:
    """
    Extracts the lung insert mask using Canny edge detection, with anatomically consistent positioning.

    Parameters
    ----------
    image : np.ndarray
        3D image array.
    voxel_size : float
        Voxel size in millimeters. Default is 2.0644.
    fantoma_z_center : int
        Z-coordinate of the phantom center. Default is 157.
    phantom_center_yx : Tuple[int, int], optional
        Y, X coordinates of the phantom center for reference.

    Returns
    -------
    np.ndarray
        Binary mask array corresponding to the lung insert region.
    """
    pixel_distance = int(56 / voxel_size)
    lung_insert_pixel_distance = (
        fantoma_z_center - pixel_distance,
        fantoma_z_center + pixel_distance,
    )

    if logger.isEnabledFor(logging.DEBUG):
        logging.debug(f" Image size {image.shape}")
        logging.debug(f" Expected lung insert Z range: {lung_insert_pixel_distance}")

    if phantom_center_yx is None:
        mid_slice = image[fantoma_z_center]
        smoothed_mid = gaussian_filter(mid_slice, sigma=2.0)
        phantom_mask = smoothed_mid > (np.max(smoothed_mid) * 0.3)
        phantom_center_raw = center_of_mass(phantom_mask)
        phantom_center_yx = (int(phantom_center_raw[0]), int(phantom_center_raw[1]))

    if logger.isEnabledFor(logging.DEBUG):
        logging.debug(f" Phantom center (Y, X): {phantom_center_yx}")

    lung_centers = []
    all_slices = list(
        range(lung_insert_pixel_distance[0], lung_insert_pixel_distance[1] + 1)
    )

    search_radius_pixels = int(30 / voxel_size)

    for z in all_slices:
        axial_slice = image[z]

        smoothed = gaussian_filter(axial_slice, sigma=3.0)

        phantom_mask_smooth = smoothed > (np.max(smoothed) * 0.4)
        filled_phantom = binary_fill_holes(phantom_mask_smooth)

        y_min = max(0, phantom_center_yx[0] - search_radius_pixels)
        y_max = min(axial_slice.shape[0], phantom_center_yx[0] + search_radius_pixels)
        x_min = max(0, phantom_center_yx[1] - search_radius_pixels)
        x_max = min(axial_slice.shape[1], phantom_center_yx[1] + search_radius_pixels)

        roi_slice = smoothed[y_min:y_max, x_min:x_max]
        roi_phantom = filled_phantom[y_min:y_max, x_min:x_max]

        lung_insert_detected = False
        centroid_y, centroid_x = None, None

        if np.sum(roi_phantom) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            roi_phantom_uint8 = (roi_phantom * 255).astype(np.uint8)
            eroded_phantom = cv2.erode(roi_phantom_uint8, kernel, iterations=3)

            if np.sum(eroded_phantom) > 100:
                masked_roi = np.where(eroded_phantom > 0, roi_slice, np.max(roi_slice))

                lung_threshold = np.percentile(
                    masked_roi[eroded_phantom > 0], 15
                )  # Bottom 15%
                lung_candidates = (masked_roi <= lung_threshold) & (eroded_phantom > 0)

                if np.sum(lung_candidates) > 50:
                    lung_labels, num_regions = ndimage_label(lung_candidates)  # type: ignore[misc]

                    if num_regions > 0:
                        best_score = -1
                        best_centroid = None

                        expected_y = roi_slice.shape[0] // 2  # Center of ROI
                        expected_x = roi_slice.shape[1] // 2

                        for region_label in range(1, num_regions + 1):
                            region_mask = lung_labels == region_label
                            region_size = np.sum(region_mask)

                            if region_size < 50 or region_size > 2000:
                                continue

                            region_centroid = center_of_mass(region_mask)
                            if np.isnan(region_centroid[0]) or np.isnan(
                                region_centroid[1]
                            ):
                                continue

                            dist_from_center = np.sqrt(
                                (region_centroid[0] - expected_y) ** 2
                                + (region_centroid[1] - expected_x) ** 2
                            )

                            region_uint8 = (region_mask * 255).astype(np.uint8)
                            contours, _ = cv2.findContours(
                                region_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )

                            circularity = 0.0
                            if contours:
                                contour = contours[0]
                                area = cv2.contourArea(contour)
                                perimeter = cv2.arcLength(contour, True)
                                if perimeter > 0:
                                    circularity = (
                                        4 * np.pi * area / (perimeter * perimeter)
                                    )

                            score = circularity * (1 / (1 + dist_from_center * 0.1))

                            if score > best_score and circularity > 0.3:
                                best_score = score
                                best_centroid = region_centroid

                        if best_centroid is not None:
                            centroid_y = int(best_centroid[0]) + y_min
                            centroid_x = int(best_centroid[1]) + x_min
                            lung_insert_detected = True
                            if logger.isEnabledFor(logging.DEBUG):
                                logging.debug(
                                    f"  Slice {z}: Lung insert detected at ({centroid_y}, {centroid_x}), score: {best_score:.3f}"
                                )

        if lung_insert_detected and centroid_y is not None and centroid_x is not None:
            lung_centers.append((z, centroid_y, centroid_x))
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logging.debug(f"  Slice {z}: No valid lung insert detected")

    if lung_centers:
        y_coords = [center[1] for center in lung_centers]
        x_coords = [center[2] for center in lung_centers]

        median_y = int(np.median(y_coords))
        median_x = int(np.median(x_coords))

        outlier_threshold = 10
        filtered_centers = []
        for center in lung_centers:
            if (
                abs(center[1] - median_y) <= outlier_threshold
                and abs(center[2] - median_x) <= outlier_threshold
            ):
                filtered_centers.append(center)

        if filtered_centers:
            final_avg_y = int(np.mean([center[1] for center in filtered_centers]))
            final_avg_x = int(np.mean([center[2] for center in filtered_centers]))

            if logger.isEnabledFor(logging.DEBUG):
                logging.debug(
                    f"  Filtered {len(lung_centers) - len(filtered_centers)} outliers"
                )
                logging.debug(
                    f"  Final lung insert center (Y, X): ({final_avg_y}, {final_avg_x})"
                )

            complete_centers = []
            for z in all_slices:
                complete_centers.append((z, final_avg_y, final_avg_x))

            return np.array(complete_centers)

    logging.warning("\nNo reliable lung insert detection - using anatomical fallback")
    fallback_y = phantom_center_yx[0] + int(15 / voxel_size)  # 15mm offset
    fallback_x = phantom_center_yx[1]

    complete_centers = [(z, fallback_y, fallback_x) for z in all_slices]
    return np.array(complete_centers)


def calculate_weighted_cbr_fom(results):
    """
    Calcula el CBR y FOM ponderados para una lista de resultados tipo:
    {
        "diameter_mm": float,
        "percentaje_constrast_QH": float,
        "background_variability_N": float,
        "avg_hot_counts_CH": float,
        "avg_bkg_counts_CB": float,
        "bkg_std_dev_SD": float,
    }
    Devuelve un diccionario con CBR y FOM ponderados y listas individuales.
    """
    if not results:
        return {"weighted_CBR": None, "weighted_FOM": None}

    # Extrae valores
    diameters = [r["diameter_mm"] for r in results]
    contrasts = [r["percentaje_constrast_QH"] for r in results]
    variabilities = [r["background_variability_N"] for r in results]

    # Pesos: inverso del diámetro, normalizado
    weights = [1 / d for d in diameters]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Cálculo de CBR y FOM por diámetro
    CBRs = [c / v if v != 0 else 0 for c, v in zip(contrasts, variabilities)]
    FOMs = [(c**2) / v if v != 0 else 0 for c, v in zip(contrasts, variabilities)]

    weighted_CBR = sum(w * cbr for w, cbr in zip(weights, CBRs))
    weighted_FOM = sum(w * fom for w, fom in zip(weights, FOMs))

    return {
        "weighted_CBR": weighted_CBR,
        "weighted_FOM": weighted_FOM,
        "CBRs": CBRs,
        "FOMs": FOMs,
        "weights": weights,
        "diameters": diameters,
    }


if __name__ == "__main__":
    # --- Ejemplo de Uso y Verificación con los valores del usuario ---
    dims = (391, 391, 346)
    spacing = (2.0644, 2.0644, 2.0644)

    center_voxel_indices = mm_to_voxel((0.0, 0.0, 0.0), dims, spacing)

    print(f"Dimensiones de la imagen (x,y,z): {dims}")
    print(f"Espaciado de vóxel (x,y,z): {spacing}")
    print("-" * 30)

    # --- Prueba de conversión: mm -> vóxel ---
    print(
        f"El vóxel más cercano al centro físico (0,0,0) es:"
        f" {center_voxel_indices} (z,y,x)"
    )

    # --- Prueba de conversión inversa: vóxel -> mm ---
    print(
        f"\nConvirtiendo los índices del vóxel central"
        f"{center_voxel_indices} de vuelta a mm..."
    )
    calculated_mm = voxel_to_mm(center_voxel_indices, dims, spacing)
    print(
        f"  -> Coordenada calculada: ({calculated_mm[0]:.2f}, "
        f"{calculated_mm[1]:.2f}, {calculated_mm[2]:.2f}) (x,y,z)"
    )

    # --- Verificación de simetría ---
    print("\nVerificando que las conversiones" " son inversas la una de la otra...")
    test_mm = (-50.00, 34.00, 0.00)
    print(f"Test 1: {test_mm} -> Vóxel (z,y,x) -> mm (x,y,z)")
    voxel_result = mm_to_voxel(test_mm, dims, spacing)
    mm_result = voxel_to_mm(voxel_result, dims, spacing)
    print(
        f"  {test_mm} -> {voxel_result} -> "
        f"({mm_result[0]:.2f}, {mm_result[1]:.2f}, {mm_result[2]:.2f})"
    )
    if not np.allclose(test_mm, mm_result, atol=1.1):
        raise ValueError(
            "Conversión inversa falló: los valores no coinciden dentro de la tolerancia"
        )
    print("  -> Verificación exitosa!")

    test_voxel = (169, 211, 171)
    print(f"Test 2: {test_voxel} -> mm -> Vóxel")
    mm_result_2 = voxel_to_mm(test_voxel, dims, spacing)
    voxel_result_2 = mm_to_voxel(mm_result_2, dims, spacing)
    print(f"  {test_voxel} -> {mm_result_2} -> {voxel_result_2}")
    if test_voxel != voxel_result_2:
        raise ValueError(
            f"Conversión inversa falló: esperado {test_voxel}, obtenido {voxel_result_2}"
        )
    print("  -> Verificación exitosa!")

    print("37 mm")
    print(mm_to_voxel((58.84, 23.74, -30.97), dims, spacing))
    print("28 mm")
    print(mm_to_voxel((29.93, -25.80, 30.97), dims, spacing))
    print("22 mm")
    print(mm_to_voxel((-25.80, -25.80, 30.97), dims, spacing))
    print("17 mm")
    print(mm_to_voxel((-56.77, 25.80, 30.97), dims, spacing))
    print("13 mm")
    print(mm_to_voxel((-25.80, 75.35, 30.97), dims, spacing))
    print("10 mm")
    print(mm_to_voxel((29.93, 77.41, 30.97), dims, spacing))

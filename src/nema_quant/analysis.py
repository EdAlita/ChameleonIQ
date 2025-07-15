import yacs.config
import numpy as np
from typing import Tuple, Dict, List, Any
from .phantom import NemaPhantom


def extract_circular_mask_2d(
    slice_dims: Tuple[int, int],
    roi_center_vox: Tuple[float, float],
    roi_radius_vox: float
) -> np.ndarray:
    """
    Creates a 2D boolean mask for a circular ROI on a single slice.

    This function is highly efficient as it avoids loops. It first creates
    two 1D arrays representing all y and x coordinates in the slice. It then
    calculates the squared distance of every pixel from the given center
    and returns a 2D boolean array where True indicates the pixel is inside
    the circle.

    Parameters
    ----------
    slice_dims : tuple of int, shape (2,)
        The (y, x) dimensions of the 2D slice.
    roi_center_vox : tuple of float, shape (2,)
        The (y, x) coordinates of the circle's center in voxels.
    roi_radius_vox : float
        The radius of the circle in voxels.

    Returns
    -------
    numpy.ndarray
        A 2D boolean NumPy array where pixels inside the circle are True.
    """
    y_coords, x_coords = np.ogrid[:slice_dims[0], :slice_dims[1]]
    center_y, center_x = roi_center_vox

    squared_dist = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2

    return squared_dist <= roi_radius_vox ** 2


def _calculate_background_stats(
    image_data: np.ndarray,
    phantom: NemaPhantom,
    slices_indices: List[int],
    centers: List[Tuple[int, int]]
) -> Dict[int, Dict[str, float]]:
    """
    Internal function to calculate background mean (C_B) and std dev (SD_B)
    """
    slices_dims_yx = (image_data.shape[1], image_data.shape[2])
    bkg_counts_per_size: Dict[int, List[float]] = {
        int(round(s['diameter'])):
        [] for s in phantom.rois.values() if "sphere" not in s['name']
    }

    for x, y in centers:
        for sphere_name, sphere_def in phantom.rois.items():
            if "sphere" not in sphere_name:
                continue

            roi_mask = extract_circular_mask_2d(
                slices_dims_yx, (y, x), sphere_def['radius_vox']
            )

            for slices_idx in slices_indices:
                if 0 <= slices_idx < image_data.shape[0]:
                    img_slice = image_data[slices_idx, :, :]
                    avg_count = np.mean(img_slice[roi_mask])
                    sphere_diam_mm = int(round(sphere_def['diameter']))
                    bkg_counts_per_size[sphere_diam_mm].append(avg_count)

    bkg_stats = {}
    for diam, counts_list in bkg_counts_per_size.items():
        bkg_stats[diam] = {
            'C_B': float(np.mean(counts_list)),
            'SD_B': float(np.std(counts_list))
        }

    return bkg_stats


def _calculate_hot_sphere_counts(
    image_data: np.ndarray,
    phantom: NemaPhantom,
    central_slice_idx: int
) -> Dict[str, float]:
    """Internal function to calculate the mean
    counts (C_H) for each hot rod."""

    hot_sphere_counts = {}
    central_slice = image_data[central_slice_idx, :, :]
    slice_dims_yx = central_slice.shape

    for name, sphere_def in phantom.rois.items():
        if "sphere" not in name:
            continue

        center_yx = (sphere_def['center_vox'][1], sphere_def['center_vox'][0])
        roi_mask = extract_circular_mask_2d(
            slice_dims_yx, center_yx, sphere_def['radius_vox']
        )

        hot_sphere_counts[name] = np.mean(central_slice[roi_mask])

    return hot_sphere_counts


def calculate_nema_metrics(
    image_data: np.ndarray,
    phantom: NemaPhantom,
    cfg: yacs.config.CfgNode
) -> List[Dict[str, Any]]:
    """
    Calculates NEMA Image Quality metrics: Percent Contrast (Q_H) and
    Background Variability (N).

    This function orchestrates the analysis by calling helper functions to
    measure background and hot sphere regions, then computes the final
    metrics as defined in the NEMA NU 2-2018 standard.

    Parameters
    ----------
    image_data : np.ndarray
        The 3D image data array with shape (z, y, x).
    phantom : NemaPhantom
        An initialized NemaPhantom object with ROI definitions.
    cfg : yacs.config.CfgNode
        Configuration parameters for dataset processing.

    Returns
    -------
    list of dict
        A list containing the calculated metrics for each sphere.

    Notes
    -----
    Author: EdAlita
    Date: 2025-07-08 06:47:01
    """
    activity_hot = cfg.ACTIVITY.HOT
    activity_bkg = cfg.ACTIVITY.BACKGROUND

    if activity_bkg <= 0 or (activity_hot / activity_bkg) <= 1:
        raise ValueError(
            "Activity ratio (a_H / a_B) must be greater than 1 and"
            "background activity must be positive."
        )

    central_slices_idx = cfg.ROIS.CENTRAL_SLICE
    cm_in_z_vox = phantom._mm_to_voxels(10, 2)
    slices_indices = sorted(list({
        central_slices_idx,
        int(round(central_slices_idx + cm_in_z_vox)),
        int(round(central_slices_idx - cm_in_z_vox)),
        int(round(central_slices_idx + 2 * cm_in_z_vox)),
        int(round(central_slices_idx - 2 * cm_in_z_vox))
    }))

    background_stats = _calculate_background_stats(
        image_data, phantom, slices_indices, cfg.ROIS.BACKGROUND_CENTER_YX
    )

    hot_sphere_counts = _calculate_hot_sphere_counts(
        image_data, phantom, central_slices_idx
    )

    results = []
    activity_ratio_term = (activity_hot / activity_bkg) - 1.0

    for name, C_H in hot_sphere_counts.items():
        sphere_def = phantom.get_roi(name)
        if sphere_def is None:
            continue
        sphere_diam_mm = int(round(sphere_def['diameter']))

        C_B = background_stats[sphere_diam_mm]['C_B']
        SD_B = background_stats[sphere_diam_mm]['SD_B']

        percent_contrast = ((C_H / C_B) - 1.0) / activity_ratio_term * 100.0
        percent_variability = (SD_B / C_B) * 100.0

        results.append(
            {
                'diameter_mm': sphere_diam_mm,
                'percentaje_constrast_QH': percent_contrast,
                'background_variability_N': percent_variability,
                'avg_hot_counts_CH': C_H,
                'avg_bkg_counts_CB': C_B,
                'bkg_std_dev_SD': SD_B,
            }
        )

    return results

"""
Core analysis functions for image quality quantification.

Author: Edwing Ulin-Briseno
Date: 2025-07-16
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import yacs.config

from .metrics import get_values
from .phantom import NemaPhantom
from .statistics import Estimator
from .utils import (
    extract_canny_mask,
    find_phantom_center,
    find_phantom_center_cv2_threshold,
)

_logger = logging.getLogger(__name__)


def _propagate_ratio(
    mu1: float,
    mu2: float,
    var1: float,
    var2: float,
    cov12: float = 0.0,
    eps: float = 1e-12,
) -> float:
    """
    First-order Taylor variance propagation for ratio mu1 / mu2.
    Returns standard deviation.
    """

    if abs(mu2) < eps:
        return 0.0

    grad1 = 1.0 / mu2
    grad2 = -mu1 / (mu2**2)

    var = grad1**2 * var1 + grad2**2 * var2 + 2.0 * grad1 * grad2 * cov12

    return float(np.sqrt(max(var, 0.0)))


def extract_circular_mask_2d(
    slice_dims: Tuple[int, int],
    roi_center_vox: Tuple[float, float],
    roi_radius_vox: float,
) -> npt.NDArray[np.bool_]:
    """Create a 2D circular region-of-interest (ROI) mask.

    Generates a 2D boolean mask for a circular ROI using efficient NumPy vectorization
    instead of loops. Computes the Euclidean distance from each pixel to the circle's
    center and marks pixels within the radius as True.

    Parameters
    ----------
    slice_dims : tuple[int, int]
        (height, width) dimensions of the 2D slice.
    roi_center_vox : tuple[float, float]
        (y, x) coordinates of the circle's center in voxels.
    roi_radius_vox : float
        Radius of the circular ROI in voxels.

    Returns
    -------
    numpy.ndarray
        2D boolean mask with shape matching slice_dims. True marks pixels inside
        the circular ROI, False outside.

    Examples
    --------
    Create a mask for a 100x100 pixel slice with a circle of radius 20 at center (50, 50):

    >>> mask = extract_circular_mask_2d((100, 100), (50.0, 50.0), 20.0)
    >>> mask.shape
    (100, 100)
    >>> mask.sum()  # Number of pixels inside circle
    1257

    Notes
    -----
    Uses NumPy's ogrid for efficient coordinate generation and vectorized distance
    computation for optimal performance on large images.
    """
    y_coords, x_coords = np.ogrid[: slice_dims[0], : slice_dims[1]]
    center_y, center_x = roi_center_vox

    squared_dist = (y_coords - center_y) ** 2 + (x_coords - center_x) ** 2

    return squared_dist <= roi_radius_vox**2


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


def _calculate_background_stats(
    image_data: npt.NDArray[Any],
    phantom: NemaPhantom,
    slices_indices: List[int],
    centers_offset: List[Tuple[int, int]],
    save_visualizations: bool = False,
    viz_dir: Optional[Path] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Internal function to calculate background mean (C_B) and std dev (SD_B)
    """
    reference_sphere = None

    for name, sphere_def in phantom.rois.items():
        if "sphere" in name:
            reference_sphere = sphere_def
            break

    if reference_sphere is None:
        raise ValueError("No sphere ROI found in phantom")

    pivot_point_yx = reference_sphere["center_vox"]
    slices_dims_yx = (image_data.shape[1], image_data.shape[2])

    bkg_voxels_per_size: Dict[int, List[float]] = {}

    for name, _ in phantom.rois.items():
        if "sphere" in name:
            sphere_roi = phantom.get_roi(name)
            if sphere_roi:
                diam_mm = int(round(sphere_roi["diameter"]))
                if diam_mm not in bkg_voxels_per_size:
                    bkg_voxels_per_size[diam_mm] = []

    if save_visualizations and viz_dir:
        central_slice = image_data[slices_indices[len(slices_indices) // 2], :, :]
        ref_radius = None
        for name, _ in phantom.rois.items():
            if "sphere" in name:
                sphere_roi = phantom.get_roi(name)
                if sphere_roi:
                    ref_radius = sphere_roi["radius_vox"]
                    break

        if ref_radius:
            save_background_visualization(
                central_slice,
                centers_offset,
                pivot_point_yx,
                ref_radius,
                viz_dir,
                slices_indices[len(slices_indices) // 2],
            )

    for y_offset, x_offset in centers_offset:
        for name, _ in phantom.rois.items():
            if "sphere" not in name:
                continue

            sphere_roi = phantom.get_roi(name)
            if sphere_roi is None:
                continue

            roi_mask = extract_circular_mask_2d(
                slices_dims_yx,
                (pivot_point_yx[0] + y_offset, pivot_point_yx[1] + x_offset),
                sphere_roi["radius_vox"],
            )

            for slice_idx in slices_indices:
                if 0 <= slice_idx < image_data.shape[0]:
                    img_slice = image_data[slice_idx, :, :]
                    sphere_diam_mm = int(round(sphere_roi["diameter"]))
                    voxels = img_slice[roi_mask]
                    bkg_voxels_per_size[sphere_diam_mm].extend(voxels.tolist())  # type: ignore[arg-type]
    bkg_stats = {}
    for diam, voxels_list in bkg_voxels_per_size.items():
        if len(voxels_list) > 0:
            voxels_array = np.array(voxels_list)  # Convert list of scalars to array
            bkg_stats[diam] = {
                "C_B": float(np.mean(voxels_array)),
                "SD_B": float(np.std(voxels_array, ddof=1)),
                "n_B": len(voxels_list),
            }
        else:
            bkg_stats[diam] = {"C_B": 100.0, "SD_B": 0.0, "n_B": 0}

    return bkg_stats


def _calculate_hot_sphere_counts_offset_zxy(
    image_data: npt.NDArray[Any],
    phantom: NemaPhantom,
    central_slice_idx: int,
    save_visualizations: bool = False,
    viz_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:  # Changed return type
    """Internal function to calculate the mean counts (C_H) for each hot sphere."""

    offsets_xy = [(dy, dx) for dy in range(-10, 11) for dx in range(-10, 11)]
    offsets_z = list(range(-10, 11))

    hot_sphere_counts = {}

    z_dim, y_dim, x_dim = image_data.shape

    for name, _ in phantom.rois.items():

        if "sphere" not in name:
            continue

        sphere_roi = phantom.get_roi(name)
        if sphere_roi is None:
            continue

        center_y, center_x = sphere_roi["center_vox"]
        radius = sphere_roi["radius_vox"]

        diameter = int(np.ceil(radius * 2))
        mask_size = diameter if diameter % 2 == 1 else diameter + 1

        base_mask = extract_circular_mask_2d(
            (mask_size, mask_size),
            (mask_size // 2, mask_size // 2),
            radius,
        )

        mask_half = mask_size // 2

        max_mean = -np.inf
        best_std = 0.0
        best_n = 1
        best_mask = None
        best_slice = None
        best_offset = None
        for dz in offsets_z:

            z_idx = central_slice_idx + dz
            if z_idx < 0 or z_idx >= z_dim:
                continue

            current_slice = image_data[z_idx]

            for dy, dx in offsets_xy:

                target_y = int(round(center_y + dy))
                target_x = int(round(center_x + dx))

                y_min = target_y - mask_half
                y_max = target_y + mask_half + 1
                x_min = target_x - mask_half
                x_max = target_x + mask_half + 1

                if y_min < 0 or x_min < 0 or y_max > y_dim or x_max > x_dim:
                    continue

                shifted_region = current_slice[y_min:y_max, x_min:x_max]

                values = shifted_region[base_mask]

                if values.size == 0:
                    continue

                mean_val = np.mean(values)

                if mean_val > max_mean:
                    max_mean = mean_val
                    best_std = np.std(values, ddof=1)
                    best_n = values.size
                    best_mask = base_mask
                    best_slice = shifted_region
                    best_offset = (dz, dy, dx)

        if max_mean == -np.inf:
            _logger.warning(f"No valid ROI found for {name}")
            hot_sphere_counts[name] = {
                "mean": 0.0,
                "std_H": 0.0,
                "n_H": 1,
            }
            continue

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                f"{name}: best offset {best_offset}, "
                f"mean={max_mean:.3f}, std={best_std:.3f}, n={best_n}"
            )

        hot_sphere_counts[name] = {
            "mean": float(max_mean),
            "std_H": float(best_std),
            "n_H": int(best_n),
        }

        if save_visualizations and viz_dir and best_slice is not None:
            save_sphere_visualization(
                best_slice,
                name,
                (mask_half, mask_half),
                radius,
                best_mask if best_mask is not None else np.zeros_like(best_slice, dtype=bool),  # type: ignore[arg-type]
                viz_dir,
                central_slice_idx,
            )

    return hot_sphere_counts


def _calculate_crc_std(
    image_data: npt.NDArray[Any],
    phantom: NemaPhantom,
    central_slice_idx: int,
    measure_in_px: int,
    uniform_region_mask: npt.NDArray[Any],
    cfg: yacs.config.CfgNode,
) -> Dict[str, Dict[str, Any]]:  # type: ignore[return-value]
    """
    Calculate Recovery Coefficients following NEMA NU4-2008 standard.

    Per NEMA NU4-2008:
    - "Averaging the voxels along the axial axis and determining the maximum
       average activity inside each VOI."
    - "Five VOIs with a diameter TWICE the rod diameter and length of 10 mm."

    Important: The phantom config specifies the rod diameter, but per NEMA standard
    the VOI must be TWICE that diameter. This is critical for accurate RC calculation.

    Algorithm:
    1. Create VOI with diameter = 2 × rod diameter (radius = 2 × rod radius)
    2. For each (y,x) position in the VOI, create an axial line profile (along z)
    3. Find the (y,x) position with maximum average along z-axis
    4. Extract that specific line profile
    5. Compute mean and std dev from that line profile only

    This differs from computing over entire 2D slices and reduces std dev by
    focusing on axial uniformity at the peak location.
    """

    mode = cfg.STATISTICS.MODE
    eps = getattr(cfg.STATISTICS, "EPSILON", 1e-12)

    uniform_values = image_data[uniform_region_mask]
    uniform_est = Estimator.from_samples(uniform_values, mode=mode)

    _logger.debug(
        " RC params: center=%d, measure_in_px=%d, z_dim=%d",
        central_slice_idx,
        measure_in_px,
        image_data.shape[0],
    )

    num_slices = [
        i
        for i in range(
            central_slice_idx - measure_in_px, central_slice_idx + measure_in_px + 1
        )
        if 0 <= i < image_data.shape[0]
    ]
    _logger.debug(f" RC slices in bounds: {num_slices}")

    crc_results: Dict[str, Dict[str, Any]] = {}

    for name, _ in phantom.rois.items():
        sphere_roi = phantom.get_roi(name)

        if sphere_roi is None:
            continue

        # NEMA NU4-2008: VOI diameter must be TWICE the rod diameter
        # Config contains rod diameter, so VOI radius = 2 × rod radius
        voi_radius_vox = sphere_roi["radius_vox"] * 2.0

        y, x = sphere_roi["center_vox"]

        # Create 2D mask for the ROD VOI
        roi_mask = extract_circular_mask_2d(
            (image_data.shape[1], image_data.shape[2]),
            sphere_roi["center_vox"],
            voi_radius_vox,
        )

        # Get (y,x) coordinates where mask is True
        y_coords, x_coords = np.where(roi_mask)

        if len(y_coords) == 0:
            _logger.warning(f"  No voxels found in ROI mask for {name}")
            crc_results[name] = {  # type: ignore[index]
                "mean_signal": 0.0,
                "std_signal": 0.0,
                "uniform_mean": 0.0,  # type: ignore[name-defined]
                "uniform_std": 0.0,  # type: ignore[name-defined]
                "recovery_coeff": 0.0,
                "percentage_STD_rc": 0.0,
                "cError": 0.0,
            }
            continue

        _logger.debug(
            f" Calculating RC for {name}:"
            f" rod_diameter={sphere_roi['diameter']}mm,"  # type: ignore[index]
            f" voi_radius_vox={voi_radius_vox:.3f},"
            f" mask_pixels={len(y_coords)}"
        )

    for name, _ in phantom.rois.items():

        sphere_roi = phantom.get_roi(name)
        if sphere_roi is None:
            continue

        voi_radius_vox = sphere_roi["radius_vox"] * 2.0

        roi_mask = extract_circular_mask_2d(
            (image_data.shape[1], image_data.shape[2]),
            sphere_roi["center_vox"],
            voi_radius_vox,
        )

        y_coords, x_coords = np.where(roi_mask)

        if len(y_coords) == 0:
            continue

        max_avg = -np.inf
        best_profile = None

        for y, x in zip(y_coords, x_coords):

            profile = [float(image_data[z, y, x]) for z in num_slices]

            avg = np.mean(profile)

            if avg > max_avg:
                max_avg = float(avg)
                best_profile = profile

        if best_profile is None:
            continue

        rod_est = Estimator.from_samples(
            np.array(best_profile),
            mode=mode,
        )

        if abs(uniform_est.mean) < eps:
            rc_est = Estimator(0.0, 0.0, 1)
        else:
            rc_est = rod_est.ratio(uniform_est)

        crc_results[name] = {
            "mean_signal": rod_est.mean,
            "std_signal": rod_est.std,
            "uniform_mean": uniform_est.mean,
            "uniform_std": uniform_est.std,
            "recovery_coeff": rc_est.mean,
            "percentage_STD_rc": (
                100 * rc_est.std / rc_est.mean if abs(rc_est.mean) > eps else 0.0
            ),
            "cError": rc_est.std,
        }

    return crc_results


def _calculate_spillover_ratio(
    image_data: npt.NDArray[Any],
    phantom: NemaPhantom,
    uniform_region_mask: npt.NDArray[Any],
    air_region_mask: npt.NDArray[Any],
    water_region_mask: npt.NDArray[Any],
    cfg: yacs.config.CfgNode,
) -> Dict[str, Dict[str, Any]]:  # type: ignore[return-value]
    """Internal function to calculate the spillover ratio for lung inserts."""
    mode = cfg.STATISTICS.MODE
    eps = getattr(cfg.STATISTICS, "EPSILON", 1e-12)

    uniform_values = image_data[uniform_region_mask]
    uniform_est = Estimator.from_samples(uniform_values, mode=mode)

    spillover_ratios = {}

    for region_name, region_mask in [
        ("air", air_region_mask),
        ("water", water_region_mask),
    ]:

        region_values = image_data[region_mask]
        region_est = Estimator.from_samples(region_values, mode=mode)

        if abs(uniform_est.mean) < eps:
            sor_est = Estimator(0.0, 0.0, 1)
        else:
            sor_est = region_est.ratio(uniform_est)

        _logger.info(
            f"ROI {region_name.capitalize()}:"
            f" voxels={np.sum(region_mask)},"
            f" mean={region_est.mean:.6f},"
            f" std={region_est.std:.6f},"
            f" perc_std={100 * region_est.std / region_est.mean if region_est.mean != 0 else 0:.2f}"
        )

        spillover_ratios[region_name] = {
            "SOR": sor_est.mean,
            "SOR_error": sor_est.std,
            "%STD": (
                100 * sor_est.std / sor_est.mean if abs(sor_est.mean) > eps else 0.0
            ),
        }

    return spillover_ratios


def _calculate_lung_insert_counts(
    image_data: npt.NDArray[Any],
    lung_inserts_centers: npt.NDArray[Any],
    CB_37: float,
    voxel: float,
) -> Dict[int, float]:
    """Internal function to calculate the mean counts (C_lung) for each axial slice within the permitted lung bounds."""

    lung_insert = {}

    for z, y, x in lung_inserts_centers:
        axial_cut = image_data[z, :, :]
        slice_dims_yx = axial_cut.shape
        roi_mask = extract_circular_mask_2d(
            (slice_dims_yx[0], slice_dims_yx[1]), (y, x), 15 / voxel
        )
        lung_insert[z] = (np.mean(axial_cut[roi_mask]) / CB_37) * 100

    return lung_insert


def calculate_weighted_cbr_from(results):
    """
    Calculates weighted Contrast-to-Background Ratio (CBR) and Figure of Merit (FOM) from sphere results.

    Orchestrates weighted metric calculation by processing individual sphere measurements and computing
    diameter-weighted averages as defined in the NEMA NU 2-2018 standard for overall image quality assessment.

    Parameters
    ----------
    results : list of dict
        List of calculated metrics for each sphere containing diameter_mm, percentaje_constrast_QH,
        and background_variability_N keys.

    Returns
    -------
    dict
        Dictionary containing weighted_CBR, weighted_FOM, individual CBRs, FOMs, weights, and diameters.
        Returns None values for weighted metrics if results list is empty.

    Notes
    -----
    Author: EdAlita
    Date: 2025-01-08 16:15:00

    The weighting scheme uses inverse diameter weighting (1/d) normalized to sum to 1.0.
    CBR is calculated as contrast/variability, FOM as contrast²/variability.
    """
    if not results:
        return {"weighted_CBR": None, "weighted_FOM": None}

    diameters = [r["diameter_mm"] for r in results]
    contrasts = [r["percentaje_constrast_QH"] for r in results]
    variabilities = [r["background_variability_N"] for r in results]

    weights = [1 / (d**2) for d in diameters]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    CBRs = [c / v if v != 0 else 0 for c, v in zip(contrasts, variabilities)]
    FOMs = [(c**2) / v if v != 0 else 0 for c, v in zip(contrasts, variabilities)]

    weighted_CBR = sum(w * cbr for w, cbr in zip(weights, CBRs))
    weighted_FOM = sum(w * fom for w, fom in zip(weights, FOMs))

    return {
        "weighted_CBR": weighted_CBR,
        "weighted_FOM": weighted_FOM,
        "CBRs": CBRs,
        "FOMs": FOMs,
        "weigths": weights,
        "diameters": diameters,
    }


def calculate_nema_metrics(
    image_data: npt.NDArray[Any],
    phantom: NemaPhantom,
    cfg: yacs.config.CfgNode,
    save_visualizations: bool = False,
    visualizations_dir: str = "visualizations",
) -> Tuple[List[Dict[str, Any]], Dict[int, float]]:
    """Calculate NEMA NU 2-2018 image quality metrics.

    Orchestrates the complete NEMA image quality analysis pipeline:

    1. Identifies background regions at multiple slice positions
    2. Extracts hot sphere region counts
    3. Computes Percent Contrast (Q_H) and Background Variability (N)
    4. Analyzes lung insert counts for spillover ratio assessment

    This is the primary function for quantifying image quality according to the
    NEMA NU 2-2018 standard.

    Parameters
    ----------
    image_data : numpy.ndarray
        3D PET image array with shape (z, y, x) in voxel units.
    phantom : NemaPhantom
        Initialized phantom model with ROI definitions and coordinate transforms.
    cfg : yacs.config.CfgNode
        Configuration object containing:

        - ACTIVITY.RATIO: Hot/background activity ratio (a_H / a_B)
        - ROIS.CENTRAL_SLICE: Central slice index in z-axis
        - ROIS.BACKGROUND_OFFSET_YX: Background region offsets
        - ROIS.ORIENTATION_YX: Orientation multipliers for background regions
        - ROIS.SPACING: Voxel spacing in mm

    save_visualizations : bool, optional
        If True, saves ROI mask visualizations to disk. Default is False.
    visualizations_dir : str, optional
        Directory path for saving visualization images. Default is "visualizations".

    Returns
    -------
    results : list[dict]
        List of metric dictionaries for each hot sphere:

        - diameter_mm: Sphere diameter in mm
        - percentaje_constrast_QH: Percent contrast (%)
        - background_variability_N: Background variability (%)
        - avg_hot_counts_CH: Average hot sphere counts
        - avg_bkg_counts_CB: Average background counts
        - bkg_std_dev_SD: Background standard deviation

    results_lung : dict[int, float]
        Lung insert spillover ratios indexed by slice number.

    Raises
    ------
    ValueError
        If activity ratio is not greater than 1.

    Notes
    -----
    The function analyzes multiple slice positions (±10mm and ±20mm from central slice)
    to account for axial variation in the phantom and image statistics.

    References
    ----------
    - NEMA NU 2-2018: Performance Measurements of Positron Emission Tomographs

    Examples
    --------
    Calculate NEMA metrics for a loaded image:

    >>> from nema_quant.phantom import NemaPhantom
    >>> from nema_quant.config import CfgNode
    >>> phantom = NemaPhantom(image_path='pet_image.nii.gz')
    >>> cfg = get_default_config()
    >>> metrics, lung_results = calculate_nema_metrics(
    ...     image_data, phantom, cfg, save_visualizations=True
    ... )
    >>> for m in metrics:
    ...     print(f"Sphere {m['diameter_mm']}mm: Q_H={m['percentaje_constrast_QH']:.1f}%")
    """
    viz_dir = None
    if save_visualizations:
        viz_dir = Path(visualizations_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)
        _logger.info(f" Saving visualizations to: {viz_dir}")

    activity_ratio = cfg.ACTIVITY.RATIO

    if activity_ratio <= 0 or activity_ratio <= 1:
        raise ValueError(
            "Activity ratio (a_H / a_B) must be greater than 1 and"
            "background activity must be positive."
        )

    central_slices_idx = cfg.ROIS.CENTRAL_SLICE
    cm_in_z_vox = phantom._mm_to_voxels(10, 2)
    slices_indices = sorted(
        {
            central_slices_idx,
            int(round(central_slices_idx + cm_in_z_vox)),
            int(round(central_slices_idx - cm_in_z_vox)),
            int(round(central_slices_idx + 2 * cm_in_z_vox)),
            int(round(central_slices_idx - 2 * cm_in_z_vox)),
        }
    )

    background_stats = _calculate_background_stats(
        image_data,
        phantom,
        slices_indices,
        [
            (y * cfg.ROIS.ORIENTATION_YX[0], x * cfg.ROIS.ORIENTATION_YX[1])
            for y, x in cfg.ROIS.BACKGROUND_OFFSET_YX
        ],
        save_visualizations=save_visualizations,
        viz_dir=viz_dir,
    )

    hot_sphere_counts = _calculate_hot_sphere_counts_offset_zxy(
        image_data,
        phantom,
        central_slices_idx,
        save_visualizations=save_visualizations,
        viz_dir=viz_dir,
    )

    results = []
    activity_ratio_term = activity_ratio - 1.0
    CB_37 = 0.0

    eps = getattr(cfg.STATISTICS, "EPSILON", 1e-12)
    mode = cfg.STATISTICS.MODE
    sd_model = cfg.STATISTICS.SD_VARIANCE_MODEL

    for name, hot_data in hot_sphere_counts.items():

        sphere_def = phantom.get_roi(name)
        if sphere_def is None:
            continue

        sphere_diam_mm = int(round(sphere_def["diameter"]))

        sphere_diam_mm = int(round(sphere_def["diameter"]))

        hot_est = Estimator.from_mean_std(
            mean=hot_data["mean"],
            std=hot_data["std_H"],
            n=int(hot_data["n_H"]),
            mode=mode,
        )

        bkg_data = background_stats[sphere_diam_mm]

        bkg_est = Estimator.from_mean_std(
            mean=bkg_data["C_B"],
            std=bkg_data["SD_B"],
            n=int(bkg_data["n_B"]),
            mode=mode,
        )

        if abs(bkg_est.mean) < eps:
            _logger.warning(
                f"Sphere {sphere_diam_mm}mm skipped: C_B too small ({bkg_est.mean:.3e})"
            )
            continue

        ratio_est = hot_est.ratio(bkg_est)
        qh_est = ratio_est.subtract_constant(1.0).scale(100.0 / activity_ratio_term)

        # Build SD estimator
        SD_B = bkg_data["SD_B"]  # type: ignore[index]
        n_B = max(int(bkg_data["n_B"]), 1)  # type: ignore[arg-type]
        C_B = bkg_est.mean

        if sd_model == "poisson":
            var_sd = C_B / (2.0 * n_B)
        else:
            var_sd = (SD_B**2) / (2.0 * n_B) if n_B > 1 else 0.0

        sd_est = Estimator(SD_B, var_sd, n_B)  # type: ignore[arg-type]

        n_est = sd_est.ratio(bkg_est).scale(100.0)

        # Save 37 mm background for lung insert
        if sphere_diam_mm == 37:
            CB_37 = bkg_est.mean
        _logger.info(
            f"{sphere_diam_mm:2d} mm | "
            f"RC={qh_est.mean:.2f} ± {qh_est.std:.2f}% | "
            f"BV={n_est.mean:.2f} ± {n_est.std:.2f}%"
        )

        results.append(
            {
                "diameter_mm": sphere_diam_mm,
                "percentaje_constrast_QH": qh_est.mean,
                "percentaje_constrast_QH_error": qh_est.std,
                "percentaje_constrast_QH_%STD": (
                    100 * qh_est.std / qh_est.mean if abs(qh_est.mean) > eps else 0.0
                ),
                "background_variability_N": n_est.mean,
                "background_variability_error": n_est.std,
                "background_variability_%STD": (
                    100 * n_est.std / n_est.mean if abs(n_est.mean) > eps else 0.0
                ),
                "avg_hot_counts_CH": hot_est.mean,
                "avg_bkg_counts_CB": bkg_est.mean,
                "bkg_std_dev_SD": SD_B,
            }
        )

    phantom_center_zyx = find_phantom_center(
        image_data, threshold=(np.max(image_data) * 0.41)
    )

    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug(f" Phantom Center found at (z,y,x) : {phantom_center_zyx}")

    lung_insert_centers = extract_canny_mask(
        image_data, cfg.ROIS.SPACING, int(phantom_center_zyx[0])
    )

    results_lung = _calculate_lung_insert_counts(
        image_data, lung_insert_centers, CB_37, cfg.ROIS.SPACING
    )

    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug(" Lung Insert Results")
        for k, v in results_lung.items():
            _logger.debug(f"  Slice {int(k)}: {float(v):.3f}")

    return results, results_lung


def calculate_nema_metrics_nu4_2008(
    image_data: npt.NDArray[Any],
    phantom: NemaPhantom,
    cfg: yacs.config.CfgNode,
    save_visualizations: bool = False,
    visualizations_dir: str = "visualizations",
) -> Tuple[List[Dict[str, Any]], Dict[int, float], Dict[str, float]]:
    """Calculate NEMA NU 4-2008 image quality metrics.

    This function is a placeholder for the implementation of NEMA NU 4-2008 metrics calculation.
    The actual logic for processing the image data according to the NU 4-2008 standard needs to be implemented.

    Parameters
    ----------
    image_data : numpy.ndarray
        3D PET image array with shape (z, y, x) in voxel units.
    phantom : NemaPhantom
        Initialized phantom model with ROI definitions and coordinate transforms.
    cfg : yacs.config.CfgNode
        Configuration object containing necessary parameters for NU 4-2008 analysis.
    save_visualizations : bool, optional
        If True, saves ROI mask visualizations to disk. Default is False.
    visualizations_dir : str, optional
        Directory path for saving visualization images. Default is "visualizations".

    Returns
    -------
    Tuple[List[Dict[str, Any]], Dict[int, float], Dict[str, float]]
    """
    mode = cfg.STATISTICS.MODE
    eps = getattr(cfg.STATISTICS, "EPSILON", 1e-12)

    dim_z, dim_y, dim_x = image_data.shape
    _logger.debug(f" Image data dimensions (z,y,x): {dim_z}, {dim_y}, {dim_x}")

    # Find phantom center in xy plane only (for centering cylindrical ROIs)
    center_method = getattr(cfg.ROIS, "PHANTOM_CENTER_METHOD", "weighted_slices")
    center_threshold = getattr(cfg.ROIS, "PHANTOM_CENTER_THRESHOLD_FRACTION", 0.41)

    ce_z, ce_y, ce_x = find_phantom_center_cv2_threshold(
        image_data,
        threshold_fraction=center_threshold,
        method=center_method,
    )

    phantom_center_z = int(ce_z)
    phantom_center_x = int(ce_x)
    phantom_center_y = int(ce_y)

    rods_center_z = cfg.ROIS.CENTRAL_SLICE

    _logger.info(
        f" Phantom center (z, x,y) found at: ({phantom_center_z}, {phantom_center_x}, {phantom_center_y})"
    )
    _logger.info(
        f" Using CENTRAL_SLICE z={rods_center_z} as reference (hot rods center)"
    )

    # Calculate uniform region center and bounds
    uniform_center_z = rods_center_z - cfg.ROIS.ORIENTATION_Z * (
        cfg.ROIS.UNIFORM_OFFSET_MM / cfg.ROIS.SPACING
    )
    half_height_vox = (cfg.ROIS.UNIFORM_HEIGHT_MM / cfg.ROIS.SPACING) / 2.0
    uniform_z_min = int(np.floor(uniform_center_z - half_height_vox))
    uniform_z_max = int(np.ceil(uniform_center_z + half_height_vox))
    _logger.debug(
        f" Uniform region: center_z={uniform_center_z:.1f}, "
        f"range z={uniform_z_min}-{uniform_z_max}, "
        f"offset={cfg.ROIS.UNIFORM_OFFSET_MM}mm, "
        f"height={cfg.ROIS.UNIFORM_HEIGHT_MM}mm"
    )

    uniform_region_mask = create_cylindrical_mask(
        shape_zyx=(image_data.shape[0], image_data.shape[1], image_data.shape[2]),  # type: ignore[arg-type]
        center_zyx=(
            phantom_center_z
            + cfg.ROIS.ORIENTATION_Z * (cfg.ROIS.UNIFORM_OFFSET_MM / cfg.ROIS.SPACING),
            phantom_center_y,
            phantom_center_x,
        ),
        radius_mm=cfg.ROIS.UNIFORM_RADIUS_MM,
        height_mm=cfg.ROIS.UNIFORM_HEIGHT_MM,
        spacing_xyz=np.array([cfg.ROIS.SPACING, cfg.ROIS.SPACING, cfg.ROIS.SPACING]),  # type: ignore[arg-type]
    )
    uniform_values = image_data[uniform_region_mask]
    uniform_est = Estimator.from_samples(uniform_values, mode=mode)
    uniformity_results = {
        "mean": uniform_est.mean,
        "maximum": float(np.max(uniform_values)) if uniform_values.size > 0 else 0.0,
        "minimum": float(np.min(uniform_values)) if uniform_values.size > 0 else 0.0,
        "%STD": (
            100 * uniform_est.std / uniform_est.mean
            if abs(uniform_est.mean) > eps
            else 0.0
        ),
    }

    air_region_mask = create_cylindrical_mask(
        shape_zyx=(image_data.shape[0], image_data.shape[1], image_data.shape[2]),  # type: ignore[arg-type]
        center_zyx=(
            phantom_center_z
            - cfg.ROIS.ORIENTATION_Z * (cfg.ROIS.AIRWATER_OFFSET_MM / cfg.ROIS.SPACING),
            phantom_center_y
            - cfg.ROIS.ORIENTATION_YX[0]
            * (cfg.ROIS.AIRWATER_SEPARATION_MM / cfg.ROIS.SPACING),
            phantom_center_x,
        ),
        radius_mm=cfg.ROIS.AIR_RADIUS_MM,
        height_mm=cfg.ROIS.AIR_HEIGHT_MM,
        spacing_xyz=np.array([cfg.ROIS.SPACING, cfg.ROIS.SPACING, cfg.ROIS.SPACING]),  # type: ignore[arg-type]
    )

    water_region_mask = create_cylindrical_mask(
        shape_zyx=(image_data.shape[0], image_data.shape[1], image_data.shape[2]),  # type: ignore[arg-type]
        center_zyx=(
            phantom_center_z
            - cfg.ROIS.ORIENTATION_Z * (cfg.ROIS.AIRWATER_OFFSET_MM / cfg.ROIS.SPACING),
            phantom_center_y
            + cfg.ROIS.ORIENTATION_YX[0]
            * (cfg.ROIS.AIRWATER_SEPARATION_MM / cfg.ROIS.SPACING),
            phantom_center_x,
        ),
        radius_mm=cfg.ROIS.WATER_RADIUS_MM,
        height_mm=cfg.ROIS.WATER_HEIGHT_MM,
        spacing_xyz=np.array([cfg.ROIS.SPACING, cfg.ROIS.SPACING, cfg.ROIS.SPACING]),  # type: ignore[arg-type]
    )

    spillover_ratio = _calculate_spillover_ratio(
        image_data=image_data,
        phantom=phantom,
        uniform_region_mask=uniform_region_mask,
        air_region_mask=air_region_mask,
        water_region_mask=water_region_mask,
        cfg=cfg,
    )

    _logger.info(" Spillover Ratios:")
    for region, metrics in spillover_ratio.items():
        _logger.info(
            f"  {region.capitalize()}: SOR={metrics['SOR']:.4f} ± {metrics['SOR_error']:.4f}, %STD={metrics['%STD']:.2f}%"
        )

    crc_results = _calculate_crc_std(
        image_data=image_data,
        phantom=phantom,
        central_slice_idx=rods_center_z,
        measure_in_px=10,  # 10mm height / 0.5mm spacing = 20 voxels → ±10 slices = 21 total
        uniform_region_mask=uniform_region_mask,
        cfg=cfg,
    )

    _logger.info(" CRC Results:")
    for region, metrics in crc_results.items():
        _logger.info(
            f"  {str(region).capitalize()}:"  # type: ignore[attr-defined]
            f" RC={metrics['recovery_coeff']:.2f} ± {metrics['cError']:.2f},"  # type: ignore[index]
            f" %STD={metrics['percentage_STD_rc']:.2f}%"  # type: ignore[index]
        )
    return crc_results, spillover_ratio, uniformity_results  # type: ignore[return-value]


def save_sphere_visualization(
    image_slice: npt.NDArray[Any],
    sphere_name: str,
    center_yx: Tuple[float, float],
    radius_vox: float,
    roi_mask: npt.NDArray[Any],
    output_dir: Path,
    slice_idx: int,
) -> None:
    """
    Saves a visualization of the sphere ROI and mask for debugging purposes.

    Generates and stores an image showing the analyzed sphere, its center and radius, and the ROI mask overlay. Useful for verifying
    ROI placement and mask accuracy.

    Parameters
    ----------
    image_slice : np.ndarray
        2D image slice containing the sphere.
    sphere_name : str
        Name of the sphere being analyzed.
    center_yx : Tuple[float, float]
        Center coordinates (y, x) of the sphere.
    radius_vox : float
        Radius of the sphere in voxels.
    roi_mask : np.ndarray
        Boolean mask indicating the ROI.
    output_dir : Path
        Directory to save the visualization.
    slice_idx : int
        Index of the slice being analyzed.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(
        image_slice, cmap="gray", vmin=0, vmax=np.percentile(image_slice, 99)
    )
    circle = patches.Circle(
        (center_yx[1], center_yx[0]),
        radius_vox,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    axes[0].add_patch(circle)
    axes[0].plot(center_yx[1], center_yx[0], "r+", markersize=10, markeredgewidth=2)
    axes[0].set_title(f"{sphere_name}\nOriginal Image with ROI")
    axes[0].axis("off")

    axes[1].imshow(roi_mask, cmap="Reds", alpha=0.8)
    axes[1].set_title(f"{sphere_name}\nROI Mask")
    axes[1].axis("off")

    masked_image = image_slice.copy()
    masked_image[~roi_mask] = 0
    axes[2].imshow(
        masked_image, cmap="gray", vmin=0, vmax=np.percentile(image_slice, 99)
    )
    axes[2].set_title(f"{sphere_name}\nMasked Region Only")
    axes[2].axis("off")

    mean_counts = np.mean(image_slice[roi_mask])
    std_counts = np.std(image_slice[roi_mask])
    num_pixels = np.sum(roi_mask)

    fig.suptitle(
        f"Slice {slice_idx} - {sphere_name}\n"
        f"Mean: {mean_counts:.2f}, Std: {std_counts:.2f}, Pixels: {num_pixels}",
        fontsize=12,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{sphere_name}_slice_{slice_idx}_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved sphere visualization: {output_file}")


def save_background_visualization(
    image_slice: npt.NDArray[Any],
    centers_offset: List[Tuple[int, int]],
    pivot_point_yx: Tuple[float, float],
    radius_vox: float,
    output_dir: Path,
    slice_idx: int,
) -> None:
    """
    Saves a visualization of the background ROIs for debugging purposes.

    Generates and stores an image displaying background ROI locations, offsets, reference pivot point, and radii. Useful for verifying
    placement and mask accuracy.

    Parameters
    ----------
    image_slice : np.ndarray
        2D image slice.
    centers_offset : List[Tuple[int, int]]
        List of offset positions for background ROIs.
    pivot_point_yx : Tuple[float, float]
        Pivot point (y, x) of the reference sphere.
    radius_vox : float
        Radius for background ROIs in voxels.
    output_dir : Path
        Directory to save the visualization.
    slice_idx : int
        Index of the slice being analyzed.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.imshow(image_slice, cmap="gray", vmin=0, vmax=np.percentile(image_slice, 99))

    for i, (y_offset, x_offset) in enumerate(centers_offset):
        center_y = pivot_point_yx[0] + y_offset
        center_x = pivot_point_yx[1] + x_offset

        circle = patches.Circle(
            (center_x, center_y),
            radius_vox,
            linewidth=1.5,
            edgecolor="cyan",
            facecolor="none",
            alpha=0.8,
        )
        ax.add_patch(circle)
        ax.text(
            center_x,
            center_y,
            str(i + 1),
            ha="center",
            va="center",
            color="yellow",
            fontweight="bold",
            fontsize=8,
        )

    ax.plot(
        pivot_point_yx[1], pivot_point_yx[0], "r+", markersize=15, markeredgewidth=3
    )
    ax.text(
        pivot_point_yx[1] + 10,
        pivot_point_yx[0],
        "Pivot",
        color="red",
        fontweight="bold",
    )

    ax.set_title(
        f"Background ROIs - Slice {slice_idx}\n{len(centers_offset)} ROIs shown"
    )
    ax.axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"background_rois_slice_{slice_idx}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved background visualization: {output_file}")


def calculate_advanced_metrics(
    image_data: npt.NDArray[Any],
    gt_data: npt.NDArray[Any],
    measures: Tuple[str, ...],
    cfg: yacs.config.CfgNode,
) -> Dict[str, Any]:
    """Compute advanced image-quality metrics against a reference mask.

    Parameters
    ----------
    image_data : npt.NDArray[Any]
        3D reconstructed image volume.
    gt_data : npt.NDArray[Any]
        Ground-truth or reference mask aligned with `image_data`.
    measures : Tuple[str, ...]
        Metric names to compute (passed to `get_values()`).
    cfg : yacs.config.CfgNode
        Configuration containing `ROIS.SPACING` (voxel size in mm).

    Returns
    -------
    Dict[str, Any]
        Mapping of metric names to computed values.

    Notes
    -----
    Logs each computed metric at INFO level.
    """

    values = dict(
        get_values(
            image_data,
            gt_data,
            measures=measures,
            voxelspacing=(cfg.ROIS.SPACING, cfg.ROIS.SPACING, cfg.ROIS.SPACING),
        )
    )
    _logger.info("Advanced Metrics:")
    for k, v in values.items():
        _logger.info(f" {k}: {v:.7f}")
    return values

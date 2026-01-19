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
from .utils import extract_canny_mask, find_phantom_center

logger = logging.getLogger(__name__)


def extract_circular_mask_2d(
    slice_dims: Tuple[int, int],
    roi_center_vox: Tuple[float, float],
    roi_radius_vox: float,
) -> npt.NDArray[np.bool_]:
    """
    Creates a 2D boolean mask for a circular ROI on a slice.

    Highly efficient: avoids loops by leveraging NumPy vectorization. It generates arrays for y and x coordinates,
    computes the squared distance from each pixel to the center, and returns a boolean array marking pixels inside the circle.

    Parameters
    ----------
    slice_dims : tuple of int, shape (2,)
        (y, x) dimensions of the 2D slice.
    roi_center_vox : tuple of float, shape (2,)
        (y, x) coordinates of the circle's center in voxels.
    roi_radius_vox : float
        Radius of the circle in voxels.

    Returns
    -------
    numpy.ndarray
        2D boolean array: True for pixels inside the circle, False otherwise.
    """
    y_coords, x_coords = np.ogrid[: slice_dims[0], : slice_dims[1]]
    center_y, center_x = roi_center_vox

    squared_dist = (y_coords - center_y) ** 2 + (x_coords - center_x) ** 2

    return squared_dist <= roi_radius_vox**2


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

    bkg_counts_per_size: Dict[int, List[float]] = {}

    for name, _ in phantom.rois.items():
        if "sphere" in name:
            sphere_roi = phantom.get_roi(name)
            if sphere_roi:
                diam_mm = int(round(sphere_roi["diameter"]))
                if diam_mm not in bkg_counts_per_size:
                    bkg_counts_per_size[diam_mm] = []

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
                    avg_count = np.mean(img_slice[roi_mask])
                    sphere_diam_mm = int(round(sphere_roi["diameter"]))
                    bkg_counts_per_size[sphere_diam_mm].append(avg_count)

    bkg_stats = {}
    for diam, counts_list in bkg_counts_per_size.items():
        if counts_list:
            bkg_stats[diam] = {
                "C_B": float(np.mean(counts_list)),
                "SD_B": float(np.std(counts_list)),
            }
        else:
            bkg_stats[diam] = {"C_B": 100.0, "SD_B": 0.0}

    return bkg_stats


def _calculate_hot_sphere_counts_offset_zxy(
    image_data: npt.NDArray[Any],
    phantom: NemaPhantom,
    central_slice_idx: int,
    save_visualizations: bool = False,
    viz_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Internal function to calculate the mean counts (C_H) for each hot sphere."""

    offsets = [(x, y) for x in range(-10, 11) for y in range(-10, 11)]

    offsets_z = list(range(-10, 11))

    hot_sphere_counts = {}

    for name, _ in phantom.rois.items():
        if "sphere" not in name:
            continue

        sphere_roi = phantom.get_roi(name)
        if sphere_roi is None:
            continue

        center_yx = sphere_roi["center_vox"]
        max_mean = -np.inf
        best_offset_zyx = None

        for dz in offsets_z:
            slice_idx = central_slice_idx + dz
            if slice_idx < 0 or slice_idx >= image_data.shape[0]:
                continue

            current_slice = image_data[slice_idx, :, :]
            slice_dims_yx = current_slice.shape

            for offset in offsets:
                offset_center = (center_yx[0] + offset[0], center_yx[1] + offset[1])
                roi_mask = extract_circular_mask_2d(
                    (slice_dims_yx[0], slice_dims_yx[1]),
                    offset_center,
                    sphere_roi["radius_vox"],
                )
                mean_count = np.mean(current_slice[roi_mask])
                if mean_count > max_mean:
                    max_mean = mean_count
                    best_offset_zyx = (dz, offset[0], offset[1])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"  Found the best average counts for {name} with offset {best_offset_zyx}: {max_mean:.2f}"
            )
        hot_sphere_counts[name] = max_mean

        # Save visualization if requested
        if save_visualizations and viz_dir:
            save_sphere_visualization(
                current_slice,
                name,
                center_yx,
                sphere_roi["radius_vox"],
                roi_mask,
                viz_dir,
                central_slice_idx,
            )

    return hot_sphere_counts


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
    """
    Calculates NEMA Image Quality metrics: Percent Contrast (Q_H) and Background Variability (N).

    Orchestrates metric analysis by calling helper functions to measure background and hot sphere regions, then computes
    final metrics as defined in the NEMA NU 2-2018 standard.

    Parameters
    ----------
    image_data : np.ndarray
        3D image data array, shape (z, y, x).
    phantom : NemaPhantom
        Initialized NemaPhantom object with ROI definitions.
    cfg : yacs.config.CfgNode
        Configuration settings for dataset processing.
    save_visualizations : bool, optional
        If True, saves visualizations of ROI masks. Default is False.
    visualizations_dir : str, optional
        Directory for saving ROI mask images. Default is "visualizations".

    Returns
    -------
    list of dict
        List of calculated metrics for each sphere.

    Notes
    -----
    Author: EdAlita
    Date: 2025-07-08 06:47:01
    """
    viz_dir = None
    if save_visualizations:
        viz_dir = Path(visualizations_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f" Saving visualizations to: {viz_dir}")

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

    for name, C_H in hot_sphere_counts.items():
        sphere_def = phantom.get_roi(name)
        if sphere_def is None:
            continue
        sphere_diam_mm = int(round(sphere_def["diameter"]))
        C_B = background_stats[sphere_diam_mm]["C_B"]
        SD_B = background_stats[sphere_diam_mm]["SD_B"]

        percent_contrast = ((C_H / C_B) - 1.0) / activity_ratio_term * 100.0
        percent_variability = (SD_B / C_B) * 100.0

        if sphere_diam_mm == 37:
            CB_37 = C_B

        logging.info(
            f" Diameter Sphere {sphere_diam_mm}: Percentaje Contrast {percent_contrast:.1f}% Background Variability {percent_variability:.1f}%"
        )
        results.append(
            {
                "diameter_mm": sphere_diam_mm,
                "percentaje_constrast_QH": percent_contrast,
                "background_variability_N": percent_variability,
                "avg_hot_counts_CH": C_H,
                "avg_bkg_counts_CB": C_B,
                "bkg_std_dev_SD": SD_B,
            }
        )

    phantom_center_zyx = find_phantom_center(
        image_data, threshold=(np.max(image_data) * 0.41)
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f" Phantom Center found at (z,y,x) : {phantom_center_zyx}")

    lung_insert_centers = extract_canny_mask(
        image_data, cfg.ROIS.SPACING, int(phantom_center_zyx[0])
    )

    results_lung = _calculate_lung_insert_counts(
        image_data, lung_insert_centers, CB_37, cfg.ROIS.SPACING
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(" Lung Insert Results")
        for k, v in results_lung.items():
            logger.debug(f"  Slice {int(k)}: {float(v):.3f}")

    return results, results_lung


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

    values = dict(
        get_values(
            image_data,
            gt_data,
            measures=measures,
            voxelspacing=(cfg.ROIS.SPACING, cfg.ROIS.SPACING, cfg.ROIS.SPACING),
        )
    )
    logger.info("Advanced Metrics:")
    for k, v in values.items():
        logger.info(f" {k}: {v:.7f}")
    return values

from typing import Any, Dict, Optional, Tuple

import numpy as np
import yacs.config

from config.defaults import get_cfg_defaults


class NemaPhantom:
    """
    Defines the geometry of the NEMA NU 2-2018 phantom and its ROIs.

    Translates physical dimensions (in mm) of the phantom's features (hot spheres, lung insert, background ROIs)
    into voxel-based coordinates and sizes using the properties of a PET image.

    Parameters
    ----------
    image_dims : tuple of int, shape (3,)
        Dimensions of the image (x, y, z) in voxels.
    voxel_spacing : tuple of float, shape (3,)
        Voxel size (x, y, z) in mm.

    Attributes
    ----------
    image_dims : tuple of int
        Image dimensions in voxels.
    voxel_spacing : tuple of float
        Voxel spacing in mm.
    phantom_center_voxels : np.ndarray
        Center of the phantom in voxel coordinates.
    rois : list of dict
        List of dictionaries defining each ROI: name, center (voxels), and radius (voxels).
    """

    def __init__(
        self,
        cfg: yacs.config.CfgNode,
        image_dims: Tuple[int, int, int],
        voxel_spacing: Tuple[float, float, float],
    ) -> None:
        """
        Initializes the phantom with the properties of the target image.

        Sets up the phantom geometry using the provided image dimensions and voxel spacing, ensuring both are valid 3-element tuples.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            Configuration settings.
        image_dims : tuple of int, shape (3,)
            Dimensions of the image (x, y, z) in voxels.
        voxel_spacing : tuple of float, shape (3,)
            Size of each voxel (x, y, z) in mm.

        Raises
        ------
        ValueError
            If 'image_dims' does not contain exactly 3 elements.
        ValueError
            If 'voxel_spacing' does not contain exactly 3 elements.
        """
        if len(image_dims) != 3:
            raise ValueError(
                f"Expected 3 elements for 'image_dims' but got {len(image_dims)}. Value: {image_dims}"
            )
        if len(voxel_spacing) != 3:
            raise ValueError(
                f"Expected 3 elements for 'voxel_spacing' but got {len(voxel_spacing)}. Value: {voxel_spacing}"
            )

        self.image_dims = image_dims
        self.voxel_spacing = voxel_spacing
        self.phantom_center_voxels = np.array(image_dims) / 2.0
        self.roi_definitions_mm = cfg.PHANTHOM.ROI_DEFINITIONS_MM
        self.rois = self._initialize_rois()

    def _mm_to_voxels(self, distance_mm: float, axis_index: int) -> float:
        """
        Converts a distance in millimeters to voxels along a specified axis.

        Uses voxel spacing for the given axis to compute the equivalent voxel count.

        Parameters
        ----------
        distance_mm : float
            Distance in millimeters to convert.
        axis_index : int
            Axis index: 0 for x, 1 for y, 2 for z.

        Returns
        -------
        float
            Equivalent distance in voxels.
        """
        return distance_mm / self.voxel_spacing[axis_index]

    def _initialize_rois(self) -> Dict[str, Dict[str, Any]]:
        """
        Defines ROIs according to the NEMA standard and converts them to voxel space.

        Sets the physical locations and sizes of hot spheres and the lung insert, then computes their voxel-based coordinates for image analysis.

        Returns
        -------
        dict
            Dictionary with ROI names as keys and each value containing the ROI's center and radius in voxel space.
        """
        processed_rois: Dict[str, Dict[str, Any]] = {}
        for roi_def in self.roi_definitions_mm:
            roi_name = roi_def["name"]
            radius_mm = float(roi_def["diameter_mm"]) / 2.0
            radius_vox = self._mm_to_voxels(radius_mm, 0)
            center_yx = roi_def["center_yx"]

            processed_rois[roi_name] = {
                "diameter": roi_def["diameter_mm"],
                "center_vox": tuple(center_yx),
                "radius_vox": radius_vox,
            }

        return processed_rois

    def get_roi(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the definition of a specific ROI by its name.

        Provides efficient O(1) average time complexity for accessing ROI data by name.

        Parameters
        ----------
        name : str
            Name of the ROI to retrieve (e.g., 'hot_sphere_10mm').

        Returns
        -------
        dict or None
            Dictionary with ROI properties ('center_vox', 'radius_vox') if found; otherwise, None.
        """
        return self.rois.get(name)


if __name__ == "__main__":
    IMAGE_DIMS: Tuple[int, int, int] = (391, 391, 346)
    VOXEL_SPACING: Tuple[float, float, float] = (2.0644, 2.0644, 2.0644)

    cfg = get_cfg_defaults()

    phantom = NemaPhantom(cfg=cfg, image_dims=IMAGE_DIMS, voxel_spacing=VOXEL_SPACING)

    roi_name_to_check = "hot_sphere_10mm"
    sphere_10mm_roi = phantom.get_roi(roi_name_to_check)

    if sphere_10mm_roi:
        # Format the coordinates for clean printing
        center_coords = sphere_10mm_roi["center_vox"]
        center_str = f"({center_coords[0]:.2f}, {center_coords[1]:.2f})"

        print(f"ROI: {roi_name_to_check}")
        print(f"  -> Center (voxels): {center_str} [y,x]")
        print(f"  -> Radius (voxels): {sphere_10mm_roi['radius_vox']:.2f}")

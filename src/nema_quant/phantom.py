import numpy as np
from typing import Tuple, List, Dict, Optional, Any


class NemaPhantom:
    """
    Defines the geometry of the NEMA NU 2-2018 phantom and its ROIs.

    This class translates the physical dimensions (in mm) of the NEMA phantom's
    features (hot spheres, lung insert, background ROIs) into voxel-based
    coordinates and sizes, based on the properties of a specific PET image.

    Parameters
    ----------
    image_dims : tuple of int, shape (3,)
        The dimensions of the image (x, y, z) in voxels.
    voxel_spacing : tuple of float, shape (3,)
        The size of the voxel (x, y, z) in mm.

    Attributes
    ----------
    image_dims : tuple of int
        Stores the image dimensions.
    voxel_spacing : tuple of float
        Stores the voxel spacing.
    phantom_center_voxels : np.ndarray
        The calculated center of the phantom in voxel coordinates.
    rois : list of dict
        A list of dictionaries, each defining a specific ROI with its
        name, center in voxels, and radius in voxels.

    """
    def __init__(
        self,
        image_dims: Tuple[int, int, int],
        voxel_spacing: Tuple[float, float, float]
    ) -> None:
        """
        Initializes the phantom with the properties of the target image.

        Parameters
        ----------
        image_dims : tuple of int, shape (3,)
            The dimensions of the image (x, y, z) in voxels.
        voxel_spacing : tuple of float, shape (3,)
            The size of the voxel (x, y, z) in mm.
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
        self.rois = self._initialize_rois()

    def _mm_to_voxels(self, distance_mm: float, axis_index: int) -> float:
        """
        Convert a distance in mm to voxels for a specific axis.

        Parameters
        ----------
        distance_mm : float
            The distance in millimeters to convert.
        axis_index : int
            The index of the axis (0 for x, 1 for y, 2 for z)

        Returns
        -------
        float
            The equivalent distance in voxels.
        """
        return distance_mm / self.voxel_spacing[axis_index]

    def _initialize_rois(self) -> Dict[str, Dict[str, Any]]:
        """
        Define ROIs based on the NEMA standard and convert them to voxel space.

        This method sets the physical locations and sizes of the hot spheres
        and lung insert, then calls the necessary functions to convert these
        properties into voxel-based coordinates for image analysis.

        Returns
        -------
        dict
            A dictionary where keys are ROI names and values contain the
            ROI's center and radius in voxel space.
        """
        roi_definitions_mm: List[Dict[str, Any]] = [
            {'name': 'hot_sphere_10mm', 'diameter': 10,
             'position': (57.0, 0.0, 0.0)},
            {'name': 'hot_sphere_13mm', 'diameter': 13,
             'position': (28.5, 49.37, 0.0)},
            {'name': 'hot_sphere_17mm', 'diameter': 17,
             'position': (-28.5, 49.37, 0.0)},
            {'name': 'hot_sphere_22mm', 'diameter': 22,
             'position': (-57.0, 0.0, 0.0)},
            {'name': 'hot_sphere_28mm', 'diameter': 28,
             'position': (-28.5, -49.37, 0.0)},
            {'name': 'hot_sphere_37mm', 'diameter': 37,
             'position': (-49.77, 35.93, -2.9)},
            {'name': 'lung_insert', 'diameter': 30,
             'position': (0.0, 0.0, 0.0)},
        ]

        processed_rois: Dict[str, Dict[str, Any]] = {}
        for roi_def in roi_definitions_mm:
            roi_name = roi_def['name']
            radius_mm = float(roi_def['diameter']) / 2.0
            radius_vox = self._mm_to_voxels(radius_mm, 0)

            position_offset_vox = np.array([
                self._mm_to_voxels(float(roi_def['position'][i]), i)
                for i in range(3)
            ])
            center_vox = self.phantom_center_voxels + position_offset_vox

            processed_rois[roi_name] = {
                'center_vox': tuple(center_vox),
                'radius_vox': radius_vox
            }

        return processed_rois

    def get_roi(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the definition of a specific ROI by its name.

        This method provides O(1) average time complexity access to ROI data.

        Parameters
        ----------
        name : str
            The name of the ROI to retrieve (e.g., 'hot_sphere_10mm').

        Returns
        -------
        dict or None
            A dictionary with ROI properties ('center_vox', 'radius_vox')
            if found, otherwise None.
        """
        return self.rois.get(name)


if __name__ == '__main__':
    IMAGE_DIMS: Tuple[int, int, int] = (391, 391, 346)
    VOXEL_SPACING: Tuple[float, float, float] = (2.0644, 2.0644, 2.0644)

    phantom = NemaPhantom(image_dims=IMAGE_DIMS, voxel_spacing=VOXEL_SPACING)

    roi_name_to_check = 'hot_sphere_10mm'
    sphere_10mm_roi = phantom.get_roi(roi_name_to_check)

    if sphere_10mm_roi:
        # Format the coordinates for clean printing
        center_coords = sphere_10mm_roi['center_vox']
        center_str = (
            f"({center_coords[0]:.2f}, {center_coords[1]:.2f}, "
            f"{center_coords[2]:.2f})"
        )

        print(f"ROI: {roi_name_to_check}")
        print(f"  -> Center (voxels): {center_str}")
        print(f"  -> Radius (voxels): {sphere_10mm_roi['radius_vox']:.2f}")

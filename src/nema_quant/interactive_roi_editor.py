#!/usr/bin/env python3
"""
Interactive ROI Editor for NEMA Phantom Configuration

This tool automatically detects sphere centers and allows interactive adjustment
of ROI positions to generate configuration values for nema_phantom_config.yaml.

Author: Edwing Ulin-Briseno
Date: 2026-01-17
"""

import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, TextBox
from scipy.ndimage import center_of_mass
from scipy.ndimage import label as ndimage_label

from nema_quant.io import load_nii_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKGROUND_OFFSET_YX: List[Tuple[int, int]] = [
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
DEFAULT_PIXEL_SPACING = 2.0644


class InteractiveROIEditor:
    """Interactive editor for NEMA phantom ROI positions."""

    def __init__(
        self,
        image: npt.NDArray[Any],
        initial_slice: int = 100,
        threshold_percentile: float = 50.0,
        pixel_spacing: float = DEFAULT_PIXEL_SPACING,
        background_offset_yx: Optional[List[Tuple[int, int]]] = None,
    ):
        """
        Initialize the interactive ROI editor.

        Parameters
        ----------
        image : npt.NDArray[Any]
            3D image array
        initial_slice : int
            Starting central slice
        threshold_percentile : float
            Percentile for auto-detection threshold
        """
        self.image = image
        self.central_slice = initial_slice
        self.orientation_yx = [1, 1]
        self.threshold_percentile = threshold_percentile
        self.pixel_spacing = pixel_spacing
        self.background_offset_yx = (
            list(background_offset_yx)
            if background_offset_yx is not None
            else list(BACKGROUND_OFFSET_YX)
        )

        # Standard NEMA phantom sphere diameters and colors
        self.sphere_diameters = [37, 28, 22, 17, 13, 10]
        self.sphere_colors = ["red", "orange", "gold", "lime", "cyan", "blue"]
        self.sphere_names = [
            "hot_sphere_37mm",
            "hot_sphere_28mm",
            "hot_sphere_22mm",
            "hot_sphere_17mm",
            "hot_sphere_13mm",
            "hot_sphere_10mm",
        ]

        # Auto-detect sphere centers
        self.roi_centers = self._auto_detect_centers()

        # Initialize figure and widgets
        self._setup_figure()

    def _auto_detect_centers(self) -> List[List[int]]:
        """
        Auto-detect sphere centers using thresholding and labeling.

        Returns
        -------
        List[List[int]]
            List of [y, x] center coordinates for each detected sphere
        """
        logger.info("Auto-detecting sphere centers...")

        # Get slice for detection
        slice_img = self.image[self.central_slice]

        # Calculate threshold
        threshold = float(np.max(slice_img) * self.threshold_percentile)
        logger.info(f"Detection threshold: {threshold:.6f}")

        # Create binary mask and label objects
        binary_mask = slice_img > threshold
        labeled_mask, num_features = ndimage_label(binary_mask)  # type: ignore
        num_features = int(num_features)
        logger.info(f"Number of objects found: {num_features}")

        if num_features == 0:
            logger.warning("No objects detected. Using default positions.")
            return [[100, 100]] * 6

        # Calculate center of mass for each region
        centers = []
        for i in range(1, num_features + 1):
            region_mask = labeled_mask == i
            com = center_of_mass(region_mask)
            com_rounded = [round(com[0]), round(com[1])]
            centers.append(com_rounded)
            logger.debug(f"Region {i}: center = {com_rounded}")

        # Sort centers by size (largest first) and take top 6
        # Calculate region sizes
        region_sizes = []
        for i in range(
            1, min(num_features + 1, 20)
        ):  # Limit to first 20 for performance
            region_mask = labeled_mask == i
            size = np.sum(region_mask)
            region_sizes.append((i, size))

        # Sort by size descending
        region_sizes.sort(key=lambda x: x[1], reverse=True)

        # Take centers of 6 largest regions
        final_centers = []
        for i, _ in region_sizes[:6]:
            region_mask = labeled_mask == i
            com = center_of_mass(region_mask)
            com_rounded = [round(com[0]), round(com[1])]
            final_centers.append(com_rounded)

        # Pad with defaults if less than 6 found
        while len(final_centers) < 6:
            final_centers.append([100, 100])

        logger.info(f"Detected {len(final_centers)} ROI centers")
        return final_centers

    def _setup_figure(self) -> None:
        """Setup matplotlib figure with image and interactive widgets."""
        self.fig = plt.figure(figsize=(16, 10))

        # Main image axes
        self.ax_img = plt.subplot2grid((8, 3), (0, 0), rowspan=8, colspan=2)

        # Control panel axes
        self.ax_controls = plt.subplot2grid((8, 3), (0, 2), rowspan=8)
        self.ax_controls.axis("off")

        # Draw initial image
        self._redraw()

        # Setup control widgets
        self._setup_widgets()

        plt.tight_layout()

    def _setup_widgets(self) -> None:
        """Setup interactive text boxes and buttons."""
        # Central Slice control
        ax_slice = plt.axes((0.70, 0.90, 0.15, 0.03))
        self.textbox_slice = TextBox(
            ax_slice, "Central Slice:", initial=str(self.central_slice)
        )
        self.textbox_slice.on_submit(self._update_slice)

        # Orientation Y control
        ax_orient_y = plt.axes((0.70, 0.85, 0.07, 0.03))
        self.textbox_orient_y = TextBox(
            ax_orient_y, "Orient Y:", initial=str(self.orientation_yx[0])
        )
        self.textbox_orient_y.on_submit(self._update_orientation)

        # Orientation X control
        ax_orient_x = plt.axes((0.78, 0.85, 0.07, 0.03))
        self.textbox_orient_x = TextBox(
            ax_orient_x, "X:", initial=str(self.orientation_yx[1])
        )
        self.textbox_orient_x.on_submit(self._update_orientation)

        # ROI center textboxes
        self.textbox_rois = []
        y_start = 0.75
        y_step = 0.08

        for i in range(6):
            # Y coordinate
            ax_y = plt.axes((0.68, y_start - i * y_step, 0.08, 0.03))
            tb_y = TextBox(
                ax_y,
                f"{self.sphere_diameters[i]}mm Y:",
                initial=str(self.roi_centers[i][0]),
            )
            tb_y.on_submit(lambda text, idx=i, fn=self._update_roi: fn(idx, text, True))  # type: ignore

            # X coordinate
            ax_x = plt.axes((0.77, y_start - i * y_step, 0.08, 0.03))
            tb_x = TextBox(ax_x, "X:", initial=str(self.roi_centers[i][1]))
            tb_x.on_submit(lambda text, idx=i, fn=self._update_roi: fn(idx, text, False))  # type: ignore

            self.textbox_rois.append((tb_y, tb_x))

        # Generate YAML button
        ax_button = plt.axes((0.70, 0.05, 0.15, 0.05))
        self.btn_generate = Button(ax_button, "Generate YAML")
        self.btn_generate.on_clicked(self._generate_yaml)

        # Re-detect button
        ax_redetect = plt.axes((0.70, 0.12, 0.15, 0.05))
        self.btn_redetect = Button(ax_redetect, "Re-detect ROIs")
        self.btn_redetect.on_clicked(self._redetect_rois)

    def _update_slice(self, text: str) -> None:
        """Update central slice from textbox."""
        try:
            new_slice = int(text)
            if 0 <= new_slice < self.image.shape[0]:
                self.central_slice = new_slice
                self._redraw()
            else:
                logger.warning(f"Slice {new_slice} out of range")
        except ValueError:
            logger.warning(f"Invalid slice value: {text}")

    def _update_orientation(self, text: str) -> None:
        """Update orientation from textboxes."""
        try:
            orient_y = int(self.textbox_orient_y.text)
            orient_x = int(self.textbox_orient_x.text)
            self.orientation_yx = [orient_y, orient_x]
            self._redraw()
        except ValueError:
            logger.warning("Invalid orientation values")

    def _update_roi(self, idx: int, text: str, is_y: bool) -> None:
        """Update ROI coordinate from textbox."""
        try:
            value = int(text)
            if is_y:
                self.roi_centers[idx][0] = value
            else:
                self.roi_centers[idx][1] = value
            self._redraw()
        except ValueError:
            logger.warning(f"Invalid coordinate value: {text}")

    def _redetect_rois(self, event: Any) -> None:
        """Re-run auto-detection on current slice."""
        self.roi_centers = self._auto_detect_centers()

        # Update textboxes
        for i, (tb_y, tb_x) in enumerate(self.textbox_rois):
            tb_y.set_val(str(self.roi_centers[i][0]))
            tb_x.set_val(str(self.roi_centers[i][1]))

        self._redraw()

    def _redraw(self) -> None:
        """Redraw the image with current ROI positions."""
        self.ax_img.clear()

        # Display image
        slice_img = self.image[self.central_slice]
        self.ax_img.imshow(slice_img, cmap="binary", origin="lower")

        # Draw ROI circles
        centro_37 = None
        for i in range(6):
            y, x = self.roi_centers[i]
            diameter = self.sphere_diameters[i]
            radius_pix = (diameter / 2) / self.pixel_spacing

            circle = Circle(
                (x, y),
                radius_pix,
                edgecolor=self.sphere_colors[i],
                facecolor="none",
                linewidth=2,
                alpha=0.8,
            )
            self.ax_img.add_patch(circle)

            # Add center marker
            self.ax_img.plot(x, y, "+", color=self.sphere_colors[i], markersize=12)

            if self.sphere_names[i] == "hot_sphere_37mm":
                centro_37 = (y, x)

            # Add label
            self.ax_img.text(
                x,
                y - radius_pix - 5,
                f"{diameter}mm",
                color=self.sphere_colors[i],
                ha="center",
                fontsize=10,
                weight="bold",
            )

        if centro_37 is not None:
            background_offset = [
                (y * self.orientation_yx[0], x * self.orientation_yx[1])
                for y, x in self.background_offset_yx
            ]
            background_radius = (37 / 2) / self.pixel_spacing
            for dy, dx in background_offset:
                background_y, background_x = centro_37[0] + dy, centro_37[1] + dx
                circle = Circle(
                    (background_x, background_y),
                    background_radius,
                    edgecolor="orange",
                    facecolor="none",
                    lw=2,
                    linestyle="--",
                    label="Background" if (dy, dx) == background_offset[0] else "",
                )
                self.ax_img.add_patch(circle)
                self.ax_img.plot(
                    background_x, background_y, "o", color="orange", markersize=7
                )

        self.ax_img.set_title(
            f"Central Slice: {self.central_slice} | Orientation: {self.orientation_yx} | Spacing: {self.pixel_spacing} mm",
            fontsize=12,
        )
        self.ax_img.set_xlabel("X (pixels)")
        self.ax_img.set_ylabel("Y (pixels)")
        self.ax_img.grid(False)

        self.fig.canvas.draw_idle()

    def _generate_yaml(self, event: Any) -> None:
        """Generate and print YAML configuration."""
        print("\n" + "=" * 80)
        print("GENERATED YAML CONFIGURATION - Copy and paste into your config file:")
        print("=" * 80)
        print("\nPHANTHOM:")
        print("  ROI_DEFINITIONS_MM:")

        for i in range(6):
            y, x = self.roi_centers[i]
            print(f"    - center_yx: [{y}, {x}]")
            print(f"      diameter_mm: {self.sphere_diameters[i]}")
            print(f'      color: "{self.sphere_colors[i]}"')
            print("      alpha: 0.18")
            print(f'      name: "{self.sphere_names[i]}"')

        print("\nROIS:")
        print(f"  CENTRAL_SLICE: {self.central_slice}")
        print(f"  ORIENTATION_YX: {self.orientation_yx}")
        print(f"  SPACING: {self.pixel_spacing}")
        print("  BACKGROUND_OFFSET_YX:")
        for y, x in self.background_offset_yx:
            print(f"    - [{y}, {x}]")

        print("\n" + "=" * 80)
        logger.info("YAML configuration generated!")

    def show(self) -> None:
        """Display the interactive editor."""
        plt.show()


def main() -> None:
    """Main entry point for the interactive ROI editor."""
    parser = argparse.ArgumentParser(
        description="Interactive ROI Editor for NEMA Phantom Configuration"
    )
    parser.add_argument(
        "input_image", type=str, help="Path to input NIfTI image (.nii or .nii.gz)"
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=None,
        help="Initial central slice (default: auto-detect middle slice)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.41,
        help="Threshold percentile for auto-detection (default: 0.41)",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=DEFAULT_PIXEL_SPACING,
        help="Pixel spacing in mm used to size ROIs (default: 2.0644)",
    )

    args = parser.parse_args()

    # Load image
    logger.info(f"Loading image: {args.input_image}")
    image_path = Path(args.input_image)

    if not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        return

    image_array_3d, affine = load_nii_image(filepath=image_path, return_affine=True)

    # Determine initial slice
    if args.slice is None:
        initial_slice = image_array_3d.shape[0] // 2
        logger.info(f"Auto-detected middle slice: {initial_slice}")
    else:
        initial_slice = args.slice

    # Create and show editor
    logger.info("Starting interactive ROI editor...")
    editor = InteractiveROIEditor(
        image=image_array_3d,
        initial_slice=initial_slice,
        threshold_percentile=args.threshold,
        pixel_spacing=args.spacing,
    )
    editor.show()


if __name__ == "__main__":
    main()

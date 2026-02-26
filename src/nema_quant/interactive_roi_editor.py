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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
import yacs.config
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.ndimage import center_of_mass
from scipy.ndimage import label as ndimage_label

from config.defaults import get_cfg_defaults
from nema_quant.analysis import create_cylindrical_mask
from nema_quant.io import load_nii_image
from nema_quant.utils import find_phantom_center, find_phantom_center_cv2_threshold

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


class InteractiveROIEditor(QtWidgets.QMainWindow):
    """GPU-accelerated interactive editor for NEMA NU 2-2018 phantom ROI positions using PyQtGraph."""

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
        pixel_spacing : float
            Voxel spacing in mm
        background_offset_yx : Optional[List[Tuple[int, int]]]
            Background ROI offsets from center 37mm sphere
        """
        super().__init__()
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
        self.statusBar().showMessage("Ready")

        # Initialize center detection parameters
        self.center_method = "weighted_slices"
        self.center_threshold = 0.41

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

        # Initialize ACQUISITION parameters
        self.emission_image_time_minutes = 10

        # Initialize ACTIVITY parameters
        self.activity_hot = 0.000765697
        self.activity_background = 0.0000772253
        self.activity_units = "mCi/mL"
        self.activity_ratio = 9.915
        self.activity_total = "29.24 MBq"

        # Initialize FILE parameters
        self.file_user_pattern = "frame(\\d+)"
        self.file_case = "Test"

        # Auto-detect sphere centers
        self.roi_centers = self._auto_detect_centers()

        # Setup Qt UI with PyQtGraph
        self.setWindowTitle("NEMA NU 2-2018 ROIs Editor")
        self.setGeometry(100, 100, 1600, 900)
        self._setup_ui()

        # Initialize ROI circles list
        self._roi_circles: List[pg.CircleROI] = []

        self._update_display()

    def _auto_detect_centers(self) -> List[List[int]]:
        """
        Auto-detect sphere centers using thresholding and labeling.

        Returns
        -------
        List[List[int]]
            List of [y, x] center coordinates for each detected sphere
        """
        logger.info("Auto-detecting sphere centers...")
        slice_img = self.image[self.central_slice]

        # Calculate threshold from percentile
        threshold = float(np.max(slice_img) * self.threshold_percentile)
        logger.info(
            f"Detection threshold: {threshold:.6f} (percentile {self.threshold_percentile}%)"
        )

        # Create binary mask and label objects
        binary_mask = slice_img > threshold
        labeled_mask, num_features = ndimage_label(binary_mask)  # type: ignore[misc]
        num_features = int(num_features)
        logger.info(f"Number of objects found: {num_features}")

        if num_features == 0:
            logger.warning("No objects detected. Using default positions.")
            result: List[List[int]] = [[100, 100]] * 6  # type: ignore[misc]
            return result

        # Calculate region sizes and centers
        region_info = []
        for i in range(1, num_features + 1):
            region_mask = labeled_mask == i
            com = center_of_mass(region_mask)
            com_rounded = [round(com[0]), round(com[1])]
            size = np.sum(region_mask)
            region_info.append((i, size, com_rounded))
            logger.debug(f"Region {i}: center = {com_rounded}, size = {size}")

        # Sort by size descending (largest first = largest diameter sphere)
        region_info.sort(key=lambda x: x[1], reverse=True)

        # Take centers of 6 largest regions (diameter 37, 28, 22, 17, 13, 10mm)
        final_centers = []
        for _i, _, com_rounded in region_info[:6]:
            final_centers.append(com_rounded)

        # Pad with defaults if less than 6 found
        while len(final_centers) < 6:
            final_centers.append([100, 100])

        logger.info(
            f"Detected {len([c for c in final_centers if c != [100, 100]])} ROI centers (expected 6)"
        )
        return final_centers

    def _setup_ui(self) -> None:
        """Setup PyQtGraph UI with ImageView and controls."""
        # Main widget and layout
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QHBoxLayout(main_widget)

        # Left side: Image views (PyQtGraph)
        image_layout = QtWidgets.QVBoxLayout()

        # Axial view
        self.view_axial = pg.ImageView()
        self.view_axial.ui.roiBtn.hide()
        self.view_axial.ui.menuBtn.hide()
        image_layout.addWidget(self.view_axial, 1)

        # Right side: Controls (Qt widgets)
        control_widget = QtWidgets.QWidget()
        control_widget.setStyleSheet(
            "QWidget { font-size: 11pt; } QGroupBox { font-size: 11pt; font-weight: bold; } QLabel { font-weight: bold; }"
        )
        control_layout = QtWidgets.QVBoxLayout(control_widget)
        control_layout.setSpacing(8)
        control_layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QtWidgets.QLabel("NEMA NU 2-2018 Editor")
        title.setStyleSheet("font-weight: bold; font-size: 14pt;")
        control_layout.addWidget(title)

        # ACTIVITY parameters (collapsible)
        act_group = QtWidgets.QGroupBox("ACTIVITY")
        act_layout = QtWidgets.QGridLayout()

        self.activity_hot_spin = QtWidgets.QDoubleSpinBox()
        self.activity_hot_spin.setLocale(QtCore.QLocale("en_US"))
        self.activity_hot_spin.setRange(0, 1.0)
        self.activity_hot_spin.setDecimals(10)
        self.activity_hot_spin.setValue(self.activity_hot)
        self.activity_hot_spin.setSingleStep(0.00001)

        self.activity_bg_spin = QtWidgets.QDoubleSpinBox()
        self.activity_bg_spin.setLocale(QtCore.QLocale("en_US"))
        self.activity_bg_spin.setRange(0, 0.001)
        self.activity_bg_spin.setDecimals(10)
        self.activity_bg_spin.setValue(self.activity_background)
        self.activity_bg_spin.setSingleStep(0.000001)

        self.activity_ratio_spin = QtWidgets.QDoubleSpinBox()
        self.activity_ratio_spin.setLocale(QtCore.QLocale("en_US"))
        self.activity_ratio_spin.setRange(1.0, 100.0)
        self.activity_ratio_spin.setValue(self.activity_ratio)
        self.activity_ratio_spin.setSingleStep(0.1)

        self.activity_units_edit = QtWidgets.QLineEdit()
        self.activity_units_edit.setText(self.activity_units)

        self.activity_total_edit = QtWidgets.QLineEdit()
        self.activity_total_edit.setText(self.activity_total)

        act_layout.addWidget(QtWidgets.QLabel("HOT:"), 0, 0)
        act_layout.addWidget(self.activity_hot_spin, 0, 1)
        act_layout.addWidget(QtWidgets.QLabel("BACKGROUND:"), 1, 0)
        act_layout.addWidget(self.activity_bg_spin, 1, 1)
        act_layout.addWidget(QtWidgets.QLabel("RATIO:"), 2, 0)
        act_layout.addWidget(self.activity_ratio_spin, 2, 1)
        act_layout.addWidget(QtWidgets.QLabel("UNITS:"), 3, 0)
        act_layout.addWidget(self.activity_units_edit, 3, 1)
        act_layout.addWidget(QtWidgets.QLabel("TOTAL:"), 4, 0)
        act_layout.addWidget(self.activity_total_edit, 4, 1)

        act_group.setLayout(act_layout)
        control_layout.addWidget(act_group)

        # Slice control
        slice_group = QtWidgets.QGroupBox("Slice")
        slice_layout = QtWidgets.QHBoxLayout()
        self.slice_spinbox = QtWidgets.QSpinBox()
        self.slice_spinbox.setRange(0, self.image.shape[0] - 1)
        self.slice_spinbox.setValue(self.central_slice)
        self.slice_spinbox.valueChanged.connect(self._on_slice_changed)
        slice_layout.addWidget(QtWidgets.QLabel("Z:"))
        slice_layout.addWidget(self.slice_spinbox)
        slice_group.setLayout(slice_layout)
        control_layout.addWidget(slice_group)

        # Orientation controls
        orient_group = QtWidgets.QGroupBox("Orientation")
        orient_layout = QtWidgets.QGridLayout()
        self.orient_y_spinbox = QtWidgets.QSpinBox()
        self.orient_y_spinbox.setRange(-1, 1)
        self.orient_y_spinbox.setValue(self.orientation_yx[0])
        self.orient_y_spinbox.valueChanged.connect(self._on_orientation_changed)
        self.orient_x_spinbox = QtWidgets.QSpinBox()
        self.orient_x_spinbox.setRange(-1, 1)
        self.orient_x_spinbox.setValue(self.orientation_yx[1])
        self.orient_x_spinbox.valueChanged.connect(self._on_orientation_changed)
        orient_layout.addWidget(QtWidgets.QLabel("Y:"), 0, 0)
        orient_layout.addWidget(self.orient_y_spinbox, 0, 1)
        orient_layout.addWidget(QtWidgets.QLabel("X:"), 1, 0)
        orient_layout.addWidget(self.orient_x_spinbox, 1, 1)
        orient_group.setLayout(orient_layout)
        control_layout.addWidget(orient_group)

        # ROI Centers (editable spinboxes)
        roi_group = QtWidgets.QGroupBox("ROI Centers (6 Spheres)")
        roi_layout = QtWidgets.QGridLayout()
        self.roi_spinboxes: List[Tuple[QtWidgets.QSpinBox, QtWidgets.QSpinBox]] = []

        for i in range(6):
            y, x = self.roi_centers[i]
            diameter = self.sphere_diameters[i]

            label = QtWidgets.QLabel(f"{diameter}mm")
            y_spin = QtWidgets.QSpinBox()
            y_spin.setRange(0, self.image.shape[1] - 1)
            y_spin.setValue(y)
            y_spin.valueChanged.connect(
                lambda val, idx=i: self._on_roi_center_changed(idx, val, True)
            )

            x_spin = QtWidgets.QSpinBox()
            x_spin.setRange(0, self.image.shape[2] - 1)
            x_spin.setValue(x)
            x_spin.valueChanged.connect(
                lambda val, idx=i: self._on_roi_center_changed(idx, val, False)
            )

            roi_layout.addWidget(label, i, 0)
            roi_layout.addWidget(QtWidgets.QLabel("Y:"), i, 1)
            roi_layout.addWidget(y_spin, i, 2)
            roi_layout.addWidget(QtWidgets.QLabel("X:"), i, 3)
            roi_layout.addWidget(x_spin, i, 4)

            self.roi_spinboxes.append((y_spin, x_spin))

        roi_group.setLayout(roi_layout)
        control_layout.addWidget(roi_group)

        # ACQUISITION parameters (collapsible)
        acq_group = QtWidgets.QGroupBox("ACQUISITION")
        acq_group.setCheckable(True)
        acq_group.setChecked(False)
        acq_layout = QtWidgets.QGridLayout()

        self.emission_time_spin = QtWidgets.QDoubleSpinBox()
        self.emission_time_spin.setLocale(QtCore.QLocale("en_US"))
        self.emission_time_spin.setRange(0.1, 60.0)
        self.emission_time_spin.setValue(self.emission_image_time_minutes)
        self.emission_time_spin.setSingleStep(0.5)
        acq_layout.addWidget(QtWidgets.QLabel("Emission Time (min):"), 0, 0)
        acq_layout.addWidget(self.emission_time_spin, 0, 1)

        acq_group.setLayout(acq_layout)
        control_layout.addWidget(acq_group)

        # FILE parameters (collapsible)
        file_group = QtWidgets.QGroupBox("FILE")
        file_group.setCheckable(True)
        file_group.setChecked(False)
        file_layout = QtWidgets.QGridLayout()

        self.file_pattern_edit = QtWidgets.QLineEdit()
        self.file_pattern_edit.setText(self.file_user_pattern)

        self.file_case_edit = QtWidgets.QLineEdit()
        self.file_case_edit.setText(self.file_case)

        file_layout.addWidget(QtWidgets.QLabel("USER_PATTERN:"), 0, 0)
        file_layout.addWidget(self.file_pattern_edit, 0, 1)
        file_layout.addWidget(QtWidgets.QLabel("CASE:"), 1, 0)
        file_layout.addWidget(self.file_case_edit, 1, 1)

        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)

        center_group = QtWidgets.QGroupBox("Phantom Center")
        center_group.setCheckable(True)
        center_group.setChecked(False)
        center_layout = QtWidgets.QGridLayout()
        self.center_method_combo = QtWidgets.QComboBox()
        self.center_method_combo.addItems(["weighted_slices", "max_slice"])
        self.center_method_combo.setCurrentText(self.center_method)
        self.center_method_combo.currentTextChanged.connect(
            self._on_center_params_changed
        )
        self.center_threshold_spinbox = QtWidgets.QDoubleSpinBox()
        self.center_threshold_spinbox.setLocale(QtCore.QLocale("en_US"))
        self.center_threshold_spinbox.setRange(0.01, 1.0)
        self.center_threshold_spinbox.setSingleStep(0.01)
        self.center_threshold_spinbox.setValue(self.center_threshold)
        self.center_threshold_spinbox.valueChanged.connect(
            self._on_center_params_changed
        )
        center_layout.addWidget(QtWidgets.QLabel("Method:"), 0, 0)
        center_layout.addWidget(self.center_method_combo, 0, 1)
        center_layout.addWidget(QtWidgets.QLabel("Threshold:"), 1, 0)
        center_layout.addWidget(self.center_threshold_spinbox, 1, 1)
        center_group.setLayout(center_layout)
        control_layout.addWidget(center_group)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.btn_redetect = QtWidgets.QPushButton("Re-detect")
        self.btn_redetect.clicked.connect(self._on_redetect)
        self.btn_generate = QtWidgets.QPushButton("Generate & Save YAML")
        self.btn_generate.clicked.connect(self._on_generate_yaml)
        button_layout.addWidget(self.btn_redetect)
        button_layout.addWidget(self.btn_generate)
        control_layout.addLayout(button_layout)

        control_layout.addStretch()

        # Wrap control layout in scroll area for better usability
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(control_widget)

        # Combine layouts
        splitter = QtWidgets.QSplitter()
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(image_layout)
        splitter.addWidget(left_widget)
        splitter.addWidget(scroll_area)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def _update_display(self) -> None:
        """Update all image displays."""
        # Axial view
        img_axial = self.image[self.central_slice]
        self.view_axial.setImage(img_axial, autoRange=False)

        cmap = pg.ColorMap(pos=[0.0, 1.0], color=[(255, 255, 255), (0, 0, 0)])

        self.view_axial.setColorMap(cmap)
        self.view_axial.getView().scene().sigMouseMoved.connect(self._mouse_moved)

        # Draw ROI overlays
        self._draw_roi_circles()

    def _mouse_moved(self, pos):
        vb = self.view_axial.getView()

        if not vb.sceneBoundingRect().contains(pos):
            return

        mouse_point = vb.mapSceneToView(pos)
        x = int(mouse_point.x())
        y = int(mouse_point.y())

        img = self.view_axial.imageItem.image
        if img is None:
            return

        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            val = img[y, x]
            self.statusBar().showMessage(f"x={x}, y={y}, val={val:.4f}")

    def _draw_roi_circles(self) -> None:
        """Draw interactive ROI circles on axial view."""
        # Clear previous ROIs
        if not hasattr(self, "_roi_circles"):
            self._roi_circles = []
        for circle in self._roi_circles:
            try:
                self.view_axial.removeItem(circle)
            except (ValueError, RuntimeError):
                pass  # Already removed
        self._roi_circles = []

        # Draw circles for each ROI sphere (centers are in Y,X format)
        for i, (x, y) in enumerate(self.roi_centers):
            diameter = self.sphere_diameters[i]
            radius_pix = (diameter / 2) / self.pixel_spacing
            color = self.sphere_colors[i]

            # Create circle ROI (PyQtGraph expects x, y)
            circle_roi = pg.CircleROI(
                (x - radius_pix, y - radius_pix),
                (radius_pix * 2, radius_pix * 2),
                pen=pg.mkPen(color, width=2),
                movable=False,
                rotatable=False,
                resizable=False,
            )
            self.view_axial.addItem(circle_roi)
            self._roi_circles.append(circle_roi)
            label = pg.TextItem(text=f"{diameter}mm", color=color, anchor=(0.5, -0.5))
            self.view_axial.addItem(label)
            label.setPos(x, y - radius_pix - 10)  # Position label above the circle
            self.view_axial.addItem(label)
            self._roi_circles.append(label)

        # Draw background ROIs offset from 37mm sphere center
        # Find the 37mm sphere (hot_sphere_37mm) - should be first in list (index 0)
        if len(self.roi_centers) > 0:
            centro_37_y, centro_37_x = self.roi_centers[0]
            background_radius = (37 / 2) / self.pixel_spacing

            for dy, dx in self.background_offset_yx:
                # Apply orientation to offsets
                dy_oriented = dy * self.orientation_yx[0]
                dx_oriented = dx * self.orientation_yx[1]

                # Calculate background ROI position
                bg_y = centro_37_y + dy_oriented
                bg_x = centro_37_x + dx_oriented

                # Create dashed circle for background ROI
                pen = pg.mkPen("orange", width=2)
                pen.setDashPattern([5, 5])
                background_roi = pg.CircleROI(
                    (bg_y - background_radius, bg_x - background_radius),
                    (background_radius * 2, background_radius * 2),
                    pen=pen,
                    movable=False,
                    rotatable=False,
                    resizable=False,
                )
                self.view_axial.addItem(background_roi)
                self._roi_circles.append(background_roi)

    def _on_slice_changed(self, value: int) -> None:
        """Handle slice change."""
        self.central_slice = value
        self._update_display()

    def _on_threshold_changed(self, value: float) -> None:
        """Handle threshold percentile change."""
        self.threshold_percentile = value
        # Re-detect centers with updated threshold
        self.roi_centers = self._auto_detect_centers()
        for i, (y, x) in enumerate(self.roi_centers):
            if i < len(self.roi_spinboxes):
                self.roi_spinboxes[i][0].setValue(y)
                self.roi_spinboxes[i][1].setValue(x)
        self._update_display()

    def _on_center_params_changed(self, _value: object = None) -> None:
        """Handle phantom center parameter changes."""
        self.center_method = self.center_method_combo.currentText()
        self.center_threshold = float(self.center_threshold_spinbox.value())
        # Re-detect centers with updated threshold
        self.roi_centers = self._auto_detect_centers()
        for i, (y, x) in enumerate(self.roi_centers):
            if i < len(self.roi_spinboxes):
                self.roi_spinboxes[i][0].setValue(y)
                self.roi_spinboxes[i][1].setValue(x)
        self._update_display()

    def _on_orientation_changed(self, value: int) -> None:
        """Handle orientation change."""
        self.orientation_yx = [
            self.orient_y_spinbox.value(),
            self.orient_x_spinbox.value(),
        ]
        self._update_display()

    def _on_roi_center_changed(self, idx: int, value: int, is_y: bool) -> None:
        """Handle ROI center change."""
        if is_y:
            self.roi_centers[idx][0] = value
        else:
            self.roi_centers[idx][1] = value
        self._draw_roi_circles()

    def _on_redetect(self) -> None:
        """Re-detect sphere centers."""
        self.roi_centers = self._auto_detect_centers()
        for i, (y, x) in enumerate(self.roi_centers):
            self.roi_spinboxes[i][0].setValue(y)
            self.roi_spinboxes[i][1].setValue(x)
        logger.info(f"Detected {len(self.roi_centers)} sphere centers")
        self._update_display()

    def _on_generate_yaml(self) -> None:
        """Generate and save YAML configuration."""
        # Update values from UI fields
        self.emission_image_time_minutes = self.emission_time_spin.value()
        self.activity_hot = self.activity_hot_spin.value()
        self.activity_background = self.activity_bg_spin.value()
        self.activity_ratio = self.activity_ratio_spin.value()
        self.activity_units = self.activity_units_edit.text()
        self.activity_total = self.activity_total_edit.text()
        self.file_user_pattern = self.file_pattern_edit.text()
        self.file_case = self.file_case_edit.text()

        # Generate YAML content
        yaml_content = self._generate_yaml_content()

        # Show file save dialog
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save YAML Configuration",
            "nema_config.yaml",
            "YAML Files (*.yaml);;All Files (*)",
        )

        if not file_path:
            logger.info("Save cancelled by user")
            return

        # Save to file
        try:
            with open(file_path, "w") as f:
                f.write(yaml_content)
            logger.info(f"YAML configuration saved to: {file_path}")
            QtWidgets.QMessageBox.information(
                self, "Success", f"Configuration saved successfully to:\n{file_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save YAML: {e}")
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to save configuration:\n{str(e)}"
            )

    def _generate_yaml_content(self) -> str:
        """Generate YAML configuration content."""
        yaml_lines = []

        # ACQUISITION section
        yaml_lines.append("ACQUISITION:")
        yaml_lines.append(
            f"  EMMISION_IMAGE_TIME_MINUTES: {self.emission_image_time_minutes}"
        )

        # ACTIVITY section
        yaml_lines.append("")
        yaml_lines.append("ACTIVITY:")
        yaml_lines.append(f"  HOT: {self.activity_hot:.10f}".rstrip("0").rstrip("."))
        yaml_lines.append(
            f"  BACKGROUND: {self.activity_background:.10f}".rstrip("0").rstrip(".")
        )
        yaml_lines.append(f'  UNITS: "{self.activity_units}"')
        yaml_lines.append(f"  RATIO: {self.activity_ratio}")
        yaml_lines.append(f'  ACTIVITY_TOTAL: "{self.activity_total}"')

        # PHANTHOM section
        yaml_lines.append("")
        yaml_lines.append("PHANTHOM:")
        yaml_lines.append("  ROI_DEFINITIONS_MM:")

        for i in range(6):
            y, x = self.roi_centers[i]
            yaml_lines.append(f"    - center_yx: [{y}, {x}]")
            yaml_lines.append(f"      diameter_mm: {self.sphere_diameters[i]}")
            yaml_lines.append(f'      color: "{self.sphere_colors[i]}"')
            yaml_lines.append("      alpha: 0.18")
            yaml_lines.append(f'      name: "{self.sphere_names[i]}"')

        # ROIS section
        yaml_lines.append("")
        yaml_lines.append("ROIS:")
        yaml_lines.append(f"  CENTRAL_SLICE: {self.central_slice}")
        yaml_lines.append(f"  ORIENTATION_YX: {self.orientation_yx}")

        # FILE section
        yaml_lines.append("")
        yaml_lines.append("FILE:")
        yaml_lines.append(f'  USER_PATTERN: "{self.file_user_pattern}"')
        yaml_lines.append(f'  CASE: "{self.file_case}"')

        return "\n".join(yaml_lines) + "\n"

    def show(self) -> None:
        """Show the editor window."""
        super().show()


class InteractiveROIEditorNU4(QtWidgets.QMainWindow):
    """GPU-accelerated interactive editor for NU 4-2008 IQ ROIs using PyQtGraph."""

    def __init__(
        self,
        image: npt.NDArray[Any],
        cfg: yacs.config.CfgNode,
        initial_slice: Optional[int] = None,
        spacing_override: Optional[float] = None,
        center_method: Optional[str] = None,
        center_threshold: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.image = image
        self.cfg = cfg

        self.central_slice = (
            int(initial_slice)
            if initial_slice is not None
            else int(getattr(cfg.ROIS, "CENTRAL_SLICE", image.shape[0] // 2))
        )

        self.spacing = float(
            spacing_override
            if spacing_override is not None
            else float(getattr(cfg.ROIS, "SPACING", 0.5))
        )

        self.orientation_yx = list(getattr(cfg.ROIS, "ORIENTATION_YX", [1, 1]))
        self.orientation_z = int(getattr(cfg.ROIS, "ORIENTATION_Z", 1))

        self.uniform_radius_mm = float(getattr(cfg.ROIS, "UNIFORM_RADIUS_MM", 11.25))
        self.uniform_height_mm = float(getattr(cfg.ROIS, "UNIFORM_HEIGHT_MM", 10.0))
        self.air_radius_mm = float(getattr(cfg.ROIS, "AIR_RADIUS_MM", 2.0))
        self.air_height_mm = float(getattr(cfg.ROIS, "AIR_HEIGHT_MM", 7.5))
        self.water_radius_mm = float(getattr(cfg.ROIS, "WATER_RADIUS_MM", 2.0))
        self.water_height_mm = float(getattr(cfg.ROIS, "WATER_HEIGHT_MM", 7.5))
        self.uniform_offset_mm = float(getattr(cfg.ROIS, "UNIFORM_OFFSET_MM", 2.5))
        self.airwater_offset_mm = float(getattr(cfg.ROIS, "AIRWATER_OFFSET_MM", 10.0))
        self.airwater_separation_mm = float(
            getattr(cfg.ROIS, "AIRWATER_SEPARATION_MM", 7.5)
        )
        self.statusBar().showMessage("Ready")

        self.center_method = center_method or getattr(
            cfg.ROIS, "PHANTOM_CENTER_METHOD", "weighted_slices"
        )
        self.center_threshold = float(
            center_threshold
            if center_threshold is not None
            else float(getattr(cfg.ROIS, "PHANTOM_CENTER_THRESHOLD_FRACTION", 0.41))
        )

        # Initialize RECONSTRUCTION parameters
        self.reconstruction_algorithm = getattr(cfg, "ALGORITHM", "OSEM")
        self.reconstruction_iterations = int(getattr(cfg, "ITERATIONS", 2))
        self.reconstruction_filter = getattr(cfg, "FILTER", "Gaussian")

        # Initialize ACTIVITY parameters for NU 4-2008
        self.activity_phantom_activity = "10 MBq"
        self.activity_time = "16:20:00"

        # Initialize FILE parameters
        self.file_user_pattern = "frame(\\d+)"
        self.file_case = "NU4_2008"

        self.roi_defs = [
            dict(roi, center_yx=list(roi["center_yx"]))
            for roi in getattr(cfg.PHANTHOM, "ROI_DEFINITIONS_MM", [])
        ]
        if not self.roi_defs:
            raise ValueError("PHANTHOM.ROI_DEFINITIONS_MM is empty in config")

        self._compute_phantom_center()
        if initial_slice is None:
            self.central_slice = int(self.phantom_center_z) + 42

        max_slice = self.image.shape[0] - 1
        if self.central_slice < 0 or self.central_slice > max_slice:
            clamped_slice = max(0, min(self.central_slice, max_slice))
            logger.info(
                "Central slice %s out of range, clamped to %s",
                self.central_slice,
                clamped_slice,
            )
            self.central_slice = clamped_slice

        # Auto-detect ROI centers on initial slice
        detected_centers = self._auto_detect_centers()
        for i, center in enumerate(detected_centers):
            if i < len(self.roi_defs):
                self.roi_defs[i]["center_yx"] = center

        # Cache for masks
        self._masks_cache: Dict[str, Any] = {}
        self._masks_cache_key: Optional[str] = None

        # Initialize ROI circles list
        self._roi_circles: List[pg.CircleROI] = []
        self._sagittal_overlays: List[pg.GraphicsObject] = []

        # Setup UI
        self.setWindowTitle("NEMA NU 4-2008 IQ ROIs")
        self.setGeometry(100, 100, 1600, 900)
        self._setup_ui()
        self._update_display()

    def _compute_phantom_center(self) -> None:
        ce_z, ce_y, ce_x = find_phantom_center_cv2_threshold(
            self.image,
            threshold_fraction=self.center_threshold,
            method=self.center_method,
        )
        self.phantom_center_z = int(ce_z)
        self.phantom_center_y = int(ce_y)
        self.phantom_center_x = int(ce_x)

    def _auto_detect_centers(self) -> List[List[int]]:
        """Auto-detect sphere centers using thresholding and labeling."""
        logger.info("Auto-detecting sphere centers...")
        slice_img = self.image[self.central_slice]
        detection_threshold = 0.25
        threshold = float(np.max(slice_img) * detection_threshold)
        logger.info(
            f"Detection threshold: {threshold:.6f} (using {detection_threshold*100}%)"
        )

        binary_mask = slice_img > threshold
        labeled_mask, num_features = ndimage_label(binary_mask)  # type: ignore[misc]
        num_features = int(num_features)
        logger.info(f"Number of objects found: {num_features}")

        if num_features == 0:
            logger.warning("No objects detected. Using default positions.")
            result: List[List[int]] = [[100, 100]] * len(self.roi_defs)  # type: ignore[misc]
            return result

        region_info = []
        for i in range(1, num_features + 1):
            region_mask = labeled_mask == i
            com = center_of_mass(region_mask)
            com_rounded = [round(com[0]), round(com[1])]
            size = np.sum(region_mask)
            region_info.append((i, size, com_rounded))
            logger.debug(f"Region {i}: center = {com_rounded}, size = {size}")

        region_info.sort(key=lambda x: x[1], reverse=False)
        final_centers = []
        num_rois = len(self.roi_defs)
        for _, _, com_rounded in region_info[:num_rois]:
            final_centers.append(com_rounded)

        while len(final_centers) < num_rois:
            final_centers.append([100, 100])

        logger.info(
            f"Detected {len([c for c in final_centers if c != [100, 100]])} ROI centers (expected {num_rois})"
        )
        return final_centers

    def _setup_ui(self) -> None:
        """Setup PyQtGraph UI with ImageView and controls."""
        # Main widget and layout
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QHBoxLayout(main_widget)

        # Left side: Image views (PyQtGraph)
        image_layout = QtWidgets.QVBoxLayout()

        # Axial view
        self.view_axial = pg.ImageView()
        self.view_axial.ui.roiBtn.hide()
        self.view_axial.ui.menuBtn.hide()
        image_layout.addWidget(self.view_axial, 1)

        # Sagittal view
        self.view_sagittal = pg.ImageView()
        self.view_sagittal.ui.roiBtn.hide()
        self.view_sagittal.ui.menuBtn.hide()
        image_layout.addWidget(self.view_sagittal, 1)

        # Right side: Controls (Qt widgets)
        control_widget = QtWidgets.QWidget()
        control_widget.setStyleSheet(
            "QWidget { font-size: 11pt; } QGroupBox { font-size: 11pt; font-weight: bold; } QLabel { font-weight: bold; }"
        )
        control_layout = QtWidgets.QVBoxLayout(control_widget)
        control_layout.setSpacing(8)
        control_layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QtWidgets.QLabel("NEMA NU 4-2008 Editor")
        title.setStyleSheet("font-weight: bold; font-size: 14pt;")
        control_layout.addWidget(title)

        # ACTIVITY parameters (collapsible)
        act_group = QtWidgets.QGroupBox("ACTIVITY")
        act_layout = QtWidgets.QGridLayout()

        self.activity_phantom_edit = QtWidgets.QLineEdit()
        self.activity_phantom_edit.setText(self.activity_phantom_activity)

        self.activity_time_edit = QtWidgets.QLineEdit()
        self.activity_time_edit.setText(self.activity_time)

        act_layout.addWidget(QtWidgets.QLabel("PHANTOM_ACTIVITY:"), 0, 0)
        act_layout.addWidget(self.activity_phantom_edit, 0, 1)
        act_layout.addWidget(QtWidgets.QLabel("ACTIVITY_TIME:"), 1, 0)
        act_layout.addWidget(self.activity_time_edit, 1, 1)

        act_group.setLayout(act_layout)
        control_layout.addWidget(act_group)

        # Slice control
        slice_group = QtWidgets.QGroupBox("Slice")
        slice_layout = QtWidgets.QHBoxLayout()
        self.slice_spinbox = QtWidgets.QSpinBox()
        self.slice_spinbox.setRange(0, self.image.shape[0] - 1)
        self.slice_spinbox.setValue(self.central_slice)
        self.slice_spinbox.valueChanged.connect(self._on_slice_changed)
        slice_layout.addWidget(QtWidgets.QLabel("Z:"))
        slice_layout.addWidget(self.slice_spinbox)
        slice_group.setLayout(slice_layout)
        control_layout.addWidget(slice_group)

        # Orientation controls
        orient_group = QtWidgets.QGroupBox("Orientation")
        orient_layout = QtWidgets.QHBoxLayout()

        self.orient_y_spinbox = QtWidgets.QSpinBox()
        self.orient_y_spinbox.setRange(-1, 1)
        self.orient_y_spinbox.setValue(self.orientation_yx[0])
        self.orient_y_spinbox.valueChanged.connect(self._on_orientation_changed)

        self.orient_x_spinbox = QtWidgets.QSpinBox()
        self.orient_x_spinbox.setRange(-1, 1)
        self.orient_x_spinbox.setValue(self.orientation_yx[1])
        self.orient_x_spinbox.valueChanged.connect(self._on_orientation_changed)

        self.orient_z_spinbox = QtWidgets.QSpinBox()
        self.orient_z_spinbox.setRange(-1, 1)
        self.orient_z_spinbox.setValue(self.orientation_z)
        self.orient_z_spinbox.valueChanged.connect(self._on_orientation_changed)

        orient_layout.addWidget(QtWidgets.QLabel("Y:"))
        orient_layout.addWidget(self.orient_y_spinbox)
        orient_layout.addWidget(QtWidgets.QLabel("X:"))
        orient_layout.addWidget(self.orient_x_spinbox)
        orient_layout.addWidget(QtWidgets.QLabel("Z:"))
        orient_layout.addWidget(self.orient_z_spinbox)

        orient_group.setLayout(orient_layout)
        control_layout.addWidget(orient_group)

        # ROI Centers (editable spinboxes)
        roi_group = QtWidgets.QGroupBox("ROI Centers")
        roi_layout = QtWidgets.QGridLayout()
        self.roi_spinboxes: List[Tuple[QtWidgets.QSpinBox, QtWidgets.QSpinBox]] = []
        roi_names = ["1mm", "2mm", "3mm", "4mm", "5mm"]

        for i, roi in enumerate(self.roi_defs[:5]):
            y, x = roi["center_yx"]
            roi_label = roi_names[i] if i < len(roi_names) else f"R{i+1}"

            label = QtWidgets.QLabel(roi_label)
            y_spin = QtWidgets.QSpinBox()
            y_spin.setRange(0, self.image.shape[1] - 1)
            y_spin.setValue(y)
            y_spin.valueChanged.connect(
                lambda val, idx=i: self._on_roi_center_changed(idx, val, True)
            )

            x_spin = QtWidgets.QSpinBox()
            x_spin.setRange(0, self.image.shape[2] - 1)
            x_spin.setValue(x)
            x_spin.valueChanged.connect(
                lambda val, idx=i: self._on_roi_center_changed(idx, val, False)
            )

            roi_layout.addWidget(label, i, 0)
            roi_layout.addWidget(QtWidgets.QLabel("Y:"), i, 1)
            roi_layout.addWidget(y_spin, i, 2)
            roi_layout.addWidget(QtWidgets.QLabel("X:"), i, 3)
            roi_layout.addWidget(x_spin, i, 4)

            self.roi_spinboxes.append((y_spin, x_spin))

        roi_group.setLayout(roi_layout)
        control_layout.addWidget(roi_group)

        # RECONSTRUCTION parameters (collapsible)
        recon_group = QtWidgets.QGroupBox("RECONSTRUCTION")
        recon_group.setCheckable(True)
        recon_group.setChecked(False)
        recon_layout = QtWidgets.QGridLayout()

        self.recon_algorithm_edit = QtWidgets.QLineEdit()
        self.recon_algorithm_edit.setText(self.reconstruction_algorithm)

        self.recon_iterations_spin = QtWidgets.QSpinBox()
        self.recon_iterations_spin.setRange(1, 100)
        self.recon_iterations_spin.setValue(self.reconstruction_iterations)

        self.recon_filter_edit = QtWidgets.QLineEdit()
        self.recon_filter_edit.setText(self.reconstruction_filter)

        recon_layout.addWidget(QtWidgets.QLabel("ALGORITHM:"), 0, 0)
        recon_layout.addWidget(self.recon_algorithm_edit, 0, 1)
        recon_layout.addWidget(QtWidgets.QLabel("ITERATIONS:"), 1, 0)
        recon_layout.addWidget(self.recon_iterations_spin, 1, 1)
        recon_layout.addWidget(QtWidgets.QLabel("FILTER:"), 2, 0)
        recon_layout.addWidget(self.recon_filter_edit, 2, 1)

        recon_group.setLayout(recon_layout)
        control_layout.addWidget(recon_group)

        # Cylinder parameters (editable)
        cyl_group = QtWidgets.QGroupBox("Cylinder Parameters")
        cyl_group.setCheckable(True)
        cyl_group.setChecked(False)
        cyl_layout = QtWidgets.QGridLayout()

        def _add_cyl_param(
            row: int,
            label: str,
            value: float,
            min_val: float = 0.01,
            max_val: float = 50.0,
        ) -> QtWidgets.QDoubleSpinBox:
            label_widget = QtWidgets.QLabel(label)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setLocale(QtCore.QLocale("en_US"))
            spin.setRange(min_val, max_val)
            spin.setValue(value)
            spin.setSingleStep(0.1)
            spin.valueChanged.connect(self._on_cylinder_params_changed)
            cyl_layout.addWidget(label_widget, row, 0)
            cyl_layout.addWidget(spin, row, 1)
            return spin

        uniform_header = QtWidgets.QLabel("Uniform Cylinder")
        uniform_header.setStyleSheet("font-weight: bold; font-size: 11pt;")
        cyl_layout.addWidget(uniform_header, 0, 0, 1, 2)
        self.uniform_radius_spin = _add_cyl_param(
            1, "Uniform Radius (mm):", self.uniform_radius_mm
        )
        self.uniform_height_spin = _add_cyl_param(
            2, "Uniform Height (mm):", self.uniform_height_mm
        )
        self.uniform_offset_spin = _add_cyl_param(
            3, "Uniform Offset (mm):", self.uniform_offset_mm
        )

        air_header = QtWidgets.QLabel("Air Cylinder")
        air_header.setStyleSheet("font-weight: bold; font-size: 11pt;")
        cyl_layout.addWidget(air_header, 4, 0, 1, 2)
        self.air_radius_spin = _add_cyl_param(5, "Air Radius (mm):", self.air_radius_mm)
        self.air_height_spin = _add_cyl_param(6, "Air Height (mm):", self.air_height_mm)

        water_header = QtWidgets.QLabel("Water Cylinder")
        water_header.setStyleSheet("font-weight: bold; font-size: 11pt;")
        cyl_layout.addWidget(water_header, 7, 0, 1, 2)
        self.water_radius_spin = _add_cyl_param(
            8, "Water Radius (mm):", self.water_radius_mm
        )
        self.water_height_spin = _add_cyl_param(
            9, "Water Height (mm):", self.water_height_mm
        )

        offsets_header = QtWidgets.QLabel("Air/Water Offsets")
        offsets_header.setStyleSheet("font-weight: bold; font-size: 11pt;")
        cyl_layout.addWidget(offsets_header, 10, 0, 1, 2)
        self.airwater_offset_spin = _add_cyl_param(
            11, "Air/Water Offset (mm):", self.airwater_offset_mm
        )
        self.airwater_sep_spin = _add_cyl_param(
            12, "Air/Water Separation (mm):", self.airwater_separation_mm
        )

        cyl_group.setLayout(cyl_layout)
        control_layout.addWidget(cyl_group)

        # Spacing control
        spacing_group = QtWidgets.QGroupBox("Spacing")
        spacing_layout = QtWidgets.QHBoxLayout()
        spacing_group.setCheckable(True)
        spacing_group.setChecked(False)
        self.spacing_spinbox = QtWidgets.QDoubleSpinBox()
        self.spacing_spinbox.setLocale(QtCore.QLocale("en_US"))
        self.spacing_spinbox.setRange(0.01, 10.0)
        self.spacing_spinbox.setValue(self.spacing)
        self.spacing_spinbox.setSingleStep(0.01)
        self.spacing_spinbox.valueChanged.connect(self._on_spacing_changed)
        spacing_layout.addWidget(QtWidgets.QLabel("mm:"))
        spacing_layout.addWidget(self.spacing_spinbox)
        spacing_group.setLayout(spacing_layout)
        control_layout.addWidget(spacing_group)

        # FILE parameters (collapsible)
        file_group = QtWidgets.QGroupBox("FILE")
        file_group.setCheckable(True)
        file_group.setChecked(False)
        file_layout = QtWidgets.QGridLayout()

        self.file_pattern_edit = QtWidgets.QLineEdit()
        self.file_pattern_edit.setText(self.file_user_pattern)

        self.file_case_edit = QtWidgets.QLineEdit()
        self.file_case_edit.setText(self.file_case)

        file_layout.addWidget(QtWidgets.QLabel("USER_PATTERN:"), 0, 0)
        file_layout.addWidget(self.file_pattern_edit, 0, 1)
        file_layout.addWidget(QtWidgets.QLabel("CASE:"), 1, 0)
        file_layout.addWidget(self.file_case_edit, 1, 1)

        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.btn_recompute = QtWidgets.QPushButton("Recompute")
        self.btn_recompute.clicked.connect(self._on_recompute)
        self.btn_generate = QtWidgets.QPushButton("Generate & Save YAML")
        self.btn_generate.clicked.connect(self._on_generate_yaml)
        button_layout.addWidget(self.btn_recompute)
        button_layout.addWidget(self.btn_generate)
        control_layout.addLayout(button_layout)

        control_layout.addStretch()

        # Wrap control layout in scroll area for better usability
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(control_widget)

        # Combine layouts
        splitter = QtWidgets.QSplitter()
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(image_layout)
        splitter.addWidget(left_widget)
        splitter.addWidget(scroll_area)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def _update_display(self) -> None:
        """Update all image displays."""
        # Axial view
        img_axial = self.image[self.central_slice]
        self.view_axial.setImage(img_axial, autoRange=False)
        # Apply grayscale lookup table
        cmap = pg.ColorMap(pos=[0.0, 1.0], color=[(255, 255, 255), (0, 0, 0)])
        self.view_axial.setColorMap(cmap)

        # Add ROI overlays as circular regions
        self._draw_roi_circles()

        # Sagittal view (Y,Z at fixed X)
        sagittal_img = self.image[:, :, int(self.phantom_center_x)]
        self.view_sagittal.setImage(sagittal_img, autoRange=False)
        self.view_sagittal.setColorMap(cmap)
        self._draw_sagittal_overlays()

    def _draw_roi_circles(self) -> None:
        """Draw interactive ROI circles on axial view."""
        # Clear previous ROIs
        if not hasattr(self, "_roi_circles"):
            self._roi_circles = []
        for circle in self._roi_circles:
            try:
                self.view_axial.removeItem(circle)
            except (ValueError, RuntimeError):
                pass  # Already removed
        self._roi_circles = []

        # Draw circles for each ROI (centers are in Y,X format)
        for roi in self.roi_defs:
            y, x = roi["center_yx"]
            radius_pix = (roi["diameter_mm"] / 2) / self.spacing
            color = roi.get("color", "yellow")

            # Create circle ROI
            circle_roi = pg.CircleROI(
                (y - radius_pix, x - radius_pix),
                (radius_pix * 2, radius_pix * 2),
                pen=pg.mkPen(color, width=2),
                movable=False,
                rotatable=False,
                resizable=False,
            )
            self.view_axial.addItem(circle_roi)
            self._roi_circles.append(circle_roi)
            label_text = f"{roi.get('diameter_mm', 0)}mm"
            label_item = pg.TextItem(text=label_text, color=color, anchor=(0.5, 0.5))
            label_item.setPos(y, x + radius_pix + 1)  # Position label above the circle
            self.view_axial.addItem(label_item)
            self._roi_circles.append(label_item)
            self.view_axial.getView().scene().sigMouseMoved.connect(self._mouse_moved)

    def _draw_sagittal_overlays(self) -> None:
        """Draw sagittal cylinder ROIs and phantom center marker."""
        view_box = self.view_sagittal.getView()
        for item in self._sagittal_overlays:
            try:
                view_box.removeItem(item)
            except (ValueError, RuntimeError):
                pass
        self._sagittal_overlays = []

        uniform_mask, air_mask, water_mask = self._build_masks()
        x_idx = int(self.phantom_center_x)
        uniform_slice = uniform_mask[:, :, x_idx].astype(np.float32)
        air_slice = air_mask[:, :, x_idx].astype(np.float32)
        water_slice = water_mask[:, :, x_idx].astype(np.float32)

        for mask_slice, color, label in (
            (uniform_slice, (255, 0, 0), "Uniform"),
            (air_slice, (0, 0, 255), "Air"),
            (water_slice, (0, 160, 0), "Water"),
        ):
            item = pg.ImageItem(mask_slice)
            item.setLookupTable(self._mask_lut(color, alpha=180))
            item.setLevels([0, 1])
            item.setZValue(10)
            view_box.addItem(item)
            self._sagittal_overlays.append(item)

            label_pos = self._mask_label_pos(mask_slice)
            if label_pos is not None:
                label_item = pg.TextItem(
                    text=label, color=(0, 0, 0), anchor=(0.5, -0.2)
                )
                label_item.setPos(label_pos[1], label_pos[0])
                label_item.setZValue(15)
                view_box.addItem(label_item)
                self._sagittal_overlays.append(label_item)

        plane_x = np.array([0, self.image.shape[1] - 1], dtype=float)
        plane_y = np.array([self.central_slice, self.central_slice], dtype=float)
        plane_pen = pg.mkPen((255, 140, 0), width=2)
        plane_pen.setDashPattern([6, 4])
        plane_item = pg.PlotDataItem(plane_y, plane_x, pen=plane_pen)
        plane_item.setZValue(12)
        view_box.addItem(plane_item)
        self._sagittal_overlays.append(plane_item)

        plane_label = pg.TextItem(
            text=f"Axial ROI slice (Z={self.central_slice})",
            color=(0, 0, 0),
            anchor=(0, 0.5),
        )

        plane_label.setPos(float(self.central_slice), 5)
        plane_label.setZValue(5)
        view_box.addItem(plane_label)
        self._sagittal_overlays.append(plane_label)

        center_item = pg.ScatterPlotItem(
            [self.phantom_center_z],
            [self.phantom_center_y],
            pen=pg.mkPen("red"),
            brush=pg.mkBrush("red"),
            size=10,
            symbol="x",
        )
        center_item.setZValue(20)
        view_box.addItem(center_item)
        self._sagittal_overlays.append(center_item)

    def _mask_lut(
        self, color: Tuple[int, int, int], alpha: int = 180
    ) -> npt.NDArray[np.uint8]:
        """Create an RGBA LUT with transparent zero for mask overlays."""
        lut = np.zeros((256, 4), dtype=np.uint8)
        lut[1:, 0] = color[0]
        lut[1:, 1] = color[1]
        lut[1:, 2] = color[2]
        lut[1:, 3] = alpha
        return lut

    def _mask_label_pos(
        self, mask_slice: npt.NDArray[np.float32]
    ) -> Optional[Tuple[float, float]]:
        """Return (y, x) centroid for a mask slice, if any pixels are set."""
        coords = np.argwhere(mask_slice > 0)
        if coords.size == 0:
            return None
        center_yx = coords.mean(axis=0)
        return float(center_yx[1]), float(center_yx[0])

    def _mouse_moved(self, pos):
        vb = self.view_axial.getView()

        if not vb.sceneBoundingRect().contains(pos):
            return

        mouse_point = vb.mapSceneToView(pos)
        x = int(mouse_point.x())
        y = int(mouse_point.y())

        img = self.view_axial.imageItem.image
        if img is None:
            return

        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            val = img[y, x]
            self.statusBar().showMessage(f"x={x}, y={y}, val={val:.4f}")

    def _on_slice_changed(self, value: int) -> None:
        """Handle slice change."""
        self.central_slice = value
        self._update_display()

    def _on_spacing_changed(self, value: float) -> None:
        """Handle spacing change."""
        self.spacing = value
        self._update_display()

    def _on_cylinder_params_changed(self, _value: float) -> None:
        """Handle cylinder parameter changes."""
        self.uniform_radius_mm = float(self.uniform_radius_spin.value())
        self.uniform_height_mm = float(self.uniform_height_spin.value())
        self.uniform_offset_mm = float(self.uniform_offset_spin.value())
        self.air_radius_mm = float(self.air_radius_spin.value())
        self.air_height_mm = float(self.air_height_spin.value())
        self.water_radius_mm = float(self.water_radius_spin.value())
        self.water_height_mm = float(self.water_height_spin.value())
        self.airwater_offset_mm = float(self.airwater_offset_spin.value())
        self.airwater_separation_mm = float(self.airwater_sep_spin.value())
        self._masks_cache_key = None
        self._update_display()

    def _on_center_params_changed(self, _value: object = None) -> None:
        """Handle phantom center parameter changes."""
        self.center_method = self.center_method_combo.currentText()
        self.center_threshold = float(self.center_threshold_spinbox.value())
        self._compute_phantom_center()
        self._update_display()

    def _on_orientation_changed(self, value: int) -> None:
        """Handle orientation change."""
        self.orientation_yx = [
            self.orient_y_spinbox.value(),
            self.orient_x_spinbox.value(),
        ]
        self.orientation_z = self.orient_z_spinbox.value()
        self._update_display()

    def _on_roi_center_changed(self, idx: int, value: int, is_y: bool) -> None:
        """Handle ROI center change."""
        if is_y:
            self.roi_defs[idx]["center_yx"][0] = value
        else:
            self.roi_defs[idx]["center_yx"][1] = value
        self._draw_roi_circles()

    def _on_recompute(self) -> None:
        """Recompute sphere centers."""
        detected_centers = self._auto_detect_centers()
        for i, center in enumerate(detected_centers):
            if i < len(self.roi_defs):
                self.roi_defs[i]["center_yx"] = center
                self.roi_spinboxes[i][0].setValue(center[0])
                self.roi_spinboxes[i][1].setValue(center[1])
        logger.info(f"Detected {len(detected_centers)} ROI centers")
        self._update_display()

    def _on_generate_yaml(self) -> None:
        """Generate and save YAML configuration."""
        # Update values from UI fields
        self.reconstruction_algorithm = self.recon_algorithm_edit.text()
        self.reconstruction_iterations = self.recon_iterations_spin.value()
        self.reconstruction_filter = self.recon_filter_edit.text()
        self.activity_phantom_activity = self.activity_phantom_edit.text()
        self.activity_time = self.activity_time_edit.text()
        self.file_user_pattern = self.file_pattern_edit.text()
        self.file_case = self.file_case_edit.text()

        # Generate YAML content
        yaml_content = self._generate_yaml_content()

        # Show file save dialog
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save YAML Configuration",
            "nema_nu4_config.yaml",
            "YAML Files (*.yaml);;All Files (*)",
        )

        if not file_path:
            logger.info("Save cancelled by user")
            return

        # Save to file
        try:
            with open(file_path, "w") as f:
                f.write(yaml_content)
            logger.info(f"YAML configuration saved to: {file_path}")
            QtWidgets.QMessageBox.information(
                self, "Success", f"Configuration saved successfully to:\n{file_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save YAML: {e}")
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to save configuration:\n{str(e)}"
            )

    def _generate_yaml_content(self) -> str:
        """Generate YAML configuration content."""
        yaml_lines = []

        # RECONSTRUCTION section
        yaml_lines.append("RECONSTRUCTION:")
        yaml_lines.append(f'  ALGORITHM: "{self.reconstruction_algorithm}"')
        yaml_lines.append(f"  ITERATIONS: {self.reconstruction_iterations}")
        yaml_lines.append(f'  FILTER: "{self.reconstruction_filter}"')

        # ACTIVITY section
        yaml_lines.append("")
        yaml_lines.append("ACTIVITY:")
        yaml_lines.append(f'  PHANTOM_ACTIVITY: "{self.activity_phantom_activity}"')
        yaml_lines.append(f'  ACTIVITY_TIME: "{self.activity_time}"')

        # PHANTHOM section
        yaml_lines.append("")
        yaml_lines.append("PHANTHOM:")
        yaml_lines.append("  ROI_DEFINITIONS_MM:")

        for roi in self.roi_defs:
            y, x = roi["center_yx"]
            yaml_lines.append(f"    - center_yx: [{y}, {x}]")
            yaml_lines.append(f"      diameter_mm: {roi['diameter_mm']}")
            yaml_lines.append(f'      color: "{roi.get("color", "red")}"')
            yaml_lines.append(f"      alpha: {roi.get('alpha', 0.18)}")
            yaml_lines.append(f'      name: "{roi.get("name", "roi")}"')

        # ROIS section
        yaml_lines.append("")
        yaml_lines.append("ROIS:")
        yaml_lines.append(f"  CENTRAL_SLICE: {self.central_slice}")
        yaml_lines.append(f"  ORIENTATION_YX: {self.orientation_yx}")
        yaml_lines.append(f"  ORIENTATION_Z: {self.orientation_z}")
        yaml_lines.append(f"  SPACING: {self.spacing}")
        yaml_lines.append(f"  UNIFORM_RADIUS_MM: {self.uniform_radius_mm}")
        yaml_lines.append(f"  UNIFORM_HEIGHT_MM: {self.uniform_height_mm}")
        yaml_lines.append(f"  AIR_RADIUS_MM: {self.air_radius_mm}")
        yaml_lines.append(f"  AIR_HEIGHT_MM: {self.air_height_mm}")
        yaml_lines.append(f"  WATER_RADIUS_MM: {self.water_radius_mm}")
        yaml_lines.append(f"  WATER_HEIGHT_MM: {self.water_height_mm}")
        yaml_lines.append(f"  UNIFORM_OFFSET_MM: {self.uniform_offset_mm}")
        yaml_lines.append(f"  AIRWATER_OFFSET_MM: {self.airwater_offset_mm}")
        yaml_lines.append(f"  AIRWATER_SEPARATION_MM: {self.airwater_separation_mm}")

        # FILE section
        yaml_lines.append("")
        yaml_lines.append("FILE:")
        yaml_lines.append(f'  USER_PATTERN: "{self.file_user_pattern}"')
        yaml_lines.append(f'  CASE: "{self.file_case}"')

        return "\n".join(yaml_lines) + "\n"

    def _build_masks(
        self,
    ) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """Build cylinder masks with caching."""
        cache_key = (
            f"{self.uniform_radius_mm}_{self.uniform_height_mm}_"
            f"{self.air_radius_mm}_{self.air_height_mm}_"
            f"{self.water_radius_mm}_{self.water_height_mm}_"
            f"{self.uniform_offset_mm}_{self.airwater_offset_mm}_"
            f"{self.airwater_separation_mm}_{self.orientation_yx}_{self.orientation_z}"
        )

        if self._masks_cache_key == cache_key and "uniform" in self._masks_cache:
            return (
                self._masks_cache["uniform"],
                self._masks_cache["air"],
                self._masks_cache["water"],
            )

        spacing_xyz = (self.spacing, self.spacing, self.spacing)
        uniform_center_z = self.phantom_center_z + self.orientation_z * (
            self.uniform_offset_mm / self.spacing
        )
        airwater_center_z = self.phantom_center_z - self.orientation_z * (
            self.airwater_offset_mm / self.spacing
        )

        uniform_center = (
            uniform_center_z,
            self.phantom_center_y,
            self.phantom_center_x,
        )
        air_center = (
            airwater_center_z,
            self.phantom_center_y
            - self.orientation_yx[0] * (self.airwater_separation_mm / self.spacing),
            self.phantom_center_x,
        )
        water_center = (
            airwater_center_z,
            self.phantom_center_y
            + self.orientation_yx[0] * (self.airwater_separation_mm / self.spacing),
            self.phantom_center_x,
        )

        uniform_mask = create_cylindrical_mask(
            shape_zyx=self.image.shape,
            center_zyx=uniform_center,
            radius_mm=self.uniform_radius_mm,
            height_mm=self.uniform_height_mm,
            spacing_xyz=spacing_xyz,
        )
        air_mask = create_cylindrical_mask(
            shape_zyx=self.image.shape,
            center_zyx=air_center,
            radius_mm=self.air_radius_mm,
            height_mm=self.air_height_mm,
            spacing_xyz=spacing_xyz,
        )
        water_mask = create_cylindrical_mask(
            shape_zyx=self.image.shape,
            center_zyx=water_center,
            radius_mm=self.water_radius_mm,
            height_mm=self.water_height_mm,
            spacing_xyz=spacing_xyz,
        )

        self._masks_cache = {
            "uniform": uniform_mask,
            "air": air_mask,
            "water": water_mask,
        }
        self._masks_cache_key = cache_key

        return uniform_mask, air_mask, water_mask

    def show(self) -> None:
        """Show the editor window."""
        super().show()


def main() -> None:
    """Main entry point for the interactive ROI editor."""
    parser = argparse.ArgumentParser(
        description="Interactive ROI Editor for NEMA Phantom Configuration"
    )
    parser.add_argument(
        "--standard",
        choices=["NU_2_2018", "NU_4_2008"],
        default="NU_2_2018",
        help="NEMA standard to edit (default: NU_2_2018)",
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
        default=None,
        help="Pixel spacing in mm used to size ROIs (default: from NIfTI or 2.0644)",
    )
    parser.add_argument(
        "--center-method",
        type=str,
        default=None,
        help="Phantom center detection method (NU 4-2008)",
    )
    parser.add_argument(
        "--center-threshold",
        type=float,
        default=None,
        help="Phantom center detection threshold fraction (NU 4-2008)",
    )

    args = parser.parse_args()

    # Load image
    logger.info(f"Loading image: {args.input_image}")
    image_path = Path(args.input_image)

    if not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        return

    image_array_3d, affine = load_nii_image(filepath=image_path, return_affine=True)

    if args.standard == "NU_4_2008":
        # Create QApplication for PyQtGraph
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        cfg = get_cfg_defaults()
        spacing_override = None
        if affine is not None:
            voxel_spacing = (
                float(np.abs(affine[0, 0])),
                float(np.abs(affine[1, 1])),
                float(np.abs(affine[2, 2])),
            )
            spacing_override = voxel_spacing[0]
            if not np.isclose(voxel_spacing[0], voxel_spacing[1]) or not np.isclose(
                voxel_spacing[0], voxel_spacing[2]
            ):
                logger.info(
                    "Non-isotropic spacing detected (x=%.4f, y=%.4f, z=%.4f); using x spacing",
                    voxel_spacing[0],
                    voxel_spacing[1],
                    voxel_spacing[2],
                )
        if args.spacing is not None:
            spacing_override = float(args.spacing)

        logger.info("Starting NU 4-2008 interactive ROI editor...")
        editor = InteractiveROIEditorNU4(
            image=image_array_3d,
            cfg=cfg,
            initial_slice=args.slice,
            spacing_override=spacing_override,
            center_method=args.center_method,
            center_threshold=args.center_threshold,
        )
        editor.show()
        app.exec()
        return

    # NU 2-2018 uses PyQtGraph, so create QApplication
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    # Determine initial slice for NU 2-2018
    if args.slice is None:
        try:
            z_com, _, _ = find_phantom_center(
                image_array_3d, threshold=np.max(image_array_3d) * 0.41
            )
            initial_slice = int(np.ceil(z_com))
            logger.info(
                f"Auto-detected phantom COM slice: {initial_slice} (z={z_com:.3f})"
            )
        except (ValueError, RuntimeError) as exc:
            initial_slice = image_array_3d.shape[0] // 2
            logger.warning(
                "Phantom COM could not be computed. Falling back to middle slice: "
                f"{initial_slice}. Reason: {exc}"
            )
    else:
        initial_slice = args.slice

    # Create and show editor
    logger.info("Starting NU 2-2018 interactive ROI editor...")
    spacing_value = (
        float(args.spacing) if args.spacing is not None else DEFAULT_PIXEL_SPACING
    )
    editor = InteractiveROIEditor(
        image=image_array_3d,
        initial_slice=initial_slice,
        threshold_percentile=args.threshold,
        pixel_spacing=spacing_value,
    )
    editor.show()
    app.exec()


if __name__ == "__main__":
    main()

"""
defaults.py

This module sets up the default configuration for training and evaluating 3D
image models using the yacs library. It includes model architecture parameters,
training settings, data paths, optimizer configurations, and
other miscellaneous options.

Dependencies:
    - yacs.config: For hierarchical configuration management.

Owner:
    Edwing Ulin

Version:
    v1.0.0
"""

from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------- #
# Nema Tools Options           #
# ---------------------------- #

_C.ACQUISITION = CN()
_C.ACQUISITION.EMMISION_IMAGE_TIME_MINUTES = 10

_C.ACTIVITY = CN()
_C.ACTIVITY.HOT = 0.79
_C.ACTIVITY.BACKGROUND = 0.079
_C.ACTIVITY.RATIO = 9.91
_C.ACTIVITY.UNITS = "mCi"
_C.ACTIVITY.ACTIVITY_TOTAL = "29.24 MBq"

_C.PHANTHOM = CN()
_C.PHANTHOM.ROI_DEFINITIONS_MM = [
    {
        "center_yx": (211, 171),
        "diameter_mm": 37,
        "color": "red",
        "alpha": 0.18,
        "name": "hot_sphere_37mm",
    },
    {
        "center_yx": (187, 184),
        "diameter_mm": 28,
        "color": "orange",
        "alpha": 0.18,
        "name": "hot_sphere_28mm",
    },
    {
        "center_yx": (187, 212),
        "diameter_mm": 22,
        "color": "gold",
        "alpha": 0.18,
        "name": "hot_sphere_22mm",
    },
    {
        "center_yx": (211, 226),
        "diameter_mm": 17,
        "color": "lime",
        "alpha": 0.18,
        "name": "hot_sphere_17mm",
    },
    {
        "center_yx": (235, 212),
        "diameter_mm": 13,
        "color": "cyan",
        "alpha": 0.18,
        "name": "hot_sphere_13mm",
    },
    {
        "center_yx": (235, 184),
        "diameter_mm": 10,
        "color": "blue",
        "alpha": 0.18,
        "name": "hot_sphere_10mm",
    },
]

_C.ROIS = CN()
_C.ROIS.CENTRAL_SLICE = 172

_C.ROIS.BACKGROUND_OFFSET_YX = [
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
_C.ROIS.ORIENTATION_YX = [1, 1]

_C.ROIS.SPACING = 2.0644

_C.FILE = CN()
_C.FILE.USER_PATTERN = r"frame(\d+)"
_C.FILE.CASE = "Test"

_C.STYLE = CN()
_C.STYLE.COLORS = [
    "#023743FF",
    "#72874EFF",
    "#476F84FF",
    "#A4BED5FF",
    "#453947FF",
    "#8C7A6BFF",
    "#C97D60FF",
    "#F0B533FF",
]
_C.STYLE.PLT_STYLE = "seaborn-v0_8-talk"
_C.STYLE.RCPARAMS = [
    ("font.size", 24),
    ("axes.titlesize", 24),
    ("axes.labelsize", 24),
    ("xtick.labelsize", 24),
    ("ytick.labelsize", 24),
    ("legend.fontsize", 24),
    ("legend.title_fontsize", 24),
    ("lines.linewidth", 2.5),
    ("lines.markersize", 8),
    ("axes.linewidth", 1.2),
    ("font.family", "DejaVu Sans"),
]

_C.STYLE.LEGEND = CN()
_C.STYLE.LEGEND.LABELPAD = 20
_C.STYLE.LEGEND.FONTWEIGHT = "bold"  # or Normal, Light, Heavy, etc.

_C.STYLE.GRID = CN()
_C.STYLE.GRID.LINESTYLE = "--"
_C.STYLE.GRID.LINEWIDTH = 2.0
_C.STYLE.GRID.ALPHA = 0.3
_C.STYLE.GRID.COLOR = "gray"

_C.STYLE.PLOT = CN()
_C.STYLE.PLOT.DEFAULT = CN()
_C.STYLE.PLOT.DEFAULT.COLOR = "#666666FF"
_C.STYLE.PLOT.DEFAULT.LINEWIDTH = 1.0
_C.STYLE.PLOT.DEFAULT.ALPHA = 0.6
_C.STYLE.PLOT.DEFAULT.ZORDER = 5
_C.STYLE.PLOT.DEFAULT.LINESTYLE = "--"
_C.STYLE.PLOT.DEFAULT.MARKERSIZE = 4
_C.STYLE.PLOT.DEFAULT.MARKEREDGEWIDTH = 0.5

_C.STYLE.PLOT.ENHANCED = CN()
_C.STYLE.PLOT.ENHANCED.LINEWIDTH = 4.0
_C.STYLE.PLOT.ENHANCED.ALPHA = 1.0
_C.STYLE.PLOT.ENHANCED.ZORDER = 30
_C.STYLE.PLOT.ENHANCED.LINESTYLE = "-"
_C.STYLE.PLOT.ENHANCED.MARKERSIZE = 15
_C.STYLE.PLOT.ENHANCED.MARKEREDGEWIDTH = 2.0


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()

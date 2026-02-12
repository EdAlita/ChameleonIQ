"""
defaults
========

Default configuration settings for NEMA phantom analysis tools.

This module establishes the default configuration hierarchy for NEMA (National
Electrical Manufacturers Association) phantom analysis using the yacs (Yet Another
Configuration System) library. It provides sensible defaults for acquisition
parameters, activity measurements, phantom geometry, ROI definitions, file patterns,
and visualization styles that can be easily overridden for different datasets and
use cases.

The configuration is organized into logical sections using nested CfgNode objects,
enabling hierarchical access and modification of settings without affecting other
configuration areas.

Configuration Sections
----------------------
ACQUISITION : CfgNode
    Acquisition parameters for PET imaging.

    EMMISION_IMAGE_TIME_MINUTES : int
        Duration of emission image acquisition in minutes (default: 10).

ACTIVITY : CfgNode
    Activity concentration measurements for phantom calibration.

    HOT : float
        Activity concentration in hot sphere regions (default: 0.79 mCi/mL).
    BACKGROUND : float
        Activity concentration in background region (default: 0.079 mCi/mL).
    RATIO : float
        Ratio of hot to background activity (default: 9.91).
    UNITS : str
        Activity measurement units (default: "mCi").
    ACTIVITY_TOTAL : str
        Total activity in the phantom (default: "29.24 MBq").

PHANTHOM : CfgNode
    EARL NEMA phantom geometry and ROI definitions.

    ROI_DEFINITIONS_MM : list of dict
        List of 6 hot sphere definitions with keys:

        - center_yx : tuple of int
            Center position in (y, x) image coordinates
        - diameter_mm : float
            Sphere diameter in millimeters
        - color : str
            Visualization color name
        - alpha : float
            Transparency value [0, 1]
        - name : str
            Descriptive sphere identifier

ROIS : CfgNode
    Region of Interest specifications for analysis.

    CENTRAL_SLICE : int
        Central axial slice index for analysis (default: 172).
    BACKGROUND_OFFSET_YX : list of tuple
        12 background sampling positions as (y, x) offsets from phantom center.
    ORIENTATION_YX : list of int
        Image orientation indicators [y_orient, x_orient] (default: [1, 1]).
    SPACING : float
        Pixel spacing in mm (default: 2.0644).

FILE : CfgNode
    File naming patterns and identifiers.

    USER_PATTERN : str
        Regular expression to extract frame numbers from filenames
        (default: r"frame(\\d+)").
    CASE : str
        Case identifier for output labeling (default: "Test").

STYLE : CfgNode
    Visualization and matplotlib styling parameters.

    COLORS : list of str
        List of 8 hex color codes for plot series.
    PLT_STYLE : str
        Matplotlib style template (default: "seaborn-v0_8-talk").
    RCPARAMS : list of tuple
        Matplotlib rcParams as (key, value) pairs for font sizes,
        line widths, and font family.
    LEGEND : CfgNode
        Legend styling with LABELPAD (int) and FONTWEIGHT (str).
    GRID : CfgNode
        Grid styling with LINESTYLE, LINEWIDTH, ALPHA, and COLOR.
    PLOT : CfgNode
        Plot styling with DEFAULT and ENHANCED substyles.

        DEFAULT : CfgNode
            Base plot styling (dashed lines, lower alpha).
        ENHANCED : CfgNode
            Highlighted plot styling (solid lines, full opacity).

Functions
---------
get_cfg_defaults()
    Returns a cloned copy of the default configuration object.

Returns
-------
CfgNode
    A yacs CfgNode object containing all default configuration values.

Notes
-----
- All configuration values are stored in the module-level `_C` variable
- Use `get_cfg_defaults()` to obtain a modifiable clone rather than the
  original `_C` object
- Hot spheres follow EARL NEMA IQ phantom specification (6 spheres: 37, 28,
  22, 17, 13, 10 mm diameters)
- Background sampling uses 12 distributed offset positions around phantom center
- Visualization colors use 8-digit hex format (#RRGGBBAA) with alpha channel
- ROI positions use (y, x) convention to match NumPy array indexing

Examples
--------
Access default configuration:

    >>> from src.config.defaults import get_cfg_defaults
    >>> cfg = get_cfg_defaults()
    >>> acquisition_time = cfg.ACQUISITION.EMMISION_IMAGE_TIME_MINUTES
    >>> print(acquisition_time)
    10

Modify configuration:

    >>> cfg.ACQUISITION.EMMISION_IMAGE_TIME_MINUTES = 20
    >>> cfg.ACTIVITY.UNITS = "MBq/mL"
    >>> num_spheres = len(cfg.PHANTHOM.ROI_DEFINITIONS_MM)
    >>> print(num_spheres)
    6

Access nested styling parameters:

    >>> default_color = cfg.STYLE.PLOT.DEFAULT.COLOR
    >>> enhanced_linewidth = cfg.STYLE.PLOT.ENHANCED.LINEWIDTH

References
----------
- NEMA NU 2-2018 Standard for PET imaging performance
- EARL NEMA IQ phantom specifications
- yacs documentation: https://github.com/rbgirshick/yacs

See Also
--------
yacs.config.CfgNode : Base configuration node class
src.config : Package containing all configuration modules

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
    """
    Get a cloned yacs CfgNode object with default values.

    Returns a deep copy of the module-level default configuration object `_C`,
    containing all NEMA analysis tool settings. Cloning prevents modifications
    from affecting the original defaults.

    Returns
    -------
    CfgNode
        A cloned configuration node containing:

        - ACQUISITION: Emission imaging parameters
        - ACTIVITY: Activity measurements and ratios
        - PHANTHOM: Phantom geometry and ROI definitions
        - ROIS: Region of interest specifications
        - FILE: File pattern matching and naming
        - STYLE: Visualization and plotting parameters

    Notes
    -----
    Always use this function rather than accessing `_C` directly to ensure
    configuration isolation between independent uses.

    Examples
    --------
    Create independent configuration instances:

        >>> cfg1 = get_cfg_defaults()
        >>> cfg2 = get_cfg_defaults()
        >>> cfg1.ACQUISITION.EMMISION_IMAGE_TIME_MINUTES = 15
        >>> cfg2.ACQUISITION.EMMISION_IMAGE_TIME_MINUTES
        10

    """
    return _C.clone()

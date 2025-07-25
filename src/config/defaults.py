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

_C.PHANTHOM = CN()
_C.PHANTHOM.ROI_DEFINITIONS_MM = [
    {'center_yx': (211, 171), 'diameter_mm': 37, 'color':'red',    'alpha':0.18, 'name':"hot_sphere_37mm"},
    {'center_yx': (187, 184), 'diameter_mm': 28, 'color':'orange', 'alpha':0.18, 'name':"hot_sphere_28mm"},
    {'center_yx': (187, 212), 'diameter_mm': 22, 'color':'gold',   'alpha':0.18, 'name':"hot_sphere_22mm"},
    {'center_yx': (211, 226), 'diameter_mm': 17, 'color':'lime',   'alpha':0.18, 'name':"hot_sphere_17mm"},
    {'center_yx': (235, 212), 'diameter_mm': 13, 'color':'cyan',   'alpha':0.18, 'name':"hot_sphere_13mm"},
    {'center_yx': (235, 184), 'diameter_mm': 10, 'color':'blue',   'alpha':0.18, 'name':"hot_sphere_10mm"}
]

_C.ROIS = CN()
_C.ROIS.CENTRAL_SLICE = 172
_C.ROIS.BACKGROUND_OFFSET_YX = [
    (-16, -28), (-33, -19), (-40, -1), (-35, 28), (-39, 50), (-32, 69),
    (-15, 79), (3, 76), (19, 65), (34, 51), (38, 28), (25, -3)
]
_C.ROIS.SPACING = 2.0644

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()

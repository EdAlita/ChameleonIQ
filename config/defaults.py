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
# Nema Tools Options
# ---------------------------- #
_C.ACTIVITY = CN()

_C.ACTIVITY.HOT = 0.79
_C.ACTIVITY.BACKGROUND = 0.079
_C.ACTIVITY.RATIO = 9.91

_C.ROIS = CN()
_C.ROIS.CENTRAL_SLICE = 171
_C.ROIS.BACKGROUND_CENTER_YX = [
    (161, 231), (200, 250), (200, 170), (155, 205), (153, 193), (162, 185),
    (235, 235), (242, 225), (245, 214), (245, 202), (242, 190), (235, 180)
]


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

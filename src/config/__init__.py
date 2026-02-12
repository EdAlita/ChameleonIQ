"""
config
======

Configuration management module for NEMA analysis tools.

This package provides centralized configuration management for the NEMA (National
Electrical Manufacturers Association) phantom analysis tools. It handles default
settings for acquisition parameters, activity measurements, phantom definitions,
ROI (Region of Interest) specifications, file patterns, and visualization styles.

The module uses the yacs (Yet Another Configuration System) library to provide
hierarchical configuration management, allowing easy override of default values
for different use cases and datasets.

Submodules
----------
defaults : module
    Contains the default configuration settings for all NEMA tools, including:

    - Acquisition parameters (emission image time)
    - Activity measurements (hot/background ratios, units)
    - Phantom definitions (ROI positions and dimensions)
    - ROI configurations (central slice, background offsets, orientation)
    - File naming patterns and case identifiers
    - Visualization styles (colors, plot parameters, grid settings)

Configuration Structure
-----------------------
The configuration is organized into logical sections:

ACQUISITION
    Emission image acquisition parameters
ACTIVITY
    Activity concentration measurements and ratios
PHANTHOM
    Phantom geometry and ROI definitions
ROIS
    Region of Interest specifications
FILE
    File pattern matching and naming conventions
STYLE
    Visualization and plotting parameters

Key Configuration Groups
------------------------
- **ACQUISITION**: Emission imaging time (default: 10 minutes)
- **ACTIVITY**: Activity concentrations with units (mCi/mL or MBq)
- **PHANTHOM**: 6 hot spheres with diameters (37, 28, 22, 17, 13, 10 mm)
- **ROIS**: Central slice definition and background sampling points
- **FILE**: User pattern for frame numbering and case identifiers
- **STYLE**: Color schemes, matplotlib rcParams, legend/grid/plot styling

Usage
-----
Import the default configuration:

    from src.config.defaults import get_cfg_defaults
    cfg = get_cfg_defaults()

Access and modify configuration values:

    cfg.ACQUISITION.EMMISION_IMAGE_TIME_MINUTES = 15
    activity_ratio = cfg.ACTIVITY.RATIO

Notes
-----
- Configuration uses hierarchical CfgNode structure from yacs
- All YAML configuration files in this directory can be merged with defaults
- ROI definitions include center coordinates (y, x), diameter, color, and name
- Style settings apply to matplotlib visualization output

See Also
--------
defaults : Default configuration implementation
yacs : Yet Another Configuration System for hierarchical configs

"""

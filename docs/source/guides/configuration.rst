Configuration Guide
====================

Overview
--------

ChameleonIQ is configuration-driven so it can adapt to different scanners,
phantoms, and activity protocols. Most calculations and reports depend on the
YAML values you provide.

Why Configuration Matters
-------------------------

- Ensures correct activity ratio for percent contrast
- Converts millimeters to voxels with the right spacing
- Aligns ROI positions with your phantom setup
- Controls plot styling for consistent reports

Configuration Sections
----------------------

ACQUISITION
~~~~~~~~~~~

Emission imaging parameters::

    ACQUISITION:
      EMMISION_IMAGE_TIME_MINUTES: 10

- **EMMISION_IMAGE_TIME_MINUTES** (int): Duration of acquisition in minutes

ACTIVITY
~~~~~~~~

Activity concentration measurements::

    ACTIVITY:
      HOT: 0.79
      BACKGROUND: 0.079
      RATIO: 9.91
      UNITS: "mCi"
      ACTIVITY_TOTAL: "29.24 MBq"

- **HOT** (float): Activity in hot sphere regions
- **BACKGROUND** (float): Activity in background regions
- **RATIO** (float): Hot-to-background ratio used in the formula
- **UNITS** (str): Activity units (mCi, MBq, etc.)
- **ACTIVITY_TOTAL** (str): Total phantom activity

PHANTHOM
~~~~~~~~

Phantom geometry and ROI definitions::

    PHANTHOM:
      ROI_DEFINITIONS_MM:
        - center_yx: [211, 171]
          diameter_mm: 37
          color: "red"
          alpha: 0.18
          name: "hot_sphere_37mm"

ROIS
~~~~

Region of Interest specifications::

    ROIS:
      CENTRAL_SLICE: 172
      BACKGROUND_OFFSET_YX: [[-16, -28], ...]
      ORIENTATION_YX: [1, 1]
      SPACING: 2.0644

FILE
~~~~

File naming and patterns::

    FILE:
      USER_PATTERN: "frame(\\d+)"
      CASE: "Test"

STYLE
~~~~~

Visualization parameters::

    STYLE:
      COLORS: ["#023743FF", ...]
      PLT_STYLE: "seaborn-v0_8-talk"
      RCPARAMS:
        - ["font.size", 24]
        - ["axes.titlesize", 24]

Real-World Examples
-------------------

- High-resolution scanner (1mm voxels): ``SPACING: 1.0``
- Standard clinical scanner: ``SPACING: 2.0644``
- High-contrast protocol: increase ``ACTIVITY.RATIO``

Full Example YAML
-----------------

::

    ACQUISITION:
      EMMISION_IMAGE_TIME_MINUTES: 34

    ACTIVITY:
      HOT: 31.22
      BACKGROUND: 3.76
      UNITS: "kBq/mL"
      RATIO: 8.30
      ACTIVITY_TOTAL: "37.42 MBq"

    PHANTHOM:
      ROI_DEFINITIONS_MM:
        - center_yx: [213, 225]
          diameter_mm: 37
          color: "red"
          alpha: 0.18
          name: "hot_sphere_37mm"

    ROIS:
      SPACING: 2.0644
      CENTRAL_SLICE: 165
      ORIENTATION_YX: [1, -1]

    FILE:
      USER_PATTERN: "frame(\\d+)"
      CASE: "Test"

Using Configuration
-------------------

From Python
~~~~~~~~~~~

.. code-block:: python

    from config.defaults import get_cfg_defaults

    # Get default config
    cfg = get_cfg_defaults()

    # Modify settings
    cfg.ACQUISITION.EMMISION_IMAGE_TIME_MINUTES = 20
    cfg.ACTIVITY.UNITS = "MBq/mL"

    # Use in analysis
    metrics = analysis.calculate_image_quality_metrics(phantom, cfg=cfg)

From Command Line
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    nema_quant --input image.nii.gz --config custom.yaml --output results/

Custom YAML File
~~~~~~~~~~~~~~~~

Create ``custom_config.yaml``::

    ACQUISITION:
      EMMISION_IMAGE_TIME_MINUTES: 15

    ACTIVITY:
      HOT: 0.85
      BACKGROUND: 0.085
      RATIO: 10.0
      UNITS: "MBq/mL"

    FILE:
      CASE: "My_Study"

For detailed configuration options, see :doc:`../api/config`.

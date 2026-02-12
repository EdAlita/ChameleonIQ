nema_quant
==================

Overview
--------

``nema_quant`` provides comprehensive tools for automated analysis of PET/CT image quality
according to NEMA NU 2-2018 standard. It includes phantom detection, segmentation, quality
metrics computation, ROI analysis, and professional reporting capabilities.

.. contents:: Table of Contents
   :local:
   :depth: 2

Main Features
-------------

- **Phantom Detection**: Automatic detection and segmentation of NEMA phantoms
- **Image Quality Metrics**: Contrast, noise, recovery coefficient, spillover ratio
- **ROI Analysis**: Interactive and automated region-of-interest extraction
- **Report Generation**: Automated PDF/TXT reports with visualizations
- **Batch Processing**: CLI for high-throughput analysis
- **NEMA NU 2-2018 Compliance**: Follows international standards

.. Module Contents
.. ---------------

.. .. automodule:: nema_quant
..    :members:
..    :undoc-members:
..    :show-inheritance:
..    :no-index:

Submodules
----------

The package is organized into the following submodules:

.. toctree::
   :maxdepth: 1

   nema_quant_analysis
   nema_quant_phantom
   nema_quant_reporting
   nema_quant_interactive_roi_editor
   nema_quant_utils
   nema_quant_io

Quick Start
-----------

Command Line Usage
~~~~~~~~~~~~~~~~~~

Basic Analysis
^^^^^^^^^^^^^^

Analyze a single PET/CT image with required arguments:

.. code-block:: bash

   chameleoniq_quant input.nii --config custom_config.yaml --output results.txt

For compressed NIfTI files:

.. code-block:: bash

   chameleoniq_quant input.nii.gz --config custom_config.yaml --output results.txt

With Verbose Output
^^^^^^^^^^^^^^^^^^^^

Enable detailed logging:

.. code-block:: bash

   chameleoniq_quant input.nii --config custom_config.yaml --output results.txt --log_level DEBUG

Advanced Options
^^^^^^^^^^^^^^^^

Save visualization images of ROI masks:

.. code-block:: bash

   chameleoniq_quant input.nii --config custom_config.yaml --output results.txt --save-visualizations --visualizations-dir ./roi_masks

Provide explicit voxel spacing (if not in image header):

.. code-block:: bash

   chameleoniq_quant input.nii --config custom_config.yaml --output results.txt --spacing 3.5 3.5 3.27

Calculate advanced segmentation metrics with ground truth image:

.. code-block:: bash

   chameleoniq_quant input.nii --config custom_config.yaml --output results.txt --advanced-metrics --gt-image ground_truth.nii.gz

Command Line Arguments
^^^^^^^^^^^^^^^^^^^^^^

Required Arguments
'''''''''''''''''''

**input_image** (positional)
   Path to input NIfTI image file (``.nii`` or ``.nii.gz``)

**--output, -o**
   Path to output file for results (required)

**--config, -c**
   Path to custom YAML configuration file (required). See :doc:`../guides/configuration` for reference configuration parameters.

Optional Arguments
'''''''''''''''''''

**--save-visualizations**
   Save visualization images of ROI masks and analysis regions as PNG files.

   Default: ``False``

**--spacing**
   Voxel spacing in millimeters as three float values: ``x y z``

   Example: ``--spacing 3.5 3.5 3.27``

   Only needed if spacing is missing from image header.

**--visualizations-dir**
   Directory path where to save visualization images.

   Default: ``visualizations``

   Example: ``--visualizations-dir ./output/rois``

**--advanced-metrics, -a**
   Calculate advanced segmentation metrics (Dice, Hausdorff distance, etc.).

   Requires ground truth image with ``--gt-image`` argument.

   Default: ``False``

**--gt-image**
   Path to ground truth NIfTI image file for calculating advanced metrics.

   Only used when ``--advanced-metrics`` is specified.

   Example: ``--gt-image reference_segmentation.nii.gz``

**--log_level**
   Set logging verbosity level.

   Options: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``

   Default: ``DEBUG``

**--version**
   Display version information and exit.

Examples
''''''''

Basic workflow:

.. code-block:: bash

   chameleoniq_quant scan_001.nii.gz \
     --config analysis_config.yaml \
     --output results/scan_001_report.txt

Full workflow with visualizations and advanced metrics:

.. code-block:: bash

   chameleoniq_quant scan_001.nii.gz \
     --config analysis_config.yaml \
     --output results/scan_001_report.txt \
     --save-visualizations \
     --visualizations-dir results/roi_masks \
     --advanced-metrics \
     --gt-image reference_segmentation.nii.gz \
     --log_level INFO

Batch processing script:

.. code-block:: bash

   for image in data/*.nii.gz; do
     output="${image%.nii.gz}_results.txt"
     chameleoniq_quant "$image" \
       --config default_config.yaml \
       --output "$output" \
       --log_level INFO
   done

Configuration
-------------

Configuration is defined in a YAML file and provided to the CLI via ``--config``.
For the full schema, defaults, and examples, see :doc:`../guides/configuration`.

Image Quality Metrics
---------------------

The package computes the following NEMA NU 2-2018 metrics:

Percent Contrast (Q_H,j)
~~~~~~~~~~~~~~~~~~~~~~~~~

Measures the contrast between hot and background regions:

.. math::

   Q_{H,j} = \frac{A_H - A_B}{A_B} \times 100\%

where :math:`A_H` is mean activity in hot sphere and :math:`A_B` is mean activity in background.

Background Variability (N_j)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measures noise in background regions:

.. math::

   N_j = \frac{\sigma_B}{A_B} \times 100\%

where :math:`\sigma_B` is standard deviation and :math:`A_B` is mean activity in background.

Recovery Coefficient (RC)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Measures activity recovery in lesions:

.. math::

   RC = \frac{C_{meas} - C_{bg}}{C_{true} - C_{bg}}


Supported File Formats
----------------------

Input Formats
~~~~~~~~~~~~~

- **NIfTI**: ``.nii``, ``.nii.gz`` (Recommended)
- **DICOM**: ``.dcm``
- **MetaImage**: ``.mhd``, ``.raw``

Output Formats
~~~~~~~~~~~~~~

- **Reports**: HTML, PDF
- **Data**: JSON, CSV
- **Images**: PNG, SVG

Examples
--------

Batch Processing
~~~~~~~~~~~~~~~~

Process multiple images:

.. code-block:: python

   from pathlib import Path
   from config.defaults import get_cfg_defaults
   from nema_quant.io import load_nii_image
   from nema_quant.phantom import NemaPhantom
   from nema_quant.analysis import calculate_nema_metrics
   from nema_quant.reporting import save_results_to_txt

   cfg = get_cfg_defaults()
   image_dir = Path('images/')
   output_dir = Path('results/')
   output_dir.mkdir(parents=True, exist_ok=True)

   for image_path in image_dir.glob('*.nii.gz'):
       # Load image
       image_data, affine = load_nii_image(image_path, return_affine=True)

       # Extract properties
       image_dims = image_data.shape
       voxel_spacing = (
           float(abs(affine[0, 0])),
           float(abs(affine[1, 1])),
           float(abs(affine[2, 2]))
       )

       # Analyze
       phantom = NemaPhantom(cfg, image_dims, voxel_spacing)
       results, lung_results = calculate_nema_metrics(image_data, phantom, cfg)

       # Save results
       output_file = output_dir / f"{image_path.stem}_metrics.txt"
       save_results_to_txt(results, output_file, cfg, image_path, voxel_spacing)

Custom ROI Analysis
~~~~~~~~~~~~~~~~~~~

The phantom ROIs are automatically initialized from configuration. To work with custom ROI positions:

.. code-block:: python

   from config.defaults import get_cfg_defaults
   from nema_quant.phantom import NemaPhantom
   from nema_quant.analysis import calculate_nema_metrics

   cfg = get_cfg_defaults()
   phantom = NemaPhantom(cfg, image_dims, voxel_spacing)

   # Access pre-defined ROIs from phantom
   for roi_name, roi_data in phantom.rois.items():
       center = roi_data['center_vox']
       diameter = roi_data['diameter_mm']
       print(f"{roi_name}: center={center}, diameter={diameter}mm")

   # Perform analysis with default ROIs
   results, lung_results = calculate_nema_metrics(image_data, phantom, cfg)

Advanced Visualization
~~~~~~~~~~~~~~~~~~~~~~

Generate visualizations of analysis results:

.. code-block:: python

   import matplotlib.pyplot as plt
   from nema_quant.reporting import (
       generate_plots,
       generate_rois_plots,
       generate_torso_plot
   )

   output_dir = Path('visualizations')
   output_dir.mkdir(exist_ok=True)

   # Generate NEMA metric plots
   generate_plots(results=results, output_dir=output_dir, cfg=cfg)

   # Generate ROI location plots
   generate_rois_plots(image=image_data, output_dir=output_dir, cfg=cfg)

   # Generate torso visualization
   generate_torso_plot(image=image_data, output_dir=output_dir, cfg=cfg)

See Also
--------

- :doc:`../guides/how_it_works` - Technical details of the analysis pipeline
- :doc:`../guides/configuration` - Complete configuration reference
- :doc:`config` - Configuration package documentation
- :doc:`../usage` - Comprehensive usage guide

Standards Compliance
--------------------

This implementation follows:

- **NEMA NU 2-2018**: Standards for Performance Measurements of Positron Emission Tomographs [NEMA2018]_
- **EARL Guidelines**: European Association of Nuclear Medicine Research Limited [EARL]_
- **IQ Phantom Specifications**: NEMA Image Quality phantom standards

References
----------

.. [NEMA2018] NEMA Standards Publication NU 2-2018: Performance Measurements of
              Positron Emission Tomographs. National Electrical Manufacturers
              Association, 2018.

.. [EARL] EARL Guidelines for PET/CT Systems. European Association of Nuclear
          Medicine, https://earl.eanm.org/

Notes
-----

- All spatial coordinates use (y, x, z) convention to match NumPy array indexing
- ROI positions are in millimeters relative to phantom center
- Activity concentrations can be specified in mCi/mL, MBq/mL, or kBq/mL
- Hot spheres follow EARL NEMA IQ phantom specification (6 spheres: 37, 28, 22, 17, 13, 10 mm)

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Image not loading:**
   Check that the file format is supported and the path is correct.

**Incorrect ROI detection:**
   Verify that the phantom is properly aligned and the central slice is correct.

**Memory errors:**
   For large images, consider processing on a machine with more RAM or using batch processing.

**Configuration errors:**
   Ensure YAML syntax is correct and all required fields are present.

For more help, see the `GitHub Issues <https://github.com/EdAlita/ChameleonIQ/issues>`_.

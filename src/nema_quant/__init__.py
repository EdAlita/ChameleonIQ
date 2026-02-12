"""

``nema_quant`` is a comprehensive tool for analyzing PET/CT image quality according to
NEMA NU 2-2018 standard. It provides automated quantification of image quality metrics,
phantom segmentation, and detailed reporting capabilities.

Overview
--------

This package performs:

- **Phantom Detection & Segmentation**: Automatic detection and segmentation of NEMA phantoms
  in PET/CT images using image processing and computer vision techniques.
- **Image Quality Metrics**: Computation of standard metrics including contrast, noise,
  recovery coefficient,background variability, and Lung Insert Corrections.
- **ROI Analysis**: Interactive and automated region-of-interest (ROI) extraction and analysis
  with comprehensive visualization.
- **Report Generation**: Automated generation of detailed PDF/TXT reports with visualizations
  and statistical summaries.

Main Modules
------------

- ``analysis`` : Core analysis functions for image quality quantification and metric computation
- ``cli`` : Command-line interface for batch processing and scripting
- ``io`` : Input/output utilities for loading and saving image data and results
- ``phantom`` : Phantom models and segmentation algorithms
- ``utils`` : General utility functions for image processing and data manipulation
- ``reporting`` : Report generation and visualization tools
- ``interactive_roi_editor`` : GUI tool for interactive ROI editing and refinement

Quick Start
-----------

Command Line Usage
~~~~~~~~~~~~~~~~~~~

Execute NEMA analysis from the command line::

    nema_quant --input image.nii.gz --config config.yaml --output results/

Python API
~~~~~~~~~~

Use the Python API for programmatic access::

    from nema_quant import analysis
    from nema_quant.phantom import NemaPhantom

    # Load and analyze image
    phantom = NemaPhantom(image_path='image.nii.gz')
    metrics = analysis.calculate_nema_metrics(image_data, phantom, config.Node())

Configuration
-------------

Configuration is managed through YAML files or programmatically. See the
``config`` module
for available options and default settings.

Standards Compliance
--------------------

This implementation follows the **NEMA NU 2-2018 standard** for:

- Image Quality Assessment in Nuclear Medicine
- PET/CT System Evaluation
- Quantitative Metrics and Thresholds

"""

nema_quant_iter
=======================

Iterative NEMA image quality analysis across reconstruction iterations.

Overview
--------

``nema_quant_iter`` analyzes a folder of reconstruction iterations and produces
iteration-wise NEMA IQ metrics, plots, and reports. Iteration numbers are
extracted from filenames using the regex pattern in ``FILE.USER_PATTERN``.

.. contents:: Table of Contents
   :local:
   :depth: 2

Main Features
-------------

- **Batch Iteration Analysis**: Process a directory of iteration images
- **Weighted CBR/FOM Tracking**: Identify peak weighted CBR and FOM
- **Publication Plots**: Contrast and background variability vs sphere diameter
- **Reports & CSV Export**: Text, PDF, PNG, and CSV outputs

Module Contents
---------------

.. automodule:: nema_quant_iter
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Quick Start
-----------

Command Line Usage
~~~~~~~~~~~~~~~~~~

Basic analysis:

.. code-block:: bash

   chameleoniq_quant_iter /path/to/iterations \
     --config config.yaml \
     --output results/iter_analysis.txt

With visualizations and explicit spacing:

.. code-block:: bash

   chameleoniq_quant_iter /path/to/iterations \
     --config config.yaml \
     --output results/iter_analysis.txt \
     --save-visualizations \
     --visualizations-dir results/roi_masks \
     --spacing 2.0 2.0 2.0

Inputs
------

Iteration folder
~~~~~~~~~~~~~~~~

The input path must be a folder containing **.nii** files. Iteration numbers are
parsed from filenames using ``FILE.USER_PATTERN`` in the YAML configuration.
For example, if your files are ``frame01.nii``, ``frame02.nii``, use:

.. code-block:: yaml

   FILE:
     USER_PATTERN: "frame(\\d+)"

If no files match the pattern, the analysis stops with an error.

Configuration
-------------

The CLI requires a YAML config file (``--config``). See
:doc:`../guides/configuration` for full defaults and parameter reference. The
most important config value for iterative analysis is:

- ``FILE.USER_PATTERN``: regex with a capture group for the iteration number

Command Line Arguments
----------------------

Required
~~~~~~~~

**input_path** (positional)
   Folder containing iteration ``.nii`` files

**--output, -o**
   Output text report path (required)

**--config, -c**
   YAML configuration file (required)

Optional
~~~~~~~~

**--spacing**
   Voxel spacing in millimeters as three floats: ``x y z``

**--save-visualizations**
   Save ROI mask and analysis images

**--visualizations-dir**
   Directory to save visualization images (default: ``visualizations``)

**--log_level**
   Logging verbosity (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL)

**--verbose, -v**
   Enable verbose error output

Outputs
-------

The command produces:

- Text report at the path provided by ``--output``
- PDF report with the same base name as ``--output``
- Plots in ``<output_dir>/png`` (contrast, background, convergence, lung plots)
- CSV data in ``<output_dir>/csv``
- Log files in ``<output_dir>/logs``

Tips
----

- Use consistent iteration naming so the regex captures the correct number.
- Provide ``--spacing`` if spacing is missing from the NIfTI header.
- Start with a subset of iterations to validate the pipeline, then run full sets.

Troubleshooting
---------------

**No iteration files found:**
   Ensure the input folder has ``.nii`` files and the regex in
   ``FILE.USER_PATTERN`` matches the filenames.

**Spacing warning:**
   Provide ``--spacing x y z`` when voxel spacing is not stored in the header.

See Also
--------

- :doc:`nema_quant` - Core NEMA analysis workflow
- :doc:`../guides/configuration` - Configuration reference

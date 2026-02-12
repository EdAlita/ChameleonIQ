Usage Guide
===========

Command Line Interface
----------------------

Main Commands
~~~~~~~~~~~~~

The CLI is available under both ``nema_*`` and ``chameleoniq_*`` names. Examples
below use the branded names.

chameleoniq_quant
^^^^^^^^^^^^^^^^^

Required:

- ``input_image``
- ``--output``
- ``--config``

Optional:

- ``--save-visualizations``
- ``--spacing``
- ``--visualizations-dir``
- ``--advanced-metrics`` and ``--gt-image``
- ``--log_level`` (DEBUG, INFO, WARNING, ERROR, CRITICAL)

Example (basic)::

    chameleoniq_quant input.nii.gz --config config.yaml --output results.txt

Example (advanced)::

    chameleoniq_quant input.nii.gz --config config.yaml --output results.txt \
        --advanced-metrics --gt-image gt.nii.gz --save-visualizations

chameleoniq_quant_iter
^^^^^^^^^^^^^^^^^^^^^^

Required:

- ``input_path`` (directory of iterations)
- ``--output``
- ``--config``

Optional:

- ``--save-visualizations``
- ``--visualizations-dir``
- ``--log_level``
- ``--spacing``
- ``--verbose``

Example::

    chameleoniq_quant_iter /path/to/iterations --config config.yaml --output results.txt

chameleoniq_merge
^^^^^^^^^^^^^^^^^

Required:

- ``xml_config``
- ``--output``
- ``--config``

Example::

    chameleoniq_merge experiments.xml --config config.yaml --output merged/

XML configuration uses experiment entries with paths to result files and optional
dose and plot status flags.

chameleoniq_coord
^^^^^^^^^^^^^^^^^

Convert between mm and voxel coordinates::

    chameleoniq_coord mm2vox 58.84 23.74 -30.97 --dims 391 391 346 --spacing 2.0644 2.0644 2.0644
    chameleoniq_coord vox2mm 158 207 158 --dims 391 391 346 --spacing 2.0644 2.0644 2.0644

Python API
----------

Basic Workflow
~~~~~~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    from config.defaults import get_cfg_defaults
    from nema_quant.io import load_nii_image
    from nema_quant.phantom import NemaPhantom
    from nema_quant.analysis import calculate_nema_metrics
    from nema_quant.reporting import generate_reportlab_report, save_results_to_txt

    # Load configuration and image
    cfg = get_cfg_defaults()
    image_path = Path('image.nii.gz')
    image_data, affine = load_nii_image(image_path, return_affine=True)

    # Extract image properties
    image_dims = image_data.shape
    voxel_spacing = (
        2.0644,
        2.0644,
        2.0644
    )

    # Initialize phantom and analyze
    phantom = NemaPhantom(cfg, image_dims, voxel_spacing)
    results, lung_results = calculate_nema_metrics(image_data, phantom, cfg)

    # Save results
    output_path = Path('results.txt')
    save_results_to_txt(results, output_path, cfg, image_path, voxel_spacing)

    # Generate report
    pdf_output = output_path.with_suffix('.pdf')
    lung_results_str = {str(k): v for k, v in lung_results.items()}
    generate_reportlab_report(results, pdf_output, cfg, image_path, voxel_spacing, lung_results_str)

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from config.defaults import get_cfg_defaults
    from nema_quant.analysis import calculate_nema_metrics

    # Load and customize configuration
    cfg = get_cfg_defaults()
    cfg.ACQUISITION.EMMISION_IMAGE_TIME_MINUTES = 20
    cfg.ACTIVITY.HOT = 0.95
    cfg.ACTIVITY.BACKGROUND = 0.095

    # Use custom config in analysis
    results, lung_results = calculate_nema_metrics(
        image_data,
        phantom,
        cfg,
        save_visualizations=True,
        visualizations_dir='visualizations'
    )

Interactive ROI Editor
~~~~~~~~~~~~~~~~~~~~~~

Run the interactive ROI detector from the CLI::

    chameleoniq_roi_detector path/to/image.nii.gz

Nota (centros de esferas)
~~~~~~~~~~~~~~~~~~~~~~~~~

Si no se conocen los centros de las esferas, se debe ejecutar el
``chameleoniq_roi_detector`` para generar el YAML con las posiciones. Si ya se
conocen, se pueden ingresar directamente en el archivo YAML.

Input Formats
-------------

Supported Image Formats
~~~~~~~~~~~~~~~~~~~~~~~~

- NIfTI (.nii, .nii.gz)
- DICOM (.dcm)
- MetaImage (.mhd, .raw)

Configuration Files
~~~~~~~~~~~~~~~~~~~

YAML configuration example::

    ACQUISITION:
      EMMISION_IMAGE_TIME_MINUTES: 10

    ACTIVITY:
      HOT: 0.79
      BACKGROUND: 0.079
      RATIO: 9.91

    FILE:
      USER_PATTERN: "frame(\\d+)"
      CASE: "My_Study"

Output Results
--------------

The output text report includes:

- Analysis configuration
- Table of NEMA metrics by sphere
- Summary statistics and legend

See :doc:`guides/how_it_works` for metric interpretation.

Tips & Best Practices
---------------------

- Use NIfTI files for best compatibility
- Ensure proper image orientation before analysis
- Validate results with interactive ROI editor
- Save configuration files for reproducibility
- Use batch processing for multiple images

Need Help?
----------

- Check :doc:`guides/how_it_works`
- Review examples in the repository
- Open an issue on `GitHub <https://github.com/EdAlita/ChameleonIQ/issues>`_

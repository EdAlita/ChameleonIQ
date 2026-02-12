Getting Started
===============

What You Need
-------------

- Python 3.11+
- A PET reconstruction in NIfTI format (``.nii`` or ``.nii.gz``)
- Scanner parameters (voxel spacing, acquisition time)
- Phantom activity concentrations

Quick Installation
------------------

Install ChameleonIQ using pip::

    pip install ChameleonIQ

Verify Installation
~~~~~~~~~~~~~~~~~~~

Test your installation by running::

    chameleoniq_quant --help

Your First Analysis
-------------------

Command Line
~~~~~~~~~~~~

1) Create a YAML config (see :doc:`guides/configuration`).

2) Run a single analysis::

    chameleoniq_quant input.nii.gz --config config.yaml --output results.txt

3) Optional: enable visualizations::

    chameleoniq_quant input.nii.gz --config config.yaml --output results.txt --save-visualizations

Python API
~~~~~~~~~~

Programmatic analysis::

    from pathlib import Path
    from config.defaults import get_cfg_defaults
    from nema_quant.io import load_nii_image
    from nema_quant.phantom import NemaPhantom
    from nema_quant.analysis import calculate_nema_metrics

    # Load configuration and image
    cfg = get_cfg_defaults()
    image_data, affine = load_nii_image(Path('image.nii.gz'), return_affine=True)

    # Extract image properties
    image_dims = image_data.shape
    voxel_spacing = (
        float(abs(affine[0, 0])),
        float(abs(affine[1, 1])),
        float(abs(affine[2, 2]))
    )

    # Initialize phantom and analyze
    phantom = NemaPhantom(cfg, image_dims, voxel_spacing)
    results, lung_results = calculate_nema_metrics(image_data, phantom, cfg)

Next Steps
----------

- **Install**: :doc:`installation` for detailed setup
- **Run**: :doc:`usage` for CLI and workflows
- **Configure**: :doc:`guides/configuration` for YAML details
- **Understand**: :doc:`guides/how_it_works` for the pipeline

Common Tasks
~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   guides/batch_processing

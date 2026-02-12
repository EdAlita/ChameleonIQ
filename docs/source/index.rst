ChameleonIQ Documentation
==========================

.. image:: _static/banner.png
   :alt: ChameleonIQ Banner
   :align: center
   :width: 100%

.. image:: https://img.shields.io/badge/Python-3.11%2B-blue
   :alt: Python 3.11+
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/License-MIT%2FAPACHE%2FBSD-green
   :alt: License

Welcome to ChameleonIQ
~~~~~~~~~~~~~~~~~~~~~~

**ChameleonIQ** is a comprehensive Python tool for automated NEMA NU 2-2018 Image Quality analysis of PET/CT systems.
It provides phantom detection, segmentation, quality metrics computation, and professional reporting capabilities.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started
   installation
   usage

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   guides/how_it_works
   guides/configuration
   guides/troubleshooting
   guides/architecture

.. toctree::
   :maxdepth: 1
   :caption: Python API

   api/config
   api/nema_quant
   api/nema_quant_iter
   api/nema_merge

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   guides/changelog
   guides/license
   guides/contributing
   guides/future_work
   guides/faq

Key Features
~~~~~~~~~~~~

- **NEMA NU 2-2018 Compliance**: Automated image quality assessment following international standards
- **Phantom Detection**: Intelligent automatic segmentation of NEMA phantoms
- **Quality Metrics**: Percent Contrast, Background Variability, Recovery Coefficient, and Spillover Ratio
- **Interactive Tools**: ROI editor with visualization utilities
- **Report Generation**: Professional PDF/HTML reports with statistical summaries
- **Batch Processing**: CLI for high-throughput analysis
- **Flexible Configuration**: YAML-based configuration system

Installation
~~~~~~~~~~~~

Install the latest version::

    pip install ChameleonIQ

For development installation::

    git clone https://github.com/EdAlita/ChameleonIQ
    cd ChameleonIQ
    pip install -e .

Citation
~~~~~~~~

If you use ChameleonIQ in your research, please cite:

.. code-block:: bibtex

    @software{ulin2026chameleoniq,
        author = {Ulin-Briseno, Edwing Y.},
        title = {ChameleonIQ: NEMA Image Quality Analysis Tool},
        year = {2026},
        url = {https://github.com/EdAlita/ChameleonIQ}
    }

Support & Contribution
~~~~~~~~~~~~~~~~~~~~~~

- **Repository**: `GitHub <https://github.com/EdAlita/ChameleonIQ>`_
- **Issues**: `Report Bugs <https://github.com/EdAlita/ChameleonIQ/issues>`_
- **License**: MIT AND (Apache-2.0 OR BSD-2-Clause)

Acknowledgements
~~~~~~~~~~~~~~~~

ChameleonIQ is developed at the `Institute for Instrumentation in Molecular Imaging (i3M) <https://i3m.csic.upv.es/>`_,
a joint research center of the Universitat Politècnica de València (UPV) and the Spanish National Research Council (CSIC).

Copyright © 2026 Edwing Y. Ulin-Briseno

Built with `Sphinx <https://www.sphinx-doc.org/>`_ and `ReadTheDocs Theme <https://sphinx-rtd-theme.readthedocs.io/>`_

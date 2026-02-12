Installation Guide
==================

System Requirements
-------------------

- Python 3.11 or higher
- Linux, Windows, or macOS
- 4GB RAM minimum (8GB recommended)

Installation Methods
--------------------

We strongly recommend using a virtual environment.

Via PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~

Install the latest stable release::

    pip install ChameleonIQ

Via GitHub (Development)
~~~~~~~~~~~~~~~~~~~~~~~~

Clone and install in development mode::

    git clone https://github.com/EdAlita/ChameleonIQ.git
    cd ChameleonIQ
    pip install -e .

For development with additional tools::

    pip install -e ".[dev]"


Verify Installation
-------------------

Check available commands::

    chameleoniq_quant --help

Run the test suite (dev installs)::

    pytest tests/

Troubleshooting
---------------

Missing Dependencies
~~~~~~~~~~~~~~~~~~~~~

If you encounter import errors::

    pip install --upgrade --force-reinstall ChameleonIQ

---------------------------------------

Using venv::

    python -m venv chameleon_env
    source chameleon_env/bin/activate  # On Windows: chameleon_env\Scripts\activate
    pip install ChameleonIQ

Using conda::

    conda create -n chameleon python=3.11
    conda activate chameleon
    pip install ChameleonIQ

CLI Names
---------

Installed commands include both the short and branded names:

- ``nema_quant`` and ``chameleoniq_quant``
- ``nema_quant_iter`` and ``chameleoniq_quant_iter``
- ``nema_merge`` and ``chameleoniq_merge``
- ``nema_coord`` and ``chameleoniq_coord``
- ``chameleoniq_roi_detector`` (interactive ROI editor)
- ``chameleoniq_gui`` (GUI)

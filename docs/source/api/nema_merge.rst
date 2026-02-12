nema_merge
==================

Tools for merging and comparing NEMA analysis results across experiments.

Overview
--------

``nema_merge`` aggregates CSV outputs from multiple experiments defined in an XML
configuration. It generates merged plots and statistical summaries for QA and
longitudinal monitoring.

.. contents:: Table of Contents
   :local:
   :depth: 2

Main Features
-------------

- **XML-Driven Merging**: Define experiments and data sources in one file
- **Comparative Plots**: Contrast and background variability across datasets
- **Dose-Response Analysis**: Optional dose plots when dose values are provided
- **Lung & Advanced Metrics**: Optional lung insert and advanced metric summaries
- **Statistical Summaries**: t-tests and corrected significance heatmaps

Module Contents
---------------

.. automodule:: nema_merge
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Quick Start
-----------

Command Line Usage
~~~~~~~~~~~~~~~~~~

Basic merge:

.. code-block:: bash

   chameleoniq_merge experiments.xml \
     --config config.yaml \
     --output results/merged

Inputs
------

XML configuration
~~~~~~~~~~~~~~~~~

The CLI reads an XML file that defines experiments and their data sources. Each
``experiment`` element requires ``name`` and ``path`` attributes. Optional
attributes include ``plot_status``, ``dose``, ``lung_path``, and
``advanced_path``.

.. code-block:: xml

   <experiments>
     <experiment
       name="Scanner_A_Jan"
       path="results/scanner_a_metrics.csv"
       plot_status="enhanced"
       dose="10.0"
       lung_path="results/scanner_a_lung.csv"
       advanced_path="results/scanner_a_advanced.csv" />
     <experiment
       name="Scanner_B_Jan"
       path="results/scanner_b_metrics.csv"
       plot_status="default"
       dose="12.5" />
   </experiments>

Metrics CSV format
~~~~~~~~~~~~~~~~~~

Each metrics CSV should include these columns:

- ``diameter_mm`` (or ``diameter``, ``diam``, ``d``)
- ``percentaje_constrast_QH``
- ``background_variability_N``

Optional lung CSV
~~~~~~~~~~~~~~~~~

If ``lung_path`` is provided, the CSV must include a ``data`` column containing
lung insert accuracy values.

Optional advanced metrics CSV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``advanced_path`` is provided, the CSV should include one row per experiment
with columns such as ``Dice``, ``Jaccard``, ``VS``, ``MI``, ``Recall``, and
``ASSD``.

Configuration
-------------

The CLI requires a YAML config file (``--config``). This controls plot styling
and shared defaults. See :doc:`../guides/configuration` for details.

Command Line Arguments
----------------------

Required
~~~~~~~~

**xml_config** (positional)
   Path to XML configuration file

**--output**
   Output directory for merged plots (required)

**--config, -c**
   YAML configuration file (required)

Optional
~~~~~~~~

**--log-level**
   Logging verbosity (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR)

Outputs
-------

The command produces:

- Merged contrast/background plots in the output directory
- Dose-response plots when ``dose`` is provided in the XML
- Lung insert violin plots when ``lung_path`` is provided
- Advanced metrics boxplots and heatmaps when ``advanced_path`` is provided

Tips
----

- Use ``plot_status="enhanced"`` to highlight specific experiments.
- Provide numeric ``dose`` values to enable dose-response plots.
- Keep experiment CSV schemas consistent across runs.

Troubleshooting
---------------

**No experiments loaded:**
   Verify the XML file contains ``experiment`` elements with ``name`` and
   ``path`` attributes.

**Missing columns:**
   Ensure metrics CSVs include ``diameter_mm``, ``percentaje_constrast_QH``, and
   ``background_variability_N`` (or accepted alternatives for diameter).

See Also
--------

- :doc:`nema_quant` - Core NEMA analysis workflow
- :doc:`../guides/configuration` - Configuration reference

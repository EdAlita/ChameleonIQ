How It Works
============

This section explains the technical details of ChameleonIQ's analysis pipeline.

Understanding the NEMA Phantom
------------------------------

The NEMA IQ phantom contains six spheres (10, 13, 17, 22, 28, 37 mm). These
evaluate detectability across sizes. Default sphere positions are configurable
in the YAML file, so you can adapt to different phantom geometries.

Data Preparation Pipeline
-------------------------

1. **Image Loading**: Read the 3D image and voxel spacing.
2. **Coordinate Mapping**: Convert millimeters to voxels using ``cfg.ROIS.SPACING``.
3. **ROI Planning**: Map sphere centers and 12 background offsets.
4. **Validation**: Ensure ROIs are inside the image bounds.

Multi-Slice Strategy
--------------------

Using ``cfg.ROIS.CENTRAL_SLICE``, the tool samples multiple slices at ±10 mm and
±20 mm (converted to voxel offsets), providing robust statistics across Z.

3D Sphere Optimization
----------------------

Each sphere position is refined by a 3D grid search around the expected center.
The search evaluates a 5×5 XY grid across several Z offsets, selecting the
offset with the highest mean activity within the ROI.

Background ROI Strategy
-----------------------

Background regions are placed using ``cfg.ROIS.BACKGROUND_OFFSET_YX`` and match
each sphere size to avoid bias. Measurements are aggregated across the sampled
slices.

Image Quality Metrics
---------------------

**Percent Contrast (Q_H,j)**

Measures the contrast between hot and background regions:

.. math::

    Q_{H,j} = \frac{A_H - A_B}{A_B} \times 100\%

where:

- :math:`A_H` = mean activity in hot sphere
- :math:`A_B` = mean activity in background

**Background Variability (N_j)**

Measures noise in background regions:

.. math::

    N_j = \frac{\sigma_B}{A_B} \times 100\%

where:

- :math:`\sigma_B` = standard deviation in background
- :math:`A_B` = mean activity in background

**Lung Insert (Spillover Ratio)**

Lung insert measurements are computed as a ratio relative to the 37 mm sphere
background (:math:`C_{B,37}`):

.. math::

    	ext{Lung}_{\%} = \frac{C_{lung}}{C_{B,37}} \times 100\%

This provides a NEMA-aligned spillover assessment for the lung insert region.

Advanced Segmentation Metrics (Optional)
----------------------------------------

When ``--advanced-metrics`` and a ground-truth mask are provided, the tool
computes segmentation metrics such as Dice and Jaccard for validation.

Visualization & Reporting
-------------------------

When visualizations are enabled, the tool generates:

- ROI overlays and masks per sphere
- Background ROI maps
- Contrast and variability plots

PDF reports include configuration details, results tables, and plots for
quality assurance.

Configuration System
--------------------

Configuration uses hierarchical YAML files:

- **Default Config**: Built-in defaults from ``config.defaults`` module
- **User Config**: Custom YAML files override defaults
- **Runtime Config**: Programmatic overrides in Python code

See :doc:`configuration` for detailed configuration options.

For more technical details, see the `documentation <../../documentation/HOW_IT_WORKS.md>`_.

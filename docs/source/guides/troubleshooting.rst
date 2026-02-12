Troubleshooting
===============

Common Issues
-------------

Cannot find sphere centers
~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptoms:

- ROI optimization fails
- ROIs appear misplaced
- Missing visualization outputs

Fixes:

1. Verify ``ROIS.CENTRAL_SLICE`` using a viewer.
2. Confirm ``ROIS.SPACING`` matches the image header.
3. Adjust ``ROIS.ORIENTATION_YX`` (try flipping signs).
4. Use ``chameleoniq_coord`` to validate coordinate conversions.

Cannot write output file
~~~~~~~~~~~~~~~~~~~~~~~~

Symptoms:

- Output file missing (often on Windows)

Fixes:

- Ensure the output path ends with ``.txt`` when using the CLI.

More Help
---------

- :doc:`configuration`
- :doc:`../usage`
- https://github.com/EdAlita/ChameleonIQ/issues

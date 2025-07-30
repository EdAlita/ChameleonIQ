![Tests](https://github.com/EdAlita/nema_analysis_tool/actions/workflows/tests.yml/badge.svg)

# NEMA NU 2-2018 Image Quality Analysis Tool

This project is a Python-based tool for the automated analysis of PET image quality based on the NEMA NU 2-2018 standard, specifically focusing on Section 7.4.1.

## Features

*   Calculates Percent Contrast (Q_H,j), Percent Background Variability (N_j), and Accuracy of Corrections (ΔC_lung,i).
*   Utilizes 3D Regions of Interest (ROIs) based on the NEMA Body Phantom.
*   Includes a feature for automatic co-registration of off-center images.
*   Loads raw image data with user-defined dimensions and voxel spacing.

## Project Structure

```
nema-analysis-tool/
├── src/
│   ├── nema_quant/         # Main package source code
│   │   ├── analysis.py     # Core NEMA analysis algorithms
│   │   ├── io.py           # Medical image I/O operations
│   │   ├── utils.py        # Utility functions and helpers
│   │   ├── phantom.py      # Phantom geometry definitions
│   │   ├── reporting.py    # Report generation and visualization
│   │   └── cli.py          # Command-line interface
│   └── config/             # Configuration files moved here
├── tests/                  # Comprehensive test suite
└── docs/                   # Documentation and examples
```

## How to get Started?
Read these:
- [Installation instructions](documentation/INSTALLATION.md)
- [Usage instructions](documentation/USAGE.md)
- [How it works?](documentation/HOW_IT_WORKS.md)

Additional information:
- [What will change?](documentation/CHANGELOG.md)

[//]: # (- [Ignore label]&#40;documentation/ignore_label.md&#41;)
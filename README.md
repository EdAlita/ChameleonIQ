<p align="center">
  <a href="https://github.com/EdAlita/nema_analysis_tool/actions/workflows/tests.yml">
    <img alt="Tests" src="https://github.com/EdAlita/nema_analysis_tool/actions/workflows/tests.yml/badge.svg?branch=main">
  </a>
  <a href="https://github.com/psf/black">
    <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
  </a>
  <a href="https://github.com/EdAlita/nema_analysis_tool/blob/main/LICENSE">
    <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
  </a>
  <a>
    <img alt="Python" src="https://img.shields.io/badge/language-Python-blue?logo=python">
  </a>
</p>



# NEMA NU 2-2018 Image Quality Analysis Tool

This project is a Python-based tool for the automated analysis of PET image quality based on the NEMA NU 2-2018 standard, specifically focusing on Section 7.4.1.

## Features

*   Calculates Percent Contrast (Q_H,j), Percent Background Variability (N_j), and Accuracy of Corrections (ΔC_lung,i).
*   Utilizes 3D Regions of Interest (ROIs) based on the NEMA Body Phantom.
*   Loads raw image data with user-defined dimensions and voxel spacing.
*   Automatic postions of ROIs on given centers

## Project Structure

```bash
nema-analysis-tool
├── data
├── documentation
├── src
│  ├── config
│  ├── nema_merge
│  ├── nema_quant
│  └── nema_quant_iter
└── tests
```

- [**Config**](src/config): configuration files to run the tool
- [**Data**](src/data): Data for testing and logos used for PDF reports
- [**Documentation**](src/nema_quant_iter): Documentation, wikis and how to install the tool
- [**nema_merge**](src/nema_merge): scripts for creating fusing of individuals runs
- [**nema_quant**](src/nema_quant): scripts for individuals test runs
- [**nema_quant_iter**](src/nema_quant_iter): scripts for a iteration based analysis

## How to get Started?
Read these:
- [**Installation instructions**](documentation/INSTALLATION.md)
- [**Usage instructions**](documentation/USAGE.md)
- [**How it works?**](documentation/HOW_IT_WORKS.md)

Additional information:
- [**What will change?**](documentation/CHANGELOG.md)

## License
This project is licensed under the Apache Lincese 2.0 - see the [LICENSE.md](LICENSE) file for details.

[//]: # (- [Ignore label]&#40;documentation/ignore_label.md&#41;)

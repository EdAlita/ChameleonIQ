<p align="center">
  <img
    alt="Logo Banner"
    src="https://raw.githubusercontent.com/EdAlita/nema_analysis_tool/main/data/banner.png"
  >
</p>

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
  <a>
    <img alt="Git Download" src="https://img.shields.io/github/downloads/EdAlita/nema_analysis_tool/total">
  </a>
  <a>
    <img alt="Git Release" src="https://img.shields.io/github/v/release/EdAlita/ChameleonIQ">
  </a>
</p>


# ChameleonIQ: Nema-aware Image Quality Tool for Python

This project is a Python-based tool for the automated analysis of PET image quality based on the NEMA NU 2-2018 standard, specifically focusing on Section 7.4.1.

Please use this cite when using the sofware:

     Ulin-Briseno, E. (2026). ChameleonIQ (Version 2.0.0) [Computer software]. https://github.com/EdAlita/ChameleonIQ


## Features

*   Calculates Percent Contrast (Q_H,j), Percent Background Variability (N_j), and Accuracy of Corrections (ΔC_lung,i).
*   Utilizes 3D Regions of Interest (ROIs) based on the NEMA Body Phantom.
*   Loads nii image data with user-defined dimensions and voxel spacing.
*   Automatic postions of ROIs on given centers

## How to get Started?
Read these:
- [**Installation instructions**](https://github.com/EdAlita/ChameleonIQ/wiki/Installation)
- [**Usage instructions**](https://github.com/EdAlita/ChameleonIQ/wiki/Usage)
- [**How it works?**](https://github.com/EdAlita/ChameleonIQ/wiki/How-it-works)

Additional information:
- [**What will change?**](https://github.com/EdAlita/ChameleonIQ/wiki/Changelog)

## License
This project is licensed under the Apache Lincese 2.0 - see the [LICENSE.md](Lhttps://github.com/EdAlita/ChameleonIQ/blob/main/LICENSE.txt) file for details.

## Acknowledgements

<p align="center">
  <img
    alt="i3m logo"
    src="https://i3m.csic.upv.es/wp-content/uploads/2023/09/logo-web-i3m.png"
  >
</p>

ChameleonIQ is an open-source project developed as part of my research activities at the Institute for [Institute for Instrumentation in Molecular Imaging (i3M)](https://i3m.csic.upv.es/). I gratefully acknowledge the support of the [Detectors for Molecular Imaging Laboratory (DMIL)](https://i3m.csic.upv.es/research/stim/dmil/), i3M, and the Spanish National Research Council (CSIC). The i3M is a joint research center established in 2010 by the Universitat Politècnica de València (UPV) and CSIC, located on the Vera Campus in Valencia, Spain.

[//]: # (- [Ignore label]&#40;documentation/ignore_label.md&#41;)

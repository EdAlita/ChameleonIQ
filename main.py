#!/usr/bin/env python3
"""
NEMA NU 2-2018 Image Quality Analysis Tool

Main entry point for the NEMA analysis tool. This script performs automated
analysis of PET image quality based on the NEMA NU 2-2018 standard.

Author: Edwing Ulin-Briseno
Date: 2025-07-16
"""

import sys
from src.nema_quant.cli import main as cli_main


if __name__ == "__main__":
    sys.exit(cli_main())
"""
Entry point for running nema_quant as a module.

This allows the package to be run with:
    python -m nema_quant
"""

from .cli import main

if __name__ == "__main__":
    main()

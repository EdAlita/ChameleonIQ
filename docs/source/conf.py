import os
import sys
from pathlib import Path

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

# Add the project root and src/ to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

project = "ChameleonIQ"
copyright = "2026, Edwing Y. Ulin-Briseno"
author = "Edwing Y. Ulin-Briseno"
release = "2.1.0"

autodoc_mock_imports = getattr(globals(), "autodoc_mock_imports", []) + [
    "pyqtgraph",
    "pyqtgraph.Qt",
    "PyQt5",
    "PyQt5.QtCore",
    "PyQt5.QtGui",
]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",  # Add this line
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]

# Autosummary settings
autosummary_generate = False

# Ignore autosummary stubs (single-page module docs)
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "api/generated/**",
]

# Use Furo theme
html_theme = "sphinx_rtd_theme"

# Logo configuration
html_logo = "_static/logo.png"
html_title = "ChameleonIQ"

autodoc_default_options = {
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "show-inheritance": True,
}

autodoc_mock_imports = getattr(globals(), "autodoc_mock_imports", []) + [
    "numpy",
    "scipy",
    "nibabel",
    "matplotlib",
    "seaborn",
    "pandas",
    "yacs",
    "cv2",
    "PIL",
    "reportlab",
    "SimpleITK",
    "skimage",
    "rich",
    "statsmodels",
    "PyQt5",
]

# Some packages (pyqtgraph / PyQt5) try to access Qt internals at import time
# which can break Sphinx runs in headless/CI environments. Ensure those modules
# are present in sys.modules as mocks so imports in autodoc do not load real
# packages from the environment.
try:
    from unittest.mock import MagicMock
except Exception:  # pragma: no cover - fallback for very old pythons
    MagicMock = None

if MagicMock is not None:
    MOCK_MODULES = [
        "pyqtgraph",
        "pyqtgraph.Qt",
        "pyqtgraph.Qt.QtCore",
        "pyqtgraph.Qt.QtGui",
        "PyQt5",
        "PyQt5.QtCore",
        "PyQt5.QtGui",
    ]
    for m in MOCK_MODULES:
        if m not in sys.modules:
            sys.modules[m] = MagicMock()

napoleon_use_param = True
napoleon_use_rtype = True
napoleon_numpy_docstring = True

# MyST configuration for Markdown
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

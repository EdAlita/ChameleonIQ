import sys
from pathlib import Path

# Add the project root and src/ to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

project = "ChameleonIQ"
copyright = "2026, Edwing Y. Ulin-Briseno"
author = "Edwing Y. Ulin-Briseno"
release = "2.1.0"

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

autodoc_mock_imports = [
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

napoleon_use_param = True
napoleon_use_rtype = True
napoleon_numpy_docstring = True

# MyST configuration for Markdown
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

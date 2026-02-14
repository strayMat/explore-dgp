"""Sphinx configuration."""

import importlib
import pathlib
import sys
from datetime import datetime
from typing import List

# -- Path setup --------------------------------------------------------------
import os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. Note that we are adding an absolute
# path.
_project_directory = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_directory))
os.environ["PYTHONPATH"] = str(_project_directory) + os.pathsep + os.environ.get("PYTHONPATH", "")


# -- Project information -----------------------------------------------------
PACKAGE_NAME = "explore_dgp"
try:
    project_metadata = importlib.metadata.metadata(PACKAGE_NAME)
except importlib.metadata.PackageNotFoundError as err:
    raise RuntimeError(
        f"The package '{PACKAGE_NAME}' must be installed. "
        "Please install the package in editable mode before building docs."
    ) from err


# pylint: disable=invalid-name

# -- Project information -----------------------------------------------------
project = project_metadata["Name"]
author = project_metadata["Author-email"]
# pylint: disable=redefined-builtin
copyright = f"{datetime.now().year}, {author}"
version = release = project_metadata["Version"]

# -- General configuration ---------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".py": "myst-nb",
}

extensions = [
    "myst_nb",  # MyST-NB for notebooks and .md parsing
    "sphinx.ext.autodoc",  # Include documentation from docstrings (https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
    "sphinx.ext.autosummary",  # Generate autodoc summaries (https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html)
    "sphinx.ext.intersphinx",  # Link to other projects' documentation (https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html)
    "sphinx.ext.viewcode",  # Add documentation links to/from source code (https://www.sphinx-doc.org/en/master/usage/extensions/viewcode.html)
    "sphinx.ext.autosectionlabel",  # Allow reference sections using its title (https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html)
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings (https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
    "sphinx.ext.mathjax",  # Math rendering support for LaTeX expressions (https://www.sphinx-doc.org/en/master/usage/extensions/math.html)
    "sphinx_click",  # Automatic documentation of click based CLI (https://github.com/click-contrib/sphinx-click)
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
# Note: `custom-class-template.rst` & `custom-module-template.rst`
#   for sphinx.ext.autosummary extension `recursive` option
#   see: https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion


# List of patterns, relative to source directory, that match files and
#   directories to ignore when looking for source files.
#   This pattern also affects html_static_path and html_extra_path.
exclude_patterns: List[str] = ["_build", "Thumbs.db", ".DS_Store", ".venv"]

# Sphinx configs
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
# html_favicon = "_static/favicon.ico"
html_theme_options = {
    "logo": {
        "image_light": "logo-light.svg",
        "image_dark": "logo-dark.svg",
    },
}
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)

# -- Extension configurations ---------------------------------------------------

# sphinx.ext.autosummary configs
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# sphinx.ext.autodoc configs
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
autodoc_inherit_docstrings = True  # If no class summary, inherit base class summary
autodoc_typehints = "description"  # Show typehints as content of function or method
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_mock_imports = []  # Add any modules that can't be imported during doc build

# myst_nb configs
nb_execution_mode = "auto"  # Execute notebooks during build if they have changed
nb_execution_timeout = 60
nb_custom_formats = {".py": ["jupytext.reads", {"fmt": "py:percent"}]}
nb_execution_excludepatterns = ["conf.py"]
myst_enable_extensions = ["dollarmath", "amsmath"]

# Prefix document path to section labels, to use:
# `path/to/file:heading` instead of just `heading`
autosectionlabel_prefix_document = True


# -- External mapping --------------------------------------------------------
python_version = ".".join(map(str, sys.version_info[0:2]))
intersphinx_mapping = {
    "sphinx": ("http://www.sphinx-doc.org/en/stable", None),
    "python": ("https://docs.python.org/" + python_version, None),
}

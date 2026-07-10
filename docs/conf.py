"""Sphinx configuration for the regime-shap documentation."""

from __future__ import annotations

from importlib.metadata import version as _version

project = "regime-shap"
author = "Faith Olan-George"
copyright = "2026, Faith Olan-George"  # noqa: A001

try:
    release = _version("regime-shap")
except Exception:  # pragma: no cover
    # Fall back to a static version when the package is not installed.
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",              # author the prose pages in Markdown
    "sphinx.ext.autodoc",       # build the API reference from the docstrings
    "sphinx.ext.napoleon",      # read the Google style docstrings
    "sphinx.ext.autosummary",   # summary tables for the API
    "sphinx.ext.viewcode",      # link API entries to the source
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

html_theme = "sphinx_rtd_theme"
html_title = "regime-shap"

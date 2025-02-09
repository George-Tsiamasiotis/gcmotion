# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.abspath("."))

project = "gcmotion"
copyright = "2024, George Tsiamasiotis"
author = "George Tsiamasiotis"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "numpydoc",
]

autosummary_generate = True
autosummary_ignore_module_all = False
autosummary_imported_members = False
autosummary_ignore_module_all = False
todo_include_todos = True
numpydoc_attributes_as_param_listbool = True
numpydoc_class_members_toctree = False

autodoc_type_aliases = {
    "SupportedSpecies": "{'p', 'e', 'D', 'T', 'He3', 'He4'}"
}

autodoc_mock_imports = ["math", "ABC", "abc"]
autosummary_mock_imports = ["math", "ABC", "abc"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
# html_static_path = ["_static"]
html_theme_options = {
    # "secondary_sidebar_items": ["page-toc"],
    "collapse_navigation": False,
}

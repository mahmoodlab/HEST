# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import hest

project = 'hest'
copyright = '2024, Paul Doucet'
author = 'Paul Doucet'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinx_design',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx'
]

templates_path = ['_templates']
exclude_patterns = []


intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable", None),
}
intersphinx_disabled_reftypes = ["*"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

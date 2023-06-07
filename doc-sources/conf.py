# Copyright 2023 Vincent Jacques

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys


project = "lincs"
copyright = "Copyright 2023 Vincent Jacques"
author = "Vincent Jacques"
with open("../setup.py") as f:
    for line in f.readlines():
        if line.startswith("version = "):
            release = line[11:-2]
            break
    else:
        assert False, "Release not found in setup.py"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

# No Jekyll

extensions.append("sphinx.ext.githubpages")

# Click

os.environ["LINCS_GENERATING_SPHINX_DOC"] = "1"
extensions.append("sphinx_click")
sys.path.insert(0, os.path.abspath(".."))  # To find module 'lincs' before it's installed
sys.path.insert(0, os.path.abspath("../development"))  # To find 'cycle.py', which implements './run-development-cycle.sh'

# Details and summary directive

extensions.append("sphinxcontrib.details.directive")

# MathJax

extensions.append("sphinx.ext.mathjax")

# GraphViz

extensions.append("sphinx.ext.graphviz")

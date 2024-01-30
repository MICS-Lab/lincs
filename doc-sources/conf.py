# Copyright 2023-2024 Vincent Jacques

import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "lincs"
copyright = "Copyright 2023-2024 Vincent Jacques"
author = "Vincent Jacques"
with open("../lincs/__init__.py") as f:
    for line in f.readlines():
        if line.startswith("__version__ = "):
            release = line[15:-2]
            break
    else:
        assert False, "Release not found in lincs/__init__.py"

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

# JSON Schemas

extensions.append("sphinx-jsonschema")

# Python domain
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-the-python-domain

python_use_unqualified_type_names = True

# Markdown

extensions.append("myst_parser")

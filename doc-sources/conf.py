# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

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
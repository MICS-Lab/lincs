.. Copyright 2023 Vincent Jacques

=========
Reference
=========

@todo Generate a reference documentation using Sphinx:

- Python using autodoc
- C++ using Doxygen+Breath
- YAML file formats using JSON Schema and https://sphinx-jsonschema.readthedocs.io/en/latest/


File formats
============

*lincs* uses text-based (YAML and CSV) file formats.
The same formats are used for synthetic and real-world data.
The same formats are used when the ``lincs`` command lines outputs to actual files or to its standard output.

.. _ref-file-problem:

The problem file
----------------

@todo Write (or generate from a JSON Schema?)

.. _ref-file-alternatives:

The alternatives file
---------------------

@todo Write

.. _ref-file-model:

The model file
--------------

@todo Write (or generate from a JSON Schema?)


.. _ref-cli:

Command-line interface
======================

.. click:: lincs.command_line_interface:main
   :prog: lincs
   :nested: full

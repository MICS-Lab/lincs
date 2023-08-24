.. Copyright 2023 Vincent Jacques

=========
Reference
=========

@todo(Documentation, later) Generate a reference documentation using Sphinx:

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

The problem file is a YAML file specified by the following :download:`JSON Schema <problem-schema.yml>`:

.. jsonschema:: problem-schema.yml
   :lift_title: false

.. _ref-file-ncs-model:

The NCS model file
------------------

The model file is a YAML file specified by the following :download:`JSON Schema <model-schema.yml>`:

.. jsonschema:: model-schema.yml
   :lift_title: false

.. _ref-file-alternatives:

The alternatives file
---------------------

@todo(Documentation, soon) Write

.. _ref-cli:

Command-line interface
======================

.. click:: lincs.command_line_interface:main
   :prog: lincs
   :nested: full

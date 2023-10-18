.. Copyright 2023 Vincent Jacques

=========
Changelog
=========

Versions 0.9.0 to 0.9.1
=======================

- Pre-process the learning set before all learning algorithms.

Possible values for each criterion are listed and sorted before the actual learning starts so that learning algorithms now see all criteria as:

    - having increasing correlation with the categories
    - having values in a range of integers

This is a simplification for implementers of learning algorithms, and improves the performance of the weights-profiles-breed approach.

- Expose ``SufficientCoalitions::upset_roots`` to Python
- Fix alternative names when using the ``--max-imbalance`` option of ``lincs generate classified-alternatives``
- Produce cleaner error when ``--max-imbalance`` is too tight
- Print number of iterations at the end of WPB learnings
- Display *lincs*' version in the "Reproduction command" comment in generated files
- Various improvements to the code's readability

Version 0.8.7
=============

- Integrate CUDA parts on Windows
- Compile with OpenMP on Windows

Versions 0.8.5 to 0.8.6
=======================

- Distribute binary wheels for Windows!

Versions 0.8.0 to 0.8.4
=======================

- Rename option ``--...max-duration-seconds`` to ``--...max-duration``
- Display termination condition after learning using the ``weights-profiles-breed`` approach
- Make termination of the ``weights-profiles-breed`` approach more consistent
- Integrate `Chrones <https://pypi.org/project/Chrones/>`_ (as an optional dependency, on Linux only)
- Display iterations in ``--...verbose`` mode
- Fix pernicious memory bug

Version 0.7.0
=============

Bugfixes:

- Fix the Linux wheels: make sure they are built with GPU support
- Improve building *lincs* without ``nvcc`` (*e.g.* on macOS):

    - provide the ``lincs info has-gpu`` command
    - adapt ``lincs learn classification-model --help``

Features:

- Add "max-SAT by coalitions" and "max-SAT by separation" learning approaches (hopefully correct this time!)
- Use YAML anchors and aliases to limit repetitions in the model file format when describing :math:`U^c \textsf{-} NCS` models
- Specifying the minimum and maximum values for each criterion in the problem file:

    - Generate synthetic data using these attributes (``--denormalized-min-max``)
    - Adapt the learning algorithms to use these attributes

- Support criteria with decreasing correlation with the categories:

    - in the problem file
    - when generating synthetic data (``--allow-decreasing-criteria``)
    - in the learning algorithms

- Add a comment to all generated files stating the command-line to use to re-generate them
- Use enough decimals when storing floating point values in models to avoid any loss of precision
- Log final accuracy with ``--mrsort.weights-profiles-breed.verbose``
- Improve tests

Version 0.6.0
=============

- **Remove buggy "max-SAT by coalitions" approach**
- Add "SAT by separation" approach

Version 0.5.1
=============

- Publish wheels for macOS

Version 0.5.0
=============

- Implement "SAT by coalitions" and "max-SAT by coalitions" **removed in 0.6.0** learning methods
- Add `misclassify_alternatives` to synthesize noise on alternatives
- Expend the model file format to support specifying the sufficient coalitions by their roots
- Produce "manylinux_2_31" binary wheels
- Improve YAML schemas for problem and model file formats
- Use the "flow" YAML formatting for arrays of scalars
- Improve consistency between Python and C++ APIs (not yet documented though)
- Add more control over the "weights, profiles, breed" learning method (termination strategies, "verbose" option)
- Add an expansion point for the breeding part of "weights, profiles, breed"
- Add an exception for failed learnings

Version 0.4.5
=============

- Use JSON schemas to document and validate the problem and model files
- Support development on macOS and on machines without a GPU
- Improve documentation

Versions 0.4.1 to 0.4.4
=======================

Never properly published

Version 0.4.0
=============

- Add a GPU (CUDA) implementation of the accuracy heuristic strategy for the "weights, profiles, breed" learning method
- Introduce Alglib as a LP solver for the "weights, profiles, breed" learning method
- Publish a Docker image with *lincs* installed
- Change "domain" to "problem" everywhere
- Improve documentation
- Improve model and alternatives visualization
- Expose 'Alternative::category' properly

Versions 0.3.4 to 0.3.7
=======================

- Improve documentation

Version 0.3.3
=============

- Fix Python package description

Version 0.3.2
=============

- License (LGPLv3)

Version 0.3.1
=============

- Fix installation (missing C++ header file)

Version 0.3.0
=============

- Implement learning an MR-Sort model using Sobrie's heuristic on CPU

Version 0.2.2
=============

- Add options: `generate model --mrsort.fixed-weights-sum` and `generate classified-alternatives --max-imbalance`

Version 0.2.1
=============

- Fix images on the PyPI website

Version 0.2.0
=============

- Implement generation of pseudo-random synthetic data
- Implement classification by MR-Sort models
- Kick-off the documentation effort with a quite nice first iteration of the README

Version 0.1.3
=============

Initial publication with little functionality

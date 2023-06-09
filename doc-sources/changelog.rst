.. Copyright 2023 Vincent Jacques

=========
Changelog
=========

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

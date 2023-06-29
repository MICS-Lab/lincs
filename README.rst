.. Copyright 2023 Vincent Jacques

.. WARNING, this README is rendered to HTML in several places
    - on GitHub (https://github.com/mics-lab/lincs/)
    - on PyPI after publication of the package (https://pypi.org/project/lincs/)
    - on GitHub Pages (https://mics-lab.github.io/lincs/)
    So when you change it, take care to check all those places.

*lincs* (Learn and Infer Non Compensatory Sortings) is a collection of MCDA algorithms, usable as a C++ library, a Python (3.7+) package and a command-line utility.

*lincs* is licensed under the GNU Lesser General Public License v3.0 as indicated by the two files `COPYING <COPYING>`_ and `COPYING.LESSER <COPYING.LESSER>`_.

@todo (When we have a paper to actually cite) Add a note asking academics to kindly cite our work.

*lincs* is available for install from the `Python package index <https://pypi.org/project/lincs/>`_.
Its `documentation <http://mics-lab.github.io/lincs/>`_
and its `source code <https://github.com/mics-lab/lincs/>`_ are on GitHub.

Questions? Remarks? Bugs? Want to contribute? Open `an issue <https://github.com/MICS-Lab/lincs/issues>`_ or `a discussion <https://github.com/MICS-Lab/lincs/discussions>`_!


Contributors and previous work
==============================

*lincs* is developed by the `MICS <https://mics.centralesupelec.fr/>`_ research team at `CentraleSupélec <https://www.centralesupelec.fr/>`_.

Its main authors are (alphabetical order):

- `Laurent Cabaret <https://cabaretl.pages.centralesupelec.fr/>`_ (performance optimization)
- `Vincent Jacques <https://vincent-jacques.net>`_ (engineering)
- `Vincent Mousseau <https://www.centralesupelec.fr/fr/2EBDCB86-64A4-4747-96E8-C3066CB61F3D>`_ (domain expertise)
- `Wassila Ouerdane <https://wassilaouerdane.github.io/>`_ (domain expertise)

It's based on work by:

- `Olivier Sobrie <http://olivier.sobrie.be/>`_ (The "weights, profiles, breed" learning strategy for MR-Sort models, and the profiles improvement heuristic, developed in his `Ph.D thesis <http://olivier.sobrie.be/papers/phd_2016_sobrie.pdf>`_, and `implemented in Python <https://github.com/oso/pymcda>`_)
- Emma Dixneuf, Thibault Monsel and Thomas Vindard (`C++ implementation of Sobrie's heuristic <https://github.com/Mostah/fastPL/>`_)

@todo Add links to the fundamental articles for NCS.

@todo Add links to the articles that define other learning methods we re-implement.


Project goals
=============

Provide MCDA tools usable out of the box
----------------------------------------

You should be able to use *lincs* without being a specialist of MCDA and/or NCS models.
Just follow the `Get started <#get-started>`_ section below.

Provide a base for developing new MCDA algorithms
-------------------------------------------------

*lincs* is designed to be easy to extend with new algorithms of even replace parts of existing algorithms.
See our `contributor guide <https://mics-lab.github.io/lincs/contributor-guide.html>`_ for more details.

*lincs* also provides a benchmark framework to compare algorithms (@todo Implement and document).
This should make it easier to understand the relative strengths and weaknesses of each algorithm.


Versioning
==========

Starting with version 1.0.0, *lincs* tries to apply `semantic versioning <https://semver.org/>`_ at a *code* level:
upgrading patch and minor releases should not require changes in client code but may require you to recompile and link it.


Get started
===========

Depending on your favorite approach, you can either start with our `hands-on "Get started" guide <https://mics-lab.github.io/lincs/get-started.html>`_
or with our `conceptual overview documentation <https://mics-lab.github.io/lincs/conceptual-overview.html>`_.
The former will show you how to use our tools, the latter will explain the concepts behind them: what's MCDA, what are NCS models, *etc.*
If in doubt, start with the conceptual overview.
We highly recommend you read the other one just after.

Once you've used *lincs* a bit, you can follow up with our `user guide <https://mics-lab.github.io/lincs/user-guide.html>`_
and `reference documentation <https://mics-lab.github.io/lincs/reference.html>`_.


Develop *lincs* itself
======================

See our `contributor guide <https://mics-lab.github.io/lincs/contributor-guide.html>`_.

.. Copyright 2023 Vincent Jacques

.. WARNING, this README is rendered to HTML in several places
    - on GitHub (https://github.com/mics-lab/lincs/)
    - on PyPI after publication of the package (https://pypi.org/project/lincs/)
    - on GitHub Pages (https://mics-lab.github.io/lincs/)
    So when you change it, take care to check all those places.

*lincs* (Learn and Infer Non Compensatory Sortings) is a collection of MCDA algorithms, usable as a command-line utility.
We plan to make it usable as a C++ library and a Python (3.7+) package as well. (@todo(Feature, later))

*lincs* is licensed under the GNU Lesser General Public License v3.0 as indicated by the two files `COPYING <COPYING>`_ and `COPYING.LESSER <COPYING.LESSER>`_.

@todo(Project management, when we publish a paper) Add a note asking academics to kindly cite our work.

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

Fondamental concepts
--------------------

*lincs* is based on the following concepts.
Note that we describe them in our `conceptual overview documentation <https://mics-lab.github.io/lincs/conceptual-overview.html>`_.
This section is here to give credit to their authors.

The Non-Compensatory Sorting (NCS) model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The concept of the NCS model was first introduced by Denis Bouyssou and Thierry Marchant in their articles `An axiomatic approach to noncompensatory sorting methods in MCDM I: The case of two categories <https://hal.science/hal-00958022>`_ and `... II: More than two categories <https://hal.science/hal-00013762v1>`_.

Particular cases of the NCS model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Uc-NCS model is a particular case of the NCS model.

The MR-Sort model is a particular case of the Uc-NCS model introduced by Agnès Leroy *et al.* in `Learning the Parameters of a Multiple Criteria Sorting Method <https://link.springer.com/chapter/10.1007/978-3-642-24873-3_17>`_.

Although *lincs* can sort alternatives according to general NCS models (without veto), it only implements learning Uc-NCS and MR-Sort models.

Learning algorithms
-------------------

*lincs* provides new implementations of the following algorithms:

Learning Uc-NCS models with a SAT-based approaches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following learning algorithms were implemented using their description by Ali Tlili, Khaled Belahcène *et al.* in `Learning non-compensatory sorting models using efficient SAT/MaxSAT formulations <https://www.sciencedirect.com/science/article/abs/pii/S0377221721006858>`_:

- learning exact Uc-NCS models with a "SAT by coalitions" approach
- learning approximate Uc-NCS models with a "max-SAT by coalitions" approach
- learning exact Uc-NCS models with a "SAT by separation" approach
- learning approximate Uc-NCS models with a "max-SAT by separation" approach

Note that they were introduced in previous articles, and that this article conveniently gathers them in a single place.

Learning approximate MR-Sort with a heuristic approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This approach, described by `Olivier Sobrie <http://olivier.sobrie.be/>`_ in his `Ph.D thesis <http://olivier.sobrie.be/papers/phd_2016_sobrie.pdf>`_,
is based on splitting the learning into three phases: optimize the weights (linear programming), improve the profiles (heuristic) and breed the population of intermediate models.
We call it the "weights, profiles, breed" learning strategy in *lincs*.

It was originaly `implemented in Python <https://github.com/oso/pymcda>`_ by Olivier Sobrie.
Emma Dixneuf, Thibault Monsel and Thomas Vindard then provided a sequential `C++ implementation of Sobrie's heuristic <https://github.com/Mostah/fastPL/>`_.
*lincs* provides two parallel implementations of this approach (using OpenMP and CUDA).


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

*lincs* will also provide a benchmark framework to compare algorithms (@todo(Feature, later)).
This should make it easier to understand the relative strengths and weaknesses of each algorithm.


Get started
===========

Depending on your favorite approach, you can either start with our `hands-on "Get started" guide <https://mics-lab.github.io/lincs/get-started.html>`_
or with our `conceptual overview documentation <https://mics-lab.github.io/lincs/conceptual-overview.html>`_.
The former will show you how to use our tools, the latter will explain the concepts behind them: what's MCDA, what are NCS models, *etc.*
If in doubt, start with the conceptual overview.
We highly recommend you read the other one just after.

Once you've used *lincs* a bit, you can follow up with our `user guide <https://mics-lab.github.io/lincs/user-guide.html>`_
and `reference documentation <https://mics-lab.github.io/lincs/reference.html>`_.


Versioning
==========

Starting with version 1.0.0, *lincs* tries to apply `semantic versioning <https://semver.org/>`_ at a *code* level:
upgrading patch and minor releases should not require changes in client code but may require you to recompile and link it.


Develop *lincs* itself
======================

See our `contributor guide <https://mics-lab.github.io/lincs/contributor-guide.html>`_.

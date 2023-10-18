.. Copyright 2023 Vincent Jacques

.. This README is rendered to HTML in several places
    - on GitHub (https://github.com/mics-lab/lincs/)
    - on PyPI after publication of the package (https://pypi.org/project/lincs/)
    - on GitHub Pages (https://mics-lab.github.io/lincs/)
    So when you change it, take care to check all those places.

*lincs* (Learn and Infer Non Compensatory Sortings) is a collection of `MCDA <https://en.wikipedia.org/wiki/Multiple-criteria_decision_analysis>`_ algorithms, usable as a command-line utility.

@todo(Feature, later) Make it usable as a Python package.

@todo(Feature, later) Make it usable as a C++ library.

*lincs* supports Linux, macOS and Windows, with the exception that GPU-based algorithms are not available on macOS, because CUDA itself is not available there.

*lincs* is licensed under the GNU Lesser General Public License v3.0 as indicated by the two files `COPYING <COPYING>`_ and `COPYING.LESSER <COPYING.LESSER>`_.

@todo(Project management, when we publish a paper, later) Add a note asking academics to kindly cite our work.

*lincs* is available for install from the `Python package index <https://pypi.org/project/lincs/>`_.
Its `documentation <http://mics-lab.github.io/lincs/>`_
and its `source code <https://github.com/mics-lab/lincs/>`_ are on GitHub.

Questions? Remarks? Bugs? Want to contribute? Open `an issue <https://github.com/MICS-Lab/lincs/issues>`_ or `a discussion <https://github.com/MICS-Lab/lincs/discussions>`_!


Contributors
============

*lincs* is developed by the `MICS <https://mics.centralesupelec.fr/>`_ research team at `CentraleSupélec <https://www.centralesupelec.fr/>`_.

Its main authors are (alphabetical order):

- `Laurent Cabaret <https://cabaretl.pages.centralesupelec.fr/>`_ (performance optimization)
- `Vincent Jacques <https://vincent-jacques.net>`_ (engineering)
- `Vincent Mousseau <https://www.centralesupelec.fr/fr/2EBDCB86-64A4-4747-96E8-C3066CB61F3D>`_ (domain expertise)
- `Wassila Ouerdane <https://wassilaouerdane.github.io/>`_ (domain expertise)


Project goals
=============

Provide MCDA tools usable out of the box
----------------------------------------

You should be able to use *lincs* without being a specialist of MCDA and/or NCS models.
Just follow the "Get started" section below.

Provide a base for developing new MCDA algorithms
-------------------------------------------------

*lincs* is designed to be easy to extend with new algorithms of even replace parts of existing algorithms.
See our `contributor guide <https://mics-lab.github.io/lincs/contributor-guide.html>`_ for more details.


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

Starting with version 1.0.0, *lincs* uses `semantic versioning <https://semver.org/>`_.

*lincs*' public API (that "must be declared" according to SemVer) is constituted exclusively by its `reference documentation <https://mics-lab.github.io/lincs/reference.html>`_,
**at a code level**: we consider a change as backward compatible if the client code doesn't need to be modified to keep working,
even if that change requires recompiling the client code in some cases.

A backward compatible change might change *lincs*' behavior, especially with regards to pseudo-random behavior.

Note that we plan to make *lincs* usable as Python and C++ libraries.
When we do that, we'll add these interfaces to the public API.
In the mean time, if you chose to use *lincs* that way, you must expect unanticipated changes to these interfaces.


Develop *lincs* itself
======================

See our `contributor guide <https://mics-lab.github.io/lincs/contributor-guide.html>`_.

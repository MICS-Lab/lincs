.. Copyright 2023-2024 Vincent Jacques

.. This README is rendered to HTML in several places
    - on GitHub (https://github.com/mics-lab/lincs/)
    - on PyPI after publication of the package (https://pypi.org/project/lincs/)
    - on GitHub Pages (https://mics-lab.github.io/lincs/)
    So when you change it, take care to check all those places.

*lincs* (Learn and Infer Non Compensatory Sortings) is a collection of `MCDA <https://en.wikipedia.org/wiki/Multiple-criteria_decision_analysis>`_ algorithms, usable as a command-line utility and through a Python (3.8+) API.

*lincs* supports Linux, macOS and Windows, with the exception that GPU-based algorithms are not available on macOS, because CUDA itself is not available there.
On these 3 OSes, *lincs* only support x86_64 CPUs.

*lincs* is licensed under the GNU Lesser General Public License v3.0 as indicated by the two files `COPYING <COPYING>`_ and `COPYING.LESSER <COPYING.LESSER>`_.

*lincs* is available for install from the `Python package index <https://pypi.org/project/lincs/>`_.
Its `documentation <http://mics-lab.github.io/lincs/>`_
and its `source code <https://github.com/mics-lab/lincs/>`_ are on GitHub.

Questions? Remarks? Bugs? Want to contribute? Open `an issue <https://github.com/MICS-Lab/lincs/issues>`_ or `a discussion <https://github.com/MICS-Lab/lincs/discussions>`_!
You should probably take a look at `our roadmap <https://mics-lab.github.io/lincs/roadmap.html>`_ first.

@todo(Project management, v1.1) Add a note asking academics to kindly cite our ROADEF 2024 paper.


Contributors
============

*lincs* is developed by the `MICS <https://mics.centralesupelec.fr/>`_ research team at `CentraleSupélec <https://www.centralesupelec.fr/>`_.

Its main authors are (alphabetical order):

- Khaled Belahcène (domain expertise)
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

*lincs* is designed to be easy to extend with new algorithms or even replace parts of existing algorithms.
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

Future backward compatible changes might change *lincs*' behavior, especially with regards to pseudo-random behavior.

Note that we plan to make *lincs* usable as a C++ library.
When we do that, we'll add this interface to the public API.
In the mean time, if you chose to use *lincs* that way, you must expect unanticipated changes to this interface.

Exceptions
----------

Default values
^^^^^^^^^^^^^^

Default values of optional arguments are not considered part of the public API.
They might change in future releases if we find values that perform better for most use-cases.

We advice you write your scripts in an explicit way where it matters to you,
and rely on implicit default values only where you want the potential future improvements.

File formats
^^^^^^^^^^^^

The same specification applies to files read and produced by *lincs*.
This leads to an issue about backward compatibility:
if we allow more flexibility in input files, new versions of *lincs* will be able to read both the old and the new format, in a backward-compatible way.
But if *lincs* produces a file in the new format, existing client scripts might not be able to read it, making this change backward-incompatible.

To solve this issue, we impose an additional constraint to *lincs*' public API:
*lincs* will produce files in a new format only when the client uses the new feature that motivated the new format.

That way, we know that the client already needs to modify their scripts, so they can adapt to the new format.


Develop *lincs* itself
======================

See our `contributor guide <https://mics-lab.github.io/lincs/contributor-guide.html>`_.

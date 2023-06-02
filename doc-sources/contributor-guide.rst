.. Copyright 2023 Vincent Jacques

=================
Contributor guide
=================


Dependencies
============

To modify *lincs*, you need:

- a reasonably recent version of Bash
- a reasonably recent version of Docker

This is less than what you need to install and use it directly on a machine (see our :doc:`"Get started" guide <get-started>`), because dependencies are installed inside the Docker container.
You can even contribute to *lincs* on an OS that is not supported to run it directly, *e.g.* macOS with `Docker for Desktop <https://www.docker.com/products/docker-desktop/>`_.
If you want to change code that runs on a GPU, you need to configure the NVidia Docker runtime.
@todo Add pointers to the documentation of the NVidia Docker runtime


Development cycle
=================

The main loop when working on *lincs* is:

- make some changes
- run ``./run-development-cycle.sh``
- repeat

The ``./run-development-cycle.sh`` script first `builds a Docker image <https://github.com/MICS-Lab/lincs/blob/main/development/Dockerfile>`_ with the development dependencies.
It then runs that image as a Docker container to build the library, run its C++ and Python unit tests, install it, run its integration tests, *etc.*

Eventually, if you are a maintainer of the PyPI package, you can publish a new version of the package using... ``./publish.sh``.
Else, please `open a pull request <https://github.com/MICS-Lab/lincs/pulls>`_ on GitHub!

.. click:: cycle:main
   :prog: ./run-development-cycle.sh
   :nested: full

.. click:: publish:main
   :prog: ./publish.sh
   :nested: full


Directory structure
===================

@todo Write


Coding standards
================

We have not written coding standard yet, so please have a look around, and try to follow the existing conventions.


General design
==============

@todo Write

Focus on interfaces
-------------------

@todo Write

Strategies
----------

@todo Define, explain added value

https://en.wikipedia.org/wiki/Strategy_pattern

Customization and reusability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

But beware of virtual function calls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- known to be costly
- keep them for not-so-frequent calls; up to a few thousands per learning should be OK
- fallback to template-based static polymorphism using the CRTP for frequent calls (https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)


How-tos
=======

Update the documentation
------------------------

@todo Write

Choose Python or C++ for your change
------------------------------------

@todo Write

Main criteria: computational intensity

Examples:
- a ``WeightsProfilesBreedMrSortLearning::BreadingStrategy`` that reduces the number of iterations of the ``WeightsProfilesBreedMrSortLearning`` can be written in Python because this high-level strategy is called only a few times per learning
- a variant of ``OptimizeWeightsUsingGlop`` that spares a few CPU cycles should be written in C++ because this is where most CPU time is spent

Add a new step in an existing strategy
--------------------------------------

@todo Write

- add a null strategy (@todo Add wiki link to null object pattern)

Add a variant of a strategy
---------------------------

@todo Write

.. Copyright 2023 Vincent Jacques

=================
Contributor guide
=================

We strongly recommend you get familiar with *lincs* by reading all the user documentation before reading this guide.
This will help you contribute in a way that is consistent with the rest of the project.


Do contribute!
==============

We value contributions of any scale, from minor details to major refactorings.
If you see a typo, please fix that typo! Using the GitHub web interface spares you the need to even clone the repository.
If you think our entire architecture deserves a rewrite, please... discuss it with us `<https://github.com/MICS-Lab/lincs/discussions>`_.
(Don't spend time on something that we might reject for reasons not entirely apparent to you at the moment.)
If you have an idea but don't want (or know how) to implement it yourself, please tell us about it.
If you find a bug, please `report it <https://github.com/MICS-Lab/lincs/issues>`_.
Everything helps!

We also recognize that contributing to an open source project can be intimidating,
and that not everyone has the same experience and fluency with the tools and programming languages we use.
If you're willing to get our feedback on your contribution,
you can be assured that we will take time to provide this feedback in a kind and constructive manner.


Development dependencies
========================

To modify *lincs*, you need reasonably recent versions of:

- Bash
- Git
- Docker

This is less than what you need to install and use it directly on a machine (see our :doc:`"Get started" guide <get-started>`),
because dependencies are installed inside the Docker container.
You can even contribute to *lincs* on an OS that is not supported to run it directly,
*e.g.* macOS with `Docker for Desktop <https://www.docker.com/products/docker-desktop/>`_.
If you have an CUDA-compatible NVidia GPU and want to run code that uses it, you need to configure the NVidia Docker runtime.
@todo Add pointers to the documentation of the NVidia Docker runtime


Development cycle
=================

The main loop when working on *lincs* is:

- make some changes
- run ``./run-development-cycle.sh``
- repeat

The ``./run-development-cycle.sh`` script first `builds a Docker image <https://github.com/MICS-Lab/lincs/blob/main/development/Dockerfile>`_ with the development dependencies.
It can take a long time the first time you run it, but the Docker cache makes subsequent builds much faster.
It then runs that image as a Docker container to build the library, run its C++ and Python unit tests, install it, run its integration tests, *etc.*
It provides a few options to speed things up, see its ``--help`` option.

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

All source code is in the ``lincs`` directory.

The ``development`` directory contains scripts and files used for development.
Scripts ``./run-development-cycle.sh`` and ``./publish.sh`` at the root are thin wrappers for scripts in that directory.

The root directory also contains basic packaging files like the ``README.rst``, ``setup.py``, ``MANIFEST.in`` files, as well as the licence files.
See comments in ``.gitignore`` for details about temporary files and directories.

Documentation sources are in ``doc-sources``, and built documentation is in ``docs``, to be published in GitHub Pages.
Published documentation should only be committed when publishing a new version.
You can build it locally to check how your changes are rendered using ``./run-development-cycle.sh --with-docs``, but you must commit only ``doc-sources`` and not ``docs``.

Integration tests are in... ``integration-tests``.
Each test consists of a ``run.sh`` script and accompanying files.
Some tests are generated from the documentation to check that the code examples are correct.
This is done by the ``./run-development-cycle.sh`` script before it runs them.


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

@todo Write

But beware of virtual function calls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. START virtual-cost/run.sh
    set -o errexit
    set -o nounset
    set -o pipefail
    trap 'echo "Error on line $LINENO"' ERR

    g++ -c -O3 lib.cpp -o lib.o
    g++ -O3 no-virtual.cpp lib.o -o no-virtual
    g++ -O3 yes-virtual.cpp lib.o -o yes-virtual

    time ./no-virtual
    time ./yes-virtual
.. STOP

.. highlight:: c++

.. details:: Virtual function calls are costly (click for details)

    .. START virtual-cost/lib.hpp

    Given these classes::

        class Foo {
         public:
          virtual void yes_virtual() = 0;
          void no_virtual();
        };

        class ActualFoo : public Foo {
         public:
          void yes_virtual() override;
        };

    .. STOP

    .. EXTEND virtual-cost/lib.hpp
        Foo* makeFoo();
    .. STOP

    .. START virtual-cost/lib.cpp
        #include "lib.hpp"
    .. STOP

    .. EXTEND virtual-cost/lib.cpp

    And functions::

        void Foo::no_virtual() {}
        void ActualFoo::yes_virtual() {}

        Foo* makeFoo() { return new ActualFoo; }

    .. STOP

    .. START virtual-cost/no-virtual.cpp
        #include "lib.hpp"

        int main() {
    .. STOP

    .. EXTEND virtual-cost/no-virtual.cpp

    The following code runs in ~0.93s::

        Foo* foo = makeFoo();

        for (int i = 0; i != 1'000'000'000; ++i) {
          foo->no_virtual();
        }

    .. STOP

    .. EXTEND virtual-cost/no-virtual.cpp
        }
    .. STOP

    .. START virtual-cost/yes-virtual.cpp
        #include "lib.hpp"

        int main() {
    .. STOP

    .. EXTEND virtual-cost/yes-virtual.cpp

    .. EXTEND virtual-cost/yes-virtual.cpp

    And the following code runs in ~1.12s::

        Foo* foo = makeFoo();
        for (int i = 0; i != 1'000'000'000; ++i) {
          foo->yes_virtual();
        }

    .. STOP

    .. EXTEND virtual-cost/yes-virtual.cpp
        }
    .. STOP

So, although virtual function calls are useful, they must be used with care.
It's best to keep them for cases where they are not called too often; up to a few thousands per learning should be OK.
When polymorphism is required for frequent calls, it's best to use template-based static polymorphism.

An example of that can be found in ``lincs/liblincs/learning/weights-profiles-breed-mrsort/optimize-weights/linear-program.hpp``,
where the LP solver is injected using the ``LinearProgram`` template parameter, at no runtime cost.

So, why not all templates?
~~~~~~~~~~~~~~~~~~~~~~~~~~

One could now consider using templates everywhere, and not use virtual function calls at all.
This would have the following negative consequences:

- The number of explicit template instantiations would explode combinatorially. For example, the ``LinearProgram`` template parameter of ``.../optimize-weights/linear-program.hpp`` is currently instantiated explicitly for each LP solver in ``.../optimize-weights/linear-program.cpp``
- It would forbid customization from the Python side. By nature, Python customization happens at runtime, which requires virtual functions. (For example, from the Python side, it's possible to add a termination strategy, but it's not possible to add an LP solver)

That explains why we use virtual functions where we do.

How-tos
=======

Update the documentation
------------------------

To update the documentation, you'll have to get familiar with the following tools:

- `reStructuredText <https://docutils.sourceforge.io/rst.html>`_
- `Sphinx <https://www.sphinx-doc.org/>`_

And to less extent:

- `sphinx-click <https://sphinx-click.readthedocs.io/>`_

You can then edit ``README.rst`` and files in ``doc-sources`` and run ``./run-development-cycle.sh --with-docs``.
Open ``docs/index.html`` in your browser to check the result.

Choose Python or C++ for your change
------------------------------------

*lincs* is written partly in C++ and partly in Python.
One important reason for a Python part is usability: Python is arguably easier to get started with than C++,
so having a Python interface makes it easier for users to get started with *lincs*.
The main reason for writing the core of *lincs* in C++ is performance: for CPU-intensive tasks,
compiled C++ is definitely faster than interpreted Python; even more so for multi-threaded code.

Here is how we suggest you choose what language to use for your changes:

- Do you know both languages?

If you only know one of those languages, well, use it.
It may not be the best choice for the project, but it is the best choice for you.
If your contribution requires and deserves to be re-implemented in the other language,
then someone else may do it, or you may become motivated enough to learn the other language.

- Should the new feature be exposed in the C++ library?

The core of *lincs* is usable as a C++ library (synthetic data generation, learning, classification).
A counter example is the ``visualization`` module, which is only usable from Python.

If the new feature should be usable through the C++ library, then it must be written in C++.

- How computationally-intensive is the new feature?

Most computationally-intensive parts should be written in C++, and Python can be used for the rest.

For example, a ``WeightsProfilesBreedMrSortLearning::BreadingStrategy`` that reduces the number of iterations of the ``WeightsProfilesBreedMrSortLearning`` can be written in Python because this high-level strategy is called only a few times per learning.
On the other side, a variant of ``OptimizeWeightsUsingGlop`` that spares a few CPU cycles should be written in C++ because this is where most CPU time is spent.

Tweak an existing strategy
--------------------------

@todo Write

Add a new strategy
------------------

@todo Write

Add a new step in an existing strategy
--------------------------------------

@todo Write

- add a null strategy (@todo Add wiki link to null object pattern)

Add an external solver
----------------------

@todo Write

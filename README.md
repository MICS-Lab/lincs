This repository is an attempt to provide a parallel implementation of the meta-heuristic described in [Olivier Sobrie's thesis](https://tel.archives-ouvertes.fr/tel-01370555/document), for learning the parameters of an MR-sort classification model.

Contributors (in alphabetical order):

- [Laurent Cabaret](http://perso.ecp.fr/~cabaretl/)
- [Vincent Jacques](https://vincent-jacques.net)
- [Vincent Mousseau](https://www.centralesupelec.fr/fr/2EBDCB86-64A4-4747-96E8-C3066CB61F3D)

Previous implementations:

- [Original sequential implementation in Python by Olivier Sobrie](https://github.com/oso/pymcda)
- [Sequential implementation in C++ by Emma Dixneuf, Thibault Monsel and Thomas Vindard](https://github.com/Mostah/fastPL/)

Application domain
==================

MR-sort models are one way to automate [decisions based on several criteria](https://en.wikipedia.org/wiki/Multiple-criteria_decision_analysis).

Olivier Sobrie's meta-heuristic is a way to relatively efficiently train an MR-sort "model" from a "learning set".
This learning set is made of "alternatives" pre-classified into "categories" by a "decision maker".

The "learning" process tries to find a model that matches the decision maker's classification as closely as possible.

Example
-------

An example could be trying to predict the evolution of a patient's condition, based on their vital signs and symptoms: the algorithm would predict the outcome (*e.g.* certain death, recovery if treated, natural recovery), *i.e.* "classify" the patient, based on some "criteria" like body temperature, coughing, *etc.*
Health professionals could then focus their efforts on the patients it would benefit most.

In that example, the "alternatives" are the patients, and the "decision maker" is the ground truth.

Glossary
========

Here is a list of the main concepts manipulated in this project, with their mathematical notation and the name(s) of the variables representing them in the code.

@todo Add an actual glossary

Build everything and run all tests
==================================

Dependencies:

- a reasonably recent version of Docker with the NVidia runtime. You can run `docker run --rm --gpus all nvidia/cuda:11.2.2-base-ubuntu20.04 nvidia-smi` to ensure it's properly configured
- a reasonably recent version of Bash

With these dependencies installed,

    ./make.sh -j$(nproc)

builds a Docker image containing all the project's actual dependencies (see `builder/Dockerfile`), then invokes `make` in a container running that image, to build the project and run all tests.

The `./make.sh` script forwards all its command-line arguments to `make`, so you can invoke *e.g.* `./make.sh -j$(nproc) tools` to only build the target `tools`.
See the available targets in the `Makefile`.

Structure of the code
=====================

The code builds into two sets of tools.
The first set generates artificial (pseudo-random) data (models, learning sets, *etc.*).
The second set, arguably more important, operates on data (real or artificial) to learn models and apply then.

A dependency graph of the code is built in `build/dependency-graph.png`.

As a general rule:

- the `builder` directory contains tools to build the project
- the `tools` directory contains the source code of the tools that this project provides
- the `library` directory contains the source code of the core of this project
- everything is build in the `build` directory. That directory can be safely deleted anytime

Pseudo-random data generation
-----------------------------

The `generate-model` tool... generates a pseudo-random model.
Its `--help` option provides usage information.

Its `main` function is in `tools/generate-model.cpp`.
That function parses the command-line, delegates the actual generation to the `ppl::generate::model` function, and finally prints the generated `ppl::io::Model` on the standard output.

`ppl::generate::model` is in `library/generate.cpp`.
The `ppl::io::Model` class is dedicated to input/output of models.
It is in `library/io.cpp`.

The `generate-learning-set` tool also does what its name implies.
It's structure is very similar to the previous tool.

Learning <!-- @todo Add "and classification" -->
--------

The `learn` tool learns a model from a learning set.

Its `main` function is in `tools/learn.cpp`.
It builds a `ppl::Learning` object, sets a few of its attributes according to the options received on the command-line, then calls its `perform` method.
It finally prints the learned `ppl::io::Model` on the standard output.

The `ppl::Learning` class and its `perform` method are in `library/learning.cpp`.
The learning algorithm has three main modules:

- a heuristic for (re-)initializing models, which is in `library/initialize.cpp`
- a linear program for optimizing weights given fixed profiles, which is in `library/optimize-weights.cpp`
<!-- @todo Should we rename to "optimize weights"? (because it does find optimal weights given fixed profiles) -->
- a heuristic for improving profiles given fixed weights, which is in `library/improve-profiles.cpp`

General utilities
-----------------

`library/improve-profiles.cpp` depends on the application of a model (to assign a category to an alternative).
All the assignment algorithms are in `library/assign.cpp`.

`library/initialize.cpp`, `library/optimize-weights.cpp` and `library/improve-profiles.cpp` all use a common representation of the learning set and models in memory that is in `library/problem.hpp`.

When in doubt about a source file, have a look at the associated `-tests.*` file(s).
They contain tests that give a lot of information about the file's purpose.
Also have a look at the comments in the `*.hpp` files.

<!-- @todo Write a tool named `classify` to classify a set of alternatives using a model. -->

Roadmap
=======

Optimize learning duration
--------------------------

Initial measurements with class `Stopwatch` show that roughly 1/3 of the learning time is spent in `optimize_weights` and 2/3 is spent in `improve_profiles`.
Other parts are negligible for now, so we should focus efforts on those two functions.

Low hanging fruits in `improve_profiles`:

- (DONE) parallelize the loop on models: it is embarrassingly parallel.
- pre-compute the interesting values for profiles, and make the algorithm choose between them (see description in comment in `improve_model_profile`)
- store and update the models' accuracy instead of recomputing it again and again. (This may not be a huge improvement because `get_accuracy` is quite fast)

Then, find more intelligent things to improve.
Note that this is good news: the part we want to focus on is actually te longest part.

Low hanging fruits in `optimize_weights`:

- (DONE) parallelize the loop on models using OpenMP: it is embarrassingly parallel.
- avoid repeating some computations in `make_internal_linear_program`: keep one `LinearProgram` in memory for each model, and update it.
Also always pass it to the same `glop::LPSolver`, dedicated to this model, to benefit from GLOP's "re-use" feature, that makes it faster to solve a linear problem that's not too different from a previously solved one.
Warning: this will use more host memory.

It's probably all we can do in `optimize_weights` without significant effort: going further would require diving into solving linear programs, which is its own research domain.

Relax simplifying assumptions
-----------------------------

- relax the assumption that higher numerical values on criteria are better than lower numerical values.
This could be reversed.

- relax the assumption that numerical values for criteria are between 0 and 1.
Allow arbitrary ranges.

- generalize the criteria values to more kinds of ordered sets

- allow "single-peaked" criteria where the good values are in an interval, and high and low numerical values are both bad (*e.g.* blood pressure, where good values are between two bounds, and values outside these bounds are bad)

Learn from noisy learning sets
------------------------------

We currently learn from pseudo-random learning sets that are generated using an MR-sort model.
It is consequently always possible to reconstruct that model exactly, and to reach 100% accuracy.
We should handle generation of noisy pseudo-random learning sets that can only be *approximated* by an MR-sort model.

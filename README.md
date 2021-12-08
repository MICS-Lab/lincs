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
- a linear program for optimizing weights given fixed profiles, which is in `library/improve-weights.cpp`
<!-- @todo Should we rename to "optimize weights"? (because it does find optimal weights given fixed profiles) -->
- a heuristic for improving profiles given fixed weights, which is in `library/improve-profiles.cpp`

`library/improve-profiles.cpp` depends on the application of a model (to assign a category to an alternative).
All the assignment algorithms are in `library/assign.cpp`.

<!-- @todo Write a tool named `classify` to classify a set of alternatives using a model. -->

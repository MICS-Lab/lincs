.. Copyright 2023 Vincent Jacques

==========
User Guide
==========

Before you read this document, we strongly recommend you read our :doc:`conceptual overview <conceptual-overview>` as it establishes the bases for this guide.
We also recommend you follow our :doc:`"Get started" guide <get-started>` to get a first hands-on experience with *lincs*.

This guide is organized following what you can do with *lincs*, *i.e.* the problems you can solve with it.
Each section describes how to to it using *lincs* command-line interface, Python API and C++ API.

Temporary note: this guide is not complete yet.
It is a work in progress.
In particular, APIs are not covered.
You may find it useful to read ``lincs/command_line_interface.py`` to see how the command-line interface uses the Python API.
Similarly, reading ``lincs/liblincs/liblincs_module.cpp`` may help you understand how the Python API exposes the C++ API is, and thus how to use it.


Generate synthetic data
=======================

From the command-line interface
-------------------------------

@todo Write about ``lincs generate``

@todo Write about randomness and the ``--random-seed`` option

@todo Write about outputting to the standard output by default

Through the Python API
----------------------

@todo Write

Through the C++ API
-------------------

@todo Write


Learn a model
=============

From the command-line interface
-------------------------------

The basic command to learn a classification model with *lincs* is ``lincs learn classification-model``.
Its ``--help`` option gives you a list of the numerous options it accepts.
The first one is ``--model-type``; it tells *lincs* what type of model you want it to learn, *e.g.* MR-Sort or :math:`U^c \textsf{-} NCS`.

.. _user-learning-strategies:

Learning strategies
~~~~~~~~~~~~~~~~~~~

There can be several methods to learn a given type of model.
These methods are called "strategies" in *lincs*.
If you've chosen to learn a MR-Sort model, you can specify the learning strategy with the ``--mrsort.strategy`` option,
which can *e.g.* take the value ``weights-profiles-breed`` to select the "weights, profiles, breed" learning strategy.

Then, learning strategies can have parameters.
For example, the "weights, profiles, breed" strategy supports stopping the learning process when the accuracy of the model being learned is good enough.
The "good enough" value is specified using the ``--mrsort.weights-profiles-breed.target-accuracy`` option.

You may notice an emerging pattern in the naming of these options:
when an option makes sense only when a more general option is set to a specific value,
then the name of this more specific option starts with the value of the more general one,
followed by a dot and a name suffix for the specific option.
This naming pattern assuredly makes for some long long option names,
but it's explicit and easy to expand in a backward-compatible manner.
(And it could be worse, *e.g* if we repeated the general option name as well as its value.
So this seems like a good compromise.)

Some strategies even accept sub-strategies.
For example, the "weights, profiles, breed" strategy is a general approach where the weights and profiles of an MR-Sort model are improved alternatively, independently from each other.
It naturally accept a strategy for each of these two sub-problems, respectively through the ``--mrsort.weights-profiles-breed.weights-strategy`` and ``--mrsort.weights-profiles-breed.profiles-strategy`` options.
And if the weights strategy is ``linear-program``, then you can chose a solver using ``--mrsort.weights-profiles-breed.linear-program.solver``.

All these options have default values that we believe are the most likely to provide good results in the general case.
These default values *will change* in future releases of *lincs* when we develop better algorithms.
So, you should specify explicitly the ones that matter to your use-case, and use the default values when you want to benefit implicitly from future improvements.
Note that the general improvements will undoubtedly worsen the situation for some corner cases, but there is nothing anyone can do about that.

.. START other-learnings/run.sh
    set -o errexit
    set -o nounset
    set -o pipefail
    trap 'echo "Error on line $LINENO"' ERR

    cp ../command-line-example/{problem.yml,learning-set.csv} .
    cp ../command-line-example/expected-trained-model.yml .
.. STOP

.. START other-learnings/uses-gpu
.. STOP

.. START other-learnings/is-long
.. STOP

The following example assumes you've followed our :doc:`"Get started" guide <get-started>` and have ``problem.yml`` and ``learning-set.csv`` in your current directory.
It also assumes you have an NVidia GPU with CUDA support and its drivers correctly installed.
If you're using the Docker image, it further assumes you're running it with NVidia Docker Runtime properly configured and activated (*e.g.* with the ``--gpus all`` option of ``docker run``).
If those conditions are verified, you can tweak the "weights, profiles, breed" learning process to:

- use your GPU for the improvement of the profiles
- use the Alglib linear programming solver (instead of GLOP) for the improvement of the weights

.. highlight:: shell

.. EXTEND other-learnings/run.sh

::

    lincs learn classification-model problem.yml learning-set.csv \
      --output-model gpu+alglib-trained-model.yml \
      --mrsort.weights-profiles-breed.accuracy-heuristic.processor gpu \
      --mrsort.weights-profiles-breed.linear-program.solver alglib

.. APPEND-TO-LAST-LINE --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed 43
.. STOP

This should output a similar model, with slight numerical differences.

.. START other-learnings/expected-gpu+alglib-trained-model.yml
    kind: ncs-classification-model
    format_version: 1
    boundaries:
      - profile: [0.007700569, 0.05495565, 0.1626169, 0.1931279]
        sufficient_coalitions: &coalitions
          kind: weights
          criterion_weights: [0.01812871, 0.9818703, 0.9818703, 9.925777e-13]
      - profile: [0.03420721, 0.3244802, 0.6724876, 0.4270518]
        sufficient_coalitions: *coalitions
.. STOP

.. EXTEND other-learnings/run.sh
    diff expected-gpu+alglib-trained-model.yml gpu+alglib-trained-model.yml
.. STOP

.. EXTEND other-learnings/run.sh

You can also use an entirely different approach using SAT and max-SAT solvers::

    lincs learn classification-model problem.yml learning-set.csv \
      --output-model minisat-coalitions-trained-model.yml \
      --model-type ucncs --ucncs.approach sat-by-coalitions

    lincs learn classification-model problem.yml learning-set.csv \
      --output-model minisat-separation-trained-model.yml \
      --model-type ucncs --ucncs.approach sat-by-separation

.. STOP

.. START other-learnings/expected-minisat-coalitions-trained-model.yml

It should produce a different kind of model, with the sufficient coalitions specified explicitly by their roots::

    kind: ncs-classification-model
    format_version: 1
    boundaries:
      - profile: [1, 0.05526805, 0.1619191, 0.9954021]
        sufficient_coalitions: &coalitions
          kind: roots
          upset_roots:
            - [1, 2]
      - profile: [1, 0.3252118, 0.6726626, 0.9967546]
        sufficient_coalitions: *coalitions

.. STOP

.. START other-learnings/expected-minisat-separation-trained-model.yml
    kind: ncs-classification-model
    format_version: 1
    boundaries:
      - profile: [0.1682088, 0.05526805, 0.1619191, 0.9954021]
        sufficient_coalitions: &coalitions
          kind: roots
          upset_roots:
            - [0, 1, 2]
            - [0, 1, 2, 3]
            - [1, 2]
            - [1, 2, 3]
      - profile: [1, 0.3252118, 0.6726626, 0.9967546]
        sufficient_coalitions: *coalitions
.. STOP

.. EXTEND other-learnings/run.sh
    diff expected-minisat-coalitions-trained-model.yml minisat-coalitions-trained-model.yml
    diff expected-minisat-separation-trained-model.yml minisat-separation-trained-model.yml
.. STOP

Output location
~~~~~~~~~~~~~~~

Like synthetic data generation command, ``lincs learn classification-model`` outputs to the standard output by default,
that is if you don't specify the ``--output-model`` option, it will simply print the learned model to your console.

Randomness in heuristic strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some learning (sub-)strategies implement heuristic algorithms.
In that case, they accept a ``.random-seed`` option to initialize the pseudo-random number generator they use.
If this option is not specified, the pseudo-random number generator is initialized with a random seed.
You should use this option when you need deterministic results from the learning process, *e.g.* when you're comparing two strategies.

.. EXTEND other-learnings/run.sh

When possible when we supply several implementations of the same heuristic, we make them behave the same way when they're given the same random seed.
This is the case for example for the CPU and GPU versions of the "accuracy heuristic" profiles improvement strategy of the "weights, profiles, breed" learning strategy.
This ensures that the two following commands output exactly the same model::

    lincs learn classification-model problem.yml learning-set.csv \
      --output-model cpu-trained-model.yml \
      --mrsort.weights-profiles-breed.accuracy-heuristic.processor cpu \
      --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed 43

    lincs learn classification-model problem.yml learning-set.csv \
      --output-model gpu-trained-model.yml \
      --mrsort.weights-profiles-breed.accuracy-heuristic.processor gpu \
      --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed 43

.. STOP

.. EXTEND other-learnings/run.sh
    diff expected-trained-model.yml cpu-trained-model.yml
    diff expected-trained-model.yml gpu-trained-model.yml
.. STOP

Through the Python API
----------------------

@todo Write

Through the C++ API
-------------------

@todo Write


Use a model
===========

From the command-line interface
-------------------------------

@todo Write about ``lincs classify`` (outputting to stdout by default)

@todo Write about ``lincs classification-accuracy`` (always outputting to stdout)

@todo Write about ``lincs visualize classification-model`` (mandatory output parameter, use - to output to stdout)

Through the Python API
----------------------

@todo Write

Through the C++ API
-------------------

@todo Write

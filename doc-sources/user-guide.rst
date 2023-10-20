.. Copyright 2023 Vincent Jacques

==========
User Guide
==========

Before you read this document, we strongly recommend you read our :doc:`conceptual overview <conceptual-overview>` as it establishes the bases for this guide.
We also recommend you follow our :doc:`"Get started" guide <get-started>` to get a first hands-on experience with *lincs*.


Formatting data for *lincs*
===========================

*lincs* manipulates files for three types of data.

"Problem" files
---------------

The concept of classification problem is defined in our :ref:`conceptual overview <overview-about-classification>`.
To describe problems, *lincs* uses YAML files conforming to the `JSON schema <https://json-schema.org/>`_ you'll find in our :ref:`reference documentation <ref-file-problem>`.

.. START check-user-guide/run.sh
    set -o errexit
    set -o nounset
    set -o pipefail
    trap 'echo "Error on line $LINENO"' ERR

    lincs generate classification-model problem-example.yml --random-seed 42 --output-model model.yml
    diff model.yml expected-model.yml
.. STOP

.. START check-user-guide/problem-example.yml

Here is an example of a problem file::

    kind: classification-problem
    format_version: 1
    criteria:
      - name: Criterion 1
        value_type: real
        category_correlation: growing
        min_value: 0
        max_value: 20
      - name: Criterion 2
        value_type: real
        category_correlation: decreasing
        min_value: -5
        max_value: 5
    categories:
      - name: Low
      - name: Medium
      - name: High

.. STOP

The two first keys, ``kind`` and ``format_version`` are here to identify exactly the file format.
For now, they must always be set to ``classification-problem`` and ``1`` respectively.

Criteria
^^^^^^^^

The third key, ``criteria``, is a list of the descriptions of the criteria of the problem.
This list must contain at least one element because classification problems must have at least one criterion.

Each criterion gets a ``name`` for convenience.

Currently, criteria can only take floating point values, so their ``value_type`` is always ``real``.
We expect this could evolve to also support criteria with integer or explicitly enumerated values.

Then, the ``category_correlation`` key describe what makes "good values" for this criterion.
If it is ``growing`` (resp. ``decreasing``), then higher (resp. lower) numerical values correspond to upper categories.
Note that this correlation comes from expert knowledge about the structure of the problem,
and has nothing to do with learning a model for this problem.
We expect this could evolve to also support criteria with single-peaked correlation,
where intermediate numerical value correspond to upper categories, and extreme values to lower categories.
We also expect this could evolve to support criteria with unknown correlation,
to support the case where no expert knowledge is available and delegate this choice to the learning process.

Finally, for criteria with numerical ``value_type`` (currently all of them),
the ``min_value`` and ``max_value`` keys describe the range of values the criterion can take.

Categories
^^^^^^^^^^

The fourth key in the problem file, ``categories``, is a list of the descriptions of the categories of the problem.
It must contain at least two elements because classification problems must have at least two categories.

It must be sorted in increasing order: lower categories first and upper categories last.

Its elements are relatively simple as they only get a convenience ``name`` attribute.

"Model" files
-------------

The concept of NCS classification model is defined in our :ref:`conceptual overview <overview-ncs>`.
To describe models, *lincs* uses YAML files conforming to the JSON schema you'll find in our :ref:`reference documentation <ref-file-ncs-model>`.

"Alternatives" files
--------------------

@todo(Documentation, soon) Write

Comments in generated files
---------------------------

When the *lincs* command-line generates a file, it add a few comments describing how this file was made.
These comments are informative and can help reproducing results, but are not part of the file formats.


Generating synthetic data
=========================

@todo(Documentation, soon) Write about ``lincs generate``

@todo(Documentation, soon) Write about randomness and the ``--random-seed`` option

@todo(Documentation, soon) Write about outputting to the standard output by default


Learning a model
================

The basic command to learn a classification model with *lincs* is ``lincs learn classification-model``.
Its ``--help`` option gives you a list of the numerous options it accepts.
The first one is ``--model-type``; it tells *lincs* what type of model you want it to learn, *e.g.* MR-Sort or :math:`U^c \textsf{-} NCS`.

.. _user-learning-strategies:

Learning strategies
-------------------

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

    diff <(echo "lincs was compiled with CUDA support") <(lincs info has-gpu)
.. STOP

.. START other-learnings/uses-gpu
.. STOP

.. START other-learnings/is-long
.. STOP

The following example makes a few assumptions:

- you've followed our :doc:`"Get started" guide <get-started>` and have ``problem.yml`` and ``learning-set.csv`` in your current directory
- your installed version of lincs was built with CUDA support (check with ``lincs info has-gpu``)
- you have an NVidia GPU with CUDA support and its drivers correctly installed
- if you're using the Docker image, you're running it with NVidia Docker Runtime properly configured and activated (*e.g.* with the ``--gpus all`` option of ``docker run``)

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
    # Reproduction command (with lincs version 0.9.1): lincs learn classification-model problem.yml learning-set.csv --model-type mrsort --mrsort.strategy weights-profiles-breed --mrsort.weights-profiles-breed.models-count 9 --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed 43 --mrsort.weights-profiles-breed.initialization-strategy maximize-discrimination-per-criterion --mrsort.weights-profiles-breed.weights-strategy linear-program --mrsort.weights-profiles-breed.linear-program.solver alglib --mrsort.weights-profiles-breed.profiles-strategy accuracy-heuristic --mrsort.weights-profiles-breed.accuracy-heuristic.processor gpu --mrsort.weights-profiles-breed.breed-strategy reinitialize-least-accurate --mrsort.weights-profiles-breed.reinitialize-least-accurate.portion 0.5 --mrsort.weights-profiles-breed.target-accuracy 1.0
    # Termination condition: target accuracy reached
    # Number of iterations: 9
    kind: ncs-classification-model
    format_version: 1
    accepted_values:
      - kind: thresholds
        thresholds: [0.924693644, 0.971395075]
      - kind: thresholds
        thresholds: [0.0556534864, 0.326433569]
      - kind: thresholds
        thresholds: [0.162616938, 0.671892762]
      - kind: thresholds
        thresholds: [0.942387044, 0.988728762]
    sufficient_coalitions:
      - &coalitions
        kind: weights
        criterion_weights: [0.293799639, 0.386859566, 0.613140464, 0.304567546]
      - *coalitions
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

    # Reproduction command (with lincs version 0.9.1): lincs learn classification-model problem.yml learning-set.csv --model-type ucncs --ucncs.approach sat-by-coalitions
    kind: ncs-classification-model
    format_version: 1
    accepted_values:
      - kind: thresholds
        thresholds: [0.999706864, 0.999706864]
      - kind: thresholds
        thresholds: [0.0552680492, 0.325211823]
      - kind: thresholds
        thresholds: [0.161919117, 0.672662616]
      - kind: thresholds
        thresholds: [0.995402098, 0.996754646]
    sufficient_coalitions:
      - &coalitions
        kind: roots
        upset_roots:
          - [1, 2]
      - *coalitions

.. STOP

.. START other-learnings/expected-minisat-separation-trained-model.yml
    # Reproduction command (with lincs version 0.9.1): lincs learn classification-model problem.yml learning-set.csv --model-type ucncs --ucncs.approach sat-by-separation
    kind: ncs-classification-model
    format_version: 1
    accepted_values:
      - kind: thresholds
        thresholds: [0.0198908672, 0.999706864]
      - kind: thresholds
        thresholds: [0.0552680492, 0.325211823]
      - kind: thresholds
        thresholds: [0.161919117, 0.672662616]
      - kind: thresholds
        thresholds: [0.995402098, 0.996754646]
    sufficient_coalitions:
      - &coalitions
        kind: roots
        upset_roots:
          - [1, 2]
      - *coalitions
.. STOP

.. EXTEND other-learnings/run.sh
    diff expected-minisat-coalitions-trained-model.yml minisat-coalitions-trained-model.yml
    diff expected-minisat-separation-trained-model.yml minisat-separation-trained-model.yml
.. STOP

Output location
---------------

Like synthetic data generation command, ``lincs learn classification-model`` outputs to the standard output by default,
that is if you don't specify the ``--output-model`` option, it will simply print the learned model to your console.

Randomness in heuristic strategies
----------------------------------

Some learning (sub-)strategies implement heuristic algorithms.
In that case, they accept a ``.random-seed`` option to initialize the pseudo-random number generator they use.
If this option is not specified, the pseudo-random number generator is initialized with a random seed.
You should use this option when you need deterministic results from the learning process, *e.g.* when you're comparing two strategies.

.. EXTEND other-learnings/run.sh

When possible when we provide several implementations of the same heuristic, we make them behave the same way when they're given the same random seed.
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
    diff <(sed s/cpu/gpu/ expected-trained-model.yml) gpu-trained-model.yml
.. STOP


Using a model
=============

@todo(Documentation, soon) Write about ``lincs classify`` (outputting to stdout by default)

@todo(Documentation, soon) Write about ``lincs classification-accuracy`` (always outputting to stdout)

@todo(Documentation, soon) Write about ``lincs visualize classification-model`` (mandatory output parameter, use - to output to stdout)

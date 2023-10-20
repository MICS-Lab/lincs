.. Copyright 2023 Vincent Jacques

==========
User Guide
==========

Before you read this document, we strongly recommend you read our :doc:`conceptual overview <conceptual-overview>` as it establishes the bases for this guide.
We also recommend you follow our :doc:`"Get started" guide <get-started>` to get a first hands-on experience with *lincs*.


Formatting data for *lincs*
===========================

*lincs* manipulates files for three types of data.

.. _user-file-problem:

"Problem" files
---------------

The concept of classification problem is defined in our :ref:`conceptual overview <overview-about-classification>`.
To describe problems, *lincs* uses YAML files conforming to the `JSON schema <https://json-schema.org/>`_ you'll find in our :ref:`reference documentation <ref-file-problem>`.

.. START file-formats/run.sh
    set -o errexit
    set -o nounset
    set -o pipefail
    trap 'echo "Error on line $LINENO"' ERR

    lincs generate classification-model problem.yml --random-seed 42 --output-model mrsort-model.yml
    diff <(tail -n +2 mrsort-model.yml) expected-mrsort-model.yml

    # Check that the NCS model is correct (we don't have explicit commands for that, so we use generate classified-alternatives)
    lincs generate classified-alternatives problem.yml ncs-model.yml 1 >/dev/null

    lincs classify problem.yml mrsort-model.yml unclassified-alternatives.csv --output-classified-alternatives classified-alternatives.csv
    diff classified-alternatives.csv expected-classified-alternatives.csv
.. STOP

.. START file-formats/problem.yml

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

Each criterion has a ``name``.

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

Its elements are relatively simple as they only get a ``name``.

.. _user-file-ncs-model:

"Model" files
-------------

The concept of NCS classification model is defined in our :ref:`conceptual overview <overview-ncs>`.
To describe models, *lincs* uses YAML files conforming to the JSON schema you'll find in our :ref:`reference documentation <ref-file-ncs-model>`.

A model file is always associated to a problem file because a model exists only relatively to a given problem.
This fact is not captured by our file format for technical reasons:
embedding the problem in the model file would lead to unwanted repetitions,
referencing the problem file by name would not be robust because files can be renamed,
and referencing the problem file by content (using a hash) would forbid any change in the problem file.
So it's the user's responsibility to keep track of that information and always give *lincs* the correct problem file along with a model file.

.. START file-formats/expected-mrsort-model.yml

Here is an example of a model file corresponding to the problem file above::

    kind: ncs-classification-model
    format_version: 1
    accepted_values:
      - kind: thresholds
        thresholds: [7.49331188, 15.9249287]
      - kind: thresholds
        thresholds: [4.49812794, -3.15932083]
    sufficient_coalitions:
      - &coalitions
        kind: weights
        criterion_weights: [0.938825667, 0.343733728]
      - *coalitions

.. STOP

Like for problem files, the two first keys must take exactly these values.

Accepted values
^^^^^^^^^^^^^^^

The third key, ``accepted_values``, lists the descriptions of the accepted values according to each criterion of the problem.
It must contain exactly as many elements as the ``criteria`` list in the problem file.

For NCS models as currently defined in our conceptual overview, accepted values are simply above a profile.
The profile is a list of thresholds, one for each criterion, that separates two categories.
But this lacks generality, and we expect this could evolve, for example for single-peaked criteria.
For such a criterion, the determination of the accepted values will require two limits (upper and lower) instead of just one threshold.

So our file format takes an transposed approach and focusses on criteria instead of profiles:
for each criterion, it describes the method used to accept values at different category levels.

For current criteria (with ``growing`` or ``decreasing`` correlation), the method is always ``kind: thresholds``,
and the ``thresholds`` attribute lists the successive values required to enter an upper category.
It must have as many elements as there are boundaries between categories, *i.e.* as there are categories minus one.
It's always sorted, in increasing order for ``growing`` criteria and in decreasing order for ``decreasing`` criteria.

Note that this list is not a profile: it does not describe the limits between categories.
The matrix made of these lists is the transposed of the matrix made of the profiles.

When we support criteria with single-peaked or unknown correlation,
we'll introduce other ``kinds`` of accepted values with new attributes instead of ``thresholds``.

==================================  ========================  ==========================
Criterion ``category_correlation``  Accepted values ``kind``  Accepted values attributes
==================================  ========================  ==========================
``growing``                         ``thresholds``            ``thresholds``
``decreasing``                      ``thresholds``            ``thresholds``
==================================  ========================  ==========================

Sufficient coalitions
^^^^^^^^^^^^^^^^^^^^^

The fourth key, ``sufficient_coalitions``, describes the subsets of criteria required to get into upper categories.
It contains as many items as there are boundaries between categories, *i.e.* as there are categories minus one.

*lincs* only manipulates :math:`U^c \textsf{-} NCS` models for now, so the sufficient coalitions are identical for all categories.
To avoid repetitions in the model files, we use `YAML anchors and references <https://yaml.org/spec/1.2-old/spec.html#id2765878>`_.
All ``*coalitions`` means in the example above is "the same value as the ``&coalitions`` anchor".

Each item in the list has a first attribute, ``kind``, that tells the method used to determine the sufficient coalitions.

If its ``kind`` is ``weights``, then the sufficient coalitions are computed using an MR-sort approach,
as described in our :ref:`conceptual overview <overview-mrsort>`.
In that case, the ``criterion_weights`` attribute is a list of the criteria's weights.

If its ``kind`` is ``roots``, then the sufficient coalitions are listed explicitly as the roots of the upset they form.
This is the generic case for NCS models.
In that case, the ``upset_roots`` attribute is a list of roots, where each root is the list of the indices of the criteria in that root.

==============================  ================================
Sufficient coalitions ``kind``  Sufficient coalitions attributes
==============================  ================================
``weights``                     ``criterion_weights``
``roots``                       ``upset_roots``
==============================  ================================

.. START file-formats/ncs-model.yml

Here is another model corresponding to the problem file above, but this time using the ``roots`` kind of sufficient coalitions,
and using different coalitions for the two boundaries (so, no YAML anchor)::

    kind: ncs-classification-model
    format_version: 1
    accepted_values:
      - kind: thresholds
        thresholds: [7.49331188, 15.9249287]
      - kind: thresholds
        thresholds: [4.49812794, -3.15932083]
    sufficient_coalitions:
      - kind: roots
        upset_roots:
          - [2]
      - kind: roots
        upset_roots:
          - [1, 2]

.. STOP

"Alternatives" files
--------------------

The last file format used by *lincs* is for the description of alternatives.
It's a CSV file with a header line and one line per alternative.

Like model files, alternatives files are always associated to a problem file.

.. START file-formats/expected-classified-alternatives.csv

Here is an example corresponding to the problem above::

    name,"Criterion 1","Criterion 2",category
    "Alternative 1",10.8156891,4.39045048,Medium
    "Alternative 2",0.25551182,-1.45864725,Low
    "Alternative 3",18.4786396,4.31117153,Medium
    "Alternative 4",18.0154629,1.33949804,Medium
    "Alternative 5",9.30789757,2.66963387,Medium

.. STOP

Its header line contains the names of its columns.
Its first column, ``name``, contains the names of the alternatives.
Its intermediate columns, named after the names of criteria, contain the values of the criteria for each alternative.
Its last column, ``category``, contains the names of the categories in which each alternative is classified.

.. START file-formats/unclassified-alternatives.csv

Values in the ``category`` column can be empty to describe alternatives that are not (yet) classified::

    name,"Criterion 1","Criterion 2",category
    "Alternative 1",10.8156891,4.39045048,
    "Alternative 2",0.25551182,-1.45864725,
    "Alternative 3",18.4786396,4.31117153,
    "Alternative 4",18.0154629,1.33949804,
    "Alternative 5",9.30789757,2.66963387,

.. STOP

.. _user-comments-in-generated-files:

Comments in generated files
---------------------------

When the *lincs* command-line generates a file, it adds a few comment lines (starting with ``#``) at the beginning describing how this file was made.
These comments are informative and can help reproducing results, but they are not part of the file formats.


Generating synthetic data
=========================

The previous section described how to format your data to use it with *lincs*.
As explained in our :ref:`conceptual overview <overview-synthetic-data>`,
you can skip this step and use *lincs* to generate synthetic data.

The parent command to generate synthetic data is ``lincs generate``.
Its sub-commands specify what to generate.
Like all *lincs* commands, they output on the standard output by default,
and you can change that behavior using options to output to files.

About randomness
----------------

Most sub-commands of ``lincs generate`` use pseudo-randomness to generate their output.
By default, the pseudo-random number generator is initialized with a seed based on the current machine, time, *etc.* to favor originality.

When you need reproducibility, you can specify the seed to use with the ``--random-seed`` option.

In all cases, the :ref:`comments <user-comments-in-generated-files>` left by *lincs* in the generated files specify the seed that was used.

Generating a problem
--------------------

.. START synthetic-data/run.sh
    set -o errexit
    set -o nounset
    set -o pipefail
    trap 'echo "Error on line $LINENO"' ERR

With ``lincs generate classification-problem``, you can generate a classification problem file.
Using its default settings, you just have to pass it the numbers of criteria and categories you want, as you saw in our :doc:`get started guide <get-started>`::

    lincs generate classification-problem 4 3

.. APPEND-TO-LAST-LINE --output-problem problem.yml
.. STOP

The ``--help`` option on the command-line and our :ref:`reference documentation <ref-cli>` describe the options available to tweak the generated problem.
Most notably:

- ``--denormalized-min-max`` generates problems with pseudo-random ``min_value`` and ``max_value`` for each criterion. By default, they are always set at 0 and 1.
- ``--allow-decreasing-criteria`` chooses pseudo-randomly the ``category_corelation`` of each criterion between ``growing`` and ``decreasing``. By default, all criteria have ``growing`` correlation.

Generating a model
------------------

.. EXTEND synthetic-data/run.sh

With ``lincs generate classification-model``, you can generate a classification model file.
Using its default settings, you just have to pass it the problem file you want to use::

    lincs generate classification-model problem.yml

.. APPEND-TO-LAST-LINE --output-model model.yml
.. STOP

For now, *lincs* can only generate MR-Sort models, so the ``--model-type`` option can only take its default value: ``mrsort``.
We expect this could change if we implement the generation of other types of models.

By default, the sum of MR-Sort weights of the criteria is pseudo-random and greater than or equal to 1.
With the ``--mrsort.fixed-weight-sum`` option, you can specify a fixed value for this sum.
This effectively impacts how hard it is for alternatives to get into upper categories.

Generating alternatives
-----------------------

.. EXTEND synthetic-data/run.sh

With its default settings, ``lincs generate classified-alternatives`` requires only the problem and model files and the number of alternatives to generate::

    lincs generate classified-alternatives problem.yml model.yml 100

.. APPEND-TO-LAST-LINE --output-classified-alternatives classified-alternatives.csv
.. STOP

This generates 100 random alternatives, and then classifies them according to the model.

By default, no effort is made to balance the number of alternatives in each category.
The ``--max-imbalance`` option can be used to ensure that: it accepts a number between 0 and 1,
and ensures that the number of alternatives in each category differs from the perfectly balanced size by at most this fraction.

For example, when generating 600 alternatives for a model with 3 categories, the perfectly balanced size is 200 alternatives per category.
With ``--max-imbalance 0.2``, the number of alternatives in each category is allowed to differ by at most 20% from that perfectly balanced size,
so each category will have between 160 and 240 alternatives.

Using this option with very selective models can significantly increase the time required to generate the alternatives.
In some cases, *lincs* will even give up when it makes no progress trying to populate categories that are too hard to reach.
In that case, you can either increase the value passed to ``--max-imbalance`` or use a more lenient model.

By default, alternatives are classified exactly according to the given model.
You can introduce noise using the ``--misclassified-count`` option.
After alternatives are generated and classified, this option randomly selects the given number of alternatives and classifies them in other categories.


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

    cp ../..get-started/command-line-example/{problem.yml,learning-set.csv} .
    cp ../..get-started/command-line-example/expected-trained-model.yml .

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
    # Reproduction command (with lincs version 0.9.2-dev): lincs learn classification-model problem.yml learning-set.csv --model-type mrsort --mrsort.strategy weights-profiles-breed --mrsort.weights-profiles-breed.models-count 9 --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed 43 --mrsort.weights-profiles-breed.initialization-strategy maximize-discrimination-per-criterion --mrsort.weights-profiles-breed.weights-strategy linear-program --mrsort.weights-profiles-breed.linear-program.solver alglib --mrsort.weights-profiles-breed.profiles-strategy accuracy-heuristic --mrsort.weights-profiles-breed.accuracy-heuristic.processor gpu --mrsort.weights-profiles-breed.breed-strategy reinitialize-least-accurate --mrsort.weights-profiles-breed.reinitialize-least-accurate.portion 0.5 --mrsort.weights-profiles-breed.target-accuracy 1.0
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

    # Reproduction command (with lincs version 0.9.2-dev): lincs learn classification-model problem.yml learning-set.csv --model-type ucncs --ucncs.approach sat-by-coalitions
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
    # Reproduction command (with lincs version 0.9.2-dev): lincs learn classification-model problem.yml learning-set.csv --model-type ucncs --ucncs.approach sat-by-separation
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

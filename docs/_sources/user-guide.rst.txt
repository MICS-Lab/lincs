.. WARNING: this file is generated from 'doc-sources/user-guide.rst.tmpl'. MANUAL EDITS WILL BE LOST.

.. Copyright 2023-2024 Vincent Jacques

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

Here is an example of a problem file:

.. code:: yaml

    kind: classification-problem
    format_version: 1
    criteria:
      - name: Criterion 1
        value_type: real
        preference_direction: increasing
        min_value: 0
        max_value: 20
      - name: Criterion 2
        value_type: integer
        preference_direction: decreasing
        min_value: -5
        max_value: 5
      - name: Criterion 3
        value_type: enumerated
        ordered_values: [F, E, D, C, B, A]
    ordered_categories:
      - name: Low
      - name: Medium
      - name: High


The two first keys, ``kind`` and ``format_version`` are here to identify exactly the file format.
For now, they must always be set to ``classification-problem`` and ``1`` respectively.

Criteria
^^^^^^^^

The third key, ``criteria``, is a list of the descriptions of the criteria of the problem.
This list must contain at least one element because classification problems must have at least one criterion.

Each criterion has a ``name``.

Currently, criteria can only take floating point values, so their ``value_type`` is always ``real``.
We expect this could evolve to also support criteria with integer or explicitly enumerated values.
A criterion's ``value_type`` specifies the kind of values it can take: ``real``, ``integer`` or ``enumerated``.

For numerical criteria (*i.e* with ``real`` or ``integer`` ``value_type``),
the ``preference_direction`` key describe what makes "good values" for this criterion.
If it is ``increasing`` (resp. ``decreasing``), then higher (resp. lower) numerical values correspond to upper categories.
Note that this preference direction comes from expert knowledge about the structure of the problem,
and will be used as an absolute truth when learning a model for this problem.
We expect the supported preference directions could evolve to also support single-peaked criteria,
where intermediate numerical value correspond to upper categories, and extreme values to lower categories.
We also expect this could evolve to support criteria with unknown preference direction,
to support the case where no expert knowledge is available and delegate this choice to the learning process.

For ``enumerated`` criteria, the ``ordered_values`` key is a list of the values the criterion can take, each one as a string.
The preference direction is implied in that case: the list must be sorted from the worst value to the best one.

Finally, for numerical criteria, the ``min_value`` and ``max_value`` keys describe the range of values the criterion can take.

Categories
^^^^^^^^^^

The fourth key in the problem file, ``ordered_categories``, is a list of the descriptions of the categories of the problem.
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

Here is an example of a model file corresponding to the problem file above:

.. code:: yaml

    kind: ncs-classification-model
    format_version: 1
    accepted_values:
      - kind: thresholds
        thresholds: [6.09463787, 19.7704506]
      - kind: thresholds
        thresholds: [2, 1]
      - kind: thresholds
        thresholds: [E, D]
    sufficient_coalitions:
      - &coalitions
        kind: weights
        criterion_weights: [0.173891723, 1.97980487, 0.0961765796]
      - *coalitions


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

For current criteria (with ``increasing`` or ``decreasing`` preference direction), the method is always ``kind: thresholds``,
and the ``thresholds`` attribute lists the successive values required to enter an upper category.
It must have as many elements as there are boundaries between categories, *i.e.* as there are categories minus one.
It's always sorted, in increasing order for ``increasing`` criteria and in decreasing order for ``decreasing`` criteria.

Note that this list is not a profile: it does not describe the limits between categories.
The matrix made of these lists is the transposed of the matrix made of the profiles.

When we support single-peaked criteria or criteria with unknown preference direction,
we'll introduce other ``kinds`` of accepted values with new attributes instead of ``thresholds``.

==================================  ========================  ==========================
Criterion ``preference_direction``  Accepted values ``kind``  Accepted values attributes
==================================  ========================  ==========================
``increasing``                      ``thresholds``            ``thresholds``
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
It must contain exactly one element per criterion.

If its ``kind`` is ``roots``, then the sufficient coalitions are listed explicitly as the roots of the upset they form.
This is the generic case for NCS models.
In that case, the ``upset_roots`` attribute is a list of roots, where each root is the list of the zero-based indices of the criteria in that root.

==============================  ================================
Sufficient coalitions ``kind``  Sufficient coalitions attributes
==============================  ================================
``weights``                     ``criterion_weights``
``roots``                       ``upset_roots``
==============================  ================================

Here is another model corresponding to the problem file above, but this time using the ``roots`` kind of sufficient coalitions,
and using different coalitions for the two boundaries (so, no YAML anchor):

.. code:: yaml

    kind: ncs-classification-model
    format_version: 1
    accepted_values:
      - kind: thresholds
        thresholds: [7.49331188, 15.9249287]
      - kind: thresholds
        thresholds: [4, -3]
      - kind: thresholds
        thresholds: [D, B]
    sufficient_coalitions:
      - kind: roots
        upset_roots:
          - [1]
          - [0, 2]
      - kind: roots
        upset_roots:
          - [0, 2]


"Alternatives" files
--------------------

The last file format used by *lincs* is for the description of alternatives.
It's a CSV file with a header line and one line per alternative.

Like model files, alternatives files are always associated to a problem file.

Here is an example corresponding to the problem above:

.. code:: text

    name,"Criterion 1","Criterion 2","Criterion 3",category
    "Alternative 1",10.8156891,4,A,Low
    "Alternative 2",0.25551182,-1,D,High
    "Alternative 3",18.4786396,4,B,Low
    "Alternative 4",18.0154629,1,F,High
    "Alternative 5",9.30789757,2,A,Medium


Its header line contains the names of its columns.
Its first column, ``name``, contains the names of the alternatives.
Its intermediate columns, named after the names of criteria, contain the values of the criteria for each alternative.
Its last column, ``category``, contains the names of the categories in which each alternative is classified.

Values in the ``category`` column can be empty to describe alternatives that are not (yet) classified:

.. code:: text

    name,"Criterion 1","Criterion 2","Criterion 3",category
    "Alternative 1",10.8156891,4.39045048,A,
    "Alternative 2",0.25551182,-1.45864725,D,
    "Alternative 3",18.4786396,4.31117153,B,
    "Alternative 4",18.0154629,1.33949804,F,
    "Alternative 5",9.30789757,2.66963387,A,


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

With ``lincs generate classification-problem``, you can generate a classification problem file.
Using its default settings, you just have to pass it the numbers of criteria and categories you want, as you saw in our :doc:`get started guide <get-started>`:

.. code:: shell

    lincs generate classification-problem 4 3

The ``--help`` option on the command-line and our :ref:`reference documentation <ref-cli>` describe the options available to tweak the generated problem.

A set of options selects the ``value_type`` of the criteria in the generated problem.
By default, they are all ``real``.
The ``--allow-integer-criteria`` and ``--allow-enumerated-criteria`` options let you generate problems with ``integer`` and ``enumerated`` criteria respectively.
The ``--forbid-real-criteria``... forbids ``real`` criteria.
The ``value_type`` of each criterion is chosen pseudo-randomly among the allowed ones.

For numerical criteria, the ``--denormalized-min-max`` option generates problems with pseudo-random ``min_value`` and ``max_value`` for each criterion.
By default, they are always set at 0 and 1.

For numerical criteria, ``--allow-decreasing-criteria`` chooses pseudo-randomly the ``preference_direction`` of each criterion between ``increasing`` and ``decreasing``.
By default, all criteria have ``increasing`` preference direction.
It can be used in conjunction with ``--forbid-increasing-criteria`` to generate problems with only ``decreasing`` criteria.

Generating a model
------------------

With ``lincs generate classification-model``, you can generate a classification model file.
Using its default settings, you just have to pass it the problem file you want to use:

.. code:: shell

    lincs generate classification-model problem.yml

For now, *lincs* can only generate MR-Sort models, so the ``--model-type`` option can only take its default value: ``mrsort``.
We expect this could change if we implement the generation of other types of models.

By default, the sum of MR-Sort weights of the criteria is pseudo-random and greater than or equal to 1.
With the ``--mrsort.fixed-weight-sum`` option, you can specify a fixed value for this sum.
This effectively impacts how hard it is for alternatives to get into upper categories.

Generating alternatives
-----------------------

With its default settings, ``lincs generate classified-alternatives`` requires only the problem and model files and the number of alternatives to generate:

.. code:: shell

    lincs generate classified-alternatives problem.yml model.yml 100

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


.. _user-learning-a-model:

Learning a model
================

As you've seen in our get started guide, the basic command to learn a classification model with *lincs* is ``lincs learn classification-model``.
With its default settings, you just have to pass it a problem file and a learning set file (of classified alternatives):

.. code:: shell

    lincs learn classification-model problem.yml learning-set.csv

Its ``--help`` option and our reference documentation give you a list of the numerous options it accepts.

An whole tree of options
------------------------

The first option is ``--model-type``.
It tells *lincs* what type of model you want it to learn, *e.g.* ``mrsort`` for MR-Sort or ``ucncs`` for :math:`U^c \textsf{-} NCS`.
Then, each model type has its own set of options that are valid only for this type of model,
and this pattern goes on to form a tree of options that make sense only on a specific branch.

To capture this reality in a somewhat simple but consistent way, *lincs* uses a dot-separated naming scheme for its options:
option ``--mrsort.strategy`` is a sub-option of ``--model-type mrsort``.
It can accept the value ``weights-profiles-breed``,
and ``--mrsort.weights-profiles-breed.target-accuracy`` is a sub-option of ``--mrsort.strategy weights-profiles-breed``.
The ``model-type`` and ``strategy`` parts are not repeated to reduce verbosity a bit, but this relies on our ability to avoid naming collisions.
Each sub-option name is formed by joining with dots (``.``) the values of the options it depends on.

This pattern is arguably quite verbose, but it's explicit and relatively easy to extend in a backward-compatible manner.

Note that you've already seen an example of this scheme above, at a smaller scale, in ``lincs generate classification-model``,
where ``--mrsort.fixed-weight-sum`` is a sub-option of ``--model-type mrsort``.

Strategies
----------

Some problems can be solved using different methods.
In software, these methods are often called `"strategies" <https://en.wikipedia.org/wiki/Strategy_pattern>`_.
``lincs learn classification-model`` accepts several options named like ``--...strategy`` to let you choose among different methods for a given part of the learning.

A few of them let you choose among only one strategy... but we expect it will change when we implement more.

Available learning (sub-)strategies
-----------------------------------

Examples in this section will reuse the ``problem.yml`` and ``learning-set.csv`` files you have generated in our :doc:`"Get started" guide <get-started>`;
please make sure you have them in your current directory.

Weights, profiles, breed (WPB)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``--mrsort.strategy weights-profiles-breed`` strategy is the default for MR-Sort models.
This methods uses a small population of models, repeating the following three steps:

- improve their MR-Sort weights
- improve their boundary profiles
- breed them to keep the best models and generate new ones

It finally outputs the best model it found.

General options
...............

The size of that population is controlled by the ``--mrsort.weights-profiles-breed.models-count`` option.
Finding the optimal size is a difficult problem.
*lincs* uses a parallel implementation of the WPB loop,
so we recommend you set it to the number of physical CPU cores available on you machine.
Or maybe a small multiple of that number.

The ``--mrsort.weights-profiles-breed.verbose`` option can be used to make *lincs* display information about the progress of the learning.

The ``--mrsort.weights-profiles-breed.output-metadata`` options can be used to produce a YAML file giving information about the learning process:
the reason it stopped (accuracy reached, time limit, *etc.*), how many WPB iterations it took, *etc.*

Termination
...........

The WPB loop terminates when one of the following conditions is met:

- the ``--mrsort.weights-profiles-breed.target-accuracy`` is reached
- the ``--mrsort.weights-profiles-breed.max-duration`` is exceeded: the total duration of the learning is greater than that duration
- the ``--mrsort.weights-profiles-breed.max-duration-without-progress`` is exceeded: the accuracy of the best model so far has not improved for that duration

In all those cases, *lincs* outputs the best model it found so far.

Then, each step is controlled by its own set of options.

"Weights" step
..............

Using ``--mrsort.weights-profiles-breed.weights-strategy linear-program`` (the default and only value for that option),
the "weights" step is actually an optimization, not just an improvement.
That strategy uses a linear program, and lets you choose among several solvers with the ``--mrsort.weights-profiles-breed.linear-program.solver`` option.

By default, it uses GLOP, which is a part of `Google's OR-Tools <https://developers.google.com/optimization/>`_.

Here is an example using the `Alglib <https://www.alglib.net/>`_ solver:

.. code:: shell

    lincs learn classification-model problem.yml learning-set.csv \
      --mrsort.weights-profiles-breed.linear-program.solver alglib

It should produce a very similar model, with slight numerical differences.

"Profiles" step
...............

The "profiles" step currently only has one strategy (``--mrsort.weights-profiles-breed.profiles-strategy accuracy-heuristic``),
which is controlled by two options.

The first one is a random seed for reproducibility (``--mrsort.weights-profiles-breed.accuracy-heuristic.random-seed``).
The remarks about randomness above also apply here.

The second option lets you use your CUDA-capable GPU for increased performance: ``--mrsort.weights-profiles-breed.accuracy-heuristic.processor``.
Note that *lincs* may be built without GPU support.
This is the case for example on macOS, where CUDA is not supported.
Binary wheels for Linux and Windows do support it though.
You can check with ``lincs info has-gpu``.

Here is an example:

.. code:: shell

    lincs learn classification-model problem.yml learning-set.csv \
      --mrsort.weights-profiles-breed.accuracy-heuristic.processor gpu

If you specify the random seed, it will produce the exact same model as when using the CPU;
this is an important feature of *lincs*, that the GPU code has the same behavior as the CPU code.

"Breed" step
............

The "breed" step currently has only one strategy, that simply re-initializes the least accurate models to random ones picked according to the only ``--mrsort.weights-profiles-breed.initialization-strategy`` currently available.
Not much to be said here, but we anticipe this could evolve.

The portion of the population that is reinitialized is controlled by the ``--mrsort.weights-profiles-breed.reinitialize-least-accurate.portion`` option.

SAT-based strategies
^^^^^^^^^^^^^^^^^^^^

You can also use entirely different approaches using SAT and max-SAT solvers.
The tradeoffs offered by these methods are highlighted in our :ref:`conceptual overview <overview-learning-methods>`.

These strategies let you learn :math:`U^c \textsf{-} NCS` models, so you have to start with ``--model-type ucncs``.
Here are two examples:

.. code:: shell

    lincs learn classification-model problem.yml learning-set.csv \
      --model-type ucncs --ucncs.strategy sat-by-coalitions

And:

.. code:: shell

    lincs learn classification-model problem.yml learning-set.csv \
      --model-type ucncs --ucncs.strategy max-sat-by-separation

They produce a different kind of model, with the sufficient coalitions specified explicitly by their roots:

.. code:: yaml

    # Reproduction command (with lincs version 1.1.0a13): lincs learn classification-model problem.yml learning-set.csv --model-type ucncs --ucncs.strategy sat-by-coalitions
    kind: ncs-classification-model
    format_version: 1
    accepted_values:
      - kind: thresholds
        thresholds: [1, 1]
      - kind: thresholds
        thresholds: [0.0556534864, 0.326433569]
      - kind: thresholds
        thresholds: [0.162616938, 0.67343241]
      - kind: thresholds
        thresholds: [0.996253729, 0.997255564]
    sufficient_coalitions:
      - &coalitions
        kind: roots
        upset_roots:
          - [1, 2]
      - *coalitions


Using a model
=============

Classifying alternatives
------------------------

When you finally have a model (learned, generated or hand-crafted),
you can use it to classify alternatives with ``lincs classify problem.yml model.yml alternatives.csv``.

The ``category`` column in the input alternatives file is ignored and may be empty.

Note that the input files will not be modified: the classified alternatives will be printed on the standard output
or written in the file specified by ``--output-alternatives``.

Computing a classification accuracy
-----------------------------------

Similarly, you can use ``lincs classification-accuracy problem.yml model.yml learning-set.csv`` to compute the accuracy of a model on a learning set.

In that case, the ``category`` column must be populated as it serves as a reference to compute the accuracy.

That command displays the number of alternatives that were correctly classified and the total number of alternatives in the learning set.

Getting human-readable information about a problem or model
-----------------------------------------------------------

You can use ``lincs describe classification-problem problem.yml`` to get a human-readable description of a problem,
and ``lincs describe classification-model problem.yml model.yml`` to get one for a model, including wether it's an MR-Sort or Uc-NCS model.

Visualizing a model and alternatives
------------------------------------

And you can use ``lincs visualize classification-model problem.yml model.yml`` to create a graphical representation of a model (a ``.png`` file),
and its ``--alternatives`` and ``--alternatives-count`` options to add alternatives to the graph.
You've seen an example in our "Get started" guide.


What's next?
============

You now know pretty much everything you need to use *lincs* from the command-line.
You can learn how to use *lincs* from Python, as described in our :doc:`Python API guide <python-api>`.
You may find some additional details in our :doc:`reference documentation <reference>`.
Feel free to reach out to us if you have any question or feedback, as said at the top of the :doc:`Readme <index>`.

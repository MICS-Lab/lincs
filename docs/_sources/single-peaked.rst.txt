.. Copyright 2023-2024 Vincent Jacques

======================
Single-peaked criteria
======================

Single-peaked criteria are criteria where intermediate values are preferred to extreme values.
This is the case *e.g.* when corelating a patient's blood pressure to their global health status:
low blood pressure is bad, high blood pressure is bad, and intermediate values are good.
This kind of criterion does not fit well with the NCS models defined in our :ref:`conceptual overview <overview-ncs>`:
these models have a single lower profile for each category, which assume that criteria have monotonous preference direction.

It is however possible to generalize this definition in a way that fits both cases cleanly.

This document describes this formalism and how it's put into action in *lincs*.
It is organized like our whole user documentation, but focusing on single-peaked criteria.
It assumes you've read out :doc:`"Get started" guide <get-started>`.

Depending on your preferred learning style, you can start with the "Conceptual overview" section below,
or jump directly to the more hands-on sections below it, that build on our :doc:`user guide <user-guide>`.

Conceptual overview
===================

This section builds on our :doc:`conceptual overview documentation <conceptual-overview>`.

We define "generalized NCS model" in this section, but later, we'll simply refer to them as "NCS models".
In any context involving single-peaked criteria, it's understood that "NCS model" means "generalized NCS model".

.. admonition:: Formal definition

  A generalized NCS model is a parametric function from :math:`X` to :math:`[0..p)` defined by the following parameters:

  - for each category but the first, *i.e.* for :math:`C^h` for :math:`h \in [1..p)`:

    - the set of performance values it accepts on each criterion :math:`\mathcal{B}^h_i \subseteq X_i` for :math:`i \in [0..n)`
    - its sufficient coalitions :math:`\mathcal{F}^h \subseteq \mathcal{P}([0..n))`

  With the following constraints:

  - the set of performance must be imbricated: :math:`\mathcal{B}^h_i \supseteq \mathcal{B}^{h + 1}_i` for each category :math:`h \in [1..p-1)` and each criterion :math:`i \in [0..n)`
  - each category's set of sufficient coalitions :math:`\mathcal{F}^h` must be up-closed by inclusion: if :math:`S \in \mathcal{F}^h` and :math:`S \subset T \in \mathcal{P}([0..n))`, then :math:`T \in \mathcal{F}^h`
  - sufficient coalitions must be imbricated: :math:`\mathcal{F}^1 \supseteq ... \supseteq \mathcal{F}^{p-1}`

  This generalized NCS model assigns an alternative :math:`x = (x_0, ..., x_{n-1}) \in X` to the best category :math:`C^h`
  such that the criteria on which :math:`x` has performances in that category's accepted values are sufficient,
  defaulting to the worst category (:math:`C^0`):

  .. math::

    f: x \mapsto \max (\{0\} \cup \{ h \in [1..p): \{ i \in [0..n): x_i \in \mathcal{B}^h_i \} \in \mathcal{F}^h \})

This definition is equivalent to the previous one when :math:`\mathcal{B}^h_i = \{x_i : x_i \in X_i \land b_i \preccurlyeq_i x_i\}`
where :math:`b_i` is the lower threshold.

It also specializes nicely for single-peaked criteria, where the accepted values are imbricated intervals:
:math:`\mathcal{B}^h_i = \{x_i : x_i \in X_i \land b_i \preccurlyeq_i x_i \preccurlyeq_i B_i\}`
where :math:`b_i` is the lower bound of the interval and :math:`B_i` is its higher bound.

In the problem file format
==========================

In the problem file format, single-peaked criteria are described with:

.. code:: yaml

  - name: Criterion 1
    value_type: real
    preference_direction: single-peaked
    min_value: 0.5
    max_value: 20.5

Note that the ``single-peaked`` ``preference_direction`` is only allowed for criteria with ``real`` or ``integer`` ``value_type``.
Enumerated criteria are monotonous by design, as their values are ordered.

In the model file format
========================

In *lincs*, the model file format is designed to allow any kind of description for the accepted values.
Currently, two kinds are supported: ``thresholds`` and ``intervals``.

``thresholds`` correspond to the less generic definition, where criteria are monotonous.
For such a criterion, the model files contains the list of the :math:`b^h_i` for :math:`h \in [1..p-1)`:

.. code:: yaml

    accepted_values:
      - kind: thresholds
        thresholds: [6.09463787, 19.7704506]

For single-peaked criteria, the sets of accepted values are imbricated intervals.
In the model file, they are described like this:

.. code:: yaml

    accepted_values:
      - kind: intervals
        intervals: [[20, 80], [40, 60]]

Using the command line
======================

The only differences when using the command-line with single-peaked criteria are:

- you need to supply the ``--allow-single-peaked-criteria`` option to ``lincs generate classification-problem``.
- ``lincs visualize  classification-problem`` fails with an informative message and a link to `this discussion <https://github.com/MICS-Lab/lincs/discussions/21>`_.

I encourage you to follow our :doc:`"Get started" guide <get-started>` again, with ``--allow-single-peaked-criteria``.
Note that you may need to generate several problems before getting one with an actual single-peaked criterion, due to the random generation.

Using the Python API
====================

This section builds on our :doc:`Python API guide <python-api>`.

`lc.generate_problem` accepts a ``allowed_preference_directions`` parameter, a list of ``lc.Criterion.PreferenceDirection``s.
To generate a problem with a single-peaked criterion, you must add ``lc.Criterion.PreferenceDirection.single_peaked`` to this list.

When creating a problem manually, you can specify a criterion as taking ``lc.Criterion.IntegerValues(lc.Criterion.PreferenceDirection.single_peaked, 0, 100)``.

When creating a model manually, the accepted values for such a criterion must look like ``lc.AcceptedValues.IntegerIntervals([[20, 80], [40, 60]])``.

When creating your own strategies, if you want them to support single-peaked criteria,
you need to call the base strategy class' constructor with ``supports_single_peaked_criteria=True``.
If you don't, the learning will throw an exception before it begins.

You can then use ``ModelsBeingLearned.high_profile_ranks``,
but only for criteria that have a ``True`` value for ``PreprocessedLearningSet.single_peaked``.

Note that this attribute is indexed with an additional indirection to avoid allocating unused data:
it's only allocated for actual single-peaked criteria and must be accessed as:

.. code:: python

    assert preprocessed_learning_set.single_peaked[criterion_index]
    high_profile_rank_index = models_being_learned.high_profile_rank_indexes[criterion_index]
    models_being_learned.high_profile_ranks[model_index][boundary_index][high_profile_rank_index]

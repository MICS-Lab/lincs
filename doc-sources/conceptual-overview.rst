.. Copyright 2023 Vincent Jacques

===================
Conceptual overview
===================


About classification
====================

For now, *lincs* focuses on "classification" problems, *i.e.* the task of sorting "alternatives" into "categories".
Categories are ordered: there is a worst category and a best category, and possibly some intermediates.
Alternatives are assigned to a category based on their "performances" on a set of "criteria".
The description of the criteria and categories constitutes the "problem" itself.

This vocabulary is voluntarily abstract to allow for a wide range of applications, so a concrete example might help.
Let's say you want to assign scholarships to students based on their academic performances.
Your funding policy might be that students with the best grades should get the best scholarships.
And you might want to favor younger students, and/or students coming from more modest backgrounds.
In this example, the categories are the different scholarships, the alternatives are the students, and the criteria are grades on each topic, age and family income.
For a given student, their performances are their actual grades on each topic, their age and their family income.

The same vocabulary could apply to triaging patients in an hospital based on vital signs.

Formally, a problem is defined by:

- its number of categories :math:`p \in \mathbb{N}`
- its set of categories :math:`C = \{C^h\}_{h \in \{1, ..., p\}}`, ordered by :math:`C^1 \prec ... \prec C^p`
- its number of criteria :math:`n \in \mathbb{N}`
- its set of criteria :math:`\{X_i\}_{i \in \{1, ..., n\}}`. Each criterion is a set of values :math:`X_i` with a total pre-order :math:`\preccurlyeq_i`, for :math:`i \in \{1, ..., n\}`

In that setting, alternatives are the Cartesian product of the criteria: :math:`X = \prod_{i \in \{1, ..., n\}} X_i`.
For a given alternative :math:`x = (x_1, ..., x_n) \in X`, its performance on criterion :math:`i` is :math:`x_i \in X_i`.


Learning and classifying
========================

*lincs* provides algorithms to automate the classification of alternatives into categories.
Its approach is to first "learn" a "model" from a set of already classified alternatives, and then use that model to classify new alternatives.
The set of pre-classified alternatives is called the "training set"; it constitutes the ground truth for the learning phase.
There are then many kinds of models, and many ways to learn them.


Synthetic data
==============

It's not always practical to use real-world data when developing a new learning algorithm, so one can use synthetic data instead.
In that approach, one specifies the problem and provides a pre-known model.
They then generate pseudo-random alternatives classified according to that original model,
and use them as a training set to learn a new model.
Finally, they compare how close the learned model behaves to the original one to evaluate the quality of the algorithm.

*lincs* provides ways to generate synthetic pseudo-random problems, models and training sets.


Non-compensatory sorting (NCS)
==============================

In general, we expect alternatives with higher performances to be assigned to better categories.
But sometimes, there are some criteria that are so important that they can't be compensated by other criteria.
Non-compensatory sorting models are a way to capture that.

An NCS model defines lower performance profiles for each category.
It then assigns an alternative in a good category if it has performances above that category's lower profiles on a sufficient subset of the criteria.

Formally, an NCS model is defined by:

- for each category :math:`C^h` but the last, *i.e.* for :math:`h \in \{1, ..., p - 1\}`:

  - its upper performance profile :math:`b^h = (b^h_1, ..., b^h_n) \in X`
  - its sufficient criteria :math:`\mathcal{F}^h`, which are sets of parts of the set of criteria: :math:`\mathcal{F}^h \subseteq \mathcal{P}(\{1, ..., n\})`

The profiles must be ordered to match the order on the set of categories:
:math:`b^h_i \preccurlyeq_i b^{h + 1}_i` for each category :math:`h \in \{1, ..., p - 2\}` and each criterion :math:`i \in \{1, ..., n\}`.

Each category's sufficient criteria :math:`\mathcal{F}^h` must be up-closed by inclusion:
if :math:`S \in \mathcal{F}^h` and :math:`S \subset T \in \mathcal{P}(\{1, ..., n\})`, then :math:`T \in \mathcal{F}^h`.
This matches the intuition that they are *sufficient* criteria: if a few criteria are sufficient, then more criteria are still sufficient.

They also must be nested: :math:`\mathcal{F}^1 \subseteq ... \subseteq \mathcal{F}^{p - 1}`.
@todo Question from Vincent J to Wassila or Vincent M: why?
This means that if some criteria are sufficient for a category, then they are sufficient for the next category.
What would the consequence be of having a set of criteria that's sufficient to go to category 2 but not sufficient to go to category 3?

This NCS model assigns an alternative :math:`x = (x_1, ..., x_n) \in X` to the category :math:`C^h` if and only if:

- :math:`\{ i \in \{1, ..., n\}: x_i \succcurlyeq_i b^{h-1}_i \} \subseteq \mathcal{F}^h`, *i.e.* the criteria on which :math:`x` has performances above the upper profile of the previous category are sufficient for the current category
- and :math:`\{ i \in \{1, ..., n\}: x_i \succcurlyeq_i b^h_i \} \not\subseteq \mathcal{F}^{h + 1}`, *i.e.* the criteria on which :math:`x` has performances above the upper profile of the current category are not sufficient for the next category

@todo Add an example (with visualization) showing how an NCS models assigns alternatives to categories

Particular cases
----------------

In general, the sufficient criteria :math:`\mathcal{F}^h` can be different for each category and defined arbitrarily.

Some particular cases are quite common, namely:

- :math:`U^c \textsf{-} NCS`: where sufficient criteria are the same for all categories (:math:`\mathcal{F}^1 = ... = \mathcal{F}^{p - 1}`)
- :math:`1 \textsf{-} U^c \textsf{-} NCS` *a.k.a.* MR-Sort: where sufficient criteria are defined using weights on criteria and a threshold

@todo Add formal definitions particular cases; emphasize what they simplify *vs.* more general models

@todo Add example of NCS model that is not U^C-NCS

@todo Add example of NCS model that is not MR-Sort


Files
=====

Before starting, *lincs* needs to know basic things about the structure of the alternatives you care about and the categories they can belong to, *i.e.* your problem.
It is described in the problem file in YAML format.
It's specified in :ref:`the problem file format <ref-file-problem>`.

The training set is expected in CSV format.
It's specified in :ref:`the alternatives file format <ref-file-alternatives>`.

Finally, NCS models are described in YAML format, as specified in :ref:`the model file format <ref-file-model>`.

The same formats are used for synthetic and real-world data.


Next
====

If you haven't done so yet, we recommend you now follow our :doc:`"Get started" guide <get-started>`.

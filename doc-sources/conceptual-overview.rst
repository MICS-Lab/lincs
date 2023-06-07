.. Copyright 2023 Vincent Jacques

===================
Conceptual overview
===================


@todo Double-check strictness of inequalities (:math:`\lt` *vs.* :math:`\le`) and inclusions (:math:`\subset` *vs.* :math:`\subseteq`) in all formal definitions in this document

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

.. admonition:: Formal definition

  A problem is defined by:

  - its number of criteria :math:`n \in \mathbb{N}`
  - its set of criteria :math:`\{X_i\}_{i \in \{1, ..., n\}}`. Each criterion is a set of values :math:`X_i` with a total pre-order :math:`\preccurlyeq_i`, for :math:`i \in \{1, ..., n\}`
  - its number of categories :math:`p \in \mathbb{N}`
  - its set of categories :math:`C = \{C^h\}_{h \in \{1, ..., p\}}`, ordered by :math:`C^1 \prec ... \prec C^p`

  In that setting, alternatives are the Cartesian product of the criteria: :math:`X = \prod_{i \in \{1, ..., n\}} X_i`.
  For a given alternative :math:`x = (x_1, ..., x_n) \in X`, its performance on criterion :math:`i` is :math:`x_i \in X_i`.


Learning and classifying
========================

*lincs* provides algorithms to automate the classification of alternatives into categories.
Its approach is to first "learn" a "model" from a set of already classified alternatives, and then use that model to classify new alternatives.
The set of pre-classified alternatives is called the "training set"; it constitutes the ground truth for the learning phase.

Formally, models are functions from alternatives to categories: :math:`f: X \rightarrow C`.

Most models are parametric functions of a given form, and the learning phase consists in finding the parameters that best fit the training set.


Non-compensatory sorting (NCS)
==============================

In general, we expect alternatives with higher performances to be assigned to better categories.
But sometimes, there are some criteria that are so important that they can't be compensated by other criteria.
Non-compensatory sorting models are a way to capture that idea.

There are many "families" of models, *i.e.* sets of models that share the same general parametric form with varying parameters.
NCS models are one such family.

An NCS model defines a "lower performance profile" for each category.
It then assigns an alternative to a good category if it has performances above that category's lower profiles on a sufficient set of the criteria.
Sets of criteria are called "coalitions".
NCS models allow for several ways to reach the minimum performance level to be assigned to a category,
so sufficient criteria for a category are not a *single* coalition, but actually a *set* of coalitions.
Additionally, this set of coalitions can be different for each category.

.. admonition:: Formal definition

  For a given set :math:`S` , we note :math:`\mathcal{P}(S)` the set of all subsets of :math:`S` (*a.k.a.* its power set).

  An NCS model :math:`f: X \rightarrow \{1, ..., p\}` is a parametric function from :math:`X` to :math:`\{1, ..., p\}` defined by the following parameters:

  - for each category but the first, *i.e.* for :math:`C^h` for :math:`h \in \{2, ..., p\}`:

    - its lower performance profile :math:`b^h = (b^h_1, ..., b^h_n) \in X`
    - its sufficient coalitions :math:`\mathcal{F}^h \subseteq \mathcal{P}(\{1, ..., n\})`

  With the following constraints:

  - the profiles must be ordered to match the order on the set of categories: :math:`b^h_i \preccurlyeq_i b^{h + 1}_i` for each category :math:`h \in \{1, ..., p - 1\}` and each criterion :math:`i \in \{1, ..., n\}`
  - each category's set of sufficient coalitions :math:`\mathcal{F}^h` must be up-closed by inclusion: if :math:`S \in \mathcal{F}^h` and :math:`S \subset T \in \mathcal{P}(\{1, ..., n\})`, then :math:`T \in \mathcal{F}^h`. This matches the intuition that they are *sufficient* criteria: if a few criteria are sufficient, then more criteria are still sufficient
  - @todo Add constraint about imbrication of :math:`\mathcal{F}^h` when Vincent M answers my question by e-mail

  This NCS model assigns an alternative :math:`x = (x_1, ..., x_n) \in X` to the best category :math:`C^h`
  such that the criteria on which :math:`x` has performances above that category's lower profile are sufficient,
  defaulting to the worst category (:math:`C^1`):

  .. math::

    f: x \mapsto \max (\{1\} \cup \{ h \in \{2, ..., p\}: \{ i \in \{1, ..., n\}: x_i \succcurlyeq_i b^h_i \} \in \mathcal{F}^h \})

This definition may differ slightly from the one you're used to, but it should be formally equivalent.
We use it in *lincs* because it is somewhat simple and matches the implementation quite well.
We detail its equivalence to other common definitions in the following appendix:
@todo Write appendix about equivalence of definitions (:math:`h` is shifted by 1, assignment to category is a max instead of two conditions)

Example
-------

Let's continue on the scholarship example.
Let's say there a three levels: "no scholarship" (:math:`C^1`), "partial scholarship" (:math:`C^2`) and "full scholarship" (:math:`C^3`).
To further simplify things without sacrificing the interest of the example, we can consider four criteria:
grades in math (:math:`\mathcal{M}`), physics (:math:`\mathcal{P}`), literature (:math:`\mathcal{L}`) and history (:math:`\mathcal{H}`), all between 0 and 10,
and forget about age and family income for now.

For clarity, we'll use :math:`\mathcal{M}`, :math:`\mathcal{P}`, :math:`\mathcal{L}` and :math:`\mathcal{H}` as lower indexes instead of :math:`i` for criteria.

Let's consider the following NCS model:

- :math:`b^2 = (6, 7, 6, 5)`
- :math:`\mathcal{F}^2 = \{ \{\mathcal{M}, \mathcal{P}, \mathcal{L}\}, \{\mathcal{M}, \mathcal{P}, \mathcal{H}\}, \{\mathcal{M}, \mathcal{L}, \mathcal{H}\}, \{\mathcal{P}, \mathcal{L}, \mathcal{H}\}, ... \}`
- :math:`b^3 = (8, 8, 8, 8)`
- :math:`\mathcal{F}^2 = \{ \{\mathcal{M}, \mathcal{L}\}, \{\mathcal{M}, \mathcal{H}\}, \{\mathcal{P}, \mathcal{L}\}, \{\mathcal{P}, \mathcal{H}\}, ... \}`

where the ellipsis denotes all coalitions that contains one of the previous ones.

The profiles for this model look like this:

@todo Continue this example when Vincent M answers my questions sent by e-mail

Particular cases
----------------

Some particular cases are quite common.
They are NCS models with additional constraints, so they are slightly less general, but sufficient in many cases and computationally simpler to learn.

Here are a few that are used in *lincs*:

:math:`U^c \textsf{-} NCS`
~~~~~~~~~~~~~~~~~~~~~~~~~~

A :math:`U^c \textsf{-} NCS` model is an NCS model where all :math:`\mathcal{F}^h` are the same (:math:`\mathcal{F}`).
This simplification captures the idea that in many cases, the same criteria are sufficient for all categories, and that categories are mostly defined by their lower performance profile.

.. admonition:: Formal definition

    A :math:`U^c \textsf{-} NCS` model is an NCS model with the following additional constraint:

    - there is a single :math:`\mathcal{F} \subseteq \mathcal{P}(\{1, ..., n\})` such that :math:`\mathcal{F}^h = \mathcal{F}` for each category :math:`h \in \{2, ..., p\}`

:math:`1 \textsf{-} U^c \textsf{-} NCS` *a.k.a.* MR-Sort
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An MR-Sort model is a :math:`U^c \textsf{-} NCS` model with the additional simplification that :math:`\mathcal{F}` is defined using weights on criteria and a threshold.
A coalition is sufficient if the sum of the weights of its criteria is above 1.

.. admonition:: Formal definition

  An MR-Sort model is a :math:`U^c \textsf{-} NCS` model with the following additional parameters:

  - for each criterion :math:`i \in \{1, ..., n\}`:

    - its weight :math:`w_i \in [0, 1]`

  and the following additional constraint:

  - :math:`\mathcal{F} = \{ S \in \mathcal{P}(\{1, ..., n\}): \sum_{i \in S} w_i \geq 1 \}`

Again, this definition differs slightly from others in the literature.
We detail their equivalence in this appendix:
@todo Write appendix about equivalence of definitions (weights are de-normalized, :math:`\lambda` is 1)

@todo Add example of NCS model that is not MR-Sort


Synthetic data
==============

It's not always practical to use real-world data when developing a new learning algorithm, so one can use synthetic data instead.
In that approach, one specifies the problem and provides a pre-known model.
They then generate pseudo-random alternatives classified according to that original model,
and use them as a training set to learn a new model.
Finally, they compare how close the learned model behaves to the original one to evaluate the quality of the algorithm.

*lincs* provides ways to generate synthetic pseudo-random problems, models and training sets.


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

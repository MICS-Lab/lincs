.. Copyright 2023 Vincent Jacques

===================
Conceptual overview
===================


About classification
====================

For now, *lincs* focuses on "classification" problems, *i.e.* the task of sorting "alternatives" into "categories".
Categories are ordered: there is a worst category and a best category, and possibly some intermediates.
Alternatives are assigned to a category based on their "performances" on a set of "criteria".
The description of the criteria and categories constitutes the "domain" of the problem.

This vocabulary is voluntarily abstract to allow for a wide range of applications, so a concrete example might help.
Let's say you want to assign scholarships to students based on their academic performances.
Your funding policy might be that students with the best grades should get the best scholarships.
And you might want to favor younger students, and/or students coming from more modest backgrounds.
In this example, the categories are the different scholarships, the alternatives are the students, and the criteria are grades on each topic, age and family income.
For a given student, their performances are their actual grades on each topic, their age and their family income.

The same vocabulary could apply to triaging patients in an hospital based on vital signs.


Learning and classifying
========================

*lincs* provides algorithms to automate the classification of alternatives into categories.
Its approach is to first "learn" a "model" from a set of already classified alternatives, and then use that model to classify new alternatives.
The set of pre-classified alternatives is called the "training set"; it constitutes the ground truth for the learning phase.
There are then many kinds of models, and many ways to learn them.


Synthetic data
==============

It's not always practical to use real-world data when developing a new learning algorithm, so one can use synthetic data instead.
In that approach, one specifies the domain of the problem and provides a pre-known model.
They then generate pseudo-random alternatives classified according to that original model,
and use them as a training set to learn a new model.
Finally, they compare how close the learned model behaves to the original one to evaluate the quality of the algorithm.

*lincs* provides ways to generate synthetic pseudo-random domains, models and training sets.


Non-compensatory sorting (NCS)
==============================

In general, we expect alternatives with higher performances to be assigned to better categories.
But sometimes, there are some criteria that are so important that they can't be compensated by other criteria.
Non-compensatory sorting models are a way to capture that.

An NCS model defines lower performance profiles for each category.
It then assigns an alternative in a good category if it has performances above that category's lower profiles on a sufficient subset of the criteria.

@todo Add general formal definition of NCS models (including the fact that sufficient criteria ar upsets of the parts of the set of criteria)
@todo Add an example (with visualization) showing how an NCS models assigns alternatives to categories

Particular cases
----------------

In general, the subset of sufficient criteria is different for each category and defined arbitrarily.

Some particular cases are quite common, namely:

- 1C-NCS (@todo check name): where the same subset of criteria is sufficient for all categories
- MR-Sort: where the subset of sufficient criteria is defined using weights on criteria and a threshold

@todo Add formal definitions particular cases; emphasize what they simplify *vs.* more general models

@todo Add example of NCS model that is not 1C-NCS

@todo Add example of NCS model that is not MR-Sort


Files
=====

Before starting, *lincs* needs to know basic things about the structure of the alternatives you care about and the categories they can belong to, *i.e.* your domain.
It is described in the domain file in YAML format.
It's specified in :ref:`the domain file format <ref-file-domain>`.

The training set is expected in CSV format.
It's specified in :ref:`the alternatives file format <ref-file-alternatives>`.

Finally, NCS models are described in YAML format, as specified in :ref:`the model file format <ref-file-model>`.

The same formats are used for synthetic and real-world data.


Next
====

If you haven't done so yet, we recommend you now follow our :doc:`"Get started" guide <get-started>`.

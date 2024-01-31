# Copyright 2023-2024 Vincent Jacques

from typing import Iterable
import unittest

import matplotlib.pyplot as plt

from .classification import Problem, Model, Alternative


def visualize_classification_model(problem: Problem, model: Model, alternatives: Iterable[Alternative], axes: plt.Axes):
    """
    Create a visual representation of a classification model and classified alternatives, using Matplotlib.
    """

    # @todo(Feature, v1.2) Visualize single-peaked criteria.
    # See the possible solutions in e-mail "Visualisation des critères single-peaked" 2023-11-24

    criteria_count = len(problem.criteria)
    assert criteria_count >= 1
    categories_count = len(problem.ordered_categories)
    assert categories_count >= 2
    boundaries_count = categories_count - 1

    vertical_margin = 0.05
    horizontal_margin = 0.1

    if criteria_count == 1:
        xs = [0.5]
    else:
        xs = [horizontal_margin + criterion_index * (1 - 2 * horizontal_margin) / (criteria_count - 1) for criterion_index in range(criteria_count)]

    if criteria_count <= 2:
        criterion_index_for_alternatives = 0
    else:
        criterion_index_for_alternatives = 1

    axes.set_xlim(0, 1)
    axes.set_xticks(xs, [criterion.name for criterion in problem.criteria])

    axes.set_ylim(-vertical_margin, 1 + vertical_margin)
    axes.set_yticks([0, 1], ["worst", "best"])

    boundary_profiles = [[] for _ in problem.ordered_categories[1:]]
    for criterion, accepted_values in zip(problem.criteria, model.accepted_values):
        assert accepted_values.is_thresholds
        if criterion.is_real:
            for boundary_index in range(boundaries_count):
                boundary_profiles[boundary_index].append(accepted_values.real_thresholds.thresholds[boundary_index])
        elif criterion.is_integer:
            for boundary_index in range(boundaries_count):
                boundary_profiles[boundary_index].append(accepted_values.integer_thresholds.thresholds[boundary_index])
        else:
            assert criterion.is_enumerated
            for boundary_index in range(boundaries_count):
                boundary_profiles[boundary_index].append(criterion.enumerated_values.get_value_rank(accepted_values.enumerated_thresholds.thresholds[boundary_index]))

    def extend(ys):
        return [ys[0]] + ys + [ys[-1]]
    ys = [
        extend(normalize_profile(problem.criteria, boundary_profile))
        for boundary_profile in boundary_profiles
    ]
    ys.append(extend([1] * len(xs)))
    unstacked_ys = [ys[0]]
    for ys1, ys2 in zip(ys[1:], ys[:-1]):
        unstacked_ys.append([y1 - y2 for y1, y2 in zip(ys1, ys2)])
    collections = axes.stackplot([0] + xs + [1], unstacked_ys, alpha=0.4)
    colors = [collection.get_facecolor()[0] for collection in collections]

    for (x, criterion) in zip(xs, problem.criteria):
        secondary_ax = axes.secondary_yaxis(x)
        if criterion.is_real:
            values = criterion.real_values
            ticks = [0, 0.5, 1]
            labels = [f"{values.min_value:.1f}", f"{(values.min_value + values.max_value) / 2:.1f}", f"{values.max_value:.1f}"]
            if values.is_increasing:
                secondary_ax.set_yticks(ticks, labels)
            else:
                assert values.is_decreasing
                secondary_ax.set_yticks(ticks, reversed(labels))
        elif criterion.is_integer:
            values = criterion.integer_values
            labels = list(make_integer_labels(values.min_value, values.max_value))
            ticks = [(label - values.min_value) / (values.max_value - values.min_value) for label in labels]
            if values.is_increasing:
                secondary_ax.set_yticks(ticks, labels)
            else:
                assert values.is_decreasing
                secondary_ax.set_yticks(ticks, reversed(labels))
        else:
            assert criterion.is_enumerated
            values = criterion.enumerated_values
            ticks_count = len(values.ordered_values)
            ticks = [n / (ticks_count - 1) for n in range(ticks_count)]
            secondary_ax.set_yticks(ticks, values.ordered_values)

    for (category_index, category) in enumerate(problem.ordered_categories):
        if category_index == 0:
            low_y = 0
        else:
            low_y = normalize_value(problem.criteria[0], boundary_profiles[category_index - 1][0])
        if category_index == len(problem.ordered_categories) - 1:
            high_y = 1
        else:
            high_y = normalize_value(problem.criteria[0], boundary_profiles[category_index][0])
        y = (low_y + high_y) / 2
        color = colors[category_index]
        axes.text(
            0, y,
            category.name,
            color=color,
            alpha=1,
            fontweight="bold", verticalalignment="center"
        )

    for alternative in alternatives:
        if alternative.category_index is None:
            color = "black"
        else:
            color = colors[alternative.category_index]
        axes.plot(
            xs, normalize_profile(problem.criteria, make_numeric_profile(problem.criteria, alternative.profile)),
            "o--",
            label=alternative.name,
            color=color,
            alpha=1,
        )
        axes.text(
            xs[criterion_index_for_alternatives],
            normalize_value(
                problem.criteria[criterion_index_for_alternatives],
                make_numeric_value(
                    problem.criteria[criterion_index_for_alternatives],
                    alternative.profile[criterion_index_for_alternatives],
                ),
            ),
            alternative.name,
            color=color, alpha=1)


def make_integer_labels(min_value, max_value):
    intervals_count = max_value - min_value
    assert intervals_count >= 1
    if intervals_count == 1:
        yield min_value
        yield max_value
    elif intervals_count == 2:
        yield min_value
        yield min_value + 1
        yield max_value
    elif intervals_count == 3:
        yield min_value
        yield min_value + 1
        yield min_value + 2
        yield max_value
    elif intervals_count % 2 == 0:
        yield min_value
        yield min_value + intervals_count // 4
        yield (min_value + max_value) // 2
        yield max_value - intervals_count // 4
        yield max_value
    else:
        yield min_value
        yield min_value + intervals_count // 3
        yield max_value - intervals_count // 3
        yield max_value


class MakeIntegerLabelsTests(unittest.TestCase):
    def _test(self, min_value, max_value, expected):
        self.assertEqual(list(make_integer_labels(min_value, max_value)), expected)

    def test_all(self):
        for n in range(100):
            l = list(make_integer_labels(10, n + 11))
            self.assertLessEqual(len(l), 5)
            self.assertEqual(l[0], 10)
            self.assertEqual(l[-1], n + 11)
            if len(l) % 2 == 1:
                self.assertEqual(l[len(l) // 2], (10 + n + 11) // 2)
            self.assertEqual(l, sorted(l))

    def test_0_1(self):
        self._test(0, 1, [0, 1])

    def test_0_2(self):
        self._test(0, 2, [0, 1, 2])

    def test_0_3(self):
        self._test(0, 3, [0, 1, 2, 3])

    def test_0_4(self):
        self._test(0, 4, [0, 1, 2, 3, 4])

    def test_10_14(self):
        self._test(10, 14, [10, 11, 12, 13, 14])

    def test_0_5(self):
        self._test(0, 5, [0, 1, 4, 5])

    def test_0_6(self):
        self._test(0, 6, [0, 1, 3, 5, 6])

    def test_0_7(self):
        self._test(0, 7, [0, 2, 5, 7])

    def test_0_70(self):
        self._test(0, 70, [0, 17, 35, 53, 70])

    def test_0_8(self):
        self._test(0, 8, [0, 2, 4, 6, 8])

    def test_0_8(self):
        self._test(0, 80, [0, 20, 40, 60, 80])

    def test_bug_1(self):
        self._test(-4949, 9942, [-4949, 14, 4979, 9942])

    def test_0_9000(self):
        self._test(0, 9000, [0, 2250, 4500, 6750, 9000])

    def test_0_9001(self):
        self._test(0, 9001, [0, 3000, 6001, 9001])

    def test_0_9002(self):
        self._test(0, 9002, [0, 2250, 4501, 6752, 9002])

    def test_0_9003(self):
        self._test(0, 9003, [0, 3001, 6002, 9003])

    def test_0_9004(self):
        self._test(0, 9004, [0, 2251, 4502, 6753, 9004])


def make_numeric_profile(criteria, profile):
    return [
        make_numeric_value(criterion, y)
        for (criterion, y) in zip(criteria, profile)
    ]


def make_numeric_value(criterion, y):
    if criterion.is_real:
        return y.real.value
    elif criterion.is_integer:
        return y.integer.value
    else:
        assert criterion.is_enumerated
        return criterion.enumerated_values.get_value_rank(y.enumerated.value)


def normalize_profile(criteria, ys):
    return [
        normalize_value(criterion, y)
        for (criterion, y) in zip(criteria, ys)
    ]


def normalize_value(criterion, y):
    is_increasing = None
    if criterion.is_real:
        values = criterion.real_values
        is_increasing = values.is_increasing
        y = (y - values.min_value) / (values.max_value - values.min_value)
    elif criterion.is_integer:
        values = criterion.integer_values
        is_increasing = values.is_increasing
        y = (y - values.min_value) / (values.max_value - values.min_value)
    else:
        assert criterion.is_enumerated
        values = criterion.enumerated_values
        is_increasing = True
        y = y / (len(values.ordered_values) - 1)

    if is_increasing:
        return y
    else:
        return 1 - y

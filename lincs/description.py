from typing import Iterable
import unittest

from .classification import Problem, Criterion, Category, Model, AcceptedValues, SufficientCoalitions


def describe_classification_problem(problem: Problem) -> Iterable[str]:
    """
    Generate a human-readable description of a classification problem.
    """

    categories_count = len(problem.ordered_categories)
    assert categories_count >= 2
    criteria_count = len(problem.criteria)
    assert criteria_count >= 1

    category_names = [f'"{category.name}"' for category in problem.ordered_categories]
    category_names_joined = ", ".join(category_names[:-1]) + " and " + category_names[-1]
    yield f"This a classification problem into {categories_count} ordered categories named {category_names_joined}."
    yield f"The best category is {category_names[-1]} and the worst category is {category_names[0]}."

    if criteria_count == 1:
        yield "There is 1 classification criterion."
    else:
        yield f"There are {criteria_count} classification criteria (in no particular order)."
    for criterion in problem.criteria:
        match criterion.values:
            case Criterion.RealValues() as values:
                yield f'Criterion "{criterion.name}" takes real values between {values.min_value:.1f} and {values.max_value:.1f} included.'
                yield from describe_preference_direction(criterion.name, values)
            case Criterion.IntegerValues() as values:
                yield f'Criterion "{criterion.name}" takes integer values between {values.min_value} and {values.max_value} included.'
                yield from describe_preference_direction(criterion.name, values)
            case Criterion.EnumeratedValues(ordered_values):
                yield f'Criterion "{criterion.name}" takes values in the following set: {", ".join(f"{value}" for value in ordered_values)}.'
                yield f'The best value for criterion "{criterion.name}" is "{ordered_values[-1]}" and the worst value is "{ordered_values[0]}".'


def describe_preference_direction(criterion_name, values):
    if values.is_increasing:
        yield f'Higher values of "{criterion_name}" are known to be better.'
    elif values.is_decreasing:
        yield f'Lower values of "{criterion_name}" are known to be better.'
    else:
        assert values.is_single_peaked
        yield f'Intermediate values of "{criterion_name}" are known to be better.'


class DescribeClassificationProblemTestCase(unittest.TestCase):
    maxDiff = None

    def _test(self, problem, expected):
        self.assertEqual(list(describe_classification_problem(problem)), expected)

    def test_simplest(self):
        self._test(
            Problem(
                [
                    Criterion("Criterion", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 1)),
                ],
                [Category("Bad"), Category("Good")],
            ),
            [
                'This a classification problem into 2 ordered categories named "Bad" and "Good".',
                'The best category is "Good" and the worst category is "Bad".',
                'There is 1 classification criterion.',
                'Criterion "Criterion" takes real values between 0.0 and 1.0 included.',
                'Higher values of "Criterion" are known to be better.'
            ]
        )

    def test_many_categories(self):
        self._test(
            Problem(
                [
                    Criterion("Criterion", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 1)),
                ],
                [Category("Worsestest"), Category("Interm 1"), Category("Interm 2"), Category("Interm 3"), Category("Bestestest")],
            ),
            [
                'This a classification problem into 5 ordered categories named "Worsestest", "Interm 1", "Interm 2", "Interm 3" and "Bestestest".',
                'The best category is "Bestestest" and the worst category is "Worsestest".',
                'There is 1 classification criterion.',
                'Criterion "Criterion" takes real values between 0.0 and 1.0 included.',
                'Higher values of "Criterion" are known to be better.',
            ]
        )

    def test_criteria_variety(self):
        self._test(
            Problem(
                [
                    Criterion("Increasing real criterion", Criterion.RealValues(Criterion.PreferenceDirection.increasing, -5.2, 10.3)),
                    Criterion("Decreasing real criterion", Criterion.RealValues(Criterion.PreferenceDirection.decreasing, 5, 15)),
                    Criterion("Increasing integer criterion", Criterion.IntegerValues(Criterion.PreferenceDirection.increasing, 0, 10)),
                    Criterion("Decreasing integer criterion", Criterion.IntegerValues(Criterion.PreferenceDirection.decreasing, 4, 16)),
                    Criterion("Enumerated criterion", Criterion.EnumeratedValues(["A", "B", "C"])),
                    Criterion("Single-peaked real criterion", Criterion.RealValues(Criterion.PreferenceDirection.single_peaked, 5, 15)),
                    Criterion("Single-peaked integer criterion", Criterion.IntegerValues(Criterion.PreferenceDirection.single_peaked, 0, 10)),
                ],
                [Category("Bad"), Category("Good")],
            ),
            [
                'This a classification problem into 2 ordered categories named "Bad" and "Good".',
                'The best category is "Good" and the worst category is "Bad".',
                'There are 7 classification criteria (in no particular order).',
                'Criterion "Increasing real criterion" takes real values between -5.2 and 10.3 included.',
                'Higher values of "Increasing real criterion" are known to be better.',
                'Criterion "Decreasing real criterion" takes real values between 5.0 and 15.0 included.',
                'Lower values of "Decreasing real criterion" are known to be better.',
                'Criterion "Increasing integer criterion" takes integer values between 0 and 10 included.',
                'Higher values of "Increasing integer criterion" are known to be better.',
                'Criterion "Decreasing integer criterion" takes integer values between 4 and 16 included.',
                'Lower values of "Decreasing integer criterion" are known to be better.',
                'Criterion "Enumerated criterion" takes values in the following set: A, B, C.',
                'The best value for criterion "Enumerated criterion" is "C" and the worst value is "A".',
                'Criterion "Single-peaked real criterion" takes real values between 5.0 and 15.0 included.',
                'Intermediate values of "Single-peaked real criterion" are known to be better.',
                'Criterion "Single-peaked integer criterion" takes integer values between 0 and 10 included.',
                'Intermediate values of "Single-peaked integer criterion" are known to be better.',
            ]
        )


def describe_classification_model(problem: Problem, model: Model) -> Iterable[str]:
    """
    Generate a human-readable description of a classification model.
    """

    model.check_consistency_with(problem)

    criteria_count = len(problem.criteria)
    assert len(model.accepted_values) == criteria_count
    assert criteria_count > 0
    categories_count = len(problem.ordered_categories)
    boundaries_count = categories_count - 1
    assert boundaries_count > 0
    assert len(model.sufficient_coalitions) == boundaries_count

    def comma_and(s):
        s = list(s)
        if len(s) == 1:
            return s[0]
        elif len(s) == 2:
            return " and ".join(s)
        else:
            return ", ".join(s[:-1]) + ", and " + s[-1]  # https://en.wikipedia.org/wiki/Serial_comma

    def make_upset_roots(upset_roots):
        for coalition in upset_roots:
            criterion_names = []
            for criterion_index in coalition:
                criterion = problem.criteria[criterion_index]
                criterion_names.append(f'"{criterion.name}"')
            yield f'  - {comma_and(criterion_names)}'

    def make_profile(accepted_values, boundary_index):
        for criterion_index, criterion in enumerate(problem.criteria):
            match accepted_values[criterion_index].descriptor:
                case AcceptedValues.RealThresholds(thresholds):
                    assert len(thresholds) == boundaries_count
                    constraint = "at least" if criterion.values.is_increasing else "at most"
                    yield f'{constraint} {thresholds[boundary_index]:.2f} on criterion "{criterion.name}"'
                case AcceptedValues.IntegerThresholds(thresholds):
                    assert len(thresholds) == boundaries_count
                    constraint = "at least" if criterion.values.is_increasing else "at most"
                    yield f'{constraint} {thresholds[boundary_index]} on criterion "{criterion.name}"'
                case AcceptedValues.EnumeratedThresholds(thresholds):
                    assert len(thresholds) == boundaries_count
                    yield f'at least "{thresholds[boundary_index]}" on criterion "{criterion.name}"'
                case AcceptedValues.RealIntervals(intervals):
                    assert len(intervals) == boundaries_count
                    interval = intervals[boundary_index]
                    yield f'between {interval[0]:.2f} and {interval[1]:.2f} on criterion "{criterion.name}"'
                case AcceptedValues.IntegerIntervals(intervals):
                    assert len(intervals) == boundaries_count
                    interval = intervals[boundary_index]
                    yield f'between {interval[0]} and {interval[1]} on criterion "{criterion.name}"'

    is_uc = all(sufficient_coalitions == model.sufficient_coalitions[0] for sufficient_coalitions in model.sufficient_coalitions[1:])
    if is_uc:
        match model.sufficient_coalitions[0].descriptor:
            case SufficientCoalitions.Weights(criterion_weights):
                yield "This is a MR-Sort (a.k.a. 1-Uc-NCS) model: an NCS model where the sufficient coalitions are specified using the same criterion weights for all boundaries."
                yield "The weights associated to each criterion are:"
                assert len(criterion_weights) == criteria_count
                for criterion, weight in zip(problem.criteria, criterion_weights):
                    yield f'  - Criterion "{criterion.name}": {weight:.2f}'
                yield "To get into an upper category, an alternative must be accepted by the following boundaries on a set of criteria whose weights add up to at least 1:"
            case SufficientCoalitions.Roots(upset_roots):
                yield "This is a Uc-NCS model: an NCS model with the same sufficient coalitions for all boundaries."
                yield "The sufficient coalitions of criteria are the following, as well as any of their unions:"
                yield from make_upset_roots(upset_roots)
                yield "To get into an upper category, an alternative must be accepted by the following boundaries on a sufficient coalition of criteria:"
        for boundary_index, category in enumerate(problem.ordered_categories[1:]):
            yield f'  - For category "{category.name}": {comma_and(make_profile(model.accepted_values, boundary_index))}'
    else:
        yield "This is a generic NCS model; sufficient coalitions are specified for each boundary."
        for boundary_index, (category, sufficient_coalitions) in enumerate(zip(problem.ordered_categories[1:], model.sufficient_coalitions)):
            match sufficient_coalitions.descriptor:
                case SufficientCoalitions.Weights(criterion_weights):
                    yield f'To get into category "{category.name}", an alternative must be accepted by the following boundaries on a set of criteria whose weights add up to at least 1:'
                    for profile, weight in zip(make_profile(model.accepted_values, boundary_index), criterion_weights):
                        yield f'  - {profile} (weight: {weight:.2f})'
                case SufficientCoalitions.Roots(upset_roots):
                    yield f'The sufficient coalitions for category "{category.name}" are the following, as well as any of their unions:'
                    yield from make_upset_roots(upset_roots)
                    yield f'To get into category "{category.name}", an alternative must be accepted by the following boundaries on a sufficient coalition of criteria: {comma_and(make_profile(model.accepted_values, boundary_index))}'


class DescribeClassificationModelTestCase(unittest.TestCase):
    maxDiff = None

    problem = Problem(
        [
            Criterion("Criterion 1", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 1)),
            Criterion("Criterion 2", Criterion.RealValues(Criterion.PreferenceDirection.decreasing, 0, 1)),
            Criterion("Criterion 3", Criterion.RealValues(Criterion.PreferenceDirection.single_peaked, 0, 8)),
            Criterion("Criterion 4", Criterion.RealValues(Criterion.PreferenceDirection.decreasing, 0, 1)),
        ],
        [
            Category("Bad"),
            Category("Intermediate"),
            Category("Good"),
        ],
    )

    def _test(self, model, expected):
        self.assertEqual(list(describe_classification_model(self.problem, model)), expected)

    def test_mrsort(self):
        self._test(
            Model(
                self.problem,
                [
                    AcceptedValues(AcceptedValues.RealThresholds([0.2, 0.7])),
                    AcceptedValues(AcceptedValues.RealThresholds([0.8, 0.7])),
                    AcceptedValues(AcceptedValues.RealIntervals([(1, 7), (3, 5)])),
                    AcceptedValues(AcceptedValues.RealThresholds([0.7, 0.3])),
                ],
                [
                    SufficientCoalitions(SufficientCoalitions.Weights([0.7, 0.5, 0.4, 0.2])),
                    SufficientCoalitions(SufficientCoalitions.Weights([0.7, 0.5, 0.4, 0.2])),
                ],
            ),
            [
                'This is a MR-Sort (a.k.a. 1-Uc-NCS) model: an NCS model where the sufficient coalitions are specified using the same criterion weights for all boundaries.',
                'The weights associated to each criterion are:',
                '  - Criterion "Criterion 1": 0.70',
                '  - Criterion "Criterion 2": 0.50',
                '  - Criterion "Criterion 3": 0.40',
                '  - Criterion "Criterion 4": 0.20',
                'To get into an upper category, an alternative must be accepted by the following boundaries on a set of criteria whose weights add up to at least 1:',
                '  - For category "Intermediate": at least 0.20 on criterion "Criterion 1", at most 0.80 on criterion "Criterion 2", between 1.00 and 7.00 on criterion "Criterion 3", and at most 0.70 on criterion "Criterion 4"',
                '  - For category "Good": at least 0.70 on criterion "Criterion 1", at most 0.70 on criterion "Criterion 2", between 3.00 and 5.00 on criterion "Criterion 3", and at most 0.30 on criterion "Criterion 4"'
            ],
        )

    def test_ucncs(self):
        self._test(
            Model(
                self.problem,
                [
                    AcceptedValues(AcceptedValues.RealThresholds([0.2, 0.7])),
                    AcceptedValues(AcceptedValues.RealThresholds([0.8, 0.7])),
                    AcceptedValues(AcceptedValues.RealIntervals([(1, 7), (3, 5)])),
                    AcceptedValues(AcceptedValues.RealThresholds([0.7, 0.3])),
                ],
                [
                    SufficientCoalitions(SufficientCoalitions.Roots(self.problem, [[0, 1], [0, 2], [1, 2, 3]])),
                    SufficientCoalitions(SufficientCoalitions.Roots(self.problem, [[0, 1], [0, 2], [1, 2, 3]])),
                ],
            ),
            [
                'This is a Uc-NCS model: an NCS model with the same sufficient coalitions for all boundaries.',
                'The sufficient coalitions of criteria are the following, as well as any of their unions:',
                '  - "Criterion 1" and "Criterion 2"',
                '  - "Criterion 1" and "Criterion 3"',
                '  - "Criterion 2", "Criterion 3", and "Criterion 4"',
                'To get into an upper category, an alternative must be accepted by the following boundaries on a sufficient coalition of criteria:',
                '  - For category "Intermediate": at least 0.20 on criterion "Criterion 1", at most 0.80 on criterion "Criterion 2", between 1.00 and 7.00 on criterion "Criterion 3", and at most 0.70 on criterion "Criterion 4"',
                '  - For category "Good": at least 0.70 on criterion "Criterion 1", at most 0.70 on criterion "Criterion 2", between 3.00 and 5.00 on criterion "Criterion 3", and at most 0.30 on criterion "Criterion 4"'
            ],
        )

    def test_mixed(self):
        self._test(
            Model(
                self.problem,
                [
                    AcceptedValues(AcceptedValues.RealThresholds([0.2, 0.7])),
                    AcceptedValues(AcceptedValues.RealThresholds([0.8, 0.7])),
                    AcceptedValues(AcceptedValues.RealIntervals([(1, 7), (3, 5)])),
                    AcceptedValues(AcceptedValues.RealThresholds([0.7, 0.3])),
                ],
                [
                    SufficientCoalitions(SufficientCoalitions.Roots(self.problem, [[0, 1], [0, 2], [1, 2, 3]])),
                    SufficientCoalitions(SufficientCoalitions.Weights([0.7, 0.5, 0.4, 0.2])),
                ],
            ),
            [
                'This is a generic NCS model; sufficient coalitions are specified for each boundary.',
                'The sufficient coalitions for category "Intermediate" are the following, as well as any of their unions:',
                '  - "Criterion 1" and "Criterion 2"',
                '  - "Criterion 1" and "Criterion 3"',
                '  - "Criterion 2", "Criterion 3", and "Criterion 4"',
                'To get into category "Intermediate", an alternative must be accepted by the following boundaries on a sufficient coalition of criteria: at least 0.20 on criterion "Criterion 1", at most 0.80 on criterion "Criterion 2", between 1.00 and 7.00 on criterion "Criterion 3", and at most 0.70 on criterion "Criterion 4"',
                'To get into category "Good", an alternative must be accepted by the following boundaries on a set of criteria whose weights add up to at least 1:',
                '  - at least 0.70 on criterion "Criterion 1" (weight: 0.70)',
                '  - at most 0.70 on criterion "Criterion 2" (weight: 0.50)',
                '  - between 3.00 and 5.00 on criterion "Criterion 3" (weight: 0.40)',
                '  - at most 0.30 on criterion "Criterion 4" (weight: 0.20)'
            ]
        )

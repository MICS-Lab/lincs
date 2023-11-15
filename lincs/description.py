import unittest

from . import Problem, Criterion, Category, Model, SufficientCoalitions


def describe_problem(problem: Problem):
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
        yield f'Criterion "{criterion.name}" takes {criterion.value_type.name} values between {criterion.min_value:.1f} and {criterion.max_value:.1f} included.'
        if criterion.preference_direction == criterion.PreferenceDirection.increasing:
            yield f'Higher values of "{criterion.name}" are known to be better.'
        else:
            yield f'Lower values of "{criterion.name}" are known to be better.'


class DescribeProblemTestCase(unittest.TestCase):
    def _test(self, problem, expected):
        self.assertEqual(list(describe_problem(problem)), expected)

    def test_simplest(self):
        self._test(
            Problem(
                [
                    Criterion("Criterion", Criterion.ValueType.real, Criterion.PreferenceDirection.increasing, 0, 1),
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
                    Criterion("Criterion", Criterion.ValueType.real, Criterion.PreferenceDirection.increasing, 0, 1),
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
                    Criterion("Increasing criterion", Criterion.ValueType.real, Criterion.PreferenceDirection.increasing, -5.2, 10.3),
                    Criterion("Decreasing criterion", Criterion.ValueType.real, Criterion.PreferenceDirection.decreasing, 5, 15),
                ],
                [Category("Bad"), Category("Good")],
            ),
            [
                'This a classification problem into 2 ordered categories named "Bad" and "Good".',
                'The best category is "Good" and the worst category is "Bad".',
                'There are 2 classification criteria (in no particular order).',
                'Criterion "Increasing criterion" takes real values between -5.2 and 10.3 included.',
                'Higher values of "Increasing criterion" are known to be better.',
                'Criterion "Decreasing criterion" takes real values between 5.0 and 15.0 included.',
                'Lower values of "Decreasing criterion" are known to be better.',
            ]
        )


def describe_model(problem: Problem, model: Model):
    categories_count = len(problem.ordered_categories)
    assert categories_count >= 2
    assert len(model.boundaries) == categories_count - 1
    criteria_count = len(problem.criteria)
    assert criteria_count >= 1

    def comma_and(s):
        s = list(s)
        if len(s) == 1:
            return s[0]
        elif len(s) == 2:
            return " and ".join(s)
        else:
            return ", ".join(s[:-1]) + ", and " + s[-1]  # https://en.wikipedia.org/wiki/Serial_comma

    def make_coalitions(boundary):
        assert boundary.sufficient_coalitions.kind == SufficientCoalitions.Kind.roots
        for coalition in boundary.sufficient_coalitions.upset_roots:
            criterion_names = []
            for criterion_index in coalition:
                criterion = problem.criteria[criterion_index]
                criterion_names.append(f'"{criterion.name}"')
            yield f'  - {comma_and(criterion_names)}'

    def make_profile(boundary):
        assert len(boundary.profile) == criteria_count
        for criterion, limit in zip(problem.criteria, boundary.profile):
            constraint = "at least" if criterion.preference_direction == criterion.PreferenceDirection.increasing else "at most"
            yield f'{constraint} {limit:.2f} on criterion "{criterion.name}"'

    is_uc = all(
        # @todo Provide equality operator (on the C++ side?).
        boundary.sufficient_coalitions.kind == model.boundaries[0].sufficient_coalitions.kind
        and list(boundary.sufficient_coalitions.criterion_weights) == list(model.boundaries[0].sufficient_coalitions.criterion_weights)
        and list(boundary.sufficient_coalitions.upset_roots) == list(model.boundaries[0].sufficient_coalitions.upset_roots)
        for boundary in model.boundaries
    )
    if is_uc:
        first_boundary = model.boundaries[0]
        if first_boundary.sufficient_coalitions.kind == SufficientCoalitions.Kind.weights:
            yield "This is a MR-Sort (a.k.a. 1-Uc-NCS) model: an NCS model where the sufficient coalitions are specified using the same criterion weights for all boundaries."
            yield "The weights associated to each criterion are:"
            assert len(first_boundary.sufficient_coalitions.criterion_weights) == criteria_count
            for criterion, weight in zip(problem.criteria, first_boundary.sufficient_coalitions.criterion_weights):
                yield f'  - Criterion "{criterion.name}": {weight:.2f}'
            yield "To get into an upper category, an alternative must be better than the following profiles on a set of criteria whose weights add up to at least 1:"
        else:
            assert first_boundary.sufficient_coalitions.kind == SufficientCoalitions.Kind.roots
            yield "This is a Uc-NCS model: an NCS model with the same sufficient coalitions for all boundaries."
            yield "The sufficient coalitions of criteria are the following, as well as any of their unions:"
            yield from make_coalitions(first_boundary)
            yield "To get into an upper category, an alternative must be better than the following profiles on a sufficient coalition of criteria:"
        for category, boundary in zip(problem.ordered_categories[1:], model.boundaries):
            yield f'  - For category "{category.name}": {comma_and(make_profile(boundary))}'
    else:
        yield "This is a generic NCS model; sufficient coalitions are specified for each boundary."
        for category, boundary in zip(problem.ordered_categories[1:], model.boundaries):
            if boundary.sufficient_coalitions.kind == SufficientCoalitions.Kind.weights:
                yield f'To get into category "{category.name}", an alternative must be better than the following profile on a set of criteria whose weights add up to at least 1:'
                for profile, weight in zip(make_profile(boundary), boundary.sufficient_coalitions.criterion_weights):
                    yield f'  - {profile} (weight: {weight:.2f})'
            else:
                assert boundary.sufficient_coalitions.kind == SufficientCoalitions.Kind.roots
                yield f'The sufficient coalitions for category "{category.name}" are the following, as well as any of their unions:'
                yield from make_coalitions(boundary)
                yield f'To get into category "{category.name}", an alternative must be better than the following profile on a sufficient coalition of criteria: {comma_and(make_profile(boundary))}'


class DescribeModelTestCase(unittest.TestCase):
    maxDiff = None

    problem = Problem(
        [
            Criterion("Criterion 1", Criterion.ValueType.real, Criterion.PreferenceDirection.increasing, 0, 1),
            Criterion("Criterion 2", Criterion.ValueType.real, Criterion.PreferenceDirection.decreasing, 0, 1),
            Criterion("Criterion 3", Criterion.ValueType.real, Criterion.PreferenceDirection.increasing, 0, 1),
            Criterion("Criterion 4", Criterion.ValueType.real, Criterion.PreferenceDirection.decreasing, 0, 1),
        ],
        [
            Category("Bad"),
            Category("Intermediate"),
            Category("Good"),
        ],
    )

    def _test(self, model, expected):
        self.assertEqual(list(describe_model(self.problem, model)), expected)

    def test_mrsort(self):
        self._test(
            Model(
                self.problem,
                [
                    Model.Boundary([0.2, 0.8, 0.4, 0.7], SufficientCoalitions(SufficientCoalitions.weights, [0.7, 0.5, 0.4, 0.2])),
                    Model.Boundary([0.7, 0.7, 0.5, 0.3], SufficientCoalitions(SufficientCoalitions.weights, [0.7, 0.5, 0.4, 0.2])),
                ],
            ),
            [
                'This is a MR-Sort (a.k.a. 1-Uc-NCS) model: an NCS model where the sufficient coalitions are specified using the same criterion weights for all boundaries.',
                'The weights associated to each criterion are:',
                '  - Criterion "Criterion 1": 0.70',
                '  - Criterion "Criterion 2": 0.50',
                '  - Criterion "Criterion 3": 0.40',
                '  - Criterion "Criterion 4": 0.20',
                'To get into an upper category, an alternative must be better than the following profiles on a set of criteria whose weights add up to at least 1:',
                '  - For category "Intermediate": at least 0.20 on criterion "Criterion 1", at most 0.80 on criterion "Criterion 2", at least 0.40 on criterion "Criterion 3", and at most 0.70 on criterion "Criterion 4"',
                '  - For category "Good": at least 0.70 on criterion "Criterion 1", at most 0.70 on criterion "Criterion 2", at least 0.50 on criterion "Criterion 3", and at most 0.30 on criterion "Criterion 4"'
            ],
        )

    def test_ucncs(self):
        self._test(
            Model(
                self.problem,
                [
                    Model.Boundary([0.2, 0.8, 0.4, 0.7], SufficientCoalitions(SufficientCoalitions.roots, 4, [[0, 1], [0, 2], [1, 2, 3]])),
                    Model.Boundary([0.7, 0.7, 0.5, 0.3], SufficientCoalitions(SufficientCoalitions.roots, 4, [[0, 1], [0, 2], [1, 2, 3]])),
                ],
            ),
            [
                'This is a Uc-NCS model: an NCS model with the same sufficient coalitions for all boundaries.',
                'The sufficient coalitions of criteria are the following, as well as any of their unions:',
                '  - "Criterion 1" and "Criterion 2"',
                '  - "Criterion 1" and "Criterion 3"',
                '  - "Criterion 2", "Criterion 3", and "Criterion 4"',
                'To get into an upper category, an alternative must be better than the following profiles on a sufficient coalition of criteria:',
                '  - For category "Intermediate": at least 0.20 on criterion "Criterion 1", at most 0.80 on criterion "Criterion 2", at least 0.40 on criterion "Criterion 3", and at most 0.70 on criterion "Criterion 4"',
                '  - For category "Good": at least 0.70 on criterion "Criterion 1", at most 0.70 on criterion "Criterion 2", at least 0.50 on criterion "Criterion 3", and at most 0.30 on criterion "Criterion 4"'
            ],
        )

    def test_mixed(self):
        self._test(
            Model(
                self.problem,
                [
                    Model.Boundary([0.2, 0.8, 0.4, 0.7], SufficientCoalitions(SufficientCoalitions.roots, 4, [[0, 1], [0, 2], [1, 2, 3]])),
                    Model.Boundary([0.7, 0.7, 0.5, 0.3], SufficientCoalitions(SufficientCoalitions.weights, [0.7, 0.5, 0.4, 0.2])),
                ],
            ),
            [
                'This is a generic NCS model; sufficient coalitions are specified for each boundary.',
                'The sufficient coalitions for category "Intermediate" are the following, as well as any of their unions:',
                '  - "Criterion 1" and "Criterion 2"',
                '  - "Criterion 1" and "Criterion 3"',
                '  - "Criterion 2", "Criterion 3", and "Criterion 4"',
                'To get into category "Intermediate", an alternative must be better than the following profile on a sufficient coalition of criteria: at least 0.20 on criterion "Criterion 1", at most 0.80 on criterion "Criterion 2", at least 0.40 on criterion "Criterion 3", and at most 0.70 on criterion "Criterion 4"',
                'To get into category "Good", an alternative must be better than the following profile on a set of criteria whose weights add up to at least 1:',
                '  - at least 0.70 on criterion "Criterion 1" (weight: 0.70)',
                '  - at most 0.70 on criterion "Criterion 2" (weight: 0.50)',
                '  - at least 0.50 on criterion "Criterion 3" (weight: 0.40)',
                '  - at most 0.30 on criterion "Criterion 4" (weight: 0.20)'
            ]
        )

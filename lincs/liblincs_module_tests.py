import unittest

from . import *


class DomainTestCase(unittest.TestCase):
    def test_init_empty(self):
        domain = Domain([], [])
        self.assertEqual(len(domain.criteria), 0)
        self.assertEqual(len(domain.categories), 0)

    def test_init_wrong_types(self):
        with self.assertRaises(TypeError):
            Domain()
        with self.assertRaises(TypeError):
            Domain([])
        with self.assertRaises(TypeError):
            Domain([0], [])
        with self.assertRaises(TypeError):
            Domain([], [0])
        with self.assertRaises(TypeError):
            Domain([], [], 0)

    def test_init_one_criterion(self):
        domain = Domain([Criterion("Criterion name", ValueType.real, CategoryCorrelation.growing)], [])
        self.assertEqual(domain.criteria[0].name, "Criterion name")
        self.assertEqual(domain.criteria[0].value_type, ValueType.real)
        self.assertEqual(domain.criteria[0].category_correlation, CategoryCorrelation.growing)

    def test_assign_criterion_attributes(self):
        # @todo (When there are more values in ValueType and CategoryCorrelation) Use other values in constructor
        domain = Domain([Criterion("Wrong criterion", ValueType.real, CategoryCorrelation.growing)], [])
        domain.criteria[0].name = "Criterion name"
        domain.criteria[0].value_type = ValueType.real
        domain.criteria[0].category_correlation = CategoryCorrelation.growing
        self.assertEqual(domain.criteria[0].name, "Criterion name")
        self.assertEqual(domain.criteria[0].value_type, ValueType.real)
        self.assertEqual(domain.criteria[0].category_correlation, CategoryCorrelation.growing)

    def test_assign_criterion(self):
        domain = Domain([Criterion("Wrong criterion", ValueType.real, CategoryCorrelation.growing)], [])
        domain.criteria[0] = Criterion("Criterion name", ValueType.real, CategoryCorrelation.growing)
        self.assertEqual(domain.criteria[0].name, "Criterion name")

    def test_append_criterion(self):
        domain = Domain([], [])
        domain.criteria.append(Criterion("Criterion name", ValueType.real, CategoryCorrelation.growing))
        self.assertEqual(len(domain.criteria), 1)

    def test_assign_criteria_slice(self):
        domain = Domain([], [])
        domain.criteria[:] = [Criterion("Criterion name", ValueType.real, CategoryCorrelation.growing)]
        self.assertEqual(len(domain.criteria), 1)

    def test_init_one_category(self):
        domain = Domain([], [Category("Category name")])
        self.assertEqual(len(domain.categories), 1)
        self.assertEqual(domain.categories[0].name, "Category name")

    def test_assign_category_attributes(self):
        domain = Domain([], [Category("Wrong category")])
        domain.categories[0].name = "Category name"
        self.assertEqual(domain.categories[0].name, "Category name")

    def test_assign_category(self):
        domain = Domain([], [Category("Wrong category")])
        domain.categories[0] = Category("Category name")
        self.assertEqual(domain.categories[0].name, "Category name")

    def test_append_category(self):
        domain = Domain([], [])
        domain.categories.append(Category("Category name"))
        self.assertEqual(len(domain.categories), 1)

    def test_assign_categories_slice(self):
        domain = Domain([], [])
        domain.categories[:] = [Category("Category name")]
        self.assertEqual(len(domain.categories), 1)


class ModelTestCase(unittest.TestCase):
    def test_init_empty(self):
        domain = Domain([], [])
        model = Model(domain, [])
        self.assertEqual(len(model.boundaries), 0)

    def test_init_wrong_types(self):
        domain = Domain([], [])
        with self.assertRaises(TypeError):
            Model()
        with self.assertRaises(TypeError):
            Model(domain)
        with self.assertRaises(TypeError):
            Model(domain, 0)
        with self.assertRaises(TypeError):
            Model(0, [])
        with self.assertRaises(TypeError):
            Model(domain, [0])

    def test_init_one_empty_boundary(self):
        domain = Domain([], [])
        model = Model(domain, [Boundary([], SufficientCoalitions(SufficientCoalitionsKind.weights, []))])
        self.assertEqual(len(model.boundaries), 1)
        self.assertEqual(len(model.boundaries[0].profile), 0)
        self.assertEqual(model.boundaries[0].sufficient_coalitions.kind, SufficientCoalitionsKind.weights)
        self.assertEqual(len(model.boundaries[0].sufficient_coalitions.criterion_weights), 0)

    def test_init_three_criteria_two_categories(self):
        domain = Domain(
            [
                Criterion("Criterion 1", ValueType.real, CategoryCorrelation.growing),
                Criterion("Criterion 2", ValueType.real, CategoryCorrelation.growing),
                Criterion("Criterion 3", ValueType.real, CategoryCorrelation.growing),
            ], [
                Category("Category 1"),
                Category("Category 2"),
            ],
        )
        model = Model(
            domain,
            [
                Boundary(
                    [5., 5., 5],
                    SufficientCoalitions(SufficientCoalitionsKind.weights, [0.7, 0.7, 1])
                ),
            ],
        )
        self.assertEqual(len(model.boundaries), 1)
        self.assertEqual(len(model.boundaries[0].profile), 3)
        self.assertEqual(len(model.boundaries[0].sufficient_coalitions.criterion_weights), 3)

    def test_assign_model_attributes(self):
        domain = Domain([], [])
        model = Model(domain, [])
        model.boundaries = [Boundary([], SufficientCoalitions(SufficientCoalitionsKind.weights, []))]
        self.assertEqual(len(model.boundaries), 1)


class AlternativesTestCase(unittest.TestCase):
    def test_init_empty(self):
        domain = Domain([], [])
        alternatives = Alternatives(domain, [])
        self.assertEqual(len(alternatives.alternatives), 0)

    def test_init_wrong_types(self):
        domain = Domain([], [])
        with self.assertRaises(TypeError):
            Alternatives()
        with self.assertRaises(TypeError):
            Alternatives(domain)
        with self.assertRaises(TypeError):
            Alternatives(domain, 0)
        with self.assertRaises(TypeError):
            Alternatives(0, [])
        with self.assertRaises(TypeError):
            Alternatives(domain, [0])

    def test_init_three_criteria_two_categories(self):
        domain = Domain(
            [
                Criterion("Criterion 1", ValueType.real, CategoryCorrelation.growing),
                Criterion("Criterion 2", ValueType.real, CategoryCorrelation.growing),
                Criterion("Criterion 3", ValueType.real, CategoryCorrelation.growing),
            ], [
                Category("Category 1"),
                Category("Category 2"),
            ],
        )
        alternatives = Alternatives(
            domain,
            [
                Alternative("Alternative name", [5., 5., 5], "Category 1"),
            ],
        )
        self.assertEqual(len(alternatives.alternatives), 1)
        self.assertEqual(len(alternatives.alternatives[0].profile), 3)

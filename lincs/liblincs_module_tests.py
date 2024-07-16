# Copyright 2023-2024 Vincent Jacques

import copy
import io
import pickle
import textwrap
import unittest
import os

from . import DataValidationException, LearningFailureException
from .classification import *


forbid_gpu = os.environ.get("LINCS_DEV_FORBID_GPU", "false") == "true"

class ProblemTestCase(unittest.TestCase):
    def test_init_simplest(self):
        problem = Problem(
            criteria=[
                Criterion(name="Criterion name", values=Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 1)),
            ],
            ordered_categories=[
                Category("Bad"),
                Category("Good"),
            ],
        )
        self.assertEqual(len(problem.criteria), 1)
        self.assertEqual(problem.criteria[0].name, "Criterion name")
        self.assertEqual(problem.criteria[0].value_type, Criterion.ValueType.real)
        self.assertTrue(problem.criteria[0].is_real)
        self.assertFalse(problem.criteria[0].is_integer)
        self.assertFalse(problem.criteria[0].is_enumerated)
        self.assertEqual(problem.criteria[0].real_values.preference_direction, Criterion.PreferenceDirection.increasing)
        self.assertEqual(problem.criteria[0].real_values.min_value, 0)
        self.assertEqual(problem.criteria[0].real_values.max_value, 1)
        self.assertEqual(len(problem.ordered_categories), 2)
        self.assertEqual(problem.ordered_categories[0].name, "Bad")
        self.assertEqual(problem.ordered_categories[1].name, "Good")

    def test_init_not_enough_categories(self):
        criterion = Criterion("Criterion name", Criterion.RealValues(preference_direction=Criterion.PreferenceDirection.increasing, min_value=0, max_value=1))
        with self.assertRaises(DataValidationException) as cm:
            Problem([criterion], [])
        self.assertEqual(cm.exception.args[0], "A problem must have at least 2 categories")
        with self.assertRaises(DataValidationException) as cm:
            Problem([criterion], [Category("Single")])
        self.assertEqual(cm.exception.args[0], "A problem must have at least 2 categories")

    def test_init_no_criterion(self):
        with self.assertRaises(DataValidationException) as cm:
            Problem([], [Category("Bad"), Category("Good")])
        self.assertEqual(cm.exception.args[0], "A problem must have at least one criterion")

    def test_init_wrong_types(self):
        with self.assertRaises(TypeError):
            Problem()
        with self.assertRaises(TypeError):
            Problem([])
        with self.assertRaises(TypeError):
            Problem([0], [])
        with self.assertRaises(TypeError):
            Problem([], [0])
        with self.assertRaises(TypeError):
            Problem([], [], 0)

    def test_iso_antitone(self):
        self.assertEqual(Criterion.PreferenceDirection.isotone, Criterion.PreferenceDirection.increasing)
        self.assertEqual(Criterion.PreferenceDirection.antitone, Criterion.PreferenceDirection.decreasing)

    def test_real_criterion(self):
        criterion = Criterion("Criterion name", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0.25, 2.75))
        self.assertEqual(criterion.name, "Criterion name")
        self.assertEqual(criterion.value_type, Criterion.ValueType.real)
        self.assertTrue(criterion.is_real)
        self.assertFalse(criterion.is_integer)
        self.assertFalse(criterion.is_enumerated)
        self.assertEqual(criterion.real_values.preference_direction, Criterion.PreferenceDirection.increasing)
        self.assertEqual(criterion.real_values.min_value, 0.25)
        self.assertEqual(criterion.real_values.max_value, 2.75)
        with self.assertRaises(RuntimeError):
            criterion.integer_values
        with self.assertRaises(RuntimeError):
            criterion.enumerated_values

    def test_integer_criterion(self):
        criterion = Criterion("Criterion name", Criterion.IntegerValues(Criterion.PreferenceDirection.increasing, 0, 20))
        self.assertEqual(criterion.name, "Criterion name")
        self.assertEqual(criterion.value_type, Criterion.ValueType.integer)
        self.assertFalse(criterion.is_real)
        self.assertTrue(criterion.is_integer)
        self.assertFalse(criterion.is_enumerated)
        self.assertEqual(criterion.integer_values.preference_direction, Criterion.PreferenceDirection.increasing)
        self.assertEqual(criterion.integer_values.min_value, 0)
        self.assertEqual(criterion.integer_values.max_value, 20)

    def test_enumerated_criterion(self):
        criterion = Criterion("Criterion name", Criterion.EnumeratedValues(["a a", "b", "c"]))
        self.assertEqual(criterion.name, "Criterion name")
        self.assertEqual(criterion.value_type, Criterion.ValueType.enumerated)
        self.assertFalse(criterion.is_real)
        self.assertFalse(criterion.is_integer)
        self.assertTrue(criterion.is_enumerated)
        self.assertEqual(list(criterion.enumerated_values.ordered_values), ["a a", "b", "c"])
        self.assertEqual(criterion.enumerated_values.get_value_rank("a a"), 0)
        self.assertEqual(criterion.enumerated_values.get_value_rank("b"), 1)
        self.assertEqual(criterion.enumerated_values.get_value_rank("c"), 2)

    def test_single_peaked_real_criteria(self):
        criterion = Criterion("Criterion name", Criterion.RealValues(Criterion.PreferenceDirection.single_peaked, 0.25, 2.75))
        self.assertEqual(criterion.name, "Criterion name")
        self.assertEqual(criterion.value_type, Criterion.ValueType.real)
        self.assertTrue(criterion.is_real)
        self.assertFalse(criterion.is_integer)
        self.assertFalse(criterion.is_enumerated)
        self.assertTrue(criterion.real_values.is_single_peaked)
        self.assertFalse(criterion.real_values.is_increasing)
        self.assertFalse(criterion.real_values.is_decreasing)
        self.assertEqual(criterion.real_values.preference_direction, Criterion.PreferenceDirection.single_peaked)
        self.assertEqual(criterion.real_values.min_value, 0.25)
        self.assertEqual(criterion.real_values.max_value, 2.75)

    def test_single_peaked_integer_criteria(self):
        criterion = Criterion("Criterion name", Criterion.IntegerValues(Criterion.PreferenceDirection.single_peaked, 25, 75))
        self.assertEqual(criterion.name, "Criterion name")
        self.assertEqual(criterion.value_type, Criterion.ValueType.integer)
        self.assertFalse(criterion.is_real)
        self.assertTrue(criterion.is_integer)
        self.assertFalse(criterion.is_enumerated)
        self.assertTrue(criterion.integer_values.is_single_peaked)
        self.assertFalse(criterion.integer_values.is_increasing)
        self.assertFalse(criterion.integer_values.is_decreasing)
        self.assertEqual(criterion.integer_values.preference_direction, Criterion.PreferenceDirection.single_peaked)
        self.assertEqual(criterion.integer_values.min_value, 25)
        self.assertEqual(criterion.integer_values.max_value, 75)

    def test_pickle_and_deep_copy(self):
        problem = Problem(
            criteria=[
                Criterion(name="Real criterion", values=Criterion.RealValues(Criterion.PreferenceDirection.increasing, 5, 10)),
                Criterion(name="Integer criterion", values=Criterion.IntegerValues(Criterion.PreferenceDirection.decreasing, 15, 100)),
                Criterion(name="Enumerated criterion", values=Criterion.EnumeratedValues(["a", "b", "c"])),
                Criterion(name="Single-peaked criterion", values=Criterion.RealValues(Criterion.PreferenceDirection.single_peaked, 0, 1)),
            ],
            ordered_categories=[
                Category("Bad"),
                Category("Medium"),
                Category("Good"),
            ],
        )
        pickled_problem = pickle.dumps(problem)
        self.assertIsInstance(pickled_problem, bytes)
        unpickled_problem = pickle.loads(pickled_problem)
        copied_problem = copy.deepcopy(problem)

        for p in (problem, unpickled_problem, copied_problem):
            self.assertEqual(len(p.criteria), 4)
            self.assertEqual(p.criteria[0].name, "Real criterion")
            self.assertEqual(p.criteria[0].value_type, Criterion.ValueType.real)
            self.assertTrue(p.criteria[0].is_real)
            self.assertFalse(p.criteria[0].is_integer)
            self.assertFalse(p.criteria[0].is_enumerated)
            self.assertEqual(p.criteria[0].real_values.preference_direction, Criterion.PreferenceDirection.increasing)
            self.assertEqual(p.criteria[0].real_values.min_value, 5)
            self.assertEqual(p.criteria[0].real_values.max_value, 10)
            self.assertEqual(p.criteria[1].name, "Integer criterion")
            self.assertEqual(p.criteria[1].value_type, Criterion.ValueType.integer)
            self.assertFalse(p.criteria[1].is_real)
            self.assertTrue(p.criteria[1].is_integer)
            self.assertFalse(p.criteria[1].is_enumerated)
            self.assertEqual(p.criteria[1].integer_values.preference_direction, Criterion.PreferenceDirection.decreasing)
            self.assertEqual(p.criteria[1].integer_values.min_value, 15)
            self.assertEqual(p.criteria[1].integer_values.max_value, 100)
            self.assertEqual(p.criteria[2].name, "Enumerated criterion")
            self.assertEqual(p.criteria[2].value_type, Criterion.ValueType.enumerated)
            self.assertFalse(p.criteria[2].is_real)
            self.assertFalse(p.criteria[2].is_integer)
            self.assertTrue(p.criteria[2].is_enumerated)
            self.assertEqual(list(p.criteria[2].enumerated_values.ordered_values), ["a", "b", "c"])
            self.assertEqual(p.criteria[2].enumerated_values.get_value_rank("a"), 0)
            self.assertEqual(p.criteria[2].enumerated_values.get_value_rank("b"), 1)
            self.assertEqual(p.criteria[2].enumerated_values.get_value_rank("c"), 2)
            self.assertEqual(p.criteria[3].name, "Single-peaked criterion")
            self.assertEqual(p.criteria[3].value_type, Criterion.ValueType.real)
            self.assertTrue(p.criteria[3].is_real)
            self.assertFalse(p.criteria[3].is_integer)
            self.assertFalse(p.criteria[3].is_enumerated)
            self.assertEqual(p.criteria[3].real_values.preference_direction, Criterion.PreferenceDirection.single_peaked)
            self.assertEqual(p.criteria[3].real_values.min_value, 0)
            self.assertEqual(p.criteria[3].real_values.max_value, 1)
            self.assertEqual(len(p.ordered_categories), 3)
            self.assertEqual(p.ordered_categories[0].name, "Bad")
            self.assertEqual(p.ordered_categories[1].name, "Medium")
            self.assertEqual(p.ordered_categories[2].name, "Good")

        for p in (unpickled_problem, copied_problem):
            self.assertIsNot(p, problem)
            self.assertIsNot(p.criteria, problem.criteria)
            self.assertIsNot(p.criteria[0], problem.criteria[0])
            self.assertIsNot(p.criteria[1], problem.criteria[1])
            self.assertIsNot(p.criteria[2], problem.criteria[2])
            self.assertIsNot(p.criteria[3], problem.criteria[3])
            self.assertIsNot(p.ordered_categories, problem.ordered_categories)
            self.assertIsNot(p.ordered_categories[0], problem.ordered_categories[0])
            self.assertIsNot(p.ordered_categories[1], problem.ordered_categories[1])
            self.assertIsNot(p.ordered_categories[2], problem.ordered_categories[2])


class ModelTestCase(unittest.TestCase):
    def test_init_wrong_types(self):
        problem = Problem(
            [
                Criterion("Criterion 1", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 1)),
            ],
            [Category("Bad"), Category("Good")],
        )
        with self.assertRaises(TypeError):
            Model()
        with self.assertRaises(TypeError):
            Model(problem)
        with self.assertRaises(TypeError):
            Model(problem, 0)
        with self.assertRaises(TypeError):
            Model(0, [])
        with self.assertRaises(TypeError):
            Model(problem, [0])

    def test_init_simplest(self):
        problem = Problem(
            [
                Criterion("Criterion 1", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 1)),
            ],
            [Category("Bad"), Category("Good")],
        )
        model = Model(
            problem=problem,
            accepted_values=[AcceptedValues(AcceptedValues.RealThresholds([0.5]))],
            sufficient_coalitions=[SufficientCoalitions(SufficientCoalitions.Weights([0.75]))],
        )
        self.assertEqual(len(model.accepted_values), 1)
        self.assertEqual(model.accepted_values[0].value_type, Criterion.ValueType.real)
        self.assertTrue(model.accepted_values[0].is_real)
        self.assertFalse(model.accepted_values[0].is_integer)
        self.assertFalse(model.accepted_values[0].is_enumerated)
        self.assertEqual(model.accepted_values[0].kind, AcceptedValues.Kind.thresholds)
        self.assertTrue(model.accepted_values[0].is_thresholds)
        self.assertEqual(len(model.accepted_values[0].real_thresholds.thresholds), 1)
        self.assertEqual(model.accepted_values[0].real_thresholds.thresholds[0], 0.5)
        self.assertEqual(len(model.sufficient_coalitions), 1)
        self.assertEqual(model.sufficient_coalitions[0].kind, SufficientCoalitions.Kind.weights)
        self.assertTrue(model.sufficient_coalitions[0].is_weights)
        self.assertFalse(model.sufficient_coalitions[0].is_roots)
        self.assertEqual(len(model.sufficient_coalitions[0].weights.criterion_weights), 1)
        self.assertEqual(model.sufficient_coalitions[0].weights.criterion_weights[0], 0.75)

    def test_bad_accesses(self):
        problem = Problem(
            [
                Criterion("Criterion 1", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 1)),
            ],
            [Category("Bad"), Category("Good")],
        )
        with self.assertRaises(RuntimeError):
            problem.criteria[0].integer_values
        model = Model(
            problem,
            [AcceptedValues(AcceptedValues.RealThresholds([0.5]))],
            [SufficientCoalitions(SufficientCoalitions.Weights([0.75]))],
        )
        with self.assertRaises(RuntimeError):
            model.accepted_values[0].integer_thresholds
        with self.assertRaises(RuntimeError):
            model.sufficient_coalitions[0].roots

    def test_init_size_mismatch(self):
        problem = Problem(
            [
                Criterion("Criterion 1", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 1)),
            ],
            [Category("Bad"), Category("Good")],
        )
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem,
                [],
                [SufficientCoalitions(SufficientCoalitions.Weights([0.75]))],
            )
        self.assertEqual(cm.exception.args[0], "The number of accepted values descriptors in the model must be equal to the number of criteria in the problem")
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem,
                [
                    AcceptedValues(AcceptedValues.RealThresholds([0.5])),
                    AcceptedValues(AcceptedValues.RealThresholds([0.5])),
                ],
                [SufficientCoalitions(SufficientCoalitions.Weights([0.75]))],
            )
        self.assertEqual(cm.exception.args[0], "The number of accepted values descriptors in the model must be equal to the number of criteria in the problem")
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem,
                [AcceptedValues(AcceptedValues.RealThresholds([]))],
                [SufficientCoalitions(SufficientCoalitions.Weights([0.75]))],
            )
        self.assertEqual(cm.exception.args[0], "The number of real thresholds in an accepted values descriptor must be one less than the number of categories in the problem")
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem,
                [AcceptedValues(AcceptedValues.RealThresholds([0.5, 0.6]))],
                [SufficientCoalitions(SufficientCoalitions.Weights([0.75]))],
            )
        self.assertEqual(cm.exception.args[0], "The number of real thresholds in an accepted values descriptor must be one less than the number of categories in the problem")
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem,
                [AcceptedValues(AcceptedValues.RealThresholds([0.5]))],
                [],
            )
        self.assertEqual(cm.exception.args[0], "The number of sufficient coalitions in the model must be one less than the number of categories in the problem")
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem,
                [AcceptedValues(AcceptedValues.RealThresholds([0.5]))],
                [
                    SufficientCoalitions(SufficientCoalitions.Weights([0.75])),
                    SufficientCoalitions(SufficientCoalitions.Weights([0.75])),
                ],
            )
        self.assertEqual(cm.exception.args[0], "The number of sufficient coalitions in the model must be one less than the number of categories in the problem")
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem,
                [AcceptedValues(AcceptedValues.RealThresholds([0.5]))],
                [SufficientCoalitions(SufficientCoalitions.Weights([]))],
            )
        self.assertEqual(cm.exception.args[0], "The number of criterion weights in a sufficient coalitions descriptor must be equal to the number of criteria in the problem")
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem,
                [AcceptedValues(AcceptedValues.RealThresholds([0.5]))],
                [SufficientCoalitions(SufficientCoalitions.Weights([0.75, 0.75]))],
            )
        self.assertEqual(cm.exception.args[0], "The number of criterion weights in a sufficient coalitions descriptor must be equal to the number of criteria in the problem")

    def test_init_type_mismatch(self):
        problem = Problem(
            [
                Criterion("Criterion 1", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 1)),
            ], [
                Category("Category 1"),
                Category("Category 2"),
            ],
        )
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem,
                [
                    AcceptedValues(AcceptedValues.IntegerThresholds([50])),
                ],
                [
                    SufficientCoalitions(SufficientCoalitions.Roots(problem, [[0]])),
                ],
            )
        self.assertEqual(cm.exception.args[0], "The value type of an accepted values descriptor must be the same as the value type of the corresponding criterion")

    def test_init_roots(self):
        problem = Problem(
            [
                Criterion("Criterion 1", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 1)),
                Criterion("Criterion 2", Criterion.IntegerValues(Criterion.PreferenceDirection.decreasing, 0, 100)),
                Criterion("Criterion 3", Criterion.EnumeratedValues(["c", "b", "a"])),
            ], [
                Category("Category 1"),
                Category("Category 2"),
            ],
        )
        model = Model(
            problem,
            [
                AcceptedValues(AcceptedValues.RealThresholds([0.5])),
                AcceptedValues(AcceptedValues.IntegerThresholds([50])),
                AcceptedValues(AcceptedValues.EnumeratedThresholds(["b"])),
            ],
            [
                SufficientCoalitions(SufficientCoalitions.Roots(problem, [[0, 1], [0, 2]])),
            ],
        )
        self.assertEqual(len(model.accepted_values), 3)
        self.assertEqual(model.accepted_values[0].value_type, Criterion.ValueType.real)
        self.assertTrue(model.accepted_values[0].is_real)
        self.assertFalse(model.accepted_values[0].is_integer)
        self.assertFalse(model.accepted_values[0].is_enumerated)
        self.assertEqual(model.accepted_values[0].kind, AcceptedValues.Kind.thresholds)
        self.assertTrue(model.accepted_values[0].is_thresholds)
        self.assertEqual(len(model.accepted_values[0].real_thresholds.thresholds), 1)
        self.assertEqual(model.accepted_values[0].real_thresholds.thresholds[0], 0.5)
        self.assertEqual(model.accepted_values[1].value_type, Criterion.ValueType.integer)
        self.assertFalse(model.accepted_values[1].is_real)
        self.assertTrue(model.accepted_values[1].is_integer)
        self.assertFalse(model.accepted_values[1].is_enumerated)
        self.assertEqual(model.accepted_values[1].kind, AcceptedValues.Kind.thresholds)
        self.assertTrue(model.accepted_values[1].is_thresholds)
        self.assertEqual(len(model.accepted_values[1].integer_thresholds.thresholds), 1)
        self.assertEqual(model.accepted_values[1].integer_thresholds.thresholds[0], 50)
        self.assertEqual(model.accepted_values[2].value_type, Criterion.ValueType.enumerated)
        self.assertFalse(model.accepted_values[2].is_real)
        self.assertFalse(model.accepted_values[2].is_integer)
        self.assertTrue(model.accepted_values[2].is_enumerated)
        self.assertEqual(model.accepted_values[2].kind, AcceptedValues.Kind.thresholds)
        self.assertTrue(model.accepted_values[2].is_thresholds)
        self.assertEqual(len(model.accepted_values[2].enumerated_thresholds.thresholds), 1)
        self.assertEqual(model.accepted_values[2].enumerated_thresholds.thresholds[0], "b")
        self.assertEqual(len(model.sufficient_coalitions), 1)
        self.assertEqual(model.sufficient_coalitions[0].kind, SufficientCoalitions.Kind.roots)
        self.assertFalse(model.sufficient_coalitions[0].is_weights)
        self.assertTrue(model.sufficient_coalitions[0].is_roots)
        self.assertEqual(model.sufficient_coalitions[0].roots.upset_roots[0][0], 0)
        self.assertEqual(model.sufficient_coalitions[0].roots.upset_roots[0][1], 1)
        self.assertEqual(model.sufficient_coalitions[0].roots.upset_roots[1][0], 0)
        self.assertEqual(model.sufficient_coalitions[0].roots.upset_roots[1][1], 2)

    def test_init_size_mismatch_2(self):
        problem = Problem(
            [
                Criterion("Criterion 1", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 1)),
            ], [
                Category("Category 1"),
                Category("Category 2"),
            ],
        )
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem,
                [AcceptedValues(AcceptedValues.RealThresholds([0.5]))],
                [SufficientCoalitions(SufficientCoalitions.Roots(problem, [[3]]))],
            )
        self.assertEqual(cm.exception.args[0], "An element index in a root in a sufficient coalitions descriptor must be less than the number of criteria in the problem")

    def test_init_integer(self):
        problem = Problem(
            [
                Criterion("Criterion 1", Criterion.IntegerValues(Criterion.PreferenceDirection.increasing, 0, 10)),
            ],
            [
                Category("Category 1"),
                Category("Category 2"),
            ],
        )
        model = Model(
            problem,
            [AcceptedValues(AcceptedValues.IntegerThresholds([5]))],
            [SufficientCoalitions(SufficientCoalitions.Weights([0.75]))],
        )
        self.assertEqual(len(model.accepted_values), 1)
        self.assertEqual(model.accepted_values[0].value_type, Criterion.ValueType.integer)
        self.assertFalse(model.accepted_values[0].is_real)
        self.assertTrue(model.accepted_values[0].is_integer)
        self.assertFalse(model.accepted_values[0].is_enumerated)
        self.assertEqual(model.accepted_values[0].kind, AcceptedValues.Kind.thresholds)
        self.assertTrue(model.accepted_values[0].is_thresholds)
        self.assertEqual(len(model.accepted_values[0].integer_thresholds.thresholds), 1)
        self.assertEqual(model.accepted_values[0].integer_thresholds.thresholds[0], 5)

    def test_init_enumerated(self):
        problem = Problem(
            [
                Criterion("Criterion 1", Criterion.EnumeratedValues(["low", "mid", "high"])),
            ],
            [
                Category("Category 1"),
                Category("Category 2"),
            ],
        )
        model = Model(
            problem,
            [AcceptedValues(AcceptedValues.EnumeratedThresholds(["mid"]))],
            [SufficientCoalitions(SufficientCoalitions.Weights([0.75]))],
        )
        self.assertEqual(len(model.accepted_values), 1)
        self.assertEqual(model.accepted_values[0].value_type, Criterion.ValueType.enumerated)
        self.assertFalse(model.accepted_values[0].is_real)
        self.assertFalse(model.accepted_values[0].is_integer)
        self.assertTrue(model.accepted_values[0].is_enumerated)
        self.assertEqual(model.accepted_values[0].kind, AcceptedValues.Kind.thresholds)
        self.assertTrue(model.accepted_values[0].is_thresholds)
        self.assertEqual(len(model.accepted_values[0].enumerated_thresholds.thresholds), 1)
        self.assertEqual(model.accepted_values[0].enumerated_thresholds.thresholds[0], "mid")

    def test_init_single_peaked(self):
        problem = Problem(
            [
                Criterion("Criterion 1", Criterion.IntegerValues(Criterion.PreferenceDirection.single_peaked, 0, 10)),
                Criterion("Criterion 2", Criterion.RealValues(Criterion.PreferenceDirection.single_peaked, 0, 100)),
            ],
            [
                Category("Category 1"),
                Category("Category 2"),
                Category("Category 3"),
            ],
        )
        model = Model(
            problem,
            [
                AcceptedValues(AcceptedValues.IntegerIntervals([(2, 8), (4, 6)])),
                AcceptedValues(AcceptedValues.RealIntervals([(20, 80), (40, 60)])),
            ],
            [
                SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5])),
                SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5])),
            ],
        )
        self.assertEqual(len(model.accepted_values), 2)
        self.assertEqual(model.accepted_values[0].value_type, Criterion.ValueType.integer)
        self.assertFalse(model.accepted_values[0].is_real)
        self.assertTrue(model.accepted_values[0].is_integer)
        self.assertFalse(model.accepted_values[0].is_enumerated)
        self.assertEqual(model.accepted_values[0].kind, AcceptedValues.Kind.intervals)
        self.assertTrue(model.accepted_values[0].is_intervals)
        self.assertEqual(len(model.accepted_values[0].integer_intervals.intervals), 2)
        self.assertEqual(model.accepted_values[0].integer_intervals.intervals[0], (2, 8))
        self.assertEqual(model.accepted_values[0].integer_intervals.intervals[1], (4, 6))
        self.assertEqual(model.accepted_values[1].value_type, Criterion.ValueType.real)
        self.assertTrue(model.accepted_values[1].is_real)
        self.assertFalse(model.accepted_values[1].is_integer)
        self.assertFalse(model.accepted_values[1].is_enumerated)
        self.assertEqual(model.accepted_values[1].kind, AcceptedValues.Kind.intervals)
        self.assertTrue(model.accepted_values[1].is_intervals)
        self.assertEqual(len(model.accepted_values[1].real_intervals.intervals), 2)
        self.assertEqual(model.accepted_values[1].real_intervals.intervals[0], (20, 80))
        self.assertEqual(model.accepted_values[1].real_intervals.intervals[1], (40, 60))

    def test_pickle_and_deep_copy(self):
        problem = Problem(
            criteria=[
                Criterion(name="Real criterion", values=Criterion.RealValues(Criterion.PreferenceDirection.increasing, 5, 10)),
                Criterion(name="Integer criterion", values=Criterion.IntegerValues(Criterion.PreferenceDirection.decreasing, 15, 100)),
                Criterion(name="Enumerated criterion", values=Criterion.EnumeratedValues(["a", "b", "c"])),
                Criterion(name="Single-peaked criterion", values=Criterion.RealValues(Criterion.PreferenceDirection.single_peaked, 0, 8)),
            ],
            ordered_categories=[
                Category("Bad"),
                Category("Medium"),
                Category("Good"),
            ],
        )
        model = Model(
            problem=problem,
            accepted_values=[
                AcceptedValues(AcceptedValues.RealThresholds([7, 9])),
                AcceptedValues(AcceptedValues.IntegerThresholds([50, 25])),
                AcceptedValues(AcceptedValues.EnumeratedThresholds(["b", "c"])),
                AcceptedValues(AcceptedValues.RealIntervals([(1, 7), (3, 5)])),
            ],
            sufficient_coalitions=[
                SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5, 0.5])),
                SufficientCoalitions(SufficientCoalitions.Roots(problem, [[0, 1], [0, 2], [0, 3]])),
            ],
        )
        pickled_model = pickle.dumps(model)
        self.assertIsInstance(pickled_model, bytes)
        unpickled_model = pickle.loads(pickled_model)
        copied_model = copy.deepcopy(model)

        for m in (model, unpickled_model, copied_model):
            self.assertEqual(len(m.accepted_values), 4)
            self.assertEqual(m.accepted_values[0].value_type, Criterion.ValueType.real)
            self.assertTrue(m.accepted_values[0].is_real)
            self.assertFalse(m.accepted_values[0].is_integer)
            self.assertFalse(m.accepted_values[0].is_enumerated)
            self.assertEqual(m.accepted_values[0].kind, AcceptedValues.Kind.thresholds)
            self.assertTrue(m.accepted_values[0].is_thresholds)
            self.assertFalse(m.accepted_values[0].is_intervals)
            self.assertEqual(len(m.accepted_values[0].real_thresholds.thresholds), 2)
            self.assertEqual(m.accepted_values[0].real_thresholds.thresholds[0], 7)
            self.assertEqual(m.accepted_values[0].real_thresholds.thresholds[1], 9)
            self.assertEqual(m.accepted_values[1].value_type, Criterion.ValueType.integer)
            self.assertFalse(m.accepted_values[1].is_real)
            self.assertTrue(m.accepted_values[1].is_integer)
            self.assertFalse(m.accepted_values[1].is_enumerated)
            self.assertEqual(m.accepted_values[1].kind, AcceptedValues.Kind.thresholds)
            self.assertTrue(m.accepted_values[1].is_thresholds)
            self.assertFalse(m.accepted_values[1].is_intervals)
            self.assertEqual(len(m.accepted_values[1].integer_thresholds.thresholds), 2)
            self.assertEqual(m.accepted_values[1].integer_thresholds.thresholds[0], 50)
            self.assertEqual(m.accepted_values[1].integer_thresholds.thresholds[1], 25)
            self.assertEqual(m.accepted_values[2].value_type, Criterion.ValueType.enumerated)
            self.assertFalse(m.accepted_values[2].is_real)
            self.assertFalse(m.accepted_values[2].is_integer)
            self.assertTrue(m.accepted_values[2].is_enumerated)
            self.assertEqual(m.accepted_values[2].kind, AcceptedValues.Kind.thresholds)
            self.assertTrue(m.accepted_values[2].is_thresholds)
            self.assertFalse(m.accepted_values[2].is_intervals)
            self.assertEqual(len(m.accepted_values[2].enumerated_thresholds.thresholds), 2)
            self.assertEqual(m.accepted_values[2].enumerated_thresholds.thresholds[0], "b")
            self.assertEqual(m.accepted_values[2].enumerated_thresholds.thresholds[1], "c")
            self.assertEqual(m.accepted_values[3].value_type, Criterion.ValueType.real)
            self.assertTrue(m.accepted_values[3].is_real)
            self.assertFalse(m.accepted_values[3].is_integer)
            self.assertFalse(m.accepted_values[3].is_enumerated)
            self.assertEqual(m.accepted_values[3].kind, AcceptedValues.Kind.intervals)
            self.assertFalse(m.accepted_values[3].is_thresholds)
            self.assertTrue(m.accepted_values[3].is_intervals)
            self.assertEqual(len(m.accepted_values[3].real_intervals.intervals), 2)
            self.assertEqual(m.accepted_values[3].real_intervals.intervals[0], (1, 7))
            self.assertEqual(m.accepted_values[3].real_intervals.intervals[1], (3, 5))
            self.assertEqual(len(m.sufficient_coalitions), 2)
            self.assertEqual(m.sufficient_coalitions[0].kind, SufficientCoalitions.Kind.weights)
            self.assertTrue(m.sufficient_coalitions[0].is_weights)
            self.assertFalse(m.sufficient_coalitions[0].is_roots)
            self.assertEqual(len(m.sufficient_coalitions[0].weights.criterion_weights), 4)
            self.assertEqual(m.sufficient_coalitions[0].weights.criterion_weights[0], 0.5)
            self.assertEqual(m.sufficient_coalitions[0].weights.criterion_weights[1], 0.5)
            self.assertEqual(m.sufficient_coalitions[0].weights.criterion_weights[2], 0.5)
            self.assertEqual(m.sufficient_coalitions[0].weights.criterion_weights[3], 0.5)
            self.assertEqual(m.sufficient_coalitions[1].kind, SufficientCoalitions.Kind.roots)
            self.assertFalse(m.sufficient_coalitions[1].is_weights)
            self.assertTrue(m.sufficient_coalitions[1].is_roots)
            self.assertEqual(m.sufficient_coalitions[1].roots.upset_roots[0][0], 0)
            self.assertEqual(m.sufficient_coalitions[1].roots.upset_roots[0][1], 1)
            self.assertEqual(m.sufficient_coalitions[1].roots.upset_roots[1][0], 0)
            self.assertEqual(m.sufficient_coalitions[1].roots.upset_roots[1][1], 2)
            self.assertEqual(m.sufficient_coalitions[1].roots.upset_roots[2][0], 0)
            self.assertEqual(m.sufficient_coalitions[1].roots.upset_roots[2][1], 3)

        for m in (unpickled_model, copied_model):
            self.assertIsNot(m, model)
            self.assertIsNot(m.accepted_values, model.accepted_values)
            self.assertIsNot(m.accepted_values[0], model.accepted_values[0])
            self.assertIsNot(m.accepted_values[1], model.accepted_values[1])
            self.assertIsNot(m.accepted_values[2], model.accepted_values[2])
            self.assertIsNot(m.sufficient_coalitions, model.sufficient_coalitions)
            self.assertIsNot(m.sufficient_coalitions[0], model.sufficient_coalitions[0])
            self.assertIsNot(m.sufficient_coalitions[1], model.sufficient_coalitions[1])

    def test_pickle_empty_roots(self):
        problem = Problem(
            criteria=[
                Criterion(name="Real criterion", values=Criterion.RealValues(Criterion.PreferenceDirection.increasing, 5, 10)),
                Criterion(name="Integer criterion", values=Criterion.IntegerValues(Criterion.PreferenceDirection.decreasing, 15, 100)),
                Criterion(name="Enumerated criterion", values=Criterion.EnumeratedValues(["a", "b", "c"])),
            ],
            ordered_categories=[
                Category("Bad"),
                Category("Medium"),
                Category("Good"),
            ],
        )
        r = pickle.loads(pickle.dumps(SufficientCoalitions.Roots(problem, [])))
        self.assertEqual(r.upset_roots, [])

    def test_init_unordered_profiles(self):
        problem = Problem(
            criteria=[
                Criterion(name="Real criterion", values=Criterion.RealValues(Criterion.PreferenceDirection.increasing, 5, 10)),
                Criterion(name="Integer criterion", values=Criterion.IntegerValues(Criterion.PreferenceDirection.decreasing, 15, 100)),
                Criterion(name="Enumerated criterion", values=Criterion.EnumeratedValues(["a", "b", "c"])),
            ],
            ordered_categories=[
                Category("Bad"),
                Category("Medium"),
                Category("Good"),
            ],
        )

        Model(
            problem, [
                AcceptedValues(AcceptedValues.RealThresholds([7., 9.])),
                AcceptedValues(AcceptedValues.IntegerThresholds([50, 25])),
                AcceptedValues(AcceptedValues.EnumeratedThresholds(["b", "c"])),
            ], [
                SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
            ],
        )
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem, [
                    AcceptedValues(AcceptedValues.RealThresholds([9., 7.])),
                    AcceptedValues(AcceptedValues.IntegerThresholds([50, 25])),
                    AcceptedValues(AcceptedValues.EnumeratedThresholds(["b", "c"])),
                ], [
                    SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                    SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                ],
            )
        self.assertEqual(cm.exception.args[0], "The real thresholds in an accepted values descriptor must be in preference order")
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem, [
                    AcceptedValues(AcceptedValues.RealThresholds([7., 9.])),
                    AcceptedValues(AcceptedValues.IntegerThresholds([25, 50])),
                    AcceptedValues(AcceptedValues.EnumeratedThresholds(["b", "c"])),
                ], [
                    SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                    SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                ],
            )
        self.assertEqual(cm.exception.args[0], "The integer thresholds in an accepted values descriptor must be in preference order")
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem, [
                    AcceptedValues(AcceptedValues.RealThresholds([7., 9.])),
                    AcceptedValues(AcceptedValues.IntegerThresholds([50, 25])),
                    AcceptedValues(AcceptedValues.EnumeratedThresholds(["b", "a"])),
                ], [
                    SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                    SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                ],
            )
        self.assertEqual(cm.exception.args[0], "The enumerated thresholds in an accepted values descriptor must be in preference order")

    def test_init_profiles_outside_range(self):
        problem = Problem(
            criteria=[
                Criterion(name="Real criterion", values=Criterion.RealValues(Criterion.PreferenceDirection.increasing, 5, 10)),
                Criterion(name="Integer criterion", values=Criterion.IntegerValues(Criterion.PreferenceDirection.decreasing, 15, 100)),
                Criterion(name="Enumerated criterion", values=Criterion.EnumeratedValues(["a", "b", "c"])),
            ],
            ordered_categories=[
                Category("Bad"),
                Category("Medium"),
                Category("Good"),
            ],
        )

        Model(
            problem, [
                AcceptedValues(AcceptedValues.RealThresholds([7., 9.])),
                AcceptedValues(AcceptedValues.IntegerThresholds([50, 25])),
                AcceptedValues(AcceptedValues.EnumeratedThresholds(["b", "c"])),
            ], [
                SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
            ],
        )
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem, [
                    AcceptedValues(AcceptedValues.RealThresholds([3., 9.])),
                    AcceptedValues(AcceptedValues.IntegerThresholds([50, 25])),
                    AcceptedValues(AcceptedValues.EnumeratedThresholds(["b", "c"])),
                ], [
                    SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                    SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                ],
            )
        self.assertEqual(cm.exception.args[0], "Each threshold in an accepted values descriptor must be between the min and max values for the corresponding real criterion")
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem, [
                    AcceptedValues(AcceptedValues.RealThresholds([7., 11.])),
                    AcceptedValues(AcceptedValues.IntegerThresholds([50, 25])),
                    AcceptedValues(AcceptedValues.EnumeratedThresholds(["b", "c"])),
                ], [
                    SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                    SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                ],
            )
        self.assertEqual(cm.exception.args[0], "Each threshold in an accepted values descriptor must be between the min and max values for the corresponding real criterion")
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem, [
                    AcceptedValues(AcceptedValues.RealThresholds([7., 9.])),
                    AcceptedValues(AcceptedValues.IntegerThresholds([50, 10])),
                    AcceptedValues(AcceptedValues.EnumeratedThresholds(["b", "c"])),
                ], [
                    SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                    SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                ],
            )
        self.assertEqual(cm.exception.args[0], "Each threshold in an accepted values descriptor must be between the min and max values for the corresponding integer criterion")
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem, [
                    AcceptedValues(AcceptedValues.RealThresholds([7., 9.])),
                    AcceptedValues(AcceptedValues.IntegerThresholds([110, 25])),
                    AcceptedValues(AcceptedValues.EnumeratedThresholds(["b", "c"])),
                ], [
                    SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                    SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                ],
            )
        self.assertEqual(cm.exception.args[0], "Each threshold in an accepted values descriptor must be between the min and max values for the corresponding integer criterion")
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem, [
                    AcceptedValues(AcceptedValues.RealThresholds([7., 9.])),
                    AcceptedValues(AcceptedValues.IntegerThresholds([50, 25])),
                    AcceptedValues(AcceptedValues.EnumeratedThresholds(["b", "d"])),
                ], [
                    SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                    SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                ],
            )
        self.assertEqual(cm.exception.args[0], "Each threshold in an accepted values descriptor must be in the enumerated values for the corresponding criterion")

    def test_init_accepted_values_not_imbricated(self):
        problem = Problem(
            criteria=[
                Criterion(name="Real criterion", values=Criterion.RealValues(Criterion.PreferenceDirection.increasing, 5, 10)),
                Criterion(name="Integer criterion", values=Criterion.IntegerValues(Criterion.PreferenceDirection.decreasing, 15, 100)),
                Criterion(name="Enumerated criterion", values=Criterion.EnumeratedValues(["a", "b", "c"])),
            ],
            ordered_categories=[
                Category("Bad"),
                Category("Medium"),
                Category("Good"),
            ],
        )

        Model(
            problem, [
                AcceptedValues(AcceptedValues.RealThresholds([7., 9.])),
                AcceptedValues(AcceptedValues.IntegerThresholds([50, 25])),
                AcceptedValues(AcceptedValues.EnumeratedThresholds(["b", "c"])),
            ], [
                SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
            ],
        )
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem, [
                    AcceptedValues(AcceptedValues.RealThresholds([7., 9.])),
                    AcceptedValues(AcceptedValues.IntegerThresholds([50, 25])),
                    AcceptedValues(AcceptedValues.EnumeratedThresholds(["b", "c"])),
                ], [
                    SufficientCoalitions(SufficientCoalitions.Weights([0.5, 0.5, 0.5])),
                    SufficientCoalitions(SufficientCoalitions.Weights([1, 1, 1])),
                ],
            )
        self.assertEqual(cm.exception.args[0], "Sufficient coalitions must be imbricated")


class AlternativesTestCase(unittest.TestCase):
    def test_init_wrong_types(self):
        problem = Problem(
            [
                Criterion("Criterion 1", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 1)),
            ],
            [Category("Bad"), Category("Good")],
        )
        with self.assertRaises(TypeError):
            Alternatives()
        with self.assertRaises(TypeError):
            Alternatives(problem)
        with self.assertRaises(TypeError):
            Alternatives(problem, 0)
        with self.assertRaises(TypeError):
            Alternatives(0, [])
        with self.assertRaises(TypeError):
            Alternatives(problem, [0])

    def test_init_three_criteria_two_categories(self):
        problem = Problem(
            [
                Criterion("Criterion 1", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 10)),
                Criterion("Criterion 2", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 10)),
                Criterion("Criterion 3", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 10)),
            ], [
                Category("Category 1"),
                Category("Category 2"),
            ],
        )
        alternatives = Alternatives(
            problem,
            [
                Alternative("First alternative", [Performance(Performance.Real(5.)), Performance(Performance.Real(5.)), Performance(Performance.Real(5))], 0),
                Alternative("Second alternative", [Performance(Performance.Real(1.)), Performance(Performance.Real(2.)), Performance(Performance.Real(3.))], None),
                Alternative("Third alternative", [Performance(Performance.Real(2.)), Performance(Performance.Real(4.)), Performance(Performance.Real(6.))]),
            ],
        )
        self.assertEqual(len(alternatives.alternatives), 3)

        self.assertEqual(alternatives.alternatives[0].name, "First alternative")
        self.assertEqual(alternatives.alternatives[0].profile[0].real.value, 5.)
        self.assertEqual(alternatives.alternatives[0].profile[1].real.value, 5.)
        self.assertEqual(alternatives.alternatives[0].profile[2].real.value, 5.)
        self.assertEqual(alternatives.alternatives[0].category_index, 0)

        self.assertEqual(alternatives.alternatives[1].name, "Second alternative")
        self.assertEqual(alternatives.alternatives[1].profile[0].real.value, 1.)
        self.assertEqual(alternatives.alternatives[1].profile[1].real.value, 2.)
        self.assertEqual(alternatives.alternatives[1].profile[2].real.value, 3.)
        self.assertIsNone(alternatives.alternatives[1].category_index)

        self.assertEqual(alternatives.alternatives[2].name, "Third alternative")
        self.assertEqual(alternatives.alternatives[2].profile[0].real.value, 2.)
        self.assertEqual(alternatives.alternatives[2].profile[1].real.value, 4.)
        self.assertEqual(alternatives.alternatives[2].profile[2].real.value, 6.)
        self.assertIsNone(alternatives.alternatives[2].category_index)

    def test_init_size_mismatch(self):
        problem = Problem(
            [
                Criterion("Criterion 1", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 1)),
            ], [
                Category("Category 1"),
                Category("Category 2"),
            ],
        )
        with self.assertRaises(DataValidationException) as cm:
            Alternatives(
                problem,
                [
                    Alternative("First alternative", [], 0),
                ],
            )
        self.assertEqual(cm.exception.args[0], "The profile of an alternative must have as many performances as there are criteria in the problem")
        with self.assertRaises(DataValidationException) as cm:
            Alternatives(
                problem,
                [
                    Alternative("First alternative", [Performance(Performance.Real(5.)), Performance(Performance.Real(5.))], 0),
                ],
            )
        self.assertEqual(cm.exception.args[0], "The profile of an alternative must have as many performances as there are criteria in the problem")

    def test_init_type_mismatch(self):
        problem = Problem(
            criteria=[
                Criterion(name="Real criterion", values=Criterion.RealValues(Criterion.PreferenceDirection.increasing, 5, 10)),
                Criterion(name="Integer criterion", values=Criterion.IntegerValues(Criterion.PreferenceDirection.decreasing, 15, 100)),
                Criterion(name="Enumerated criterion", values=Criterion.EnumeratedValues(["a", "b", "c"])),
            ],
            ordered_categories=[
                Category("Bad"),
                Category("Medium"),
                Category("Good"),
            ],
        )

        Alternatives(problem, [Alternative("Name", [Performance(Performance.Real(8.)), Performance(Performance.Integer(25)), Performance(Performance.Enumerated("a"))], 0)])
        with self.assertRaises(DataValidationException) as cm:
            Alternatives(problem, [Alternative("Name", [Performance(Performance.Integer(8)), Performance(Performance.Integer(25)), Performance(Performance.Enumerated("a"))], 0)])
        self.assertEqual(cm.exception.args[0], "The type of the performance of an alternative must match the type of the real-valued criterion in the problem")
        with self.assertRaises(DataValidationException) as cm:
            Alternatives(problem, [Alternative("Name", [Performance(Performance.Real(8.)), Performance(Performance.Real(25)), Performance(Performance.Enumerated("a"))], 0)])
        self.assertEqual(cm.exception.args[0], "The type of the performance of an alternative must match the type of the integer-valued criterion in the problem")
        with self.assertRaises(DataValidationException) as cm:
            Alternatives(problem, [Alternative("Name", [Performance(Performance.Real(8.)), Performance(Performance.Integer(25)), Performance(Performance.Integer(42))], 0)])
        self.assertEqual(cm.exception.args[0], "The type of the performance of an alternative must match the type of the enumerated criterion in the problem")

    def test_pickle_and_deep_copy(self):
        problem = Problem(
            criteria=[
                Criterion(name="Real criterion", values=Criterion.RealValues(Criterion.PreferenceDirection.increasing, 5, 10)),
                Criterion(name="Integer criterion", values=Criterion.IntegerValues(Criterion.PreferenceDirection.decreasing, 15, 100)),
                Criterion(name="Enumerated criterion", values=Criterion.EnumeratedValues(["a", "b", "c"])),
            ],
            ordered_categories=[
                Category("Bad"),
                Category("Medium"),
                Category("Good"),
            ],
        )
        alternatives = Alternatives(
            problem,
            [
                Alternative("First alternative", [Performance(Performance.Real(8.)), Performance(Performance.Integer(25)), Performance(Performance.Enumerated("a"))], 0),
                Alternative("Second alternative", [Performance(Performance.Real(6.)), Performance(Performance.Integer(50)), Performance(Performance.Enumerated("b"))], None),
            ],
        )
        pickled_alternatives = pickle.dumps(alternatives)
        self.assertIsInstance(pickled_alternatives, bytes)
        unpickled_alternatives = pickle.loads(pickled_alternatives)
        copied_alternatives = copy.deepcopy(alternatives)

        for a in (alternatives, unpickled_alternatives, copied_alternatives):
            self.assertEqual(len(a.alternatives), 2)
            self.assertEqual(a.alternatives[0].name, "First alternative")
            self.assertEqual(a.alternatives[0].profile[0].real.value, 8.)
            self.assertEqual(a.alternatives[0].profile[1].integer.value, 25)
            self.assertEqual(a.alternatives[0].profile[2].enumerated.value, "a")
            self.assertEqual(a.alternatives[0].category_index, 0)
            self.assertEqual(a.alternatives[1].name, "Second alternative")
            self.assertEqual(a.alternatives[1].profile[0].real.value, 6.)
            self.assertEqual(a.alternatives[1].profile[1].integer.value, 50)
            self.assertEqual(a.alternatives[1].profile[2].enumerated.value, "b")
            self.assertIsNone(a.alternatives[1].category_index)

        for a in (unpickled_alternatives, copied_alternatives):
            self.assertIsNot(a, alternatives)
            self.assertIsNot(a.alternatives, alternatives.alternatives)
            self.assertIsNot(a.alternatives[0], alternatives.alternatives[0])
            self.assertIsNot(a.alternatives[1], alternatives.alternatives[1])

    def test_init_out_of_range(self):
        problem = Problem(
            criteria=[
                Criterion(name="Real criterion", values=Criterion.RealValues(Criterion.PreferenceDirection.increasing, 5, 10)),
                Criterion(name="Integer criterion", values=Criterion.IntegerValues(Criterion.PreferenceDirection.decreasing, 15, 100)),
                Criterion(name="Enumerated criterion", values=Criterion.EnumeratedValues(["a", "b", "c"])),
            ],
            ordered_categories=[
                Category("Bad"),
                Category("Medium"),
                Category("Good"),
            ],
        )

        Alternatives(problem, [Alternative("Name", [Performance(Performance.Real(8.)), Performance(Performance.Integer(25)), Performance(Performance.Enumerated("a"))], 0)])
        with self.assertRaises(DataValidationException) as cm:
            Alternatives(problem, [Alternative("Name", [Performance(Performance.Real(3.)), Performance(Performance.Integer(25)), Performance(Performance.Enumerated("a"))], 0)])
        self.assertEqual(cm.exception.args[0], "The performance of an alternative must be between the min and max values for the real-valued criterion in the problem")
        with self.assertRaises(DataValidationException) as cm:
            Alternatives(problem, [Alternative("Name", [Performance(Performance.Real(11.)), Performance(Performance.Integer(25)), Performance(Performance.Enumerated("a"))], 0)])
        self.assertEqual(cm.exception.args[0], "The performance of an alternative must be between the min and max values for the real-valued criterion in the problem")
        with self.assertRaises(DataValidationException) as cm:
            Alternatives(problem, [Alternative("Name", [Performance(Performance.Real(8.)), Performance(Performance.Integer(10)), Performance(Performance.Enumerated("a"))], 0)])
        self.assertEqual(cm.exception.args[0], "The performance of an alternative must be between the min and max values for the integer-valued criterion in the problem")
        with self.assertRaises(DataValidationException) as cm:
            Alternatives(problem, [Alternative("Name", [Performance(Performance.Real(8.)), Performance(Performance.Integer(110)), Performance(Performance.Enumerated("a"))], 0)])
        self.assertEqual(cm.exception.args[0], "The performance of an alternative must be between the min and max values for the integer-valued criterion in the problem")
        with self.assertRaises(DataValidationException) as cm:
            Alternatives(problem, [Alternative("Name", [Performance(Performance.Real(8.)), Performance(Performance.Integer(25)), Performance(Performance.Enumerated("d"))], 0)])
        self.assertEqual(cm.exception.args[0], "The performance of an alternative must be int the enumerated values for a criterion in the problem")


class LearningTestCase(unittest.TestCase):
    def test_access_preprocessed_learning_set(self):
        problem = generate_problem(5, 3, 41, allowed_preference_directions=[Criterion.PreferenceDirection.increasing, Criterion.PreferenceDirection.single_peaked])
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)

        preprocessed_learning_set = PreprocessedLearningSet(problem, learning_set)

        self.assertEqual(preprocessed_learning_set.criteria_count, 5)

        self.assertEqual(preprocessed_learning_set.categories_count, 3)

        self.assertEqual(preprocessed_learning_set.boundaries_count, 2)

        self.assertEqual(preprocessed_learning_set.alternatives_count, 200)

        self.assertEqual(len(preprocessed_learning_set.values_counts), 5)
        self.assertEqual(preprocessed_learning_set.values_counts[0], 202)
        self.assertEqual(list(preprocessed_learning_set.values_counts), [202, 202, 202, 202, 202])

        self.assertEqual(len(preprocessed_learning_set.performance_ranks), 5)
        self.assertEqual(len(preprocessed_learning_set.performance_ranks[0]), 200)
        self.assertEqual(preprocessed_learning_set.performance_ranks[0][0], 24)

        self.assertEqual(len(preprocessed_learning_set.assignments), 200)
        self.assertEqual(preprocessed_learning_set.assignments[0], 0)

        self.assertEqual(list(preprocessed_learning_set.single_peaked), [False, True, False, False, True])

    def test_access_wpb_models_being_learned(self):
        problem = generate_problem(5, 3, 41, allowed_preference_directions=[Criterion.PreferenceDirection.increasing, Criterion.PreferenceDirection.single_peaked])
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)

        models_being_learned = LearnMrsortByWeightsProfilesBreed.ModelsBeingLearned(PreprocessedLearningSet(problem, learning_set), 9, 44)

        self.assertEqual(models_being_learned.models_count, 9)

        self.assertEqual(len(models_being_learned.random_generators), 9)
        self.assertEqual(models_being_learned.random_generators[0](), 3585619732)

        self.assertEqual(models_being_learned.iteration_index, 0)

        self.assertEqual(len(models_being_learned.model_indexes), 9)
        self.assertEqual(models_being_learned.model_indexes[0], 0)

        self.assertEqual(len(models_being_learned.accuracies), 9)
        self.assertEqual(models_being_learned.accuracies[0], 0)

        self.assertEqual(len(models_being_learned.low_profile_ranks), 9)
        self.assertEqual(len(models_being_learned.low_profile_ranks[0]), 2)
        self.assertEqual(len(models_being_learned.low_profile_ranks[0][0]), 5)
        self.assertIsInstance(models_being_learned.low_profile_ranks[0][0][0], int)

        self.assertEqual(len(models_being_learned.high_profile_rank_indexes), 5)
        self.assertEqual(models_being_learned.high_profile_rank_indexes[1], 0)
        self.assertEqual(models_being_learned.high_profile_rank_indexes[4], 1)

        self.assertEqual(len(models_being_learned.high_profile_ranks), 9)
        self.assertEqual(len(models_being_learned.high_profile_ranks[0]), 2)
        self.assertEqual(len(models_being_learned.high_profile_ranks[0][0]), 2)
        self.assertIsInstance(models_being_learned.high_profile_ranks[0][0][0], int)

        self.assertEqual(len(models_being_learned.weights), 9)
        self.assertEqual(len(models_being_learned.weights[0]), 5)
        self.assertIsInstance(models_being_learned.weights[0][0], float)

    def test_basic_mrsort_learning(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)

        preprocessed_learning_set = PreprocessedLearningSet(problem, learning_set)
        models_being_learned = LearnMrsortByWeightsProfilesBreed.ModelsBeingLearned(preprocessed_learning_set, 9, 44)
        profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(preprocessed_learning_set, models_being_learned)
        weights_optimization_strategy = OptimizeWeightsUsingGlop(preprocessed_learning_set, models_being_learned)
        profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(preprocessed_learning_set, models_being_learned)
        breeding_strategy = ReinitializeLeastAccurate(models_being_learned, profiles_initialization_strategy, 4)
        termination_strategy = TerminateAtAccuracy(models_being_learned, len(learning_set.alternatives))
        learned_model = LearnMrsortByWeightsProfilesBreed(
            preprocessed_learning_set,
            models_being_learned,
            profiles_initialization_strategy,
            weights_optimization_strategy,
            profiles_improvement_strategy,
            breeding_strategy,
            termination_strategy,
        ).perform()

        self.assertEqual(models_being_learned.iteration_index, 8)

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 0)
        self.assertEqual(result.unchanged, 200)

        testing_set = generate_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 29)
        self.assertEqual(result.unchanged, 971)

    def test_learn_with_deleted_strategies(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)

        best_accuracies = []

        class MyObserver(LearnMrsortByWeightsProfilesBreed.Observer):
            def __init__(self, models_being_learned):
                super().__init__()
                self.models_being_learned = models_being_learned

            def after_iteration(self):
                # nonlocal best_accuracies
                best_accuracies.append(self.models_being_learned.get_best_accuracy())

            def before_return(self):
                best_accuracies.append(self.models_being_learned.get_best_accuracy())

        # This test is about a bug where strategy objects were garbage-collected before
        # the learning was 'perform'ed, causing a crash.
        preprocessed_learning_set = PreprocessedLearningSet(problem, learning_set)
        models_being_learned = LearnMrsortByWeightsProfilesBreed.ModelsBeingLearned(preprocessed_learning_set, 9, 44)
        profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(preprocessed_learning_set, models_being_learned)
        weights_optimization_strategy = OptimizeWeightsUsingGlop(preprocessed_learning_set, models_being_learned)
        profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(preprocessed_learning_set, models_being_learned)
        breeding_strategy = ReinitializeLeastAccurate(models_being_learned, profiles_initialization_strategy, 4)
        termination_strategy = TerminateAtAccuracy(models_being_learned, len(learning_set.alternatives))
        observer = MyObserver(models_being_learned)
        observers = [observer]
        learning = LearnMrsortByWeightsProfilesBreed(
            preprocessed_learning_set,
            models_being_learned,
            profiles_initialization_strategy,
            weights_optimization_strategy,
            profiles_improvement_strategy,
            breeding_strategy,
            termination_strategy,
            observers,
        )

        del preprocessed_learning_set
        del models_being_learned
        del profiles_initialization_strategy
        del weights_optimization_strategy
        del profiles_improvement_strategy
        del breeding_strategy
        del termination_strategy
        del observer
        del observers

        learning.perform()

        self.assertEqual(best_accuracies, [182, 192, 192, 193, 193, 194, 193, 199, 200])

    def test_iterations_restricted_mrsort_learning(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 1000, 43)

        preprocessed_learning_set = PreprocessedLearningSet(problem, learning_set)
        models_being_learned = LearnMrsortByWeightsProfilesBreed.ModelsBeingLearned(preprocessed_learning_set, 9, 44)
        profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(preprocessed_learning_set, models_being_learned)
        weights_optimization_strategy = OptimizeWeightsUsingGlop(preprocessed_learning_set, models_being_learned)
        profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(preprocessed_learning_set, models_being_learned)
        breeding_strategy = ReinitializeLeastAccurate(models_being_learned, profiles_initialization_strategy, 4)
        termination_strategy = TerminateAfterIterations(models_being_learned, 1)
        learned_model = LearnMrsortByWeightsProfilesBreed(
            preprocessed_learning_set,
            models_being_learned,
            profiles_initialization_strategy,
            weights_optimization_strategy,
            profiles_improvement_strategy,
            breeding_strategy,
            termination_strategy,
        ).perform()

        self.assertGreater(classify_alternatives(problem, learned_model, learning_set).changed, 0)

    def test_terminate_when_any(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)

        class MyTerminationStrategy(LearnMrsortByWeightsProfilesBreed.TerminationStrategy):
            def __init__(self):
                super().__init__()
                self.called_count = 0

            def terminate(self):
                self.called_count += 1
                return self.called_count == 6

        preprocessed_learning_set = PreprocessedLearningSet(problem, learning_set)
        models_being_learned = LearnMrsortByWeightsProfilesBreed.ModelsBeingLearned(preprocessed_learning_set, 9, 44)
        profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(preprocessed_learning_set, models_being_learned)
        weights_optimization_strategy = OptimizeWeightsUsingGlop(preprocessed_learning_set, models_being_learned)
        profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(preprocessed_learning_set, models_being_learned)
        breeding_strategy = ReinitializeLeastAccurate(models_being_learned, profiles_initialization_strategy, 4)
        my_termination_strategy = MyTerminationStrategy()
        termination_strategy = TerminateWhenAny([my_termination_strategy, TerminateAtAccuracy(models_being_learned, len(learning_set.alternatives))])
        learned_model = LearnMrsortByWeightsProfilesBreed(
            preprocessed_learning_set,
            models_being_learned,
            profiles_initialization_strategy,
            weights_optimization_strategy,
            profiles_improvement_strategy,
            breeding_strategy,
            termination_strategy,
        ).perform()

        self.assertEqual(my_termination_strategy.called_count, 6)
        self.assertEqual(classify_alternatives(problem, learned_model, learning_set).changed, 6)

    def test_python_strategies(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)

        class MyProfileInitializationStrategy(LearnMrsortByWeightsProfilesBreed.ProfilesInitializationStrategy):
            def __init__(self, preprocessed_learning_set, models_being_learned):
                super().__init__()
                self.strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(preprocessed_learning_set, models_being_learned)
                self.called_count = 0

            def initialize_profiles(self, begin, end):
                self.called_count += 1
                return self.strategy.initialize_profiles(begin, end)

        class MyWeightsOptimizationStrategy(LearnMrsortByWeightsProfilesBreed.WeightsOptimizationStrategy):
            def __init__(self, preprocessed_learning_set, models_being_learned):
                super().__init__()
                self.strategy = OptimizeWeightsUsingGlop(preprocessed_learning_set, models_being_learned)
                self.called_count = 0

            def optimize_weights(self, begin, end):
                self.called_count += 1
                return self.strategy.optimize_weights(begin, end)

        class MyProfilesImprovementStrategy(LearnMrsortByWeightsProfilesBreed.ProfilesImprovementStrategy):
            def __init__(self, preprocessed_learning_set, models_being_learned):
                super().__init__()
                self.strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(preprocessed_learning_set, models_being_learned)
                self.called_count = 0

            def improve_profiles(self, begin, end):
                self.called_count += 1
                return self.strategy.improve_profiles(begin, end)

        class MyBreedingStrategy(LearnMrsortByWeightsProfilesBreed.BreedingStrategy):
            def __init__(self, models_being_learned, profiles_initialization_strategy, count):
                super().__init__()
                self.strategy = ReinitializeLeastAccurate(models_being_learned, profiles_initialization_strategy, count)
                self.called_count = 0

            def breed(self):
                self.called_count += 1
                return self.strategy.breed()

        class MyTerminationStrategy(LearnMrsortByWeightsProfilesBreed.TerminationStrategy):
            def __init__(self, models_being_learned):
                super().__init__()
                self.models_being_learned = models_being_learned
                self.accuracies = []

            def terminate(self):
                self.accuracies.append(models_being_learned.get_best_accuracy())
                return len(self.accuracies) == 2

        preprocessed_learning_set = PreprocessedLearningSet(problem, learning_set)
        models_being_learned = LearnMrsortByWeightsProfilesBreed.ModelsBeingLearned(preprocessed_learning_set, 9, 44)
        profiles_initialization_strategy = MyProfileInitializationStrategy(preprocessed_learning_set, models_being_learned)
        weights_optimization_strategy = MyWeightsOptimizationStrategy(preprocessed_learning_set, models_being_learned)
        profiles_improvement_strategy = MyProfilesImprovementStrategy(preprocessed_learning_set, models_being_learned)
        breeding_strategy = MyBreedingStrategy(models_being_learned, profiles_initialization_strategy, 4)
        termination_strategy = MyTerminationStrategy(models_being_learned)
        learned_model = LearnMrsortByWeightsProfilesBreed(
            preprocessed_learning_set,
            models_being_learned,
            profiles_initialization_strategy,
            weights_optimization_strategy,
            profiles_improvement_strategy,
            breeding_strategy,
            termination_strategy,
        ).perform()

        self.assertEqual(profiles_initialization_strategy.called_count, 2)
        self.assertEqual(weights_optimization_strategy.called_count, 2)
        self.assertEqual(profiles_improvement_strategy.called_count, 2)
        self.assertEqual(breeding_strategy.called_count, 1)
        self.assertEqual(termination_strategy.accuracies, [182, 192])

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 8)
        self.assertEqual(result.unchanged, 192)

        testing_set = generate_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 68)
        self.assertEqual(result.unchanged, 932)

    def test_silly_strategies(self):
        class SillyProfilesInitializationStrategy(LearnMrsortByWeightsProfilesBreed.ProfilesInitializationStrategy):
            def __init__(self, log, preprocessed_learning_set, models_being_learned):
                super().__init__()
                self.log = log
                self.preprocessed_learning_set = preprocessed_learning_set
                self.models_being_learned = models_being_learned

            def initialize_profiles(self, model_indexes_begin, model_indexes_end):
                self.log.append(("initialize_profiles", model_indexes_begin, model_indexes_end))
                for model_index_index in range(model_indexes_begin, model_indexes_end):
                    model_index = models_being_learned.model_indexes[model_index_index]
                    for boundary_index in range(self.preprocessed_learning_set.boundaries_count):
                        for criterion_index in range(self.preprocessed_learning_set.criteria_count):
                            self.models_being_learned.low_profile_ranks[model_index][boundary_index][criterion_index] = 0

        class SillyWeightsOptimizationStrategy(LearnMrsortByWeightsProfilesBreed.WeightsOptimizationStrategy):
            def __init__(self, log, preprocessed_learning_set, models_being_learned):
                super().__init__()
                self.log = log
                self.preprocessed_learning_set = preprocessed_learning_set
                self.models_being_learned = models_being_learned

            def optimize_weights(self, model_indexes_begin, model_indexes_end):
                self.log.append(("optimize_weights", model_indexes_begin, model_indexes_end))
                for model_index_index in range(model_indexes_begin, model_indexes_end):
                    model_index = models_being_learned.model_indexes[model_index_index]
                    for criterion_index in range(self.preprocessed_learning_set.criteria_count):
                        self.models_being_learned.weights[model_index][criterion_index] = 1.1 / self.preprocessed_learning_set.criteria_count

        class SillyProfilesImprovementStrategy(LearnMrsortByWeightsProfilesBreed.ProfilesImprovementStrategy):
            def __init__(self, log, preprocessed_learning_set, models_being_learned):
                super().__init__()
                self.log = log
                self.preprocessed_learning_set = preprocessed_learning_set
                self.models_being_learned = models_being_learned

            def improve_profiles(self, model_indexes_begin, model_indexes_end):
                self.log.append(("improve_profiles", model_indexes_begin, model_indexes_end))
                for model_index_index in range(model_indexes_begin, model_indexes_end):
                    model_index = models_being_learned.model_indexes[model_index_index]
                    for boundary_index in range(self.preprocessed_learning_set.boundaries_count):
                        for criterion_index in range(self.preprocessed_learning_set.criteria_count):
                            rank = (boundary_index + 1) * (self.preprocessed_learning_set.values_counts[criterion_index] // (self.preprocessed_learning_set.boundaries_count + 1))
                            self.models_being_learned.low_profile_ranks[model_index][boundary_index][criterion_index] = rank

        class SillyBreedingStrategy(LearnMrsortByWeightsProfilesBreed.BreedingStrategy):
            def __init__(self, log):
                super().__init__()
                self.log = log

            def breed(self):
                self.log.append(("breed",))

        class SillyTerminationStrategy(LearnMrsortByWeightsProfilesBreed.TerminationStrategy):
            def __init__(self, log, models_being_learned):
                super().__init__()
                self.log = log
                self.models_being_learned = models_being_learned

            def terminate(self):
                self.log.append(("terminate",))
                return self.models_being_learned.iteration_index == 3

        problem = Problem(
            [
                Criterion("Criterion 1", Criterion.RealValues(Criterion.PreferenceDirection.decreasing, 0, 10)),
                Criterion("Criterion 2", Criterion.IntegerValues(Criterion.PreferenceDirection.increasing, 0, 100)),
                Criterion("Criterion 3", Criterion.EnumeratedValues(["F", "E", "D", "C", "B", "A"])),
            ],
            [Category("Bad"), Category("Medium"), Category("Good")],
        )
        learning_set = generate_alternatives(problem, generate_mrsort_model(problem, random_seed=42), alternatives_count=1000, random_seed=43)

        log = []
        preprocessed_learning_set = PreprocessedLearningSet(problem, learning_set)
        models_being_learned = LearnMrsortByWeightsProfilesBreed.ModelsBeingLearned(preprocessed_learning_set, models_count=9, random_seed=43)
        profiles_initialization_strategy = SillyProfilesInitializationStrategy(log, preprocessed_learning_set, models_being_learned)
        weights_optimization_strategy = SillyWeightsOptimizationStrategy(log, preprocessed_learning_set, models_being_learned)
        profiles_improvement_strategy = SillyProfilesImprovementStrategy(log, preprocessed_learning_set, models_being_learned)
        breeding_strategy = SillyBreedingStrategy(log)
        termination_strategy = SillyTerminationStrategy(log, models_being_learned)

        learned_model = LearnMrsortByWeightsProfilesBreed(
            preprocessed_learning_set,
            models_being_learned,
            profiles_initialization_strategy,
            weights_optimization_strategy,
            profiles_improvement_strategy,
            breeding_strategy,
            termination_strategy,
        ).perform()

        self.assertEqual(log, [
            ("initialize_profiles", 0, 9),
            ("optimize_weights", 0, 9),
            ("improve_profiles", 0, 9),
            ("terminate",),
            ("breed",),
            ("optimize_weights", 0, 9),
            ("improve_profiles", 0, 9),
            ("terminate",),
            ("breed",),
            ("optimize_weights", 0, 9),
            ("improve_profiles", 0, 9),
            ("terminate",),
            ("breed",),
            ("optimize_weights", 0, 9),
            ("improve_profiles", 0, 9),
            ("terminate",),
        ])

        self.assertAlmostEqual(learned_model.accepted_values[0].real_thresholds.thresholds[0], 6.9493637)
        self.assertAlmostEqual(learned_model.accepted_values[0].real_thresholds.thresholds[1], 3.2469211)
        self.assertEqual(learned_model.accepted_values[1].integer_thresholds.thresholds[0], 33)
        self.assertEqual(learned_model.accepted_values[1].integer_thresholds.thresholds[1], 66)
        self.assertEqual(learned_model.accepted_values[2].enumerated_thresholds.thresholds[0], "D")
        self.assertEqual(learned_model.accepted_values[2].enumerated_thresholds.thresholds[1], "B")
        self.assertAlmostEqual(learned_model.sufficient_coalitions[0].weights.criterion_weights[0], 1.1 / 3)
        self.assertAlmostEqual(learned_model.sufficient_coalitions[0].weights.criterion_weights[1], 1.1 / 3)
        self.assertAlmostEqual(learned_model.sufficient_coalitions[0].weights.criterion_weights[2], 1.1 / 3)
        self.assertAlmostEqual(learned_model.sufficient_coalitions[1].weights.criterion_weights[0], 1.1 / 3)
        self.assertAlmostEqual(learned_model.sufficient_coalitions[1].weights.criterion_weights[1], 1.1 / 3)
        self.assertAlmostEqual(learned_model.sufficient_coalitions[1].weights.criterion_weights[2], 1.1 / 3)

    def test_profiles_initialization_strategy_that_does_not_support_single_peaked_criteria(self):
        class MyProfileInitializationStrategy(LearnMrsortByWeightsProfilesBreed.ProfilesInitializationStrategy):
            def __init__(self, preprocessed_learning_set, models_being_learned):
                super().__init__()
                self.strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(preprocessed_learning_set, models_being_learned)

            def initialize_profiles(self, begin, end):
                return self.strategy.initialize_profiles(begin, end)

        def make_learning(problem, learning_set):
            preprocessed_learning_set = PreprocessedLearningSet(problem, learning_set)
            models_being_learned = LearnMrsortByWeightsProfilesBreed.ModelsBeingLearned(preprocessed_learning_set, 9, 44)
            profiles_initialization_strategy = MyProfileInitializationStrategy(preprocessed_learning_set, models_being_learned)
            weights_optimization_strategy = OptimizeWeightsUsingGlop(preprocessed_learning_set, models_being_learned)
            profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(preprocessed_learning_set, models_being_learned)
            breeding_strategy = ReinitializeLeastAccurate(models_being_learned, profiles_initialization_strategy, 4)
            termination_strategy = TerminateAtAccuracy(models_being_learned, len(learning_set.alternatives))
            return LearnMrsortByWeightsProfilesBreed(
                preprocessed_learning_set,
                models_being_learned,
                profiles_initialization_strategy,
                weights_optimization_strategy,
                profiles_improvement_strategy,
                breeding_strategy,
                termination_strategy,
            )

        problem = generate_problem(5, 3, 41, allowed_preference_directions=[Criterion.PreferenceDirection.increasing])
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)
        learning = make_learning(problem, learning_set)
        learning.perform()

        problem = generate_problem(5, 3, 41, allowed_preference_directions=[Criterion.PreferenceDirection.single_peaked])
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)
        learning = make_learning(problem, learning_set)

        with self.assertRaises(LearningFailureException) as cm:
            learning.perform()
        self.assertEqual(cm.exception.args[0], "This profiles initialization strategy doesn't support single-peaked criteria.")

    def test_profiles_initialization_strategy_that_does_support_single_peaked_criteria(self):
        class MyProfileInitializationStrategy(LearnMrsortByWeightsProfilesBreed.ProfilesInitializationStrategy):
            def __init__(self, preprocessed_learning_set, models_being_learned):
                super().__init__(supports_single_peaked_criteria=True)
                self.strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(preprocessed_learning_set, models_being_learned)

            def initialize_profiles(self, begin, end):
                return self.strategy.initialize_profiles(begin, end)

        def make_learning(problem, learning_set):
            preprocessed_learning_set = PreprocessedLearningSet(problem, learning_set)
            models_being_learned = LearnMrsortByWeightsProfilesBreed.ModelsBeingLearned(preprocessed_learning_set, 9, 44)
            profiles_initialization_strategy = MyProfileInitializationStrategy(preprocessed_learning_set, models_being_learned)
            weights_optimization_strategy = OptimizeWeightsUsingGlop(preprocessed_learning_set, models_being_learned)
            profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(preprocessed_learning_set, models_being_learned)
            breeding_strategy = ReinitializeLeastAccurate(models_being_learned, profiles_initialization_strategy, 4)
            termination_strategy = TerminateAtAccuracy(models_being_learned, len(learning_set.alternatives))
            return LearnMrsortByWeightsProfilesBreed(
                preprocessed_learning_set,
                models_being_learned,
                profiles_initialization_strategy,
                weights_optimization_strategy,
                profiles_improvement_strategy,
                breeding_strategy,
                termination_strategy,
            )

        problem = generate_problem(5, 3, 41, allowed_preference_directions=[Criterion.PreferenceDirection.increasing])
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)
        learning = make_learning(problem, learning_set)
        learning.perform()

        problem = generate_problem(5, 3, 41, allowed_preference_directions=[Criterion.PreferenceDirection.single_peaked])
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)
        learning = make_learning(problem, learning_set)
        learning.perform()

    def test_weights_optimization_strategy_that_does_not_support_single_peaked_criteria(self):
        class MyWeightsOptimizationStrategy(LearnMrsortByWeightsProfilesBreed.WeightsOptimizationStrategy):
            def __init__(self, preprocessed_learning_set, models_being_learned):
                super().__init__()
                self.strategy = OptimizeWeightsUsingGlop(preprocessed_learning_set, models_being_learned)

            def optimize_weights(self, begin, end):
                return self.strategy.optimize_weights(begin, end)

        def make_learning(problem, learning_set):
            preprocessed_learning_set = PreprocessedLearningSet(problem, learning_set)
            models_being_learned = LearnMrsortByWeightsProfilesBreed.ModelsBeingLearned(preprocessed_learning_set, 9, 44)
            profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(preprocessed_learning_set, models_being_learned)
            weights_optimization_strategy = MyWeightsOptimizationStrategy(preprocessed_learning_set, models_being_learned)
            profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(preprocessed_learning_set, models_being_learned)
            breeding_strategy = ReinitializeLeastAccurate(models_being_learned, profiles_initialization_strategy, 4)
            termination_strategy = TerminateAtAccuracy(models_being_learned, len(learning_set.alternatives))
            return LearnMrsortByWeightsProfilesBreed(
                preprocessed_learning_set,
                models_being_learned,
                profiles_initialization_strategy,
                weights_optimization_strategy,
                profiles_improvement_strategy,
                breeding_strategy,
                termination_strategy,
            )

        problem = generate_problem(5, 3, 41, allowed_preference_directions=[Criterion.PreferenceDirection.increasing])
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)
        learning = make_learning(problem, learning_set)
        learning.perform()

        problem = generate_problem(5, 3, 41, allowed_preference_directions=[Criterion.PreferenceDirection.single_peaked])
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)
        learning = make_learning(problem, learning_set)

        with self.assertRaises(LearningFailureException) as cm:
            learning.perform()
        self.assertEqual(cm.exception.args[0], "This weights optimization strategy doesn't support single-peaked criteria.")

    def test_weights_optimization_strategy_that_does_support_single_peaked_criteria(self):
        class MyWeightsOptimizationStrategy(LearnMrsortByWeightsProfilesBreed.WeightsOptimizationStrategy):
            def __init__(self, preprocessed_learning_set, models_being_learned):
                super().__init__(True)
                self.strategy = OptimizeWeightsUsingGlop(preprocessed_learning_set, models_being_learned)

            def optimize_weights(self, begin, end):
                return self.strategy.optimize_weights(begin, end)

        def make_learning(problem, learning_set):
            preprocessed_learning_set = PreprocessedLearningSet(problem, learning_set)
            models_being_learned = LearnMrsortByWeightsProfilesBreed.ModelsBeingLearned(preprocessed_learning_set, 9, 44)
            profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(preprocessed_learning_set, models_being_learned)
            weights_optimization_strategy = MyWeightsOptimizationStrategy(preprocessed_learning_set, models_being_learned)
            profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(preprocessed_learning_set, models_being_learned)
            breeding_strategy = ReinitializeLeastAccurate(models_being_learned, profiles_initialization_strategy, 4)
            termination_strategy = TerminateAtAccuracy(models_being_learned, len(learning_set.alternatives))
            return LearnMrsortByWeightsProfilesBreed(
                preprocessed_learning_set,
                models_being_learned,
                profiles_initialization_strategy,
                weights_optimization_strategy,
                profiles_improvement_strategy,
                breeding_strategy,
                termination_strategy,
            )

        problem = generate_problem(5, 3, 41, allowed_preference_directions=[Criterion.PreferenceDirection.increasing])
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)
        learning = make_learning(problem, learning_set)
        learning.perform()

        problem = generate_problem(5, 3, 41, allowed_preference_directions=[Criterion.PreferenceDirection.single_peaked])
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)
        learning = make_learning(problem, learning_set)
        learning.perform()

    def test_profiles_improvement_strategy_that_does_not_support_single_peaked_criteria(self):
        class MyProfilesImprovementStrategy(LearnMrsortByWeightsProfilesBreed.ProfilesImprovementStrategy):
            def __init__(self, preprocessed_learning_set, models_being_learned):
                super().__init__()
                self.strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(preprocessed_learning_set, models_being_learned)

            def improve_profiles(self, begin, end):
                return self.strategy.improve_profiles(begin, end)

        def make_learning(problem, learning_set):
            preprocessed_learning_set = PreprocessedLearningSet(problem, learning_set)
            models_being_learned = LearnMrsortByWeightsProfilesBreed.ModelsBeingLearned(preprocessed_learning_set, 9, 44)
            profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(preprocessed_learning_set, models_being_learned)
            weights_optimization_strategy = OptimizeWeightsUsingGlop(preprocessed_learning_set, models_being_learned)
            profiles_improvement_strategy = MyProfilesImprovementStrategy(preprocessed_learning_set, models_being_learned)
            breeding_strategy = ReinitializeLeastAccurate(models_being_learned, profiles_initialization_strategy, 4)
            termination_strategy = TerminateAtAccuracy(models_being_learned, len(learning_set.alternatives))
            return LearnMrsortByWeightsProfilesBreed(
                preprocessed_learning_set,
                models_being_learned,
                profiles_initialization_strategy,
                weights_optimization_strategy,
                profiles_improvement_strategy,
                breeding_strategy,
                termination_strategy,
            )

        problem = generate_problem(5, 3, 41, allowed_preference_directions=[Criterion.PreferenceDirection.increasing])
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)
        learning = make_learning(problem, learning_set)
        learning.perform()

        problem = generate_problem(5, 3, 41, allowed_preference_directions=[Criterion.PreferenceDirection.single_peaked])
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)
        learning = make_learning(problem, learning_set)

        with self.assertRaises(LearningFailureException) as cm:
            learning.perform()
        self.assertEqual(cm.exception.args[0], "This profiles improvement strategy doesn't support single-peaked criteria.")

    def test_profiles_improvement_strategy_that_does_support_single_peaked_criteria(self):
        class MyProfilesImprovementStrategy(LearnMrsortByWeightsProfilesBreed.ProfilesImprovementStrategy):
            def __init__(self, preprocessed_learning_set, models_being_learned):
                super().__init__(True)
                self.strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(preprocessed_learning_set, models_being_learned)

            def improve_profiles(self, begin, end):
                return self.strategy.improve_profiles(begin, end)

        def make_learning(problem, learning_set):
            preprocessed_learning_set = PreprocessedLearningSet(problem, learning_set)
            models_being_learned = LearnMrsortByWeightsProfilesBreed.ModelsBeingLearned(preprocessed_learning_set, 9, 44)
            profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(preprocessed_learning_set, models_being_learned)
            weights_optimization_strategy = OptimizeWeightsUsingGlop(preprocessed_learning_set, models_being_learned)
            profiles_improvement_strategy = MyProfilesImprovementStrategy(preprocessed_learning_set, models_being_learned)
            breeding_strategy = ReinitializeLeastAccurate(models_being_learned, profiles_initialization_strategy, 4)
            termination_strategy = TerminateAtAccuracy(models_being_learned, len(learning_set.alternatives))
            return LearnMrsortByWeightsProfilesBreed(
                preprocessed_learning_set,
                models_being_learned,
                profiles_initialization_strategy,
                weights_optimization_strategy,
                profiles_improvement_strategy,
                breeding_strategy,
                termination_strategy,
            )

        problem = generate_problem(5, 3, 41, allowed_preference_directions=[Criterion.PreferenceDirection.increasing])
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)
        learning = make_learning(problem, learning_set)
        learning.perform()

        problem = generate_problem(5, 3, 41, allowed_preference_directions=[Criterion.PreferenceDirection.single_peaked])
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)
        learning = make_learning(problem, learning_set)
        learning.perform()

    def test_breeding_strategy_that_does_not_support_single_peaked_criteria(self):
        class MyBreedingStrategy(LearnMrsortByWeightsProfilesBreed.BreedingStrategy):
            def __init__(self, models_being_learned, profiles_initialization_strategy, count):
                super().__init__()
                self.strategy = ReinitializeLeastAccurate(models_being_learned, profiles_initialization_strategy, count)

            def breed(self):
                return self.strategy.breed()

        def make_learning(problem, learning_set):
            preprocessed_learning_set = PreprocessedLearningSet(problem, learning_set)
            models_being_learned = LearnMrsortByWeightsProfilesBreed.ModelsBeingLearned(preprocessed_learning_set, 9, 44)
            profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(preprocessed_learning_set, models_being_learned)
            weights_optimization_strategy = OptimizeWeightsUsingGlop(preprocessed_learning_set, models_being_learned)
            profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(preprocessed_learning_set, models_being_learned)
            breeding_strategy = MyBreedingStrategy(models_being_learned, profiles_initialization_strategy, 4)
            termination_strategy = TerminateAtAccuracy(models_being_learned, len(learning_set.alternatives))
            return LearnMrsortByWeightsProfilesBreed(
                preprocessed_learning_set,
                models_being_learned,
                profiles_initialization_strategy,
                weights_optimization_strategy,
                profiles_improvement_strategy,
                breeding_strategy,
                termination_strategy,
            )

        problem = generate_problem(5, 3, 41, allowed_preference_directions=[Criterion.PreferenceDirection.increasing])
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)
        learning = make_learning(problem, learning_set)
        learning.perform()

        problem = generate_problem(5, 3, 41, allowed_preference_directions=[Criterion.PreferenceDirection.single_peaked])
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)
        learning = make_learning(problem, learning_set)

        with self.assertRaises(LearningFailureException) as cm:
            learning.perform()
        self.assertEqual(cm.exception.args[0], "This breeding strategy doesn't support single-peaked criteria.")

    def test_breeding_strategy_that_does_support_single_peaked_criteria(self):
        class MyBreedingStrategy(LearnMrsortByWeightsProfilesBreed.BreedingStrategy):
            def __init__(self, models_being_learned, profiles_initialization_strategy, count):
                super().__init__(True)
                self.strategy = ReinitializeLeastAccurate(models_being_learned, profiles_initialization_strategy, count)

            def breed(self):
                return self.strategy.breed()

        def make_learning(problem, learning_set):
            preprocessed_learning_set = PreprocessedLearningSet(problem, learning_set)
            models_being_learned = LearnMrsortByWeightsProfilesBreed.ModelsBeingLearned(preprocessed_learning_set, 9, 44)
            profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(preprocessed_learning_set, models_being_learned)
            weights_optimization_strategy = OptimizeWeightsUsingGlop(preprocessed_learning_set, models_being_learned)
            profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(preprocessed_learning_set, models_being_learned)
            breeding_strategy = MyBreedingStrategy(models_being_learned, profiles_initialization_strategy, 4)
            termination_strategy = TerminateAtAccuracy(models_being_learned, len(learning_set.alternatives))
            return LearnMrsortByWeightsProfilesBreed(
                preprocessed_learning_set,
                models_being_learned,
                profiles_initialization_strategy,
                weights_optimization_strategy,
                profiles_improvement_strategy,
                breeding_strategy,
                termination_strategy,
            )

        problem = generate_problem(5, 3, 41, allowed_preference_directions=[Criterion.PreferenceDirection.increasing])
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)
        learning = make_learning(problem, learning_set)
        learning.perform()

        problem = generate_problem(5, 3, 41, allowed_preference_directions=[Criterion.PreferenceDirection.single_peaked])
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)
        learning = make_learning(problem, learning_set)
        learning.perform()


    def test_observers(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)

        class MyObserver(LearnMrsortByWeightsProfilesBreed.Observer):
            def __init__(self, models_being_learned):
                super().__init__()
                self.models_being_learned = models_being_learned
                self.best_accuracies = []
                self.final_accuracy = None

            def after_iteration(self):
                self.best_accuracies.append(self.models_being_learned.get_best_accuracy())

            def before_return(self):
                self.final_accuracy = self.models_being_learned.get_best_accuracy()

        preprocessed_learning_set = PreprocessedLearningSet(problem, learning_set)
        models_being_learned = LearnMrsortByWeightsProfilesBreed.ModelsBeingLearned(preprocessed_learning_set, 9, 44)
        profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(preprocessed_learning_set, models_being_learned)
        weights_optimization_strategy = OptimizeWeightsUsingGlop(preprocessed_learning_set, models_being_learned)
        profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(preprocessed_learning_set, models_being_learned)
        breeding_strategy = ReinitializeLeastAccurate(models_being_learned, profiles_initialization_strategy, 4)
        termination_strategy = TerminateAtAccuracy(models_being_learned, len(learning_set.alternatives))
        observer = MyObserver(models_being_learned)
        LearnMrsortByWeightsProfilesBreed(
            preprocessed_learning_set,
            models_being_learned,
            profiles_initialization_strategy,
            weights_optimization_strategy,
            profiles_improvement_strategy,
            breeding_strategy,
            termination_strategy,
            [observer],
        ).perform()

        self.assertEqual(observer.best_accuracies, [182, 192, 192, 193, 193, 194, 193, 199])
        self.assertEqual(observer.final_accuracy, 200)

    def test_alglib_mrsort_learning(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)

        preprocessed_learning_set = PreprocessedLearningSet(problem, learning_set)
        models_being_learned = LearnMrsortByWeightsProfilesBreed.ModelsBeingLearned(preprocessed_learning_set, 9, 44)
        profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(preprocessed_learning_set, models_being_learned)
        weights_optimization_strategy = OptimizeWeightsUsingAlglib(preprocessed_learning_set, models_being_learned)
        profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(preprocessed_learning_set, models_being_learned)
        breeding_strategy = ReinitializeLeastAccurate(models_being_learned, profiles_initialization_strategy, 4)
        termination_strategy = TerminateAtAccuracy(models_being_learned, len(learning_set.alternatives))
        learned_model = LearnMrsortByWeightsProfilesBreed(
            preprocessed_learning_set,
            models_being_learned,
            profiles_initialization_strategy,
            weights_optimization_strategy,
            profiles_improvement_strategy,
            breeding_strategy,
            termination_strategy,
        ).perform()

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 0)
        self.assertEqual(result.unchanged, 200)

        testing_set = generate_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 21)
        self.assertEqual(result.unchanged, 979)

    @unittest.skipIf(forbid_gpu, "Can't use GPU")
    def test_gpu_mrsort_learning(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)

        preprocessed_learning_set = PreprocessedLearningSet(problem, learning_set)
        models_being_learned = LearnMrsortByWeightsProfilesBreed.ModelsBeingLearned(preprocessed_learning_set, 9, 44)
        profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(preprocessed_learning_set, models_being_learned)
        weights_optimization_strategy = OptimizeWeightsUsingGlop(preprocessed_learning_set, models_being_learned)
        profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnGpu(preprocessed_learning_set, models_being_learned)
        breeding_strategy = ReinitializeLeastAccurate(models_being_learned, profiles_initialization_strategy, 4)
        termination_strategy = TerminateAtAccuracy(models_being_learned, len(learning_set.alternatives))
        learned_model = LearnMrsortByWeightsProfilesBreed(
            preprocessed_learning_set,
            models_being_learned,
            profiles_initialization_strategy,
            weights_optimization_strategy,
            profiles_improvement_strategy,
            breeding_strategy,
            termination_strategy,
        ).perform()

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 0)
        self.assertEqual(result.unchanged, 200)

        testing_set = generate_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 29)
        self.assertEqual(result.unchanged, 971)

    def test_sat_by_coalitions_using_minisat_learning(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)

        learned_model = LearnUcncsBySatByCoalitionsUsingMinisat(problem, learning_set).perform()

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 0)
        self.assertEqual(result.unchanged, 200)

        testing_set = generate_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 26)
        self.assertEqual(result.unchanged, 974)

    def test_sat_by_separation_using_minisat_learning(self):
        problem = generate_problem(5, 2, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)

        learned_model = LearnUcncsBySatBySeparationUsingMinisat(problem, learning_set).perform()

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 0)
        self.assertEqual(result.unchanged, 200)

        testing_set = generate_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 28)
        self.assertEqual(result.unchanged, 972)

    def test_max_sat_by_coalitions_using_evalmaxsat_learning(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)

        learned_model = LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat(problem, learning_set).perform()

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 0)
        self.assertEqual(result.unchanged, 200)

        testing_set = generate_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 32)
        self.assertEqual(result.unchanged, 968)

    def test_max_sat_by_separation_using_evalmaxsat_learning(self):
        problem = generate_problem(5, 2, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)

        learned_model = LearnUcncsByMaxSatBySeparationUsingEvalmaxsat(problem, learning_set).perform()

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 0)
        self.assertEqual(result.unchanged, 200)

        testing_set = generate_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 25)
        self.assertEqual(result.unchanged, 975)

    def test_learning_failure_exception(self):
        problem = generate_problem(2, 2, 42)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 100, 42)
        misclassify_alternatives(problem, learning_set, 10, 42 + 27)

        learning = LearnUcncsBySatByCoalitionsUsingMinisat(problem, learning_set)

        with self.assertRaises(LearningFailureException):
            learned_model = learning.perform()

    def test_bug_found_by_laurent_cabaret_in_real_life_data(self):
        # Previously, in (max-)SAT learning methods, when the SAT solver returned a solution where no value was accepted
        # for a given criterion, we used the maximum value for that criterion (as configured in the problem) as the threshold.
        # This was incorrect, and would in rare occasions cause the resulting model to classify alternatives in a higher
        # category that expected. To solve this issue, we had to allow "unreachable thresholds", materialized by None in
        # Python, and null in YAML. That change broke the public interface of the library, so it required releasing a
        # major version.

        problem = Problem.load(io.StringIO(textwrap.dedent("""\
            kind: classification-problem
            format_version: 1
            criteria:
              - name: a
                value_type: integer
                preference_direction: increasing
                min_value: 0
                max_value: 1
              - name: b
                value_type: integer
                preference_direction: increasing
                min_value: 0
                max_value: 1
              - name: c
                value_type: integer
                preference_direction: increasing
                min_value: 0
                max_value: 1  # Setting to 2 improves accuracy. Why?
              - name: d
                value_type: integer
                preference_direction: increasing
                min_value: 0
                max_value: 1  # Setting to 2 improves accuracy. Why?
              - name: e
                value_type: integer
                preference_direction: increasing
                min_value: 0
                max_value: 4  # Setting to 5 improves accuracy. Why?
              - name: f
                value_type: integer
                preference_direction: increasing
                min_value: 0
                max_value: 15  # Setting to 16 improves accuracy. Why?
              - name: g
                value_type: integer
                preference_direction: increasing
                min_value: 0
                max_value: 15
              - name: h
                value_type: integer
                preference_direction: increasing
                min_value: 0
                max_value: 10
            ordered_categories:
              - name: 4
              - name: 3
              - name: 2
              - name: 1
        """)))

        learning_set = Alternatives.load(problem, io.StringIO(textwrap.dedent("""\
            name,a,b,c,d,e,f,g,h,category
            01,1,1,1,1,4,12,10,7,1
            02,1,1,1,1,0,15,9,9,1
            03,1,1,1,1,4,13,8,7,1
            04,1,1,1,1,4,12,6,5,1
            05,1,1,1,1,1,15,10,5,1
            06,1,1,1,1,4,12,1,5,1
            07,1,1,1,1,4,13,12,10,1
            08,1,1,1,1,4,14,14,10,1
            09,1,1,1,1,0,15,14,10,1
            10,1,1,1,1,4,6,5,6,1
            11,0,1,1,1,3,15,4,5,2
            12,1,1,1,1,4,8,12,10,1
            13,0,1,1,1,4,4,6,7,3
            14,1,1,1,1,3,15,14,10,1
            15,1,0,1,1,4,12,2,0,3
            16,1,1,1,1,4,12,13,4,1
            17,1,1,1,1,0,15,14,7,1
            18,1,1,1,1,4,7,11,7,1
            19,1,1,1,1,4,10,5,5,1
            20,0,1,1,1,4,2,15,7,3
            21,1,1,1,1,4,12,7,7,1
            22,1,1,1,1,0,15,7,7,1
            23,1,1,1,1,4,12,5,8,1
            24,1,0,1,1,2,15,3,1,3
            25,1,0,1,1,4,12,2,2,3
            26,0,1,1,1,4,12,9,9,2
            27,0,1,1,1,4,12,12,10,2
            28,1,1,1,1,3,15,10,9,1
            29,0,1,1,1,4,12,11,9,2
            30,0,1,1,1,4,3,4,7,3
            31,1,1,1,1,0,15,13,5,1
            32,0,1,1,1,4,12,11,4,2
            33,1,1,1,1,3,15,5,8,1
            34,0,1,1,1,4,11,7,7,2
            35,1,0,1,1,2,15,2,5,3
            36,1,1,1,1,4,11,7,7,1
            37,0,1,0,0,4,5,6,5,4
            38,1,1,1,1,4,6,13,8,1
            39,0,0,0,1,2,15,4,7,4
            40,1,0,1,1,2,15,2,1,3
            41,1,1,1,1,4,12,7,8,1
            42,0,1,1,1,4,1,4,4,3
            43,1,0,1,1,4,9,10,9,2
            44,1,0,1,1,4,12,2,7,3
            45,0,1,1,1,4,0,7,5,3
            46,0,1,1,1,3,15,11,3,2
            47,1,1,1,1,4,7,8,4,1
            48,1,0,1,1,4,11,4,1,3
            49,1,0,1,1,4,12,0,0,3
            50,1,0,1,1,3,15,4,1,3
        """)))

        preprocessed_learning_set = PreprocessedLearningSet(problem, learning_set)
        models_being_learned = LearnMrsortByWeightsProfilesBreed.ModelsBeingLearned(preprocessed_learning_set, models_count=9, random_seed=43)
        profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(preprocessed_learning_set, models_being_learned)
        weights_optimization_strategy = OptimizeWeightsUsingGlop(preprocessed_learning_set, models_being_learned)
        profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(preprocessed_learning_set, models_being_learned)
        breeding_strategy = ReinitializeLeastAccurate(models_being_learned, profiles_initialization_strategy=profiles_initialization_strategy, count=4)
        termination_strategy = TerminateAtAccuracy(models_being_learned, target_accuracy=len(learning_set.alternatives))
        model = LearnMrsortByWeightsProfilesBreed(preprocessed_learning_set, models_being_learned, profiles_initialization_strategy, weights_optimization_strategy, profiles_improvement_strategy, breeding_strategy, termination_strategy).perform()
        model_dump = io.StringIO()
        model.dump(problem, model_dump)
        self.assertEqual(model_dump.getvalue(), textwrap.dedent("""\
            kind: ncs-classification-model
            format_version: 1
            accepted_values:
              - kind: thresholds
                thresholds: [0, 1, 1]
              - kind: thresholds
                thresholds: [0, 1, 1]
              - kind: thresholds
                thresholds: [1, 1, 1]
              - kind: thresholds
                thresholds: [1, 1, 1]
              - kind: thresholds
                thresholds: [3, 3, 4]
              - kind: thresholds
                thresholds: [5, 5, 14]
              - kind: thresholds
                thresholds: [3, 9, 9]
              - kind: thresholds
                thresholds: [0, 1, 5]
            sufficient_coalitions:
              - &coalitions
                kind: weights
                criterion_weights: [0.166666836, 0.333332658, 0.166666836, 0.166666836, 0.166666836, 0.166666836, 0.166665822, 0]
              - *coalitions
              - *coalitions
        """))
        learning_set_copy = copy.deepcopy(learning_set)
        classification_result = classify_alternatives(problem, model, learning_set_copy)
        self.assertEqual(classification_result.unchanged, 50)

        model = LearnUcncsBySatBySeparationUsingMinisat(problem, learning_set).perform()
        model_dump = io.StringIO()
        model.dump(problem, model_dump)
        self.assertEqual(model_dump.getvalue(), textwrap.dedent("""\
            kind: ncs-classification-model
            format_version: 1
            accepted_values:
              - kind: thresholds
                thresholds: [0, 0, 1]
              - kind: thresholds
                thresholds: [0, 1, 1]
              - kind: thresholds
                thresholds: [1, 1, 1]
              - kind: thresholds
                thresholds: [1, null, null]
              - kind: thresholds
                thresholds: [0, 3, 3]
              - kind: thresholds
                thresholds: [0, 6, 6]
              - kind: thresholds
                thresholds: [5, 5, 5]
              - kind: thresholds
                thresholds: [0, 8, 10]
            sufficient_coalitions:
              - &coalitions
                kind: roots
                upset_roots:
                  - [0, 1, 2, 4, 5]
                  - [0, 1, 2, 5, 6]
                  - [0, 2, 4, 5, 6, 7]
              - *coalitions
              - *coalitions
        """))
        learning_set_copy = copy.deepcopy(learning_set)
        classification_result = classify_alternatives(problem, model, learning_set_copy)
        self.assertEqual(classification_result.unchanged, 50)

        model = LearnUcncsBySatByCoalitionsUsingMinisat(problem, learning_set).perform()
        model_dump = io.StringIO()
        model.dump(problem, model_dump)
        self.assertEqual(model_dump.getvalue(), textwrap.dedent("""\
            kind: ncs-classification-model
            format_version: 1
            accepted_values:
              - kind: thresholds
                thresholds: [1, 1, 1]
              - kind: thresholds
                thresholds: [0, 1, 1]
              - kind: thresholds
                thresholds: [1, null, null]
              - kind: thresholds
                thresholds: [1, 1, null]
              - kind: thresholds
                thresholds: [3, 3, null]
              - kind: thresholds
                thresholds: [6, 9, null]
              - kind: thresholds
                thresholds: [0, 4, 13]
              - kind: thresholds
                thresholds: [0, 8, null]
            sufficient_coalitions:
              - &coalitions
                kind: roots
                upset_roots:
                  - [0, 1]
                  - [1, 2, 6]
                  - [1, 4, 5, 6]
                  - [1, 2, 4, 5, 7]
                  - [1, 3, 4, 6, 7]
                  - [0, 3, 4, 5, 6, 7]
              - *coalitions
              - *coalitions
        """))
        learning_set_copy = copy.deepcopy(learning_set)
        classification_result = classify_alternatives(problem, model, learning_set_copy)
        self.assertEqual(classification_result.unchanged, 50)

        model = LearnUcncsByMaxSatBySeparationUsingEvalmaxsat(problem, learning_set).perform()
        model_dump = io.StringIO()
        model.dump(problem, model_dump)
        self.assertEqual(model_dump.getvalue(), textwrap.dedent("""\
            kind: ncs-classification-model
            format_version: 1
            accepted_values:
              - kind: thresholds
                thresholds: [1, 1, 1]
              - kind: thresholds
                thresholds: [0, 1, 1]
              - kind: thresholds
                thresholds: [1, 1, null]
              - kind: thresholds
                thresholds: [1, 1, 1]
              - kind: thresholds
                thresholds: [0, 3, 4]
              - kind: thresholds
                thresholds: [0, 6, 10]
              - kind: thresholds
                thresholds: [0, 3, 8]
              - kind: thresholds
                thresholds: [0, 3, 4]
            sufficient_coalitions:
              - &coalitions
                kind: roots
                upset_roots:
                  - [0, 1, 3, 4, 7]
                  - [0, 1, 3, 5, 7]
                  - [0, 2, 3, 4, 5, 6, 7]
                  - [1, 2, 3, 4, 5, 6, 7]
              - *coalitions
              - *coalitions
        """))
        learning_set_copy = copy.deepcopy(learning_set)
        classification_result = classify_alternatives(problem, model, learning_set_copy)
        self.assertEqual(classification_result.unchanged, 50)

        model = LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat(problem, learning_set).perform()
        model_dump = io.StringIO()
        model.dump(problem, model_dump)
        self.assertEqual(model_dump.getvalue(), textwrap.dedent("""\
            kind: ncs-classification-model
            format_version: 1
            accepted_values:
              - kind: thresholds
                thresholds: [1, 1, 1]
              - kind: thresholds
                thresholds: [1, 1, 1]
              - kind: thresholds
                thresholds: [1, null, null]
              - kind: thresholds
                thresholds: [null, null, null]
              - kind: thresholds
                thresholds: [3, null, null]
              - kind: thresholds
                thresholds: [11, 11, null]
              - kind: thresholds
                thresholds: [7, null, null]
              - kind: thresholds
                thresholds: [8, 8, null]
            sufficient_coalitions:
              - &coalitions
                kind: roots
                upset_roots:
                  - [0, 1]
                  - [1, 2, 4]
                  - [1, 3, 4]
                  - [1, 5]
                  - [2, 5]
                  - [3, 5]
                  - [4, 5]
                  - [6]
                  - [7]
              - *coalitions
              - *coalitions
        """))
        learning_set_copy = copy.deepcopy(learning_set)
        classification_result = classify_alternatives(problem, model, learning_set_copy)
        self.assertEqual(classification_result.unchanged, 50)

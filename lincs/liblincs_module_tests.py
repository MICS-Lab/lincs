# Copyright 2023 Vincent Jacques

import unittest
import os

from . import DataValidationException, LearningFailureException
from .classification import *


forbid_gpu = os.environ.get("LINCS_DEV_FORBID_GPU", "false") == "true"

# @todo(Feature, v1.1) Test using named parameters when calling the API (e.g. Criterion(name="Criterion name", ...), Problem(criteria=[...], ...))

class ProblemTestCase(unittest.TestCase):
    def test_init_simplest(self):
        problem = Problem(
            [
                Criterion("Criterion name", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 1)),
            ],
            [
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
        criterion = Criterion("Criterion name", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 1))
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
            problem,
            [AcceptedValues(AcceptedValues.RealThresholds([0.5]))],
            [SufficientCoalitions(SufficientCoalitions.Weights([0.75]))],
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
                    SufficientCoalitions(SufficientCoalitions.Roots(1, [[0]])),
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
                SufficientCoalitions(SufficientCoalitions.Roots(3, [[0, 1], [0, 2]])),
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
                [
                    AcceptedValues(AcceptedValues.RealThresholds([0.5])),
                ],
                [SufficientCoalitions(SufficientCoalitions.Roots(2, [[0]]))],
            )
        self.assertEqual(cm.exception.args[0], "The maximum number of elements in a root in a sufficient coalitions descriptor must be equal to the number of criteria in the problem")
        with self.assertRaises(DataValidationException) as cm:
            Model(
                problem,
                [AcceptedValues(AcceptedValues.RealThresholds([0.5]))],
                [SufficientCoalitions(SufficientCoalitions.Roots(3, [[3]]))],
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
                Criterion("Criterion 1", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 1)),
                Criterion("Criterion 2", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 1)),
                Criterion("Criterion 3", Criterion.RealValues(Criterion.PreferenceDirection.increasing, 0, 1)),
            ], [
                Category("Category 1"),
                Category("Category 2"),
            ],
        )
        alternatives = Alternatives(
            problem,
            [
                Alternative("First alternative", [Performance(Performance.RealPerformance(5.)), Performance(Performance.RealPerformance(5.)), Performance(Performance.RealPerformance(5))], 0),
                Alternative("Second alternative", [Performance(Performance.RealPerformance(1.)), Performance(Performance.RealPerformance(2.)), Performance(Performance.RealPerformance(3.))], None),
                Alternative("Third alternative", [Performance(Performance.RealPerformance(2.)), Performance(Performance.RealPerformance(4.)), Performance(Performance.RealPerformance(6.))]),
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
                    Alternative("First alternative", [Performance(Performance.RealPerformance(5.)), Performance(Performance.RealPerformance(5.))], 0),
                ],
            )
        self.assertEqual(cm.exception.args[0], "The profile of an alternative must have as many performances as there are criteria in the problem")

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
            Alternatives(
                problem,
                [
                    Alternative("First alternative", [Performance(Performance.IntegerPerformance(5))], 0),
                ],
            )
        self.assertEqual(cm.exception.args[0], "The type of the performance of an alternative must match the type of the criterion in the problem")


class LearningTestCase(unittest.TestCase):
    def test_basic_mrsort_learning(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_classified_alternatives(problem, model, 200, 43)

        learning_data = LearnMrsortByWeightsProfilesBreed.LearningData(problem, learning_set, 9, 44)
        profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(learning_data)
        weights_optimization_strategy = OptimizeWeightsUsingGlop(learning_data)
        profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(learning_data)
        breeding_strategy = ReinitializeLeastAccurate(learning_data, profiles_initialization_strategy, 4)
        termination_strategy = TerminateAtAccuracy(learning_data, len(learning_set.alternatives))
        learned_model = LearnMrsortByWeightsProfilesBreed(
            learning_data,
            profiles_initialization_strategy,
            weights_optimization_strategy,
            profiles_improvement_strategy,
            breeding_strategy,
            termination_strategy,
        ).perform()

        self.assertEqual(learning_data.iteration_index, 8)

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 0)
        self.assertEqual(result.unchanged, 200)

        testing_set = generate_classified_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 29)
        self.assertEqual(result.unchanged, 971)

    def test_iterations_restricted_mrsort_learning(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_classified_alternatives(problem, model, 1000, 43)

        learning_data = LearnMrsortByWeightsProfilesBreed.LearningData(problem, learning_set, 9, 44)
        profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(learning_data)
        weights_optimization_strategy = OptimizeWeightsUsingGlop(learning_data)
        profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(learning_data)
        breeding_strategy = ReinitializeLeastAccurate(learning_data, profiles_initialization_strategy, 4)
        termination_strategy = TerminateAfterIterations(learning_data, 1)
        learned_model = LearnMrsortByWeightsProfilesBreed(
            learning_data,
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
        learning_set = generate_classified_alternatives(problem, model, 200, 43)

        class MyTerminationStrategy(LearnMrsortByWeightsProfilesBreed.TerminationStrategy):
            def __init__(self):
                super().__init__()
                self.called_count = 0

            def terminate(self):
                self.called_count += 1
                return self.called_count == 6

        learning_data = LearnMrsortByWeightsProfilesBreed.LearningData(problem, learning_set, 9, 44)
        profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(learning_data)
        weights_optimization_strategy = OptimizeWeightsUsingGlop(learning_data)
        profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(learning_data)
        breeding_strategy = ReinitializeLeastAccurate(learning_data, profiles_initialization_strategy, 4)
        my_termination_strategy = MyTerminationStrategy()
        termination_strategy = TerminateWhenAny([my_termination_strategy, TerminateAtAccuracy(learning_data, len(learning_set.alternatives))])
        learned_model = LearnMrsortByWeightsProfilesBreed(
            learning_data,
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
        learning_set = generate_classified_alternatives(problem, model, 200, 43)

        class MyProfileInitializationStrategy(LearnMrsortByWeightsProfilesBreed.ProfilesInitializationStrategy):
            def __init__(self, learning_data):
                super().__init__()
                self.strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(learning_data)
                self.called_count = 0

            def initialize_profiles(self, begin, end):
                self.called_count += 1
                return self.strategy.initialize_profiles(begin, end)

        class MyWeightsOptimizationStrategy(LearnMrsortByWeightsProfilesBreed.WeightsOptimizationStrategy):
            def __init__(self, learning_data):
                super().__init__()
                self.strategy = OptimizeWeightsUsingGlop(learning_data)
                self.called_count = 0

            def optimize_weights(self):
                self.called_count += 1
                return self.strategy.optimize_weights()

        class MyProfilesImprovementStrategy(LearnMrsortByWeightsProfilesBreed.ProfilesImprovementStrategy):
            def __init__(self, learning_data):
                super().__init__()
                self.strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(learning_data)
                self.called_count = 0

            def improve_profiles(self):
                self.called_count += 1
                return self.strategy.improve_profiles()

        class MyBreedingStrategy(LearnMrsortByWeightsProfilesBreed.BreedingStrategy):
            def __init__(self, learning_data, profiles_initialization_strategy, count):
                super().__init__()
                self.strategy = ReinitializeLeastAccurate(learning_data, profiles_initialization_strategy, count)
                self.called_count = 0

            def breed(self):
                self.called_count += 1
                return self.strategy.breed()

        class MyTerminationStrategy(LearnMrsortByWeightsProfilesBreed.TerminationStrategy):
            def __init__(self, learning_data):
                super().__init__()
                self.learning_data = learning_data
                self.accuracies = []

            def terminate(self):
                self.accuracies.append(learning_data.get_best_accuracy())
                return len(self.accuracies) == 2

        learning_data = LearnMrsortByWeightsProfilesBreed.LearningData(problem, learning_set, 9, 44)
        profiles_initialization_strategy = MyProfileInitializationStrategy(learning_data)
        weights_optimization_strategy = MyWeightsOptimizationStrategy(learning_data)
        profiles_improvement_strategy = MyProfilesImprovementStrategy(learning_data)
        breeding_strategy = MyBreedingStrategy(learning_data, profiles_initialization_strategy, 4)
        termination_strategy = MyTerminationStrategy(learning_data)
        learned_model = LearnMrsortByWeightsProfilesBreed(
            learning_data,
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

        testing_set = generate_classified_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 68)
        self.assertEqual(result.unchanged, 932)

    def test_observers(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_classified_alternatives(problem, model, 200, 43)

        class MyObserver(LearnMrsortByWeightsProfilesBreed.Observer):
            def __init__(self, learning_data):
                super().__init__()
                self.learning_data = learning_data
                self.best_accuracies = []
                self.final_accuracy = None

            def after_iteration(self):
                self.best_accuracies.append(self.learning_data.get_best_accuracy())

            def before_return(self):
                self.final_accuracy = self.learning_data.get_best_accuracy()

        learning_data = LearnMrsortByWeightsProfilesBreed.LearningData(problem, learning_set, 9, 44)
        profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(learning_data)
        weights_optimization_strategy = OptimizeWeightsUsingGlop(learning_data)
        profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(learning_data)
        breeding_strategy = ReinitializeLeastAccurate(learning_data, profiles_initialization_strategy, 4)
        termination_strategy = TerminateAtAccuracy(learning_data, len(learning_set.alternatives))
        observer = MyObserver(learning_data)
        LearnMrsortByWeightsProfilesBreed(
            learning_data,
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
        learning_set = generate_classified_alternatives(problem, model, 200, 43)

        learning_data = LearnMrsortByWeightsProfilesBreed.LearningData(problem, learning_set, 9, 44)
        profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(learning_data)
        weights_optimization_strategy = OptimizeWeightsUsingAlglib(learning_data)
        profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(learning_data)
        breeding_strategy = ReinitializeLeastAccurate(learning_data, profiles_initialization_strategy, 4)
        termination_strategy = TerminateAtAccuracy(learning_data, len(learning_set.alternatives))
        learned_model = LearnMrsortByWeightsProfilesBreed(
            learning_data,
            profiles_initialization_strategy,
            weights_optimization_strategy,
            profiles_improvement_strategy,
            breeding_strategy,
            termination_strategy,
        ).perform()

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 0)
        self.assertEqual(result.unchanged, 200)

        testing_set = generate_classified_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 21)
        self.assertEqual(result.unchanged, 979)

    @unittest.skipIf(forbid_gpu, "Can't use GPU")
    def test_gpu_mrsort_learning(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_classified_alternatives(problem, model, 200, 43)

        learning_data = LearnMrsortByWeightsProfilesBreed.LearningData(problem, learning_set, 9, 44)
        profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(learning_data)
        weights_optimization_strategy = OptimizeWeightsUsingGlop(learning_data)
        profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnGpu(learning_data)
        breeding_strategy = ReinitializeLeastAccurate(learning_data, profiles_initialization_strategy, 4)
        termination_strategy = TerminateAtAccuracy(learning_data, len(learning_set.alternatives))
        learned_model = LearnMrsortByWeightsProfilesBreed(
            learning_data,
            profiles_initialization_strategy,
            weights_optimization_strategy,
            profiles_improvement_strategy,
            breeding_strategy,
            termination_strategy,
        ).perform()

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 0)
        self.assertEqual(result.unchanged, 200)

        testing_set = generate_classified_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 29)
        self.assertEqual(result.unchanged, 971)

    def test_sat_by_coalitions_using_minisat_learning(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_classified_alternatives(problem, model, 200, 43)

        learned_model = LearnUcncsBySatByCoalitionsUsingMinisat(problem, learning_set).perform()

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 0)
        self.assertEqual(result.unchanged, 200)

        testing_set = generate_classified_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 21)
        self.assertEqual(result.unchanged, 979)

    def test_sat_by_separation_using_minisat_learning(self):
        problem = generate_problem(5, 2, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_classified_alternatives(problem, model, 200, 43)

        learned_model = LearnUcncsBySatBySeparationUsingMinisat(problem, learning_set).perform()

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 0)
        self.assertEqual(result.unchanged, 200)

        testing_set = generate_classified_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 24)
        self.assertEqual(result.unchanged, 976)

    def test_max_sat_by_coalitions_using_evalmaxsat_learning(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_classified_alternatives(problem, model, 200, 43)

        learned_model = LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat(problem, learning_set).perform()

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 0)
        self.assertEqual(result.unchanged, 200)

        testing_set = generate_classified_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 28)
        self.assertEqual(result.unchanged, 972)

    def test_max_sat_by_separation_using_evalmaxsat_learning(self):
        problem = generate_problem(5, 2, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_classified_alternatives(problem, model, 200, 43)

        learned_model = LearnUcncsByMaxSatBySeparationUsingEvalmaxsat(problem, learning_set).perform()

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 0)
        self.assertEqual(result.unchanged, 200)

        testing_set = generate_classified_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 24)
        self.assertEqual(result.unchanged, 976)

    def test_learning_failure_exception(self):
        problem = generate_problem(2, 2, 42)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_classified_alternatives(problem, model, 100, 42)
        misclassify_alternatives(problem, learning_set, 10, 42 + 27)

        learning = LearnUcncsBySatByCoalitionsUsingMinisat(problem, learning_set)

        with self.assertRaises(LearningFailureException):
            learned_model = learning.perform()

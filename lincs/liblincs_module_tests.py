# Copyright 2023 Vincent Jacques

import unittest
import os

from . import *


forbid_gpu = os.environ.get("LINCS_DEV_FORBID_GPU", "false") == "true"


class ProblemTestCase(unittest.TestCase):
    def test_init_empty(self):
        problem = Problem([], [])
        self.assertEqual(len(problem.criteria), 0)
        self.assertEqual(len(problem.categories), 0)

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

    def test_init_one_criterion(self):
        problem = Problem([Criterion("Criterion name", ValueType.real, CategoryCorrelation.growing)], [])
        self.assertEqual(len(problem.criteria), 1)
        self.assertEqual(problem.criteria[0].name, "Criterion name")
        self.assertEqual(problem.criteria[0].value_type, ValueType.real)
        self.assertEqual(problem.criteria[0].category_correlation, CategoryCorrelation.growing)

    def test_assign_criterion_attributes(self):
        # @todo (When there are more values in ValueType and CategoryCorrelation) Use other values in constructor
        problem = Problem([Criterion("Wrong criterion", ValueType.real, CategoryCorrelation.growing)], [])
        problem.criteria[0].name = "Criterion name"
        problem.criteria[0].value_type = ValueType.real
        problem.criteria[0].category_correlation = CategoryCorrelation.growing
        self.assertEqual(problem.criteria[0].name, "Criterion name")
        self.assertEqual(problem.criteria[0].value_type, ValueType.real)
        self.assertEqual(problem.criteria[0].category_correlation, CategoryCorrelation.growing)

    def test_assign_criterion(self):
        problem = Problem([Criterion("Wrong criterion", ValueType.real, CategoryCorrelation.growing)], [])
        problem.criteria[0] = Criterion("Criterion name", ValueType.real, CategoryCorrelation.growing)
        self.assertEqual(problem.criteria[0].name, "Criterion name")

    def test_append_criterion(self):
        problem = Problem([], [])
        problem.criteria.append(Criterion("Criterion name", ValueType.real, CategoryCorrelation.growing))
        self.assertEqual(len(problem.criteria), 1)

    def test_assign_criteria_slice(self):
        problem = Problem([], [])
        problem.criteria[:] = [Criterion("Criterion name", ValueType.real, CategoryCorrelation.growing)]
        self.assertEqual(len(problem.criteria), 1)

    def test_init_one_category(self):
        problem = Problem([], [Category("Category name")])
        self.assertEqual(len(problem.categories), 1)
        self.assertEqual(problem.categories[0].name, "Category name")

    def test_assign_category_attributes(self):
        problem = Problem([], [Category("Wrong category")])
        problem.categories[0].name = "Category name"
        self.assertEqual(problem.categories[0].name, "Category name")

    def test_assign_category(self):
        problem = Problem([], [Category("Wrong category")])
        problem.categories[0] = Category("Category name")
        self.assertEqual(problem.categories[0].name, "Category name")

    def test_append_category(self):
        problem = Problem([], [])
        problem.categories.append(Category("Category name"))
        self.assertEqual(len(problem.categories), 1)

    def test_assign_categories_slice(self):
        problem = Problem([], [])
        problem.categories[:] = [Category("Category name")]
        self.assertEqual(len(problem.categories), 1)


class ModelTestCase(unittest.TestCase):
    def test_init_empty(self):
        problem = Problem([], [])
        model = Model(problem, [])
        self.assertEqual(len(model.boundaries), 0)

    def test_init_wrong_types(self):
        problem = Problem([], [])
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

    def test_init_one_empty_boundary(self):
        problem = Problem([], [])
        model = Model(problem, [Boundary([], SufficientCoalitions(SufficientCoalitionsKind.weights, []))])
        self.assertEqual(len(model.boundaries), 1)
        self.assertEqual(len(model.boundaries[0].profile), 0)
        self.assertEqual(model.boundaries[0].sufficient_coalitions.kind, SufficientCoalitionsKind.weights)
        self.assertEqual(len(model.boundaries[0].sufficient_coalitions.criterion_weights), 0)

    def test_init_three_criteria_two_categories(self):
        problem = Problem(
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
            problem,
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
        problem = Problem([], [])
        model = Model(problem, [])
        model.boundaries = [Boundary([], SufficientCoalitions(SufficientCoalitionsKind.weights, []))]
        self.assertEqual(len(model.boundaries), 1)


class AlternativesTestCase(unittest.TestCase):
    def test_init_empty(self):
        problem = Problem([], [])
        alternatives = Alternatives(problem, [])
        self.assertEqual(len(alternatives.alternatives), 0)

    def test_init_wrong_types(self):
        problem = Problem([], [])
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
                Criterion("Criterion 1", ValueType.real, CategoryCorrelation.growing),
                Criterion("Criterion 2", ValueType.real, CategoryCorrelation.growing),
                Criterion("Criterion 3", ValueType.real, CategoryCorrelation.growing),
            ], [
                Category("Category 1"),
                Category("Category 2"),
            ],
        )
        alternatives = Alternatives(
            problem,
            [
                Alternative("First alternative", [5., 5., 5], "Category 1"),
                Alternative("Second alternative", [1., 2., 3.], None),
                Alternative("Third alternative", [2., 4., 6.]),
            ],
        )
        self.assertEqual(len(alternatives.alternatives), 3)

        self.assertEqual(alternatives.alternatives[0].name, "First alternative")
        self.assertEqual(list(alternatives.alternatives[0].profile), [5., 5., 5.])
        self.assertEqual(alternatives.alternatives[0].category, "Category 1")

        self.assertEqual(alternatives.alternatives[1].name, "Second alternative")
        self.assertEqual(list(alternatives.alternatives[1].profile), [1., 2., 3.])
        self.assertIsNone(alternatives.alternatives[1].category)

        self.assertEqual(alternatives.alternatives[2].name, "Third alternative")
        self.assertEqual(list(alternatives.alternatives[2].profile), [2., 4., 6.])
        self.assertIsNone(alternatives.alternatives[2].category)


class MrSortLearningTestCase(unittest.TestCase):
    def test_basic_mrsort_learning(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)

        models = make_models(problem, learning_set, 9, 44)
        termination_strategy = TerminateAtAccuracy(len(learning_set.alternatives))
        profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(models)
        weights_optimization_strategy = OptimizeWeightsUsingGlop(models)
        profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(models)
        learned_model = WeightsProfilesBreedMrSortLearning(
            models,
            profiles_initialization_strategy,
            weights_optimization_strategy,
            profiles_improvement_strategy,
            termination_strategy,
        ).perform()

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 0)
        self.assertEqual(result.unchanged, 200)

        testing_set = generate_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 29)
        self.assertEqual(result.unchanged, 971)

    def test_python_termination_strategy(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)

        class MyTerminationStrategy(TerminationStrategy):
            def __init__(self):
                super().__init__()
                self.accuracies = []

            def terminate(self, iteration_index, best_accuracy):
                assert iteration_index == len(self.accuracies)
                self.accuracies.append(best_accuracy)
                return iteration_index == 2

        models = make_models(problem, learning_set, 9, 44)
        termination_strategy = MyTerminationStrategy()
        profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(models)
        weights_optimization_strategy = OptimizeWeightsUsingGlop(models)
        profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(models)
        learned_model = WeightsProfilesBreedMrSortLearning(
            models,
            profiles_initialization_strategy,
            weights_optimization_strategy,
            profiles_improvement_strategy,
            termination_strategy,
        ).perform()

        self.assertEqual(termination_strategy.accuracies, [0, 176, 186])

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 14)
        self.assertEqual(result.unchanged, 186)

        testing_set = generate_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 96)
        self.assertEqual(result.unchanged, 904)

    def test_alglib_mrsort_learning(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)

        models = make_models(problem, learning_set, 9, 44)
        termination_strategy = TerminateAtAccuracy(len(learning_set.alternatives))
        profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(models)
        weights_optimization_strategy = OptimizeWeightsUsingAlglib(models)
        profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnCpu(models)
        learned_model = WeightsProfilesBreedMrSortLearning(
            models,
            profiles_initialization_strategy,
            weights_optimization_strategy,
            profiles_improvement_strategy,
            termination_strategy,
        ).perform()

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 0)
        self.assertEqual(result.unchanged, 200)

        testing_set = generate_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 24)
        self.assertEqual(result.unchanged, 976)

    @unittest.skipIf(forbid_gpu, "Can't use GPU")
    def test_gpu_mrsort_learning(self):
        problem = generate_problem(5, 3, 41)
        model = generate_mrsort_model(problem, 42)
        learning_set = generate_alternatives(problem, model, 200, 43)

        models = make_models(problem, learning_set, 9, 44)
        termination_strategy = TerminateAtAccuracy(len(learning_set.alternatives))
        profiles_initialization_strategy = InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(models)
        weights_optimization_strategy = OptimizeWeightsUsingGlop(models)
        gpu_models = make_gpu_models(models)
        profiles_improvement_strategy = ImproveProfilesWithAccuracyHeuristicOnGpu(models, gpu_models)
        learned_model = WeightsProfilesBreedMrSortLearning(
            models,
            profiles_initialization_strategy,
            weights_optimization_strategy,
            profiles_improvement_strategy,
            termination_strategy,
        ).perform()

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 0)
        self.assertEqual(result.unchanged, 200)

        testing_set = generate_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 29)
        self.assertEqual(result.unchanged, 971)

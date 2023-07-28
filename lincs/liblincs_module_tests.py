# Copyright 2023 Vincent Jacques

import time
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
        problem = Problem([Criterion("Criterion name", Criterion.ValueType.real, Criterion.CategoryCorrelation.growing)], [])
        self.assertEqual(len(problem.criteria), 1)
        self.assertEqual(problem.criteria[0].name, "Criterion name")
        self.assertEqual(problem.criteria[0].value_type, Criterion.ValueType.real)
        self.assertEqual(problem.criteria[0].category_correlation, Criterion.CategoryCorrelation.growing)

    def test_assign_criterion_attributes(self):
        # @todo (When there are more values in ValueType and CategoryCorrelation) Use other values in constructor
        problem = Problem([Criterion("Wrong criterion", Criterion.ValueType.real, Criterion.CategoryCorrelation.growing)], [])
        problem.criteria[0].name = "Criterion name"
        problem.criteria[0].value_type = Criterion.ValueType.real
        problem.criteria[0].category_correlation = Criterion.CategoryCorrelation.growing
        self.assertEqual(problem.criteria[0].name, "Criterion name")
        self.assertEqual(problem.criteria[0].value_type, Criterion.ValueType.real)
        self.assertEqual(problem.criteria[0].category_correlation, Criterion.CategoryCorrelation.growing)

    def test_assign_criterion(self):
        problem = Problem([Criterion("Wrong criterion", Criterion.ValueType.real, Criterion.CategoryCorrelation.growing)], [])
        problem.criteria[0] = Criterion("Criterion name", Criterion.ValueType.real, Criterion.CategoryCorrelation.growing)
        self.assertEqual(problem.criteria[0].name, "Criterion name")

    def test_append_criterion(self):
        problem = Problem([], [])
        problem.criteria.append(Criterion("Criterion name", Criterion.ValueType.real, Criterion.CategoryCorrelation.growing))
        self.assertEqual(len(problem.criteria), 1)

    def test_assign_criteria_slice(self):
        problem = Problem([], [])
        problem.criteria[:] = [Criterion("Criterion name", Criterion.ValueType.real, Criterion.CategoryCorrelation.growing)]
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
        model = Model(problem, [Model.Boundary([], SufficientCoalitions(SufficientCoalitions.weights, []))])
        self.assertEqual(len(model.boundaries), 1)
        self.assertEqual(len(model.boundaries[0].profile), 0)
        self.assertEqual(model.boundaries[0].sufficient_coalitions.kind, SufficientCoalitions.Kind.weights)
        self.assertEqual(len(model.boundaries[0].sufficient_coalitions.criterion_weights), 0)

    def test_init_three_criteria_two_categories_weights_boundary(self):
        problem = Problem(
            [
                Criterion("Criterion 1", Criterion.ValueType.real, Criterion.CategoryCorrelation.growing),
                Criterion("Criterion 2", Criterion.ValueType.real, Criterion.CategoryCorrelation.growing),
                Criterion("Criterion 3", Criterion.ValueType.real, Criterion.CategoryCorrelation.growing),
            ], [
                Category("Category 1"),
                Category("Category 2"),
            ],
        )
        model = Model(
            problem,
            [
                Model.Boundary(
                    [5., 5., 5],
                    SufficientCoalitions(SufficientCoalitions.weights, [0.7, 0.7, 1])
                ),
            ],
        )
        self.assertEqual(len(model.boundaries), 1)
        self.assertEqual(len(model.boundaries[0].profile), 3)
        self.assertEqual(model.boundaries[0].sufficient_coalitions.kind, SufficientCoalitions.Kind.weights)
        self.assertEqual(len(model.boundaries[0].sufficient_coalitions.criterion_weights), 3)
        # @todo self.assertEqual(len(model.boundaries[0].sufficient_coalitions.upset_roots), 0)

    def test_init_three_criteria_two_categories_roots_boundary(self):
        problem = Problem(
            [
                Criterion("Criterion 1", Criterion.ValueType.real, Criterion.CategoryCorrelation.growing),
                Criterion("Criterion 2", Criterion.ValueType.real, Criterion.CategoryCorrelation.growing),
                Criterion("Criterion 3", Criterion.ValueType.real, Criterion.CategoryCorrelation.growing),
            ], [
                Category("Category 1"),
                Category("Category 2"),
            ],
        )
        model = Model(
            problem,
            [
                Model.Boundary(
                    [5., 5., 5],
                    SufficientCoalitions(SufficientCoalitions.roots, 3, [[0, 1], [0, 2]])
                ),
            ],
        )
        self.assertEqual(len(model.boundaries), 1)
        self.assertEqual(len(model.boundaries[0].profile), 3)
        self.assertEqual(model.boundaries[0].sufficient_coalitions.kind, SufficientCoalitions.Kind.roots)
        self.assertEqual(len(model.boundaries[0].sufficient_coalitions.criterion_weights), 0)
        # @todo self.assertEqual(len(model.boundaries[0].sufficient_coalitions.upset_roots), 2)

    def test_assign_model_attributes(self):
        problem = Problem([], [])
        model = Model(problem, [])
        model.boundaries = [Model.Boundary([], SufficientCoalitions(SufficientCoalitions.weights, []))]
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
                Criterion("Criterion 1", Criterion.ValueType.real, Criterion.CategoryCorrelation.growing),
                Criterion("Criterion 2", Criterion.ValueType.real, Criterion.CategoryCorrelation.growing),
                Criterion("Criterion 3", Criterion.ValueType.real, Criterion.CategoryCorrelation.growing),
            ], [
                Category("Category 1"),
                Category("Category 2"),
            ],
        )
        alternatives = Alternatives(
            problem,
            [
                Alternative("First alternative", [5., 5., 5], 0),
                Alternative("Second alternative", [1., 2., 3.], None),
                Alternative("Third alternative", [2., 4., 6.]),
            ],
        )
        self.assertEqual(len(alternatives.alternatives), 3)

        self.assertEqual(alternatives.alternatives[0].name, "First alternative")
        self.assertEqual(list(alternatives.alternatives[0].profile), [5., 5., 5.])
        self.assertEqual(alternatives.alternatives[0].category_index, 0)

        self.assertEqual(alternatives.alternatives[1].name, "Second alternative")
        self.assertEqual(list(alternatives.alternatives[1].profile), [1., 2., 3.])
        self.assertIsNone(alternatives.alternatives[1].category_index)

        self.assertEqual(alternatives.alternatives[2].name, "Third alternative")
        self.assertEqual(list(alternatives.alternatives[2].profile), [2., 4., 6.])
        self.assertIsNone(alternatives.alternatives[2].category_index)


class LearningTestCase(unittest.TestCase):
    def test_basic_mrsort_learning(self):
        problem = generate_classification_problem(5, 3, 41)
        model = generate_mrsort_classification_model(problem, 42)
        learning_set = generate_classified_alternatives(problem, model, 200, 43)

        learning_data = LearnMrsortByWeightsProfilesBreed.LearningData.make(problem, learning_set, 9, 44)
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

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 0)
        self.assertEqual(result.unchanged, 200)

        testing_set = generate_classified_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 29)
        self.assertEqual(result.unchanged, 971)

    def test_iterations_restricted_mrsort_learning(self):
        problem = generate_classification_problem(5, 3, 41)
        model = generate_mrsort_classification_model(problem, 42)
        learning_set = generate_classified_alternatives(problem, model, 1000, 43)

        learning_data = LearnMrsortByWeightsProfilesBreed.LearningData.make(problem, learning_set, 9, 44)
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
        problem = generate_classification_problem(5, 3, 41)
        model = generate_mrsort_classification_model(problem, 42)
        learning_set = generate_classified_alternatives(problem, model, 200, 43)

        class MyTerminationStrategy(LearnMrsortByWeightsProfilesBreed.TerminationStrategy):
            def __init__(self):
                super().__init__()
                self.called_count = 0

            def terminate(self):
                self.called_count += 1
                return self.called_count == 6

        learning_data = LearnMrsortByWeightsProfilesBreed.LearningData.make(problem, learning_set, 9, 44)
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
        self.assertEqual(classify_alternatives(problem, learned_model, learning_set).changed, 1)

    def test_python_strategies(self):
        problem = generate_classification_problem(5, 3, 41)
        model = generate_mrsort_classification_model(problem, 42)
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

        learning_data = LearnMrsortByWeightsProfilesBreed.LearningData.make(problem, learning_set, 9, 44)
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
        self.assertEqual(termination_strategy.accuracies, [176, 186])

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 14)
        self.assertEqual(result.unchanged, 186)

        testing_set = generate_classified_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 96)
        self.assertEqual(result.unchanged, 904)

    def test_observers(self):
        problem = generate_classification_problem(5, 3, 41)
        model = generate_mrsort_classification_model(problem, 42)
        learning_set = generate_classified_alternatives(problem, model, 200, 43)

        class MyObserver(LearnMrsortByWeightsProfilesBreed.Observer):
            def __init__(self, learning_data):
                super().__init__()
                self.learning_data = learning_data
                self.best_accuracies = []

            def after_iteration(self):
                self.best_accuracies.append(self.learning_data.get_best_accuracy())

        learning_data = LearnMrsortByWeightsProfilesBreed.LearningData.make(problem, learning_set, 9, 44)
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

        self.assertEqual(observer.best_accuracies, [176, 186, 193, 193, 199, 199, 199])

    def test_alglib_mrsort_learning(self):
        problem = generate_classification_problem(5, 3, 41)
        model = generate_mrsort_classification_model(problem, 42)
        learning_set = generate_classified_alternatives(problem, model, 200, 43)

        learning_data = LearnMrsortByWeightsProfilesBreed.LearningData.make(problem, learning_set, 9, 44)
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
        self.assertEqual(result.changed, 24)
        self.assertEqual(result.unchanged, 976)

    @unittest.skipIf(forbid_gpu, "Can't use GPU")
    def test_gpu_mrsort_learning(self):
        problem = generate_classification_problem(5, 3, 41)
        model = generate_mrsort_classification_model(problem, 42)
        learning_set = generate_classified_alternatives(problem, model, 200, 43)

        learning_data = LearnMrsortByWeightsProfilesBreed.LearningData.make(problem, learning_set, 9, 44)
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
        problem = generate_classification_problem(5, 3, 41)
        model = generate_mrsort_classification_model(problem, 42)
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
        problem = generate_classification_problem(5, 2, 41)
        model = generate_mrsort_classification_model(problem, 42)
        learning_set = generate_classified_alternatives(problem, model, 200, 43)

        learned_model = LearnUcncsBySatBySeparationUsingMinisat(problem, learning_set).perform()

        result = classify_alternatives(problem, learned_model, learning_set)
        self.assertEqual(result.changed, 0)
        self.assertEqual(result.unchanged, 200)

        testing_set = generate_classified_alternatives(problem, model, 1000, 44)
        result = classify_alternatives(problem, learned_model, testing_set)
        self.assertEqual(result.changed, 24)
        self.assertEqual(result.unchanged, 976)

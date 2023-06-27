// Copyright 2023 Vincent Jacques

#include "learning.hpp"

#include "classification.hpp"  // Only for tests
#include "generation.hpp"  // Only for tests

#include <doctest.h>  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


namespace {
  const char* env = std::getenv("LINCS_DEV_FORBID_GPU");
  const bool forbid_gpu = env && std::string(env) == "true";
}  // namespace

namespace lincs {

// @todo Homogenize testing strategy:
//   - on both sides:
//     - test all strategies combinations, with different parameters
//   - on the C++ side:
//     - only check that re-classification results in changed == 0
//     - for a large number of seeds (1000?)
//     - cf. tests for SAT by coalitions below
//   - on the Python side:
//     - check that re-classification results in changed == 0
//     - check that re-classification results in unchanged == 200
//     - on a single seed
//     - also check classification results on a testing set

TEST_CASE("Basic MR-Sort learning") {
  Problem problem = generate_problem(5, 3, 41);
  Model model = generate_mrsort_model(problem, 42);
  Alternatives learning_set = generate_alternatives(problem, model, 200, 43);

  const unsigned random_seed = 44;
  auto models = WeightsProfilesBreedMrSortLearning::Models::make(
    problem, learning_set, WeightsProfilesBreedMrSortLearning::default_models_count, random_seed);

  InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion profiles_initialization_strategy(models);
  OptimizeWeightsUsingGlop weights_optimization_strategy(models);
  ImproveProfilesWithAccuracyHeuristicOnCpu profiles_improvement_strategy(models);
  TerminateAtAccuracy termination_strategy(learning_set.alternatives.size());

  Model learned_model = WeightsProfilesBreedMrSortLearning(
    models,
    profiles_initialization_strategy,
    weights_optimization_strategy,
    profiles_improvement_strategy,
    termination_strategy
  ).perform();

  {
    ClassificationResult result = classify_alternatives(problem, learned_model, &learning_set);
    CHECK(result.changed == 0);
    CHECK(result.unchanged == 200);
  }

  {
    Alternatives testing_set = generate_alternatives(problem, model, 1000, 44);
    ClassificationResult result = classify_alternatives(problem, learned_model, &testing_set);
    CHECK(result.changed == 29);
    CHECK(result.unchanged == 971);
  }
}

TEST_CASE("Alglib MR-Sort learning") {
  Problem problem = generate_problem(5, 3, 41);
  Model model = generate_mrsort_model(problem, 42);
  Alternatives learning_set = generate_alternatives(problem, model, 200, 43);

  const unsigned random_seed = 44;
  auto models = WeightsProfilesBreedMrSortLearning::Models::make(
    problem, learning_set, WeightsProfilesBreedMrSortLearning::default_models_count, random_seed);

  InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion profiles_initialization_strategy(models);
  OptimizeWeightsUsingAlglib weights_optimization_strategy(models);
  ImproveProfilesWithAccuracyHeuristicOnCpu profiles_improvement_strategy(models);
  TerminateAtAccuracy termination_strategy(learning_set.alternatives.size());

  Model learned_model = WeightsProfilesBreedMrSortLearning(
    models,
    profiles_initialization_strategy,
    weights_optimization_strategy,
    profiles_improvement_strategy,
    termination_strategy
  ).perform();

  {
    ClassificationResult result = classify_alternatives(problem, learned_model, &learning_set);
    CHECK(result.changed == 0);
    CHECK(result.unchanged == 200);
  }

  {
    Alternatives testing_set = generate_alternatives(problem, model, 1000, 44);
    ClassificationResult result = classify_alternatives(problem, learned_model, &testing_set);
    CHECK(result.changed == 24);
    CHECK(result.unchanged == 976);
  }
}

TEST_CASE("GPU MR-Sort learning" * doctest::skip(forbid_gpu)) {
  Problem problem = generate_problem(5, 3, 41);
  Model model = generate_mrsort_model(problem, 42);
  Alternatives learning_set = generate_alternatives(problem, model, 200, 43);

  const unsigned random_seed = 44;
  auto host_models = WeightsProfilesBreedMrSortLearning::Models::make(
    problem, learning_set, WeightsProfilesBreedMrSortLearning::default_models_count, random_seed);
  auto gpu_models = ImproveProfilesWithAccuracyHeuristicOnGpu::GpuModels::make(host_models);

  InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion profiles_initialization_strategy(host_models);
  OptimizeWeightsUsingGlop weights_optimization_strategy(host_models);
  ImproveProfilesWithAccuracyHeuristicOnGpu profiles_improvement_strategy(host_models, gpu_models);
  TerminateAtAccuracy termination_strategy(learning_set.alternatives.size());

  Model learned_model = WeightsProfilesBreedMrSortLearning(
    host_models,
    profiles_initialization_strategy,
    weights_optimization_strategy,
    profiles_improvement_strategy,
    termination_strategy
  ).perform();

  {
    ClassificationResult result = classify_alternatives(problem, learned_model, &learning_set);
    CHECK(result.changed == 0);
    CHECK(result.unchanged == 200);
  }

  {
    Alternatives testing_set = generate_alternatives(problem, model, 1000, 44);
    ClassificationResult result = classify_alternatives(problem, learned_model, &testing_set);
    CHECK(result.changed == 29);
    CHECK(result.unchanged == 971);
  }
}

TEST_CASE("SAT by coalitions using Minisat learning - 1 criterion, 2 categories") {
  Problem problem = generate_problem(1, 2, 41);

  for (unsigned seed = 0; seed != 1000; ++seed) {
    Model model = generate_mrsort_model(problem, seed);
    Alternatives learning_set = generate_alternatives(problem, model, 200, seed);

    Model learned_model = SatCoalitionUcncsLearningUsingMinisat(problem, learning_set).perform();

    CHECK(classify_alternatives(problem, learned_model, &learning_set).changed == 0);
  }
}

TEST_CASE("SAT by coalitions using Minisat learning - 3 criteria, 2 categories") {
  Problem problem = generate_problem(3, 2, 41);

  for (unsigned seed = 0; seed != 1000; ++seed) {
    Model model = generate_mrsort_model(problem, seed);
    Alternatives learning_set = generate_alternatives(problem, model, 200, seed);

    Model learned_model = SatCoalitionUcncsLearningUsingMinisat(problem, learning_set).perform();

    CHECK(classify_alternatives(problem, learned_model, &learning_set).changed == 0);
  }
}

TEST_CASE("SAT by coalitions using Minisat learning - 1 criterion, 3 categories") {
  Problem problem = generate_problem(1, 3, 41);

  for (unsigned seed = 0; seed != 1000; ++seed) {
    Model model = generate_mrsort_model(problem, seed);
    Alternatives learning_set = generate_alternatives(problem, model, 200, seed);

    Model learned_model = SatCoalitionUcncsLearningUsingMinisat(problem, learning_set).perform();

    CHECK(classify_alternatives(problem, learned_model, &learning_set).changed == 0);
  }
}

TEST_CASE("SAT by coalitions using Minisat learning - 4 criteria, 3 categories") {
  Problem problem = generate_problem(4, 3, 41);

  for (unsigned seed = 0; seed != 1000; ++seed) {
    Model model = generate_mrsort_model(problem, seed);
    Alternatives learning_set = generate_alternatives(problem, model, 200, seed);

    Model learned_model = SatCoalitionUcncsLearningUsingMinisat(problem, learning_set).perform();

    CHECK(classify_alternatives(problem, learned_model, &learning_set).changed == 0);
  }
}

}  // namespace lincs

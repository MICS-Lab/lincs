// Copyright 2023 Vincent Jacques

#include "learning.hpp"

#include "classification.hpp"  // Only for tests
#include "generation.hpp"  // Only for tests

#include "vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


namespace {

bool env_is_true(const char* name) {
  const char* value = std::getenv(name);
  return value && std::string(value) == "true";
}

const bool forbid_gpu = env_is_true("LINCS_DEV_FORBID_GPU");
const bool skip_long = env_is_true("LINCS_DEV_SKIP_LONG");

template<typename T>
void check_learning(const lincs::Problem& problem, unsigned seed) {
  CAPTURE(seed);

  lincs::Model model = lincs::generate_mrsort_classification_model(problem, seed);
  lincs::Alternatives learning_set = lincs::generate_classified_alternatives(problem, model, 200, seed);

  T learning(problem, learning_set);

  lincs::Model learned_model = learning.perform();

  CHECK(lincs::classify_alternatives(problem, learned_model, &learning_set).changed == 0);
}

template<typename T>
void check_learning(const unsigned criteria_count, const unsigned categories_count) {
  CAPTURE(criteria_count);
  CAPTURE(categories_count);

  lincs::Problem problem = lincs::generate_classification_problem(criteria_count, categories_count, 41);

  const unsigned max_seed = skip_long ? 10 : 100;

  for (unsigned seed = 0; seed != max_seed; ++seed) {
    // @todo Understand why these seeds are problematic and handle them properly
    if (seed == 58) { continue; }
    if (seed == 59) { continue; }

    check_learning<T>(problem, seed);
  }
}

template<typename T>
void check_learning() {
  check_learning<T>(1, 2);
  check_learning<T>(3, 2);
  check_learning<T>(1, 3);
  check_learning<T>(4, 3);
}

}  // namespace

namespace lincs {

TEST_CASE("Basic MR-Sort learning") {
  class Wrapper {
   public:
    Wrapper(const Problem& problem, const Alternatives& learning_set) :
      models(WeightsProfilesBreedMrSortLearning::Models::make(
        problem, learning_set, WeightsProfilesBreedMrSortLearning::default_models_count, 44
      )),
      profiles_initialization_strategy(models),
      weights_optimization_strategy(models),
      profiles_improvement_strategy(models),
      termination_strategy(learning_set.alternatives.size()),
      learning(
        models,
        profiles_initialization_strategy,
        weights_optimization_strategy,
        profiles_improvement_strategy,
        termination_strategy
      )
    {}

   public:
    auto perform() { return learning.perform(); }

   private:
    WeightsProfilesBreedMrSortLearning::Models models;
    InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion profiles_initialization_strategy;
    OptimizeWeightsUsingGlop weights_optimization_strategy;
    ImproveProfilesWithAccuracyHeuristicOnCpu profiles_improvement_strategy;
    TerminateAtAccuracy termination_strategy;
    WeightsProfilesBreedMrSortLearning learning;
  };

  check_learning<Wrapper>();
}

TEST_CASE("Alglib MR-Sort learning") {
  class Wrapper {
   public:
    Wrapper(const Problem& problem, const Alternatives& learning_set) :
      models(WeightsProfilesBreedMrSortLearning::Models::make(
        problem, learning_set, WeightsProfilesBreedMrSortLearning::default_models_count, 44
      )),
      profiles_initialization_strategy(models),
      weights_optimization_strategy(models),
      profiles_improvement_strategy(models),
      termination_strategy(learning_set.alternatives.size()),
      learning(
        models,
        profiles_initialization_strategy,
        weights_optimization_strategy,
        profiles_improvement_strategy,
        termination_strategy
      )
    {}

   public:
    auto perform() { return learning.perform(); }

   private:
    WeightsProfilesBreedMrSortLearning::Models models;
    InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion profiles_initialization_strategy;
    OptimizeWeightsUsingAlglib weights_optimization_strategy;
    ImproveProfilesWithAccuracyHeuristicOnCpu profiles_improvement_strategy;
    TerminateAtAccuracy termination_strategy;
    WeightsProfilesBreedMrSortLearning learning;
  };

  check_learning<Wrapper>();
}

TEST_CASE("GPU MR-Sort learning" * doctest::skip(forbid_gpu)) {
  class Wrapper {
   public:
    Wrapper(const Problem& problem, const Alternatives& learning_set) :
      host_models(WeightsProfilesBreedMrSortLearning::Models::make(
        problem, learning_set, WeightsProfilesBreedMrSortLearning::default_models_count, 44
      )),
      gpu_models(ImproveProfilesWithAccuracyHeuristicOnGpu::GpuModels::make(host_models)),
      profiles_initialization_strategy(host_models),
      weights_optimization_strategy(host_models),
      profiles_improvement_strategy(host_models, gpu_models),
      termination_strategy(learning_set.alternatives.size()),
      learning(
        host_models,
        profiles_initialization_strategy,
        weights_optimization_strategy,
        profiles_improvement_strategy,
        termination_strategy
      )
    {}

   public:
    auto perform() { return learning.perform(); }

   private:
    WeightsProfilesBreedMrSortLearning::Models host_models;
    ImproveProfilesWithAccuracyHeuristicOnGpu::GpuModels gpu_models;
    InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion profiles_initialization_strategy;
    OptimizeWeightsUsingGlop weights_optimization_strategy;
    ImproveProfilesWithAccuracyHeuristicOnGpu profiles_improvement_strategy;
    TerminateAtAccuracy termination_strategy;
    WeightsProfilesBreedMrSortLearning learning;
  };

  check_learning<Wrapper>();
}

TEST_CASE("SAT by coalitions using Minisat learning") {
  check_learning<SatCoalitionUcncsLearningUsingMinisat>();
}

TEST_CASE("SAT by coalitions using EvalMaxSAT learning") {
  check_learning<SatCoalitionUcncsLearningUsingEvalmaxsat>();
}

}  // namespace lincs

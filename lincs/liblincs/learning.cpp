// Copyright 2023 Vincent Jacques

#include "learning.hpp"

#include "classification.hpp"  // Only for tests
#include "generation.hpp"  // Only for tests

#include <doctest.h>  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


namespace lincs {

TEST_CASE("Basic MR-Sort learning") {
  Domain domain = generate_domain(3, 2, 41);
  Model model = generate_mrsort_model(domain, 42);
  Alternatives learning_set = generate_alternatives(domain, model, 100, 43);

  const unsigned random_seed = 44;
  auto models = WeightsProfilesBreedMrSortLearning::Models::make(
    domain, learning_set, WeightsProfilesBreedMrSortLearning::default_models_count, random_seed);

  InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion profiles_initialization_strategy(models);
  OptimizeWeightsUsingGlop weights_optimization_strategy(models);
  ImproveProfilesWithAccuracyHeuristic profiles_improvement_strategy(models);
  TerminateAtAccuracy termination_strategy(learning_set.alternatives.size());

  Model learned_model = WeightsProfilesBreedMrSortLearning(
    models,
    profiles_initialization_strategy,
    weights_optimization_strategy,
    profiles_improvement_strategy,
    termination_strategy
  ).perform();

  {
    ClassificationResult result = classify_alternatives(domain, learned_model, &learning_set);
    CHECK(result.changed == 0);
    CHECK(result.unchanged == 100);
  }

  {
    Alternatives testing_set = generate_alternatives(domain, model, 1000, 43);
    ClassificationResult result = classify_alternatives(domain, learned_model, &testing_set);
    CHECK(result.changed == 6);
    CHECK(result.unchanged == 994);
  }
}

}  // namespace lincs

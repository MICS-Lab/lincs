// Copyright 2023 Vincent Jacques

#include "learning.hpp"

#include "classification.hpp"  // Only for tests
#include "generation.hpp"  // Only for tests
#include "learning/exception.hpp"

#include "vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


namespace {

bool env_is_true(const char* name) {
  const char* value = std::getenv(name);
  return value && std::string(value) == "true";
}

const bool forbid_gpu = env_is_true("LINCS_DEV_FORBID_GPU");
const bool skip_long = env_is_true("LINCS_DEV_SKIP_LONG");
const bool coverage = env_is_true("LINCS_DEV_COVERAGE");
const unsigned default_seeds_count = coverage ? 1 : (skip_long ? 10 : 100);

template<typename T>
void check_exact_learning(const lincs::Problem& problem, unsigned seed) {
  CAPTURE(problem.criteria.size());
  CAPTURE(problem.categories.size());
  CAPTURE(seed);

  lincs::Model model = lincs::generate_mrsort_classification_model(problem, seed);
  lincs::Alternatives learning_set = lincs::generate_classified_alternatives(problem, model, 200, seed);

  T learning(problem, learning_set);

  lincs::Model learned_model = learning.perform();

  CHECK(lincs::classify_alternatives(problem, learned_model, &learning_set).changed == 0);
}

template<typename T>
void check_exact_learning(
  const unsigned criteria_count,
  const unsigned categories_count,
  std::set<unsigned> bad_seeds = {},
  const unsigned seeds_count = default_seeds_count
) {
  lincs::Problem problem = lincs::generate_classification_problem(criteria_count, categories_count, 41);

  for (unsigned seed = 0; seed != seeds_count; ++seed) {
    if (bad_seeds.find(seed) != bad_seeds.end()) {
      CHECK_THROWS_AS(check_exact_learning<T>(problem, seed), lincs::LearningFailureException);
    } else {
      check_exact_learning<T>(problem, seed);
    }
  }
}

template<typename T>
void check_non_exact_learning(const lincs::Problem& problem, unsigned seed) {
  CAPTURE(problem.criteria.size());
  CAPTURE(problem.categories.size());
  CAPTURE(seed);

  lincs::Model model = lincs::generate_mrsort_classification_model(problem, seed);
  lincs::Alternatives learning_set = lincs::generate_classified_alternatives(problem, model, 200, seed);
  lincs::misclassify_alternatives(problem, &learning_set, 10, seed);

  T learning(problem, learning_set);

  lincs::Model learned_model = learning.perform();

  // The original model would classify with .changed == 10, so the best model must have .changed <= 10
  CHECK(lincs::classify_alternatives(problem, learned_model, &learning_set).changed <= 10);
}

template<typename T>
void check_non_exact_learning(
  const unsigned criteria_count,
  const unsigned categories_count,
  std::set<unsigned> bad_seeds = {},
  const unsigned seeds_count = default_seeds_count
) {
  lincs::Problem problem = lincs::generate_classification_problem(criteria_count, categories_count, 41);

  for (unsigned seed = 0; seed != seeds_count; ++seed) {
    if (bad_seeds.find(seed) != bad_seeds.end()) {
      CHECK_THROWS_AS(check_non_exact_learning<T>(problem, seed), lincs::LearningFailureException);
    } else {
      check_non_exact_learning<T>(problem, seed);
    }
  }
}

}  // namespace

namespace lincs {

TEST_CASE("Basic MR-Sort learning") {
  class Wrapper {
   public:
    Wrapper(const Problem& problem, const Alternatives& learning_set) :
      learning_data(LearnMrsortByWeightsProfilesBreed::LearningData::make(
        problem, learning_set, LearnMrsortByWeightsProfilesBreed::default_models_count, 44
      )),
      profiles_initialization_strategy(learning_data),
      weights_optimization_strategy(learning_data),
      profiles_improvement_strategy(learning_data),
      breeding_strategy(learning_data, profiles_initialization_strategy, LearnMrsortByWeightsProfilesBreed::default_models_count / 2),
      termination_strategy(learning_data, learning_set.alternatives.size()),
      learning(
        learning_data,
        profiles_initialization_strategy,
        weights_optimization_strategy,
        profiles_improvement_strategy,
        breeding_strategy,
        termination_strategy
      )
    {}

   public:
    auto perform() { return learning.perform(); }

   private:
    LearnMrsortByWeightsProfilesBreed::LearningData learning_data;
    InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion profiles_initialization_strategy;
    OptimizeWeightsUsingGlop weights_optimization_strategy;
    ImproveProfilesWithAccuracyHeuristicOnCpu profiles_improvement_strategy;
    ReinitializeLeastAccurate breeding_strategy;
    TerminateAtAccuracy termination_strategy;
    LearnMrsortByWeightsProfilesBreed learning;
  };

  check_exact_learning<Wrapper>(1, 2);
  check_exact_learning<Wrapper>(3, 2);
  check_exact_learning<Wrapper>(7, 2, {78});
  check_exact_learning<Wrapper>(1, 3);
  check_exact_learning<Wrapper>(4, 3, {59});
}

TEST_CASE("No termination strategy MR-Sort learning") {
  struct NeverTerminate : LearnMrsortByWeightsProfilesBreed::TerminationStrategy {
    bool terminate() override { return false; }
  };

  class Wrapper {
   public:
    Wrapper(const Problem& problem, const Alternatives& learning_set) :
      learning_data(LearnMrsortByWeightsProfilesBreed::LearningData::make(
        problem, learning_set, LearnMrsortByWeightsProfilesBreed::default_models_count, 44
      )),
      profiles_initialization_strategy(learning_data),
      weights_optimization_strategy(learning_data),
      profiles_improvement_strategy(learning_data),
      breeding_strategy(learning_data, profiles_initialization_strategy, LearnMrsortByWeightsProfilesBreed::default_models_count / 2),
      termination_strategy(),
      learning(
        learning_data,
        profiles_initialization_strategy,
        weights_optimization_strategy,
        profiles_improvement_strategy,
        breeding_strategy,
        termination_strategy
      )
    {}

   public:
    auto perform() { return learning.perform(); }

   private:
    LearnMrsortByWeightsProfilesBreed::LearningData learning_data;
    InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion profiles_initialization_strategy;
    OptimizeWeightsUsingGlop weights_optimization_strategy;
    ImproveProfilesWithAccuracyHeuristicOnCpu profiles_improvement_strategy;
    ReinitializeLeastAccurate breeding_strategy;
    NeverTerminate termination_strategy;
    LearnMrsortByWeightsProfilesBreed learning;
  };

  check_exact_learning<Wrapper>(1, 2);
  check_exact_learning<Wrapper>(3, 2);
  check_exact_learning<Wrapper>(7, 2, {78});
  check_exact_learning<Wrapper>(1, 3);
  check_exact_learning<Wrapper>(4, 3, {59});
}

TEST_CASE("Alglib MR-Sort learning") {
  class Wrapper {
   public:
    Wrapper(const Problem& problem, const Alternatives& learning_set) :
      learning_data(LearnMrsortByWeightsProfilesBreed::LearningData::make(
        problem, learning_set, LearnMrsortByWeightsProfilesBreed::default_models_count, 44
      )),
      profiles_initialization_strategy(learning_data),
      weights_optimization_strategy(learning_data),
      profiles_improvement_strategy(learning_data),
      breeding_strategy(learning_data, profiles_initialization_strategy, LearnMrsortByWeightsProfilesBreed::default_models_count / 2),
      termination_strategy(learning_data, learning_set.alternatives.size()),
      learning(
        learning_data,
        profiles_initialization_strategy,
        weights_optimization_strategy,
        profiles_improvement_strategy,
        breeding_strategy,
        termination_strategy
      )
    {}

   public:
    auto perform() { return learning.perform(); }

   private:
    LearnMrsortByWeightsProfilesBreed::LearningData learning_data;
    InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion profiles_initialization_strategy;
    OptimizeWeightsUsingAlglib weights_optimization_strategy;
    ImproveProfilesWithAccuracyHeuristicOnCpu profiles_improvement_strategy;
    ReinitializeLeastAccurate breeding_strategy;
    TerminateAtAccuracy termination_strategy;
    LearnMrsortByWeightsProfilesBreed learning;
  };

  check_exact_learning<Wrapper>(1, 2);
  check_exact_learning<Wrapper>(3, 2);
  check_exact_learning<Wrapper>(7, 2);  // @todo Investigate why seed 78 succeeds with Alglib but not with Glop
  check_exact_learning<Wrapper>(1, 3);
  check_exact_learning<Wrapper>(4, 3, {59});
}

#ifdef LINCS_HAS_NVCC

TEST_CASE("GPU MR-Sort learning" * doctest::skip(forbid_gpu)) {
  class Wrapper {
   public:
    Wrapper(const Problem& problem, const Alternatives& learning_set) :
      learning_data(LearnMrsortByWeightsProfilesBreed::LearningData::make(
        problem, learning_set, LearnMrsortByWeightsProfilesBreed::default_models_count, 44
      )),
      profiles_initialization_strategy(learning_data),
      weights_optimization_strategy(learning_data),
      profiles_improvement_strategy(learning_data),
      breeding_strategy(learning_data, profiles_initialization_strategy, LearnMrsortByWeightsProfilesBreed::default_models_count / 2),
      termination_strategy(learning_data, learning_set.alternatives.size()),
      learning(
        learning_data,
        profiles_initialization_strategy,
        weights_optimization_strategy,
        profiles_improvement_strategy,
        breeding_strategy,
        termination_strategy
      )
    {}

   public:
    auto perform() { return learning.perform(); }

   private:
    LearnMrsortByWeightsProfilesBreed::LearningData learning_data;
    InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion profiles_initialization_strategy;
    OptimizeWeightsUsingGlop weights_optimization_strategy;
    ImproveProfilesWithAccuracyHeuristicOnGpu profiles_improvement_strategy;
    ReinitializeLeastAccurate breeding_strategy;
    TerminateAtAccuracy termination_strategy;
    LearnMrsortByWeightsProfilesBreed learning;
  };

  check_exact_learning<Wrapper>(1, 2);
  check_exact_learning<Wrapper>(3, 2);
  check_exact_learning<Wrapper>(7, 2, {78});
  check_exact_learning<Wrapper>(1, 3);
  check_exact_learning<Wrapper>(4, 3, {59});
}

#endif  // LINCS_HAS_NVCC

TEST_CASE("SAT by coalitions using Minisat learning") {
  check_exact_learning<LearnUcncsBySatByCoalitionsUsingMinisat>(1, 2);
  check_exact_learning<LearnUcncsBySatByCoalitionsUsingMinisat>(3, 2);
  check_exact_learning<LearnUcncsBySatByCoalitionsUsingMinisat>(7, 2);
  check_exact_learning<LearnUcncsBySatByCoalitionsUsingMinisat>(1, 3);
  check_exact_learning<LearnUcncsBySatByCoalitionsUsingMinisat>(4, 3);
  check_exact_learning<LearnUcncsBySatByCoalitionsUsingMinisat>(3, 5);
}

TEST_CASE("SAT by separation using Minisat learning") {
  check_exact_learning<LearnUcncsBySatBySeparationUsingMinisat>(1, 2);
  check_exact_learning<LearnUcncsBySatBySeparationUsingMinisat>(3, 2);
  check_exact_learning<LearnUcncsBySatBySeparationUsingMinisat>(7, 2);
  check_exact_learning<LearnUcncsBySatBySeparationUsingMinisat>(1, 3);
  check_exact_learning<LearnUcncsBySatBySeparationUsingMinisat>(4, 3);
  check_exact_learning<LearnUcncsBySatBySeparationUsingMinisat>(3, 5);
}

TEST_CASE("Non-exact learning - MR-Sort") {
  class Wrapper {
   public:
    Wrapper(const Problem& problem, const Alternatives& learning_set) :
      learning_data(LearnMrsortByWeightsProfilesBreed::LearningData::make(
        problem, learning_set, LearnMrsortByWeightsProfilesBreed::default_models_count, 44
      )),
      profiles_initialization_strategy(learning_data),
      weights_optimization_strategy(learning_data),
      profiles_improvement_strategy(learning_data),
      breeding_strategy(learning_data, profiles_initialization_strategy, LearnMrsortByWeightsProfilesBreed::default_models_count / 2),
      termination_strategy(learning_data, 190),
      learning(
        learning_data,
        profiles_initialization_strategy,
        weights_optimization_strategy,
        profiles_improvement_strategy,
        breeding_strategy,
        termination_strategy
      )
    {}

   public:
    auto perform() { return learning.perform(); }

   private:
    LearnMrsortByWeightsProfilesBreed::LearningData learning_data;
    InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion profiles_initialization_strategy;
    OptimizeWeightsUsingGlop weights_optimization_strategy;
    ImproveProfilesWithAccuracyHeuristicOnCpu profiles_improvement_strategy;
    ReinitializeLeastAccurate breeding_strategy;
    TerminateAtAccuracy termination_strategy;
    LearnMrsortByWeightsProfilesBreed learning;
  };

  check_non_exact_learning<Wrapper>(1, 2);
  check_non_exact_learning<Wrapper>(3, 2, {21});
  check_non_exact_learning<Wrapper>(1, 3);
  check_non_exact_learning<Wrapper>(4, 3, {86, 97});
}

}  // namespace lincs

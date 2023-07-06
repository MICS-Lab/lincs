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

template<typename T>
void check_exact_learning(const lincs::Problem& problem, unsigned seed) {
  CAPTURE(seed);

  lincs::Model model = lincs::generate_mrsort_classification_model(problem, seed);
  lincs::Alternatives learning_set = lincs::generate_classified_alternatives(problem, model, 200, seed);

  T learning(problem, learning_set);

  lincs::Model learned_model = learning.perform();

  CHECK(lincs::classify_alternatives(problem, learned_model, &learning_set).changed == 0);
}

template<typename T>
void check_exact_learning(const unsigned criteria_count, const unsigned categories_count, bool is_wpb) {
  CAPTURE(criteria_count);
  CAPTURE(categories_count);

  lincs::Problem problem = lincs::generate_classification_problem(criteria_count, categories_count, 41);

  const unsigned max_seed = skip_long ? 10 : 100;

  for (unsigned seed = 0; seed != max_seed; ++seed) {
    // This set of parameters can't reach 100% accuracy with WPB approach
    if (is_wpb && criteria_count == 4 && categories_count == 3 && seed == 59) {
      CHECK_THROWS_AS(check_exact_learning<T>(problem, seed), lincs::LearningFailureException);
    } else {
      check_exact_learning<T>(problem, seed);
    }
  }
}

template<typename T>
void check_exact_learning(bool is_wpb = false) {
  check_exact_learning<T>(1, 2, is_wpb);
  check_exact_learning<T>(3, 2, is_wpb);
  check_exact_learning<T>(1, 3, is_wpb);
  check_exact_learning<T>(4, 3, is_wpb);
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

  check_exact_learning<Wrapper>(true);
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

  check_exact_learning<Wrapper>(true);
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

  check_exact_learning<Wrapper>(true);
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

  check_exact_learning<Wrapper>(true);
}

#endif  // LINCS_HAS_NVCC

TEST_CASE("SAT by coalitions using Minisat learning") {
  check_exact_learning<LearnUcncsBySatByCoalitionsUsingMinisat>();
}

TEST_CASE("SAT by coalitions using EvalMaxSAT learning") {
  check_exact_learning<LearnUcncsBySatByCoalitionsUsingEvalmaxsat>();
}

TEST_CASE("Non-exact learning - SAT by coalitions") {
  const Problem problem = generate_classification_problem(3, 2, 41);
  const Model model = generate_mrsort_classification_model(problem, 44);
  Alternatives learning_set = generate_classified_alternatives(problem, model, 100, 44);
  misclassify_alternatives(problem, &learning_set, 10, 44);

  CHECK_THROWS_AS(
    LearnUcncsBySatByCoalitionsUsingMinisat(problem, learning_set).perform(),
    LearningFailureException
  );

  LearnUcncsBySatByCoalitionsUsingEvalmaxsat learning(problem, learning_set);
  Model learned_model = learning.perform();

  CHECK(classify_alternatives(problem, learned_model, &learning_set).changed == 10);
}

TEST_CASE("Non-exact learning - MR-Sort") {
  class Wrapper {
   public:
    Wrapper(const Problem& problem, const Alternatives& learning_set, const unsigned target_accuracy) :
      learning_data(LearnMrsortByWeightsProfilesBreed::LearningData::make(
        problem, learning_set, LearnMrsortByWeightsProfilesBreed::default_models_count, 44
      )),
      profiles_initialization_strategy(learning_data),
      weights_optimization_strategy(learning_data),
      profiles_improvement_strategy(learning_data),
      breeding_strategy(learning_data, profiles_initialization_strategy, LearnMrsortByWeightsProfilesBreed::default_models_count / 2),
      termination_strategy(learning_data, target_accuracy),
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

  const Problem problem = generate_classification_problem(3, 2, 41);
  const Model model = generate_mrsort_classification_model(problem, 44);
  Alternatives learning_set = generate_classified_alternatives(problem, model, 100, 44);
  misclassify_alternatives(problem, &learning_set, 10, 44);

  CHECK_THROWS_AS(
    Wrapper(problem, learning_set, 100).perform(),
    LearningFailureException
  );

  Wrapper learning(problem, learning_set, 90);
  Model learned_model = learning.perform();

  CHECK(classify_alternatives(problem, learned_model, &learning_set).changed == 10);
}

}  // namespace lincs

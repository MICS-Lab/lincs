// Copyright 2023-2024 Vincent Jacques

#include "learning.hpp"

#include <optional>

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
const bool skip_wpb = env_is_true("LINCS_DEV_SKIP_WPB");
const bool skip_sat = env_is_true("LINCS_DEV_SKIP_SAT");
const bool skip_max_sat = env_is_true("LINCS_DEV_SKIP_MAX_SAT");
const bool coverage = env_is_true("LINCS_DEV_COVERAGE");
const unsigned default_seeds_count = coverage ? 1 : (skip_long ? 10 : 100);

template<typename T>
void check_exact_learning(const lincs::Problem& problem, const unsigned seed, const bool should_succeed) {
  lincs::Model model = lincs::generate_mrsort_classification_model(problem, seed);
  lincs::Alternatives learning_set = lincs::generate_classified_alternatives(problem, model, 200, seed);

  T learning(problem, learning_set);

  lincs::Model learned_model = learning.perform();

  const unsigned changed = lincs::classify_alternatives(problem, learned_model, &learning_set).changed;
  if (should_succeed) {
    CHECK(changed == 0);
  } else {
    CHECK(changed != 0);
  }
}

template<typename T>
void check_exact_learnings(
  const unsigned criteria_count,
  const unsigned categories_count,
  const std::vector<lincs::Criterion::PreferenceDirection>& allowed_preference_directions,
  const std::vector<lincs::Criterion::ValueType>& allowed_value_types,
  const std::set<unsigned> bad_seeds
) {
  lincs::Problem problem = lincs::generate_classification_problem(
    criteria_count, categories_count,
    41,
    false,
    allowed_preference_directions,
    allowed_value_types);

  for (unsigned seed = 0; seed != default_seeds_count; ++seed) {
    CAPTURE(seed);
    check_exact_learning<T>(problem, seed, bad_seeds.find(seed) == bad_seeds.end());
  }
}

template<typename T>
void check_non_exact_learning(const lincs::Problem& problem, const unsigned seed, const bool should_succeed) {
  // @todo(Project management, later) Should we use 'fixed_weights_sum'? Would it make the tests more significant? (By avoiding cases where one or a few criteria convey all the useful information)
  lincs::Model model = lincs::generate_mrsort_classification_model(problem, seed);
  lincs::Alternatives learning_set = lincs::generate_classified_alternatives(problem, model, 200, seed);
  lincs::misclassify_alternatives(problem, &learning_set, 10, seed);

  T learning(problem, learning_set);

  lincs::Model learned_model = learning.perform();

  // The original model would classify with .changed == 10, so the best model must have .changed <= 10
  const unsigned changed = lincs::classify_alternatives(problem, learned_model, &learning_set).changed;
  if (should_succeed) {
    CHECK(changed <= 10);
  } else {
    CHECK(changed > 10);
  }
}

template<typename T>
void check_non_exact_learnings(
  const unsigned criteria_count,
  const unsigned categories_count,
  const std::vector<lincs::Criterion::PreferenceDirection>& allowed_preference_directions,
  const std::vector<lincs::Criterion::ValueType>& allowed_value_types,
  const std::set<unsigned> bad_seeds
) {
  lincs::Problem problem = lincs::generate_classification_problem(
    criteria_count, categories_count,
    41,
    false,
    allowed_preference_directions,
    allowed_value_types);

  for (unsigned seed = 0; seed != default_seeds_count; ++seed) {
    CAPTURE(seed);
    check_non_exact_learning<T>(problem, seed, bad_seeds.find(seed) == bad_seeds.end());
  }
}

struct AccuracyObserver : lincs::LearnMrsortByWeightsProfilesBreed::Observer {
  AccuracyObserver(const LearningData& learning_data_) :
    learning_data(learning_data_),
    accuracies()
  {}

  void after_iteration() override {
    accuracies.push_back(learning_data.get_best_accuracy());
  }

  void before_return() override {
    accuracies.push_back(learning_data.get_best_accuracy());
  }

  const LearningData& learning_data;
  std::vector<unsigned> accuracies;
};

}  // namespace

namespace lincs {

namespace {

template<unsigned target_accuracy>
class BasicWpb {
  struct CpuWrapper {
    CpuWrapper(const Problem& problem, const Alternatives& learning_set) :
      learning_data(problem, learning_set, LearnMrsortByWeightsProfilesBreed::default_models_count, 44),
      profiles_initialization_strategy(learning_data),
      weights_optimization_strategy(learning_data),
      profiles_improvement_strategy(learning_data),
      breeding_strategy(learning_data, profiles_initialization_strategy, LearnMrsortByWeightsProfilesBreed::default_models_count / 2),
      termination_strategy_accuracy(learning_data, target_accuracy),
      termination_strategy_progress(learning_data, 200),
      termination_strategy({&termination_strategy_accuracy, &termination_strategy_progress}),
      observer(learning_data),
      observers{&observer},
      learning(
        learning_data,
        profiles_initialization_strategy,
        weights_optimization_strategy,
        profiles_improvement_strategy,
        breeding_strategy,
        termination_strategy,
        observers
      )
    {}

    auto perform() { return learning.perform(); }

    LearnMrsortByWeightsProfilesBreed::LearningData learning_data;
    InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion profiles_initialization_strategy;
    OptimizeWeightsUsingGlop weights_optimization_strategy;
    ImproveProfilesWithAccuracyHeuristicOnCpu profiles_improvement_strategy;
    ReinitializeLeastAccurate breeding_strategy;
    TerminateAtAccuracy termination_strategy_accuracy;
    TerminateAfterIterationsWithoutProgress termination_strategy_progress;
    TerminateWhenAny termination_strategy;
    AccuracyObserver observer;
    std::vector<LearnMrsortByWeightsProfilesBreed::Observer*> observers;
    LearnMrsortByWeightsProfilesBreed learning;
  };

  #ifdef LINCS_HAS_NVCC
  struct GpuWrapper {
    GpuWrapper(const Problem& problem, const Alternatives& learning_set) :
      learning_data(LearnMrsortByWeightsProfilesBreed::LearningData(problem, learning_set, LearnMrsortByWeightsProfilesBreed::default_models_count, 44)),
      profiles_initialization_strategy(learning_data),
      weights_optimization_strategy(learning_data),
      profiles_improvement_strategy(learning_data),
      breeding_strategy(learning_data, profiles_initialization_strategy, LearnMrsortByWeightsProfilesBreed::default_models_count / 2),
      termination_strategy_accuracy(learning_data, target_accuracy),
      termination_strategy_progress(learning_data, 200),
      termination_strategy({&termination_strategy_accuracy, &termination_strategy_progress}),
      observer(learning_data),
      observers{&observer},
      learning(
        learning_data,
        profiles_initialization_strategy,
        weights_optimization_strategy,
        profiles_improvement_strategy,
        breeding_strategy,
        termination_strategy,
        observers
      )
    {}

    auto perform() { return learning.perform(); }

    LearnMrsortByWeightsProfilesBreed::LearningData learning_data;
    InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion profiles_initialization_strategy;
    OptimizeWeightsUsingGlop weights_optimization_strategy;
    ImproveProfilesWithAccuracyHeuristicOnGpu profiles_improvement_strategy;
    ReinitializeLeastAccurate breeding_strategy;
    TerminateAtAccuracy termination_strategy_accuracy;
    TerminateAfterIterationsWithoutProgress termination_strategy_progress;
    TerminateWhenAny termination_strategy;
    AccuracyObserver observer;
    std::vector<LearnMrsortByWeightsProfilesBreed::Observer*> observers;
    LearnMrsortByWeightsProfilesBreed learning;
  };

 public:
  class Wrapper {
   public:
    Wrapper(const Problem& problem_, const Alternatives& learning_set) :
      problem(problem_),
      cpu_wrapper(problem_, learning_set),
      gpu_wrapper(problem_, learning_set)
    {}

   public:
    auto perform() {
      std::optional<Model> cpu_model;
      try {
        cpu_model = cpu_wrapper.perform();
      } catch (const LearningFailureException&) { /* Nothing */ }
      const bool cpu_success = cpu_model.has_value();

      std::optional<Model> gpu_model;
      try {
        gpu_model = gpu_wrapper.perform();
      } catch (const LearningFailureException&) { /* Nothing */ }
      bool gpu_success = gpu_model.has_value();

      CHECK(cpu_wrapper.observer.accuracies == gpu_wrapper.observer.accuracies);
      if (cpu_wrapper.observer.accuracies != gpu_wrapper.observer.accuracies) {
        std::cerr << "CPU accuracies:";
        for (unsigned accuracy: cpu_wrapper.observer.accuracies) {
          std::cerr << " " << accuracy;
        }
        std::cerr << std::endl;
        std::cerr << "GPU accuracies:";
        for (unsigned accuracy: gpu_wrapper.observer.accuracies) {
          std::cerr << " " << accuracy;
        }
        std::cerr << std::endl;
      }

      if (cpu_success == gpu_success) {
        if (cpu_success) {
          CHECK(*cpu_model == *gpu_model);
          return *cpu_model;
        } else {
          throw LearningFailureException("Both CPU and GPU failed");
        }
      } else {
        if (cpu_success) {
          FAIL("CPU succeeded but GPU failed");
          return *cpu_model;
        } else {
          FAIL("GPU succeeded but CPU failed");
          return *gpu_model;
        }
      }
    }

   private:
    const Problem& problem;
    CpuWrapper cpu_wrapper;
    GpuWrapper gpu_wrapper;
  };
  #else
 public:
  typedef CpuWrapper Wrapper;
  #endif
};

class AlglibWpbWrapper {
 public:
  AlglibWpbWrapper(const Problem& problem, const Alternatives& learning_set) :
    learning_data(LearnMrsortByWeightsProfilesBreed::LearningData(problem, learning_set, LearnMrsortByWeightsProfilesBreed::default_models_count, 44)),
    profiles_initialization_strategy(learning_data),
    weights_optimization_strategy(learning_data),
    profiles_improvement_strategy(learning_data),
    breeding_strategy(learning_data, profiles_initialization_strategy, LearnMrsortByWeightsProfilesBreed::default_models_count / 2),
    termination_strategy(learning_data, 200),
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
  TerminateAfterIterationsWithoutProgress termination_strategy;
  LearnMrsortByWeightsProfilesBreed learning;
};

}  // namespace

TEST_CASE("Basic (and GPU) WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Basic (and GPU) WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Basic (and GPU) WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Basic (and GPU) WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Basic (and GPU) WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Basic (and GPU) WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Basic (and GPU) WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Basic (and GPU) WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Basic (and GPU) WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Basic (and GPU) WPB learning - real criteria - long" * doctest::skip(skip_wpb || skip_long)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Basic (and GPU) WPB learning - real criteria - long" * doctest::skip(skip_wpb || skip_long)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Basic (and GPU) WPB learning - real criteria - long" * doctest::skip(skip_wpb || skip_long)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {41});
}

TEST_CASE("Basic (and GPU) WPB learning - real criteria - long" * doctest::skip(skip_wpb || skip_long)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {5, 59});
}

TEST_CASE("Basic (and GPU) WPB learning - real criteria - long" * doctest::skip(skip_wpb || skip_long)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Basic (and GPU) WPB learning - real criteria - long" * doctest::skip(skip_wpb || skip_long)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {55});
}

TEST_CASE("Basic (and GPU) WPB learning - discrete criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::integer, lincs::Criterion::ValueType::enumerated},
    {});
}

TEST_CASE("Basic (and GPU) WPB learning - discrete criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::integer, lincs::Criterion::ValueType::enumerated},
    {6});
}

TEST_CASE("Basic (and GPU) WPB learning - discrete criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::integer, lincs::Criterion::ValueType::enumerated},
    {});
}

TEST_CASE("Basic (and GPU) WPB learning - discrete criteria - long" * doctest::skip(skip_wpb || skip_long)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::integer, lincs::Criterion::ValueType::enumerated},
    {11});
}

TEST_CASE("Basic (and GPU) WPB learning - discrete criteria - long" * doctest::skip(skip_wpb || skip_long)) {
  check_exact_learnings<BasicWpb<200>::Wrapper>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::integer, lincs::Criterion::ValueType::enumerated},
    {14});
}

TEST_CASE("Alglib WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<AlglibWpbWrapper>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Alglib WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<AlglibWpbWrapper>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Alglib WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<AlglibWpbWrapper>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Alglib WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<AlglibWpbWrapper>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Alglib WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<AlglibWpbWrapper>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Alglib WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<AlglibWpbWrapper>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Alglib WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<AlglibWpbWrapper>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Alglib WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<AlglibWpbWrapper>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Alglib WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_exact_learnings<AlglibWpbWrapper>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Alglib WPB learning - real criteria - long" * doctest::skip(skip_wpb || skip_long)) {
  check_exact_learnings<AlglibWpbWrapper>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Alglib WPB learning - real criteria - long" * doctest::skip(skip_wpb || skip_long)) {
  check_exact_learnings<AlglibWpbWrapper>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {48});
}

TEST_CASE("Alglib WPB learning - real criteria - long" * doctest::skip(skip_wpb || skip_long)) {
  check_exact_learnings<AlglibWpbWrapper>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Alglib WPB learning - real criteria - long" * doctest::skip(skip_wpb || skip_long)) {
  check_exact_learnings<AlglibWpbWrapper>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {55, 59});
}

TEST_CASE("Alglib WPB learning - real criteria - long" * doctest::skip(skip_wpb || skip_long)) {
  check_exact_learnings<AlglibWpbWrapper>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Alglib WPB learning - real criteria - long" * doctest::skip(skip_wpb || skip_long)) {
  check_exact_learnings<AlglibWpbWrapper>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {5, 55});
}

TEST_CASE("SAT by coalitions using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - single-peaked real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - single-peaked real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - single-peaked real criteria - long" * doctest::skip(skip_sat || skip_long)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - single-peaked real criteria - long" * doctest::skip(skip_sat || skip_long)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - single-peaked integer criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::integer},
    {});
}

TEST_CASE("SAT by coalitions using Minisat learning - all" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatByCoalitionsUsingMinisat>(
    3, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing, lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real, lincs::Criterion::ValueType::integer, lincs::Criterion::ValueType::enumerated},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - single-peaked real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - single-peaked real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - single-peaked real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - single-peaked real criteria - exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - single-peaked integer criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::integer},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - all - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    3, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing, lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real, lincs::Criterion::ValueType::integer, lincs::Criterion::ValueType::enumerated},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - non-exact" * doctest::skip(skip_max_sat)) {
  check_non_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - non-exact" * doctest::skip(skip_max_sat)) {
  check_non_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - non-exact" * doctest::skip(skip_max_sat)) {
  check_non_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - non-exact" * doctest::skip(skip_max_sat)) {
  check_non_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - non-exact" * doctest::skip(skip_max_sat)) {
  check_non_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - non-exact" * doctest::skip(skip_max_sat)) {
  check_non_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - non-exact" * doctest::skip(skip_max_sat)) {
  check_non_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - non-exact" * doctest::skip(skip_max_sat)) {
  check_non_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - non-exact" * doctest::skip(skip_max_sat)) {
  check_non_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - non-exact" * doctest::skip(skip_max_sat)) {
  check_non_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - non-exact" * doctest::skip(skip_max_sat)) {
  check_non_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - non-exact" * doctest::skip(skip_max_sat)) {
  check_non_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - non-exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_non_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - non-exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_non_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - non-exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_non_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - non-exact" * doctest::skip(skip_max_sat)) {
  check_non_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - non-exact" * doctest::skip(skip_max_sat)) {
  check_non_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - real criteria - non-exact" * doctest::skip(skip_max_sat)) {
  check_non_exact_learnings<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - real criteria - long" * doctest::skip(skip_sat || skip_long)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - real criteria - long" * doctest::skip(skip_sat || skip_long)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - real criteria - long" * doctest::skip(skip_sat || skip_long)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - real criteria - long" * doctest::skip(skip_sat || skip_long)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - real criteria - long" * doctest::skip(skip_sat || skip_long)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - real criteria - long" * doctest::skip(skip_sat || skip_long)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - real criteria - long" * doctest::skip(skip_sat || skip_long)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - real criteria - long" * doctest::skip(skip_sat || skip_long)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - real criteria - long" * doctest::skip(skip_sat || skip_long)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - single-peaked real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - single-peaked real criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - single-peaked real criteria - long" * doctest::skip(skip_sat || skip_long)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - single-peaked real criteria - long" * doctest::skip(skip_sat || skip_long)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - single-peaked integer criteria" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::integer},
    {});
}

TEST_CASE("SAT by separation using Minisat learning - all" * doctest::skip(skip_sat)) {
  check_exact_learnings<LearnUcncsBySatBySeparationUsingMinisat>(
    3, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing, lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real, lincs::Criterion::ValueType::integer, lincs::Criterion::ValueType::enumerated},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using Minisat learning - single-peaked real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using Minisat learning - single-peaked real criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using Minisat learning - single-peaked real criteria - exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using Minisat learning - single-peaked real criteria - exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using Minisat learning - single-peaked integer criteria - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::integer},
    {});
}

TEST_CASE("Max-SAT by separation using Minisat learning - all - exact" * doctest::skip(skip_max_sat)) {
  check_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    3, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing, lincs::Criterion::PreferenceDirection::single_peaked},
    {lincs::Criterion::ValueType::real, lincs::Criterion::ValueType::integer, lincs::Criterion::ValueType::enumerated},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - non-exact" * doctest::skip(skip_max_sat)) {
  check_non_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - non-exact" * doctest::skip(skip_max_sat)) {
  check_non_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - non-exact" * doctest::skip(skip_max_sat)) {
  check_non_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - non-exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_non_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - non-exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_non_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - non-exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_non_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - non-exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_non_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - non-exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_non_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
 }

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - non-exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_non_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
 }

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - non-exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_non_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
 }

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - non-exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_non_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
 }

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - non-exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_non_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    4, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
 }

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - non-exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_non_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
 }

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - non-exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_non_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
 }

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - non-exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_non_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    7, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
 }

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - non-exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_non_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
 }

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - non-exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_non_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
 }

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - real criteria - non-exact - long" * doctest::skip(skip_max_sat || skip_long)) {
  check_non_exact_learnings<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(
    3, 5,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Non-exact WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_non_exact_learnings<BasicWpb<190>::Wrapper>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Non-exact WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_non_exact_learnings<BasicWpb<190>::Wrapper>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Non-exact WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_non_exact_learnings<BasicWpb<190>::Wrapper>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Non-exact WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_non_exact_learnings<BasicWpb<190>::Wrapper>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {45});
}

TEST_CASE("Non-exact WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_non_exact_learnings<BasicWpb<190>::Wrapper>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {53});
}

TEST_CASE("Non-exact WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_non_exact_learnings<BasicWpb<190>::Wrapper>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {45});
}

TEST_CASE("Non-exact WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_non_exact_learnings<BasicWpb<190>::Wrapper>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Non-exact WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_non_exact_learnings<BasicWpb<190>::Wrapper>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Non-exact WPB learning - real criteria" * doctest::skip(skip_wpb)) {
  check_non_exact_learnings<BasicWpb<190>::Wrapper>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::real},
    {});
}

TEST_CASE("Non-exact WPB learning - discrete criteria" * doctest::skip(skip_wpb)) {
  check_non_exact_learnings<BasicWpb<190>::Wrapper>(
    1, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::integer, lincs::Criterion::ValueType::enumerated},
    {});
}

TEST_CASE("Non-exact WPB learning - discrete criteria" * doctest::skip(skip_wpb)) {
  check_non_exact_learnings<BasicWpb<190>::Wrapper>(
    3, 2,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::integer, lincs::Criterion::ValueType::enumerated},
    {6});
}

TEST_CASE("Non-exact WPB learning - discrete criteria" * doctest::skip(skip_wpb)) {
  check_non_exact_learnings<BasicWpb<190>::Wrapper>(
    1, 3,
    {lincs::Criterion::PreferenceDirection::increasing, lincs::Criterion::PreferenceDirection::decreasing},
    {lincs::Criterion::ValueType::integer, lincs::Criterion::ValueType::enumerated},
    {});
}

}  // namespace lincs

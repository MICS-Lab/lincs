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
  std::set<unsigned> bad_seeds_a = {},
  std::set<unsigned> bad_seeds_b = {},
  std::set<unsigned> bad_seeds_c = {},
  const unsigned seeds_count = default_seeds_count
) {
  if (!skip_long) {
    lincs::Problem problem = lincs::generate_classification_problem(criteria_count, categories_count, 41, false, false);

    for (unsigned seed = 0; seed != seeds_count; ++seed) {
      if (bad_seeds_a.find(seed) != bad_seeds_a.end()) {
        CHECK_THROWS_AS(check_exact_learning<T>(problem, seed), lincs::LearningFailureException);
      } else {
        check_exact_learning<T>(problem, seed);
      }
    }
  }

  if (!skip_long) {
    lincs::Problem problem = lincs::generate_classification_problem(criteria_count, categories_count, 41, false, false);
    for (auto& criterion : problem.criteria) {
      criterion.category_correlation = lincs::Criterion::CategoryCorrelation::decreasing;
    }

    for (unsigned seed = 0; seed != seeds_count; ++seed) {
      if (bad_seeds_b.find(seed) != bad_seeds_b.end()) {
        CHECK_THROWS_AS(check_exact_learning<T>(problem, seed), lincs::LearningFailureException);
      } else {
        check_exact_learning<T>(problem, seed);
      }
    }
  }

  if (true) {
    lincs::Problem problem = lincs::generate_classification_problem(criteria_count, categories_count, 41, false, true);

    for (unsigned seed = 0; seed != seeds_count; ++seed) {
      if (bad_seeds_c.find(seed) != bad_seeds_c.end()) {
        CHECK_THROWS_AS(check_exact_learning<T>(problem, seed), lincs::LearningFailureException);
      } else {
        check_exact_learning<T>(problem, seed);
      }
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
  std::set<unsigned> bad_seeds_a = {},
  std::set<unsigned> bad_seeds_b = {},
  std::set<unsigned> bad_seeds_c = {},
  const unsigned seeds_count = default_seeds_count
) {
  if (!skip_long) {
    lincs::Problem problem = lincs::generate_classification_problem(criteria_count, categories_count, 41, false, false);

    for (unsigned seed = 0; seed != seeds_count; ++seed) {
      if (bad_seeds_a.find(seed) != bad_seeds_a.end()) {
        CHECK_THROWS_AS(check_non_exact_learning<T>(problem, seed), lincs::LearningFailureException);
      } else {
        check_non_exact_learning<T>(problem, seed);
      }
    }
  }

  if (!skip_long) {
    lincs::Problem problem = lincs::generate_classification_problem(criteria_count, categories_count, 41, false, false);
    for (auto& criterion : problem.criteria) {
      criterion.category_correlation = lincs::Criterion::CategoryCorrelation::decreasing;
    }

    for (unsigned seed = 0; seed != seeds_count; ++seed) {
      if (bad_seeds_b.find(seed) != bad_seeds_b.end()) {
        CHECK_THROWS_AS(check_non_exact_learning<T>(problem, seed), lincs::LearningFailureException);
      } else {
        check_non_exact_learning<T>(problem, seed);
      }
    }
  }

  if (true) {
    lincs::Problem problem = lincs::generate_classification_problem(criteria_count, categories_count, 41, false, true);

    for (unsigned seed = 0; seed != seeds_count; ++seed) {
      if (bad_seeds_c.find(seed) != bad_seeds_c.end()) {
        CHECK_THROWS_AS(check_non_exact_learning<T>(problem, seed), lincs::LearningFailureException);
      } else {
        check_non_exact_learning<T>(problem, seed);
      }
    }
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

TEST_CASE(
#ifdef LINCS_HAS_NVCC
  "Basic and GPU WPB learning"
#else
  "Basic WPB learning"
#endif
) {
  struct CpuWrapper {
    CpuWrapper(const Problem& problem, const Alternatives& learning_set) :
      learning_data(LearnMrsortByWeightsProfilesBreed::LearningData::make(
        problem, learning_set, LearnMrsortByWeightsProfilesBreed::default_models_count, 44
      )),
      profiles_initialization_strategy(learning_data),
      weights_optimization_strategy(learning_data),
      profiles_improvement_strategy(learning_data),
      breeding_strategy(learning_data, profiles_initialization_strategy, LearnMrsortByWeightsProfilesBreed::default_models_count / 2),
      termination_strategy(learning_data, learning_set.alternatives.size()),
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
    TerminateAtAccuracy termination_strategy;
    AccuracyObserver observer;
    std::vector<LearnMrsortByWeightsProfilesBreed::Observer*> observers;
    LearnMrsortByWeightsProfilesBreed learning;
  };

#ifdef LINCS_HAS_NVCC
  struct GpuWrapper {
    GpuWrapper(const Problem& problem, const Alternatives& learning_set) :
      learning_data(LearnMrsortByWeightsProfilesBreed::LearningData::make(
        problem, learning_set, LearnMrsortByWeightsProfilesBreed::default_models_count, 44
      )),
      profiles_initialization_strategy(learning_data),
      weights_optimization_strategy(learning_data),
      profiles_improvement_strategy(learning_data),
      breeding_strategy(learning_data, profiles_initialization_strategy, LearnMrsortByWeightsProfilesBreed::default_models_count / 2),
      termination_strategy(learning_data, learning_set.alternatives.size()),
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
    TerminateAtAccuracy termination_strategy;
    AccuracyObserver observer;
    std::vector<LearnMrsortByWeightsProfilesBreed::Observer*> observers;
    LearnMrsortByWeightsProfilesBreed learning;
  };

  class Wrapper {
   public:
    Wrapper(const Problem& problem_, const Alternatives& learning_set) :
      problem(problem_),
      cpu_wrapper(problem_, learning_set),
      gpu_wrapper(problem_, learning_set)
    {}

   public:
    auto perform() {
      bool cpu_success = false;
      Model cpu_model(problem, {});
      try {
        cpu_model.boundaries = cpu_wrapper.perform().boundaries;
        cpu_success = true;
      } catch (const LearningFailureException&) { /* Nothing */ }

      bool gpu_success = false;
      Model gpu_model(problem, {});
      try {
        gpu_model.boundaries = gpu_wrapper.perform().boundaries;
        gpu_success = true;
      } catch (const LearningFailureException&) { /* Nothing */ }

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
          CHECK(cpu_model.boundaries == gpu_model.boundaries);
          return cpu_model;
        } else {
          throw LearningFailureException();
        }
      } else {
        if (cpu_success) {
          FAIL("CPU succeeded but GPU failed");
          return cpu_model;
        } else {
          FAIL("GPU succeeded but CPU failed");
          return gpu_model;
        }
      }
    }

   private:
    const Problem& problem;
    CpuWrapper cpu_wrapper;
    GpuWrapper gpu_wrapper;
  };
#else
  typedef CpuWrapper Wrapper;
#endif

  check_exact_learning<Wrapper>(1, 2);
  check_exact_learning<Wrapper>(3, 2, {}, {9}, {});
  check_exact_learning<Wrapper>(7, 2, {78}, {47}, {41});
  check_exact_learning<Wrapper>(1, 3);
  check_exact_learning<Wrapper>(4, 3, {59}, {40}, {44, 55, 71});
}

TEST_CASE("No termination strategy WPB learning") {
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
  check_exact_learning<Wrapper>(3, 2, {}, {9}, {});
  check_exact_learning<Wrapper>(7, 2, {78}, {47}, {41});
  check_exact_learning<Wrapper>(1, 3);
  check_exact_learning<Wrapper>(4, 3, {59}, {40}, {44, 55, 71});
}

TEST_CASE("Alglib WPB learning") {
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
  check_exact_learning<Wrapper>(3, 2, {}, {9}, {});
  check_exact_learning<Wrapper>(7, 2, {}, {}, {});
  check_exact_learning<Wrapper>(1, 3);
  check_exact_learning<Wrapper>(4, 3, {55, 59}, {40}, {44});
}

TEST_CASE("SAT by coalitions using Minisat learning") {
  check_exact_learning<LearnUcncsBySatByCoalitionsUsingMinisat>(1, 2);
  check_exact_learning<LearnUcncsBySatByCoalitionsUsingMinisat>(3, 2);
  check_exact_learning<LearnUcncsBySatByCoalitionsUsingMinisat>(7, 2);
  check_exact_learning<LearnUcncsBySatByCoalitionsUsingMinisat>(1, 3);
  check_exact_learning<LearnUcncsBySatByCoalitionsUsingMinisat>(4, 3);
  check_exact_learning<LearnUcncsBySatByCoalitionsUsingMinisat>(3, 5);
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - exact") {
  check_exact_learning<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(1, 2);
  check_exact_learning<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(3, 2);
  check_exact_learning<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(7, 2);
  check_exact_learning<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(1, 3);
  check_exact_learning<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(4, 3);
  check_exact_learning<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(3, 5);
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - non-exact") {
  check_non_exact_learning<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(1, 2);
  check_non_exact_learning<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(3, 2);
  check_non_exact_learning<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(1, 3);
  check_non_exact_learning<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(4, 3);
}

TEST_CASE("Max-SAT by coalitions using EvalMaxSat learning - non-exact - long" * doctest::skip(skip_long)) {
  check_non_exact_learning<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(7, 2);
  check_non_exact_learning<LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat>(3, 5);
}

TEST_CASE("SAT by separation using Minisat learning") {
  check_exact_learning<LearnUcncsBySatBySeparationUsingMinisat>(1, 2);
  check_exact_learning<LearnUcncsBySatBySeparationUsingMinisat>(3, 2);
  check_exact_learning<LearnUcncsBySatBySeparationUsingMinisat>(1, 3);
  check_exact_learning<LearnUcncsBySatBySeparationUsingMinisat>(4, 3);
}

TEST_CASE("SAT by separation using Minisat learning - long" * doctest::skip(skip_long)) {
  check_exact_learning<LearnUcncsBySatBySeparationUsingMinisat>(7, 2);
  check_exact_learning<LearnUcncsBySatBySeparationUsingMinisat>(3, 5);
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - exact") {
  check_exact_learning<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(1, 2);
  check_exact_learning<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(3, 2);
  check_exact_learning<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(1, 3);
  check_exact_learning<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(4, 3);
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - exact - long" * doctest::skip(skip_long)) {
  check_exact_learning<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(7, 2);
  check_exact_learning<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(3, 5);
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - non-exact") {
  check_non_exact_learning<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(1, 2);
  check_non_exact_learning<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(1, 3);
}

TEST_CASE("Max-SAT by separation using EvalMaxSat learning - non-exact - long" * doctest::skip(skip_long)) {
  check_non_exact_learning<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(3, 2);
  check_non_exact_learning<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(4, 3);
  check_non_exact_learning<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(7, 2);
  check_non_exact_learning<LearnUcncsByMaxSatBySeparationUsingEvalmaxsat>(3, 5);
}

TEST_CASE("Non-exact WPB learning") {
  struct CpuWrapper {
    CpuWrapper(const Problem& problem, const Alternatives& learning_set) :
      learning_data(LearnMrsortByWeightsProfilesBreed::LearningData::make(
        problem, learning_set, LearnMrsortByWeightsProfilesBreed::default_models_count, 44
      )),
      profiles_initialization_strategy(learning_data),
      weights_optimization_strategy(learning_data),
      profiles_improvement_strategy(learning_data),
      breeding_strategy(learning_data, profiles_initialization_strategy, LearnMrsortByWeightsProfilesBreed::default_models_count / 2),
      termination_strategy(learning_data, 190),
      observer(learning_data),
      observers{&observer},
      learning(
        learning_data,
        profiles_initialization_strategy,
        weights_optimization_strategy,
        profiles_improvement_strategy,
        breeding_strategy,
        termination_strategy
      )
    {}

    auto perform() { return learning.perform(); }

    LearnMrsortByWeightsProfilesBreed::LearningData learning_data;
    InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion profiles_initialization_strategy;
    OptimizeWeightsUsingGlop weights_optimization_strategy;
    ImproveProfilesWithAccuracyHeuristicOnCpu profiles_improvement_strategy;
    ReinitializeLeastAccurate breeding_strategy;
    TerminateAtAccuracy termination_strategy;
    AccuracyObserver observer;
    std::vector<LearnMrsortByWeightsProfilesBreed::Observer*> observers;
    LearnMrsortByWeightsProfilesBreed learning;
  };

#ifdef LINCS_HAS_NVCC
  struct GpuWrapper {
    GpuWrapper(const Problem& problem, const Alternatives& learning_set) :
      learning_data(LearnMrsortByWeightsProfilesBreed::LearningData::make(
        problem, learning_set, LearnMrsortByWeightsProfilesBreed::default_models_count, 44
      )),
      profiles_initialization_strategy(learning_data),
      weights_optimization_strategy(learning_data),
      profiles_improvement_strategy(learning_data),
      breeding_strategy(learning_data, profiles_initialization_strategy, LearnMrsortByWeightsProfilesBreed::default_models_count / 2),
      termination_strategy(learning_data, 190),
      observer(learning_data),
      observers{&observer},
      learning(
        learning_data,
        profiles_initialization_strategy,
        weights_optimization_strategy,
        profiles_improvement_strategy,
        breeding_strategy,
        termination_strategy
      )
    {}

    auto perform() { return learning.perform(); }

    LearnMrsortByWeightsProfilesBreed::LearningData learning_data;
    InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion profiles_initialization_strategy;
    OptimizeWeightsUsingGlop weights_optimization_strategy;
    ImproveProfilesWithAccuracyHeuristicOnGpu profiles_improvement_strategy;
    ReinitializeLeastAccurate breeding_strategy;
    TerminateAtAccuracy termination_strategy;
    AccuracyObserver observer;
    std::vector<LearnMrsortByWeightsProfilesBreed::Observer*> observers;
    LearnMrsortByWeightsProfilesBreed learning;
  };

  class Wrapper {
   public:
    Wrapper(const Problem& problem_, const Alternatives& learning_set) :
      problem(problem_),
      cpu_wrapper(problem_, learning_set),
      gpu_wrapper(problem_, learning_set)
    {}

   public:
    auto perform() {
      bool cpu_success = false;
      Model cpu_model(problem, {});
      try {
        cpu_model.boundaries = cpu_wrapper.perform().boundaries;
        cpu_success = true;
      } catch (const LearningFailureException&) { /* Nothing */ }

      bool gpu_success = false;
      Model gpu_model(problem, {});
      try {
        gpu_model.boundaries = gpu_wrapper.perform().boundaries;
        gpu_success = true;
      } catch (const LearningFailureException&) { /* Nothing */ }

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
          CHECK(cpu_model.boundaries == gpu_model.boundaries);
          return cpu_model;
        } else {
          throw LearningFailureException();
        }
      } else {
        if (cpu_success) {
          FAIL("CPU succeeded but GPU failed");
          return cpu_model;
        } else {
          FAIL("GPU succeeded but CPU failed");
          return gpu_model;
        }
      }
    }

   private:
    const Problem& problem;
    CpuWrapper cpu_wrapper;
    GpuWrapper gpu_wrapper;
  };
#else
  typedef CpuWrapper Wrapper;
#endif

  check_non_exact_learning<Wrapper>(1, 2);
  check_non_exact_learning<Wrapper>(3, 2, {21, 45}, {}, {21, 45});
  check_non_exact_learning<Wrapper>(1, 3);
}

}  // namespace lincs

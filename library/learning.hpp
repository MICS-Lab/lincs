// Copyright 2021-2022 Vincent Jacques

#ifndef LEARNING_HPP_
#define LEARNING_HPP_

#include <chrono>  // NOLINT(build/c++11)
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "improve-profiles.hpp"
#include "initialize-profiles.hpp"
#include "io.hpp"
#include "observe.hpp"
#include "optimize-weights.hpp"
#include "problem.hpp"
#include "randomness.hpp"
#include "terminate.hpp"


namespace ppl {

struct LearningResult {
  LearningResult(io::Model model, uint accuracy) : best_model(model), best_model_accuracy(accuracy) {}

  io::Model best_model;
  uint best_model_accuracy;
};

// @todo Find a good default value. How?
const uint default_models_count = 9;

LearningResult perform_learning(
  Models<Host>* host_models,
  // @todo Could we use std::unique_ptr instead of std::shared_ptr?
  std::vector<std::shared_ptr<LearningObserver>> observers,
  std::shared_ptr<ProfilesInitializationStrategy> profiles_initialization_strategy,
  std::shared_ptr<WeightsOptimizationStrategy> weights_optimization_strategy,
  std::shared_ptr<ProfilesImprovementStrategy> profiles_improvement_strategy,
  std::shared_ptr<TerminationStrategy> termination_strategy
);

}  // namespace ppl

#endif  // LEARNING_HPP_

// Copyright 2021-2022 Vincent Jacques

#ifndef LEARNING_HPP_
#define LEARNING_HPP_

#include <memory>
#include <vector>

#include "improve-profiles.hpp"
#include "initialize-profiles.hpp"
#include "observe.hpp"
#include "optimize-weights.hpp"
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
  std::shared_ptr<Models<Host>> host_models,
  std::vector<std::shared_ptr<LearningObserver>> observers,
  std::shared_ptr<ProfilesInitializationStrategy> profiles_initialization_strategy,
  std::shared_ptr<WeightsOptimizationStrategy> weights_optimization_strategy,
  std::shared_ptr<ProfilesImprovementStrategy> profiles_improvement_strategy,
  std::shared_ptr<TerminationStrategy> termination_strategy
);

}  // namespace ppl

#endif  // LEARNING_HPP_

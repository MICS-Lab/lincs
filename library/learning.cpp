// Copyright 2021-2022 Vincent Jacques

#include "learning.hpp"

#include <vector>
#include <algorithm>

#include <chrones.hpp>

#include "assign.hpp"
#include "improve-profiles.hpp"
#include "optimize-weights.hpp"
#include "initialize-profiles.hpp"
#include "median-and-max.hpp"


namespace ppl {

std::vector<uint> partition_models_by_accuracy(const uint models_count, const Models<Host>& models) {
  CHRONE();

  std::vector<uint> accuracies(models_count, 0);
  for (uint model_index = 0; model_index != models_count; ++model_index) {
    accuracies[model_index] = get_accuracy(models, model_index);
  }

  std::vector<uint> model_indexes(models_count, 0);
  std::iota(model_indexes.begin(), model_indexes.end(), 0);
  ensure_median_and_max(
    model_indexes.begin(), model_indexes.end(),
    [&accuracies](uint left_model_index, uint right_model_index) {
      return accuracies[left_model_index] < accuracies[right_model_index];
    });

  return model_indexes;
}

LearningResult perform_learning(
  Models<Host>* host_models,
  RandomNumberGenerator random,
  // @todo Could we use std::unique_ptr instead of std::shared_ptr?
  std::vector<std::shared_ptr<LearningObserver>> observers,
  std::shared_ptr<ProfilesInitializationStrategy> profiles_initialization_strategy,
  std::shared_ptr<WeightsOptimizationStrategy> weights_optimization_strategy,
  std::shared_ptr<ProfilesImprovementStrategy> profiles_improvement_strategy,
  std::shared_ptr<TerminationStrategy> termination_strategy
) {
  CHRONE();

  const uint models_count = host_models->get_view().models_count;

  std::vector<uint> model_indexes(models_count, 0);
  std::iota(model_indexes.begin(), model_indexes.end(), 0);
  profiles_initialization_strategy->initialize_profiles(
    random, host_models,
    0,
    model_indexes.begin(), model_indexes.end());

  uint best_accuracy = 0;

  for (int iteration_index = 0; !termination_strategy->terminate(iteration_index, best_accuracy); ++iteration_index) {
    if (iteration_index != 0) {
      profiles_initialization_strategy->initialize_profiles(
        random, host_models,
        iteration_index,
        model_indexes.begin(), model_indexes.begin() + models_count / 2);
    }

    weights_optimization_strategy->optimize_weights(host_models);
    profiles_improvement_strategy->improve_profiles(random);

    model_indexes = partition_models_by_accuracy(models_count, *host_models);
    best_accuracy = get_accuracy(*host_models, model_indexes.back());

    for (auto observer : observers) {
      observer->after_main_iteration(iteration_index, best_accuracy, *host_models);
    }
  }

  return LearningResult(host_models->unmake_one(model_indexes.back()), best_accuracy);
}

}  // namespace ppl

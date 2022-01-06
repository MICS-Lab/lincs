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

struct LearningExecution {
  LearningExecution(
    const Domain<Host>& host_domain_,
    Models<Host>* host_models_,
    std::shared_ptr<ProfilesInitializationStrategy> profiles_initialization_strategy_,
    std::shared_ptr<WeightsOptimizationStrategy> weights_optimization_strategy_,
    std::shared_ptr<ProfilesImprovementStrategy> profiles_improvement_strategy_,
    std::shared_ptr<TerminationStrategy> termination_strategy_,
    RandomNumberGenerator random_,
    std::vector<std::shared_ptr<LearningObserver>> observers_) :
      models_count(host_models_->get_view().models_count),
      model_indexes(models_count, 0),
      profiles_initialization_strategy(profiles_initialization_strategy_),
      weights_optimization_strategy(weights_optimization_strategy_),
      profiles_improvement_strategy(profiles_improvement_strategy_),
      termination_strategy(termination_strategy_),
      host_domain(host_domain_),
      host_models(host_models_),
      random(random_),
      observers(observers_) {
    std::iota(model_indexes.begin(), model_indexes.end(), 0);
    profiles_initialization_strategy->initialize_profiles(
      random, host_models,
      0,
      model_indexes.begin(), model_indexes.end());
  }

  Learning::Result execute() {
    CHRONE();

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

    return Learning::Result(host_models->unmake_one(model_indexes.back()), best_accuracy);
  }

 private:
  // @todo Use the same naming convention in all classes (with leading underscores) (even in these "internal" classes)
  uint models_count;
  std::vector<uint> model_indexes;
  std::shared_ptr<ProfilesInitializationStrategy> profiles_initialization_strategy;
  std::shared_ptr<WeightsOptimizationStrategy> weights_optimization_strategy;
  std::shared_ptr<ProfilesImprovementStrategy> profiles_improvement_strategy;
  std::shared_ptr<TerminationStrategy> termination_strategy;

 protected:
  const Domain<Host>& host_domain;
  Models<Host>* host_models;
  RandomNumberGenerator random;

 private:
  std::vector<std::shared_ptr<LearningObserver>> observers;
};

Learning::Result Learning::perform() const {
  CHRONE();

  return LearningExecution(
    _host_domain, _host_models,
    _profiles_initialization_strategy,
    _weights_optimization_strategy,
    _profiles_improvement_strategy,
    _termination_strategy,
    _random, _observers).execute();
}

}  // namespace ppl

// Copyright 2021 Vincent Jacques

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
#include "optimize-weights.hpp"
#include "problem.hpp"
#include "randomness.hpp"
#include "terminate.hpp"


namespace ppl {

/*
The action of learning a model from a learning set.

To be configured using `set_*`, then `perform`ed.
*/
class Learning {
 public:
  // Mandatory parameters
  explicit Learning(
      const Domain<Host>& host_domain,
      Models<Host>* host_models,
      std::shared_ptr<ProfilesInitializationStrategy> profiles_initialization_strategy,
      std::shared_ptr<WeightsOptimizationStrategy> weights_optimization_strategy,
      std::shared_ptr<ProfilesImprovementStrategy> profiles_improvement_strategy,
      std::shared_ptr<TerminationStrategy> termination_strategy) :
    _host_domain(host_domain),
    _host_models(host_models),
    _random_seed(std::random_device()()),
    _profiles_initialization_strategy(profiles_initialization_strategy),
    _weights_optimization_strategy(weights_optimization_strategy),
    _profiles_improvement_strategy(profiles_improvement_strategy),
    _termination_strategy(termination_strategy)
  {}

  // Execution parameters
  Learning& set_random_seed(uint random_seed) {
    _random_seed = random_seed;
    return *this;
  }

  // Observability
  class Observer {
   public:
    virtual void after_main_iteration(int iteration_index, int best_accuracy, const Models<Host>& models) = 0;
  };

  class ProgressReporter : public Observer {
   public:
    void after_main_iteration(int iteration_index, int best_accuracy, const Models<Host>& models) override {
      std::cerr << "After iteration n°" << iteration_index << ": best accuracy = " <<
        best_accuracy << "/" << models.get_view().domain.learning_alternatives_count << std::endl;
    }
  };

  void subscribe(std::shared_ptr<Observer> observer) {
    _observers.push_back(observer);
  }

  // Execution
  struct Result {
    Result(io::Model model, uint accuracy) : best_model(model), best_model_accuracy(accuracy) {}

    io::Model best_model;
    uint best_model_accuracy;
  };

  Result perform() const;

 private:
  const Domain<Host>& _host_domain;
  Models<Host>* _host_models;
  uint _random_seed;
  std::vector<std::shared_ptr<Observer>> _observers;

  // @todo Could we use std::unique_ptr instead of std::shared_ptr?
  std::shared_ptr<ProfilesInitializationStrategy> _profiles_initialization_strategy;
  std::shared_ptr<WeightsOptimizationStrategy> _weights_optimization_strategy;
  std::shared_ptr<ProfilesImprovementStrategy> _profiles_improvement_strategy;
  std::shared_ptr<TerminationStrategy> _termination_strategy;
};

}  // namespace ppl

#endif  // LEARNING_HPP_

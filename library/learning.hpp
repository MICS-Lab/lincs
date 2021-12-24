// Copyright 2021 Vincent Jacques

#ifndef LEARNING_HPP_
#define LEARNING_HPP_

#include <chrono>  // NOLINT(build/c++11)
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "io.hpp"
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
      const io::LearningSet& learning_set,
      std::shared_ptr<TerminationStrategy> termination_strategy) :
    _host_domain(Domain<Host>::make(learning_set)),
    _models_count(9),  // @todo Decide on a good default value
    _random_seed(std::random_device()()),
    _use_gpu(UseGpu::Auto),
    _termination_strategy(termination_strategy)
  {}

  // Execution parameters
  Learning& set_models_count(uint models_count) {
    _models_count = models_count;
    return *this;
  }
  Learning& set_random_seed(uint random_seed) {
    _random_seed = random_seed;
    return *this;
  }

  // Execution space
  Learning& force_using_gpu() {
    _use_gpu = UseGpu::Force;
    return *this;
  }
  Learning& forbid_using_gpu() {
    _use_gpu = UseGpu::Forbid;
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

 public:
  enum class UseGpu { Auto, Force, Forbid };

 private:
  Domain<Host> _host_domain;
  uint _models_count;
  uint _random_seed;
  UseGpu _use_gpu;
  std::vector<std::shared_ptr<Observer>> _observers;

  // @todo Could we use a std::unique_ptr instead of this std::shared_ptr?
  std::shared_ptr<TerminationStrategy> _termination_strategy;
};

}  // namespace ppl

#endif  // LEARNING_HPP_

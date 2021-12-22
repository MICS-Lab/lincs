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


namespace ppl {

/*
The action of learning a model from a learning set.

To be configured using `set_*`, then `perform`ed.
*/
class Learning {
 public:
  // Mandatory parameters
  explicit Learning(const io::LearningSet& learning_set) :
    _host_domain(Domain<Host>::make(learning_set)),
    _target_accuracy(learning_set.alternatives_count),
    _use_gpu(UseGpu::Auto)
  {}

  // Termination criteria
  Learning& set_target_accuracy(uint target_accuracy) {
    _target_accuracy = target_accuracy;
    return *this;
  }
  Learning& set_max_iterations(uint max_iterations) {
    _max_iterations = max_iterations;
    return *this;
  }
  Learning& set_max_duration(std::chrono::steady_clock::duration max_duration) {
    _max_duration = max_duration;
    return *this;
  }

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
    virtual void after_main_iteration(int i, int best_accuracy, const Models<Host>&) = 0;
  };

  class ProgressReporter : public Observer {
   public:
    void after_main_iteration(int i, int best_accuracy, const Models<Host>& models) override {
      std::cerr << "After iteration n°" << i << ": best accuracy = " <<
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
  uint _target_accuracy;
  std::optional<uint> _max_iterations;
  std::optional<std::chrono::steady_clock::duration> _max_duration;
  std::optional<uint> _models_count;
  std::optional<uint> _random_seed;
  UseGpu _use_gpu;
  std::vector<std::shared_ptr<Observer>> _observers;
};

}  // namespace ppl

#endif  // LEARNING_HPP_

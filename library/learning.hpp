// Copyright 2021 Vincent Jacques

#ifndef LEARNING_HPP_
#define LEARNING_HPP_

#include <chrono>  // NOLINT(build/c++11)
#include <optional>
#include <utility>

#include "io.hpp"
#include "problem.hpp"
#include "randomness.hpp"


namespace ppl::learning {

class Learning {
 public:
  // Mandatory parameters
  explicit Learning(const io::LearningSet&);

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

  // Execution
  struct Result {
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
};

}  // namespace ppl::learning

#endif  // LEARNING_HPP_

// Copyright 2021-2022 Vincent Jacques

#ifndef TERMINATE_ITERATIONS_HPP_
#define TERMINATE_ITERATIONS_HPP_

#include "../terminate.hpp"


namespace ppl {

class TerminateAfterIterations : public TerminationStrategy {
 public:
  explicit TerminateAfterIterations(uint max_iterations) :
    _max_iterations(max_iterations) {}

  bool terminate(uint iteration_index, uint /*best_accuracy*/) override {
    return iteration_index >= _max_iterations;
  }

 private:
  uint _max_iterations;
};

}  // namespace ppl

#endif  // TERMINATE_ITERATIONS_HPP_

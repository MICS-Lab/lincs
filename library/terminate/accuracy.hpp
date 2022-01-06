// Copyright 2021-2022 Vincent Jacques

#ifndef TERMINATE_ACCURACY_HPP_
#define TERMINATE_ACCURACY_HPP_

#include "../terminate.hpp"


namespace ppl {

class TerminateAtAccuracy : public TerminationStrategy {
 public:
  explicit TerminateAtAccuracy(uint target_accuracy) :
    _target_accuracy(target_accuracy) {}

  bool terminate(uint /*iteration_index*/, uint best_accuracy) override {
    return best_accuracy >= _target_accuracy;
  }

 private:
  uint _target_accuracy;
};

}  // namespace ppl

#endif  // TERMINATE_ACCURACY_HPP_

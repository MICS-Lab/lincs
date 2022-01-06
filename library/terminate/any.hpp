// Copyright 2021-2022 Vincent Jacques

#ifndef TERMINATE_ANY_HPP_
#define TERMINATE_ANY_HPP_

#include <memory>
#include <vector>

#include "../terminate.hpp"


namespace ppl {

class TerminateOnAny : public TerminationStrategy {
 public:
  explicit TerminateOnAny(std::vector<std::shared_ptr<TerminationStrategy>> strategies) :
    _strategies(strategies) {}

  bool terminate(uint iteration_index, uint best_accuracy) override {
    for (auto strategy : _strategies) {
      if (strategy->terminate(iteration_index, best_accuracy)) {
        return true;
      }
    }

    return false;
  }

 private:
  std::vector<std::shared_ptr<TerminationStrategy>> _strategies;
};

}  // namespace ppl

#endif  // TERMINATE_ANY_HPP_

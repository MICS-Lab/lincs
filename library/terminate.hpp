// Copyright 2021 Vincent Jacques

#ifndef TERMINATE_HPP_
#define TERMINATE_HPP_

#include "uint.hpp"


namespace ppl {

class TerminationStrategy {
 public:
  virtual ~TerminationStrategy() {}

  virtual bool terminate(uint iteration_index, uint best_accuracy) = 0;
};

}  // namespace ppl

#endif  // TERMINATE_HPP_

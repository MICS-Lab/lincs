// Copyright 2021-2022 Vincent Jacques

#ifndef OPTIMIZE_WEIGHTS_HPP_
#define OPTIMIZE_WEIGHTS_HPP_

#include "problem.hpp"


namespace ppl {

class WeightsOptimizationStrategy {
 public:
  virtual ~WeightsOptimizationStrategy() {}

  virtual void optimize_weights(Models<Host>*) = 0;
};

}  // namespace ppl

#endif  // OPTIMIZE_WEIGHTS_HPP_

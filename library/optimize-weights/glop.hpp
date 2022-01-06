// Copyright 2021 Vincent Jacques

#ifndef OPTIMIZE_WEIGHTS_GLOP_HPP_
#define OPTIMIZE_WEIGHTS_GLOP_HPP_

#include <vector>

#include "../optimize-weights.hpp"


namespace ppl {

/*
Implement 3.3.3 of https://tel.archives-ouvertes.fr/tel-01370555/document
using GLOP to solve the linear program.
*/
class OptimizeWeightsUsingGlop : public WeightsOptimizationStrategy {
 public:
  void optimize_weights(Models<Host>*) override;
};

}  // namespace ppl

#endif  // OPTIMIZE_WEIGHTS_GLOP_HPP_

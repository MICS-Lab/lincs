// Copyright 2021 Vincent Jacques

#ifndef OPTIMIZE_WEIGHTS_HPP_
#define OPTIMIZE_WEIGHTS_HPP_

#include "problem.hpp"


namespace ppl {

/*
Implement 3.3.3 of https://tel.archives-ouvertes.fr/tel-01370555/document
*/
class WeightsOptimizer {
 public:
  explicit WeightsOptimizer(const Models<Host>&);

 public:
  void optimize_weights(Models<Host>*);
};

}  // namespace ppl

#endif  // OPTIMIZE_WEIGHTS_HPP_

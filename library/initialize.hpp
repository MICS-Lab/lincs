// Copyright 2021 Vincent Jacques

#ifndef INITIALIZE_HPP_
#define INITIALIZE_HPP_

#include <vector>

#include "problem.hpp"
#include "randomness.hpp"


namespace ppl {

/*
Implement 3.3.2 of https://tel.archives-ouvertes.fr/tel-01370555/document
*/
class ModelsInitializer {
 public:
  explicit ModelsInitializer(const Models<Host>&);

 public:
  void initialize(
    RandomNumberGenerator random,
    Models<Host>* models,
    std::vector<uint>::const_iterator model_indexes_begin,
    std::vector<uint>::const_iterator model_indexes_end);

 private:
  std::vector<std::vector<ProbabilityWeightedGenerator<float>>> _generators;
};

}  // namespace ppl

#endif  // INITIALIZE_HPP_
// Copyright 2021 Vincent Jacques

#ifndef INITIALIZE_PROFILES_HPP_
#define INITIALIZE_PROFILES_HPP_

#include <vector>

#include "problem.hpp"
#include "randomness.hpp"


namespace ppl {

/*
Implement 3.3.2 of https://tel.archives-ouvertes.fr/tel-01370555/document
*/
class ProfilesInitializer {
 public:
  explicit ProfilesInitializer(const Models<Host>&);

 public:
  void initialize_profiles(
    RandomNumberGenerator random,
    Models<Host>* models,
    uint iteration_index,
    std::vector<uint>::const_iterator model_indexes_begin,
    std::vector<uint>::const_iterator model_indexes_end);

 private:
  std::vector<std::vector<ProbabilityWeightedGenerator<float>>> _generators;
};

}  // namespace ppl

#endif  // INITIALIZE_PROFILES_HPP_

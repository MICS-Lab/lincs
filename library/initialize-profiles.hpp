// Copyright 2021 Vincent Jacques

#ifndef INITIALIZE_PROFILES_HPP_
#define INITIALIZE_PROFILES_HPP_

#include <vector>

#include "problem.hpp"
#include "randomness.hpp"


namespace ppl {

class ProfilesInitializationStrategy {
 public:
  virtual ~ProfilesInitializationStrategy() {}

  virtual void initialize_profiles(
    RandomNumberGenerator random,  // @todo Put in ctor
    Models<Host>* models,  // @todo Put in ctor
    uint iteration_index,
    std::vector<uint>::const_iterator model_indexes_begin,
    std::vector<uint>::const_iterator model_indexes_end) = 0;
};

}  // namespace ppl

#endif  // INITIALIZE_PROFILES_HPP_

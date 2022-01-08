// Copyright 2021-2022 Vincent Jacques

#ifndef INITIALIZE_PROFILES_HPP_
#define INITIALIZE_PROFILES_HPP_

#include <memory>
#include <vector>

#include "problem.hpp"


namespace ppl {

class ProfilesInitializationStrategy {
 public:
  virtual ~ProfilesInitializationStrategy() {}

  virtual void initialize_profiles(
    std::shared_ptr<Models<Host>> models,
    uint iteration_index,
    std::vector<uint>::const_iterator model_indexes_begin,
    std::vector<uint>::const_iterator model_indexes_end) = 0;
};

}  // namespace ppl

#endif  // INITIALIZE_PROFILES_HPP_

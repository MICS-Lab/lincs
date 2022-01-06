// Copyright 2021-2022 Vincent Jacques

#ifndef IMPROVE_PROFILES_HPP_
#define IMPROVE_PROFILES_HPP_

#include "problem.hpp"


namespace ppl {

class ProfilesImprovementStrategy {
 public:
  virtual ~ProfilesImprovementStrategy() {}

  virtual void improve_profiles(Models<Host>*) = 0;
};

}  // namespace ppl

#endif  // IMPROVE_PROFILES_HPP_

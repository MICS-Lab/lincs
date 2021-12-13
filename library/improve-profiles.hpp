// Copyright 2021 Vincent Jacques

#ifndef IMPROVE_PROFILES_HPP_
#define IMPROVE_PROFILES_HPP_

#include <vector>

#include "problem.hpp"
#include "randomness.hpp"


namespace ppl {

struct Desirability {
  uint v = 0;
  uint w = 0;
  uint q = 0;
  uint r = 0;
  uint t = 0;

  __host__ __device__
  float value() const {
    if (v + w + t + q + r == 0) {
      // The move has no impact. @todo What should its desirability be?
      return 0;
    } else {
      return (2 * v + w + 0.1 * t) / (v + w + t + 5 * q + r);
    }
  }
};

/*
Implement 3.3.4 (variant 2) of https://tel.archives-ouvertes.fr/tel-01370555/document
*/
class ProfilesImprover {
 public:
  explicit ProfilesImprover(const Models<Host>&);

 public:
  void improve_profiles(const RandomSource& random, Models<Host>*);
  void improve_profiles(const RandomSource& random, Models<Device>*);
};

}  // namespace ppl

#endif  // IMPROVE_PROFILES_HPP_

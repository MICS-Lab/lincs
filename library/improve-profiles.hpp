// Copyright 2021 Vincent Jacques

#ifndef IMPROVE_PROFILES_HPP_
#define IMPROVE_PROFILES_HPP_

#include <vector>

#include "problem.hpp"
#include "randomness.hpp"


namespace ppl::improve_profiles {

uint get_assignment(const Models<Host>&, uint model_index, uint alternative_index);

// Accuracy is returned as an integer between `0` and `models.domain.alternatives_count`.
// (To get the accuracy described in the thesis, it should be devided by `models.domain.alternatives_count`)
uint get_accuracy(const Models<Host>&, uint model_index);
uint get_accuracy(const Models<Device>&, uint model_index);

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

void improve_profiles(const RandomSource& random, Models<Host>*);
void improve_profiles(const RandomSource& random, Models<Device>*);

}  // namespace ppl::improve_profiles

#endif  // IMPROVE_PROFILES_HPP_

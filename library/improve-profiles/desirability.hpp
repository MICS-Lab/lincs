// Copyright 2021-2022 Vincent Jacques

#ifndef IMPROVE_PROFILES_DESIRABILITY_HPP_
#define IMPROVE_PROFILES_DESIRABILITY_HPP_

#include "../problem.hpp"


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

__host__ __device__ Desirability compute_move_desirability(
  const ModelsView&,
  uint model_index,
  uint profile_index,
  uint criterion_index,
  float destination);

}  // namespace ppl

#endif  // IMPROVE_PROFILES_DESIRABILITY_HPP_

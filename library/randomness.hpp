// Copyright 2021 Vincent Jacques

#ifndef RANDOMNESS_HPP_
#define RANDOMNESS_HPP_


#include <curand_kernel.h>

#include <random>

#include "cuda-utils.hpp"


struct RandomSource {
  RandomSource();
  ~RandomSource();

  void init_for_host(int seed);

  void init_for_device(int seed);

  curandState* rng_states;
  std::mt19937* gen;
};

struct RandomNumberGenerator {
  RandomNumberGenerator(const RandomSource& source) :  // NOLINT(runtime/explicit)
    _rng_states(source.rng_states), _gen(source.gen) {}

  __host__ __device__
  float uniform_float(const float min, const float max);

  __host__ __device__
  uint uniform_int(const uint min, const uint max);

 private:
  curandState* _rng_states;
  std::mt19937* _gen;
};


#endif  // RANDOMNESS_HPP_

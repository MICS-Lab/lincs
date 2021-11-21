// Copyright 2021 Vincent Jacques

#ifndef RANDOMNESS_HPP_
#define RANDOMNESS_HPP_


#include <curand_kernel.h>

#include "cuda-utils.hpp"


__global__ void initialize_rng(curandState* const rng_states, const uint seed);

struct RandomNumberGenerator {
  RandomNumberGenerator() : _rng_states(nullptr) {}

  void init_for_host() {
  }

  void init_for_device() {
    _rng_states = alloc_host<curandState>(1024);
    checkCudaErrors();
    initialize_rng<<<1, 1024>>>(_rng_states, 43);
    cudaDeviceSynchronize();
    checkCudaErrors();
  }

  __host__ __device__
  float uniform_float(const float min, const float max) {
    #ifdef __CUDA_ARCH__
    return min + (max - min) * curand_uniform(_rng_states);
    #else
    // @todo Implement using <random>
    return min + (max - min) * static_cast<float>(rand()) / RAND_MAX;  // NOLINT(runtime/threadsafe_fn)
    #endif
  }

  __host__ __device__
  uint uniform_int(const uint min, const uint max) {
    // @todo Implement
    return 0;
  }

 private:
  curandState* _rng_states;
};


#endif  // RANDOMNESS_HPP_

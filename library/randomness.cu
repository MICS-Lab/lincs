// Copyright 2021 Vincent Jacques

#include "randomness.hpp"

__global__ void initialize_rng(curandState* const rng_states, const uint seed) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, tid, 0, &rng_states[tid]);
}

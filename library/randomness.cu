// Copyright 2021 Vincent Jacques

#include "randomness.hpp"

#include <random>


RandomSource::RandomSource() :
  rng_states(nullptr),
  gen(nullptr) {
}

RandomSource::~RandomSource() {
  free_device(rng_states);
  if (gen != nullptr) delete gen;
}

void RandomSource::init_for_host(int seed) {
  gen = new std::mt19937(seed);
}

__global__ void initialize_rng(curandState* rng_states, const unsigned int seed) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  assert(tid < 1024);

  curand_init(seed, tid, 0, &rng_states[tid]);
}

void RandomSource::init_for_device(int seed) {
  rng_states = alloc_device<curandState>(1024);
  checkCudaErrors();
  initialize_rng<<<1, 1024>>>(rng_states, seed);
  cudaDeviceSynchronize();
  checkCudaErrors();
}

__host__ __device__
float RandomNumberGenerator::uniform_float(const float min, const float max) {
  #ifdef __CUDA_ARCH__
  const unsigned int thread_index = threadIdx.x + blockIdx.x * gridDim.x;
  #endif
  float v = max;
  do {
    #ifdef __CUDA_ARCH__
      v = min + (max - min) * curand_uniform(&_rng_states[thread_index]);
    #else
      v = std::uniform_real_distribution<float>(min, max)(*_gen);
    #endif
  } while (v == max);
  return v;
}

__host__ __device__
unsigned int RandomNumberGenerator::uniform_int(const unsigned int min, const unsigned int max) {
  #ifdef __CUDA_ARCH__
    const unsigned int thread_index = threadIdx.x + blockIdx.x * gridDim.x;
    return min + curand(&_rng_states[thread_index]) % (max - min);
  #else
    return std::uniform_int_distribution<unsigned int>(min, max - 1)(*_gen);
  #endif
}

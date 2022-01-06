// Copyright 2021-2022 Vincent Jacques

#include "randomness.hpp"

#include <random>


RandomSource::RandomSource() :
  rng_states(nullptr),
  gen(nullptr) {
}

RandomSource::~RandomSource() {
  free_device(rng_states);
  if (gen != nullptr) delete[] gen;
}

void RandomSource::init_for_host(int seed) {
  const int threads_count = omp_get_max_threads();
  gen = new std::mt19937[threads_count];

  #pragma omp parallel
  {
    const int thread_index = omp_get_thread_num();
    assert(thread_index < threads_count);
    gen[thread_index].seed(seed * (thread_index + 1));
  }
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
  #else
  const int thread_index = omp_get_thread_num();
  #endif
  float v = max;
  do {
    #ifdef __CUDA_ARCH__
      v = min + (max - min) * curand_uniform(&_rng_states[thread_index]);
    #else
      v = std::uniform_real_distribution<float>(min, max)(_gen[thread_index]);
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
    const int thread_index = omp_get_thread_num();
    return std::uniform_int_distribution<unsigned int>(min, max - 1)(_gen[thread_index]);
  #endif
}

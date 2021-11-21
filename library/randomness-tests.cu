// Copyright 2021 Vincent Jacques

#include "randomness.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "matrix-view.hpp"
#include "cuda-utils.hpp"


TEST(OnHost, UniformityOfFloat) {
  RandomSource source;
  source.init_for_host(5489);
  RandomNumberGenerator random(source);

  std::vector<unsigned int> histogram(1000, 0);
  for (unsigned int i = 0; i < 10'000'000; ++i) {
    float v = random.uniform_float(1000, 2000);
    ASSERT_GE(v, 1000);
    ASSERT_LT(v, 2000);
    ++histogram.at(std::floor(v - 1000));
  }
  for (unsigned int count : histogram) {
    EXPECT_GE(count, 9'500);
    EXPECT_LE(count, 10'500);
  }
}

__global__ void UniformityOfFloat__kernel(RandomNumberGenerator random, MatrixView1D<unsigned int> histogram) {
  for (unsigned int i = 0; i != 10000; ++i) {
    float v = random.uniform_float(1000, 2000);
    if (v == 2000) printf("Seen 2000\n");
    atomicInc(&histogram[std::floor(v - 1000)], 1000000);
  }
}

TEST(OnDevice, UniformityOfFloat) {
  RandomSource source;
  source.init_for_device(5489);
  RandomNumberGenerator random(source);

  unsigned int* device_histogram = alloc_device<unsigned int>(1000);

  UniformityOfFloat__kernel<<<1, 1000>>>(random, MatrixView1D(1000, device_histogram));
  cudaDeviceSynchronize();
  checkCudaErrors();

  std::vector<unsigned int> histogram(1000);
  copy_device_to_host(1000, device_histogram, histogram.data());
  free_device(device_histogram);

  for (unsigned int count : histogram) {
    EXPECT_GE(count, 9'500);
    EXPECT_LE(count, 10'500);
  }
}

TEST(OnHost, UniformityOfInt) {
  RandomSource source;
  source.init_for_host(5489);
  RandomNumberGenerator random(source);

  std::vector<unsigned int> histogram(1000, 0);
  for (unsigned int i = 0; i < 10000000; ++i) {
    int v = random.uniform_int(1000, 2000);
    ASSERT_GE(v, 1000);
    ASSERT_LT(v, 2000);
    ++histogram.at(v - 1000);
  }
  for (unsigned int count : histogram) {
    EXPECT_GE(count, 9500);
    EXPECT_LE(count, 10500);
  }
}

__global__ void UniformityOfInt__kernel(RandomNumberGenerator random, MatrixView1D<unsigned int> histogram) {
  for (unsigned int i = 0; i != 10000; ++i) {
    int v = random.uniform_int(1000, 2000);
    atomicInc(&histogram[v - 1000], 1000000);
  }
}

TEST(OnDevice, UniformityOfInt) {
  RandomSource source;
  source.init_for_device(5489);
  RandomNumberGenerator random(source);

  unsigned int* device_histogram = alloc_device<unsigned int>(1000);

  UniformityOfInt__kernel<<<1, 1000>>>(random, MatrixView1D(1000, device_histogram));
  cudaDeviceSynchronize();
  checkCudaErrors();

  std::vector<unsigned int> histogram(1000);
  copy_device_to_host(1000, device_histogram, histogram.data());
  free_device(device_histogram);

  for (unsigned int count : histogram) {
    EXPECT_GE(count, 9500);
    EXPECT_LE(count, 10500);
  }
}

// Copyright 2021-2022 Vincent Jacques

#include <gtest/gtest.h>

#include <cmath>

#include "cuda-utils.hpp"
#include "randomness.hpp"


TEST(OnHost, UniformityOfFloat) {
  RandomSource source;
  source.init_for_host(5489);
  RandomNumberGenerator random(source);

  std::vector<std::vector<unsigned int>> histograms(omp_get_max_threads(), std::vector<unsigned int>(1000, 0));

  #pragma omp parallel for
  for (unsigned int i = 0; i < 10'000'000; ++i) {
    float v = random.uniform_float(1000, 2000);
    assert(v >= 1000);
    assert(v < 2000);
    ++histograms[omp_get_thread_num()][std::floor(v - 1000)];
  }

  std::vector<unsigned int> histogram(1000, 0);
  for (auto& hist : histograms) {
    for (int i = 0; i != 1000; ++i) {
      histogram[i] += hist[i];
    }
  }

  ASSERT_EQ(std::accumulate(histogram.begin(), histogram.end(), 0), 10'000'000);

  for (unsigned int count : histogram) {
    EXPECT_GE(count, 9'500);
    EXPECT_LE(count, 10'500);
  }
}

__global__ void UniformityOfFloat__kernel(RandomNumberGenerator random, ArrayView1D<Device, unsigned int> histogram) {
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

  Array1D<Device, unsigned> device_histogram(1000, zeroed);

  UniformityOfFloat__kernel<<<1, 1000>>>(random, device_histogram);
  check_last_cuda_error();

  auto histogram = device_histogram.clone_to<Host>();

  for (unsigned int i = 0; i != 1000; ++i) {
    const unsigned count = histogram[i];
    EXPECT_GE(count, 9'500);
    EXPECT_LE(count, 10'500);
  }
}

TEST(OnHost, UniformityOfInt) {
  RandomSource source;
  source.init_for_host(5489);
  RandomNumberGenerator random(source);

  std::vector<std::vector<unsigned int>> histograms(omp_get_max_threads(), std::vector<unsigned int>(1000, 0));

  #pragma omp parallel for
  for (unsigned int i = 0; i < 10'000'000; ++i) {
    int v = random.uniform_int(1000, 2000);
    assert(v >= 1000);
    assert(v < 2000);
    ++histograms[omp_get_thread_num()][v - 1000];
  }

  std::vector<unsigned int> histogram(1000, 0);
  for (auto& hist : histograms) {
    for (int i = 0; i != 1000; ++i) {
      histogram[i] += hist[i];
    }
  }

  ASSERT_EQ(std::accumulate(histogram.begin(), histogram.end(), 0), 10'000'000);

  for (unsigned int count : histogram) {
    EXPECT_GE(count, 9'500);
    EXPECT_LE(count, 10'500);
  }
}

__global__ void UniformityOfInt__kernel(RandomNumberGenerator random, ArrayView1D<Device, unsigned int> histogram) {
  for (unsigned int i = 0; i != 10000; ++i) {
    int v = random.uniform_int(1000, 2000);
    atomicInc(&histogram[v - 1000], 1000000);
  }
}

TEST(OnDevice, UniformityOfInt) {
  RandomSource source;
  source.init_for_device(5489);
  RandomNumberGenerator random(source);

  Array1D<Device, unsigned> device_histogram(1000, zeroed);

  UniformityOfInt__kernel<<<1, 1000>>>(random, device_histogram);
  check_last_cuda_error();

  auto histogram = device_histogram.clone_to<Host>();

  for (unsigned int i = 0; i != 1000; ++i) {
    const unsigned count = histogram[i];
    EXPECT_GE(count, 9500);
    EXPECT_LE(count, 10500);
  }
}

TEST(ProbabilityWeightedGenerator, Single) {
  auto g = ProbabilityWeightedGenerator<uint>::make({{42, 2}});

  EXPECT_EQ(g.get_value_probabilities(), (std::map<uint, double> {{42, 1}}));

  std::mt19937 gen;
  EXPECT_EQ(g(gen), 42);
}

TEST(ProbabilityWeightedGenerator, Three) {
  auto g = ProbabilityWeightedGenerator<uint>::make({{0, 1}, {1, 10}, {2, 9}});

  EXPECT_EQ(g.get_value_probabilities(), (std::map<uint, double> {{0, 0.05}, {1, 0.5}, {2, 0.45}}));

  std::mt19937 gen(42);
  std::vector<uint> histogram(3);
  for (uint i = 0; i != 10000; ++i) {
    ++histogram[g(gen)];
  }
  EXPECT_EQ(histogram[0], 526);  // 5%
  EXPECT_EQ(histogram[1], 4947);  // 50%
  EXPECT_EQ(histogram[2], 4527);  // 45%
}

// Copyright 2021 Vincent Jacques

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include "median-and-max.hpp"


TEST(EnsureMedianAndMax, Empty) {
  std::vector<int> v;
  ensure_median_and_max(v.begin(), v.end(), std::less<int>());
  // Nothing to assert, this test just checks for invalid memory access (assuming it's run in Valgrind)
}

TEST(EnsureMedianAndMax, OneElement) {
  std::vector<int> v(1, 0);
  ensure_median_and_max(v.begin(), v.end(), std::less<int>());
  // Nothing to assert, this test just checks for invalid memory access (assuming it's run in Valgrind)
}

TEST(EnsureMedianAndMax, TwoElements) {
  std::vector<int> v {1, 0};
  ensure_median_and_max(v.begin(), v.end(), std::less<int>());
  EXPECT_EQ(v[0], 0);  // Median
  EXPECT_EQ(v[1], 1);  // Max
}

TEST(EnsureMedianAndMax, ThreeElements) {
  std::vector<int> v {2, 1, 0};
  ensure_median_and_max(v.begin(), v.end(), std::less<int>());
  EXPECT_EQ(v[0], 0);  // Below median
  EXPECT_EQ(v[1], 1);  // Median
  EXPECT_EQ(v[2], 2);  // Max
}

TEST(EnsureMedianAndMax, FourElements) {
  std::vector<int> v {3, 2, 1, 0};
  ensure_median_and_max(v.begin(), v.end(), std::less<int>());
  EXPECT_EQ(v[0], 0);  // Below median
  EXPECT_EQ(v[1], 1);  // Median
  EXPECT_EQ(v[2], 2);  // Above median
  EXPECT_EQ(v[3], 3);  // Max
}

TEST(EnsureMedianAndMax, TenElements) {
  std::vector<int> v(10, 0);
  std::iota(v.begin(), v.end(), 0);
  ensure_median_and_max(v.begin(), v.end(), std::less<int>());
  for (int i = 0; i != 5; ++i) {
    EXPECT_LT(v[i], 5);  // Below median
  }
  EXPECT_EQ(v[5], 5);  // Median
  for (int i = 6; i != 9; ++i) {
    EXPECT_GT(v[i], 5);  // Above median
  }
  EXPECT_EQ(v[9], 9);  // Max
}

TEST(EnsureMedianAndMax, ManyElements) {
  std::vector<int> v(99, 0);
  std::iota(v.begin(), v.end(), 0);
  std::random_shuffle(v.begin(), v.end());

  ensure_median_and_max(v.begin(), v.end(), std::less<int>());
  for (int i = 0; i != 49; ++i) {
    EXPECT_LT(v[i], 49);  // Below median
  }
  EXPECT_EQ(v[49], 49);  // Median
  for (int i = 50; i != 98; ++i) {
    EXPECT_GT(v[i], 49);  // Above median
  }
  EXPECT_EQ(v[98], 98);  // Max
}

TEST(EnsureMedianAndMax, ReverseOrder) {
  std::vector<int> v(99, 0);
  std::iota(v.begin(), v.end(), 0);
  std::random_shuffle(v.begin(), v.end());

  ensure_median_and_max(v.begin(), v.end(), std::greater<int>());
  for (int i = 0; i != 49; ++i) {
    EXPECT_GT(v[i], 49);  // Below median, which is actually above because we're comparing with std::greater
  }
  EXPECT_EQ(v[49], 49);  // Median
  for (int i = 50; i != 98; ++i) {
    EXPECT_LT(v[i], 49);  // Above median, which is actually below because we're comparing with std::greater
  }
  EXPECT_EQ(v[98], 0);  // Max, which is actually the min because we're comparing with std::greater
}

TEST(EnsureMedianAndMax, RepeatedRandom) {
  std::vector<int> v(99, 0);
  std::iota(v.begin(), v.end(), 0);

  for (int i = 0; i != 100; ++i) {
    std::random_shuffle(v.begin(), v.end());

    ensure_median_and_max(v.begin(), v.end(), std::less<int>());
    for (int i = 0; i != 49; ++i) {
      EXPECT_LT(v[i], 49);  // Below median
    }
    EXPECT_EQ(v[49], 49);  // Median
    for (int i = 50; i != 98; ++i) {
      EXPECT_GT(v[i], 49);  // Above median
    }
    EXPECT_EQ(v[98], 98);  // Max
  }
}

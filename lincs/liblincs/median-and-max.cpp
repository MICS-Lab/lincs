// Copyright 2023 Vincent Jacques

#include "median-and-max.hpp"

#include <numeric>
#include <random>

#include <doctest.h>  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


TEST_CASE("ensure_median_and_max(empty)") {
  std::vector<int> v;

  ensure_median_and_max(v.begin(), v.end(), std::less<int>());
  // This test only checks for invalid memory accesses
}

TEST_CASE("ensure_median_and_max(one_element)") {
  std::vector<int> v(1);

  ensure_median_and_max(v.begin(), v.end(), std::less<int>());
  // This test only checks for invalid memory accesses
}

TEST_CASE("ensure_median_and_max(two_elements)") {
  std::vector<int> v{1, 0};

  ensure_median_and_max(v.begin(), v.end(), std::less<int>());

  CHECK(v[0] == 0);  // Median
  CHECK(v[1] == 1);  // Max
}

TEST_CASE("ensure_median_and_max(three_elements)") {
  std::vector<int> v(3);
  std::iota(v.begin(), v.end(), 0);
  std::random_shuffle(v.begin(), v.end());

  ensure_median_and_max(v.begin(), v.end(), std::less<int>());

  CHECK(v[0] == 0);  // Below median
  CHECK(v[1] == 1);  // Median
  CHECK(v[2] == 2);  // Max
}

TEST_CASE("ensure_median_and_max(four_elements)") {
  std::vector<int> v(4);
  std::iota(v.begin(), v.end(), 0);
  std::random_shuffle(v.begin(), v.end());


  ensure_median_and_max(v.begin(), v.end(), std::less<int>());

  CHECK(v[0] == 0);  // Below median
  CHECK(v[1] == 1);  // Median
  CHECK(v[2] == 2);  // Above median
  CHECK(v[3] == 3);  // Max
}

TEST_CASE("ensure_median_and_max(ten_elements)") {
  std::vector<int> v(10);
  std::iota(v.begin(), v.end(), 0);
  std::random_shuffle(v.begin(), v.end());

  ensure_median_and_max(v.begin(), v.end(), std::less<int>());

  for (int i = 0; i != 5; ++i) {
    CHECK(v[i] < 5);  // Below median
  }
  CHECK(v[5] == 5);  // Median
  for (int i = 6; i != 9; ++i) {
    CHECK(v[i] > 5);  // Above median
  }
  CHECK(v[9] == 9);  // Max
}

TEST_CASE("ensure_median_and_max(many_elements)") {
  std::vector<int> v(99);
  std::iota(v.begin(), v.end(), 0);
  std::random_shuffle(v.begin(), v.end());

  ensure_median_and_max(v.begin(), v.end(), std::less<int>());

  for (int i = 0; i != 49; ++i) {
    CHECK(v[i] < 49);  // Below median
  }
  CHECK(v[49] == 49);  // Median
  for (int i = 50; i != 98; ++i) {
    CHECK(v[i] > 49);  // Above median
  }
  CHECK(v[98] == 98);  // Max
}

TEST_CASE("ensure_median_and_max(reversed_order)") {
  std::vector<int> v(99);
  std::iota(v.begin(), v.end(), 0);
  std::random_shuffle(v.begin(), v.end());

  ensure_median_and_max(v.begin(), v.end(), std::greater<int>());

  // We're comparing with std::greater, so order is reversed compared to previous test
  for (int i = 0; i != 49; ++i) {
    CHECK(v[i] > 49);  // Below median = greater than median
  }
  CHECK(v[49] == 49);  // Median
  for (int i = 50; i != 98; ++i) {
    CHECK(v[i] < 49);  // Above median = less than median
  }
  CHECK(v[98] == 0);  // Max = min
}

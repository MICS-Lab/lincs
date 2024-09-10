// Copyright 2024 Vincent Jacques

#ifndef LINCS__LINEAR_PROGRAMMING__TESTING_HPP
#define LINCS__LINEAR_PROGRAMMING__TESTING_HPP

#include <limits>
#include <random>

#include "alglib.hpp"
#include "custom-on-cpu.hpp"
#include "glop.hpp"

#include "../vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


const float infinity = std::numeric_limits<float>::infinity();

inline float relative_difference(float a, float b) {
  if (a == b) {  // Handle infinities
    return 0;
  } else if (std::isnan(a) && std::isnan(b)) {
    return 0;
  } else if (a == 0 || b == 0) {
    return std::max(std::abs(a), std::abs(b));
  } else if (std::abs(a) == infinity || std::abs(b) == infinity) {
    return 1;
  } else {
    return std::abs(a - b) / std::max(std::abs(a), std::abs(b));
  }
}

#define CHECK_NEAR(a, b) CHECK(relative_difference(a, b) < 1e-5)


typedef std::tuple<
  lincs::GlopLinearProgram,
  lincs::AlglibLinearProgram,
  lincs::CustomOnCpuLinearProgram
> LinearPrograms;

template<unsigned Index, typename... Float>
void check_all_equal_impl(const std::tuple<Float...>& costs) {
  static_assert(0 < Index);
  static_assert(Index <= sizeof...(Float));
  if constexpr (Index < sizeof...(Float)) {
    CHECK_NEAR(std::get<0>(costs), std::get<Index>(costs));
    check_all_equal_impl<Index + 1>(costs);
  }
}

template<typename... Float>
void check_all_equal(const std::tuple<Float...>& costs) {
  check_all_equal_impl<1>(costs);
}

template <typename F>
void test(F&& f) {
  LinearPrograms linear_programs;
  const auto costs = std::apply([&f](auto&... linear_program) { return std::make_tuple(f(linear_program)...); }, linear_programs);
  check_all_equal(costs);
}

#endif  // LINCS__LINEAR_PROGRAMMING__TESTING_HPP

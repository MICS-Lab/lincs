// Copyright 2024 Vincent Jacques

#ifndef LINCS__LINEAR_PROGRAMMING__TESTING_HPP
#define LINCS__LINEAR_PROGRAMMING__TESTING_HPP

#include <limits>
#include <random>

#include "alglib.hpp"
#include "custom-on-cpu.hpp"
#include "glop.hpp"

#include "../vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


constexpr float infinity = std::numeric_limits<float>::infinity();

inline float relative_difference(float a, float b) {
  assert(!std::isnan(a) && !std::isnan(b));
  assert(std::abs(a) != infinity && std::abs(b) != infinity);
  if (a == 0 || b == 0) {
    return std::max(std::abs(a), std::abs(b));
  } else {
    return std::abs(a - b) / std::max(std::abs(a), std::abs(b));
  }
}

#define CHECK_NEAR(a, b) CHECK(relative_difference(a, b) < 1e-4)


typedef std::tuple<
  lincs::GlopLinearProgram,
  lincs::AlglibLinearProgram,
  lincs::CustomOnCpuLinearProgram
> LinearPrograms;

template<unsigned Index, typename... Float>
void check_all_equal_impl(const std::tuple<std::optional<Float>...>& costs) {
  static_assert(0 < Index);
  static_assert(Index <= sizeof...(Float));
  if constexpr (Index < sizeof...(Float)) {
    if (std::get<0>(costs)) {
      if (std::get<Index>(costs)) {
        CHECK_NEAR(*std::get<0>(costs), *std::get<Index>(costs));
      } else {
        CHECK(false);
      }
    } else {
      CHECK_FALSE(std::get<Index>(costs));
    }
    check_all_equal_impl<Index + 1>(costs);
  }
}

template<typename... Float>
void check_all_equal(const std::tuple<std::optional<Float>...>& costs) {
  check_all_equal_impl<1>(costs);
}

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

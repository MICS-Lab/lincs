// Copyright 2023-2024 Vincent Jacques

#include <limits>
#include <random>

#include "alglib.hpp"
#include "glop.hpp"

#include "../vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


typedef std::tuple<
  lincs::GlopLinearProgram,
  lincs::AlglibLinearProgram
> LinearPrograms;

template<unsigned Index, typename... Float>
void check_all_equal_impl(const std::tuple<Float...>& costs) {
  static_assert(0 < Index);
  static_assert(Index <= sizeof...(Float));
  if constexpr (Index < sizeof...(Float)) {
    CHECK(std::abs(std::get<0>(costs) - std::get<Index>(costs)) < 1e-6);
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

const float infinity = std::numeric_limits<float>::infinity();

TEST_CASE("'Small house' linear program") {
  test([](auto& linear_program) {
    auto x = linear_program.create_variable();
    auto y = linear_program.create_variable();
    linear_program.mark_all_variables_created();

    // Objective: minimize F(x, y) = -0.1 * x - y
    linear_program.set_objective_coefficient(x, -0.1);
    linear_program.set_objective_coefficient(y, -1);

    // Constraints:
    // -1 <= x <= +1
    linear_program.create_constraint().set_coefficient(x, 1).set_bounds(-1, 1);
    // -1 <= y <= +1
    linear_program.create_constraint().set_coefficient(y, 1).set_bounds(-1, 1);
    // -1 <= x - y
    linear_program.create_constraint().set_coefficient(x, 1).set_coefficient(y, -1).set_bounds(-1, +infinity);
    // x + y <= 1
    linear_program.create_constraint().set_coefficient(x, 1).set_coefficient(y, 1).set_bounds(-infinity, 1);

    // Optimal solution:
    // F(0, 1) = -1
    auto solution = linear_program.solve();
    CHECK(std::abs(solution.assignments[x] - 0) < 1e-6);
    CHECK(std::abs(solution.assignments[y] - 1) < 1e-6);
    CHECK(std::abs(solution.cost - -1) < 1e-6);

    return solution.cost;
  });
}

TEST_CASE("Random linear programs with optimal solutions") {
  for (unsigned seed = 0; seed != 10'000; ++seed) {
    test([seed](auto& linear_program) {
      CAPTURE(seed);
      
      std::mt19937 mt(seed);

      std::uniform_int_distribution<unsigned> make_variables_count(2, 10);
      const unsigned variables_count = make_variables_count(mt);
      std::uniform_real_distribution<float> make_objective_coefficient(-3, 3);
      std::uniform_int_distribution<unsigned> make_constraints_count(0, 2 * variables_count);
      const unsigned constraints_count = make_constraints_count(mt);
      std::uniform_int_distribution<int> make_constraint_coefficient(-1, 1);

      std::vector<decltype(linear_program.create_variable())> variables;
      for (unsigned i = 0; i != variables_count; ++i) {
        variables.push_back(linear_program.create_variable());
      }

      linear_program.mark_all_variables_created();

      // Give a coefficient to each variable
      // @todo(Project management, when we release our in-house LP solvers) Let some variables not appear in the objective
      std::vector<float> objective_coefficients;
      objective_coefficients.resize(variables_count);
      for (unsigned i = 0; i != variables_count; ++i) {
        const float coefficient = make_objective_coefficient(mt);
        objective_coefficients[i] = coefficient;
        linear_program.set_objective_coefficient(variables[i], coefficient);
      }

      // Box all variables to ensure the problem is bounded
      for (const auto& v : variables) {
        auto c = linear_program.create_constraint();
        c.set_bounds(-1, 1);
        c.set_coefficient(v, 1);
      }

      for (unsigned i = 0; i != constraints_count; ++i) {
        auto c = linear_program.create_constraint();
        c.set_bounds(-1, 1);
        // @todo(Project management, when we release our in-house LP solvers) Let some variables not appear in some constraints
        for (const auto& v : variables) {
          c.set_coefficient(v, make_constraint_coefficient(mt));
        }
      }

      auto solution = linear_program.solve();

      float expected_cost = 0;
      for (unsigned i = 0; i != variables_count; ++i) {
        expected_cost += objective_coefficients[i] * solution.assignments[variables[i]];
      }
      CHECK(std::abs(solution.cost - expected_cost) < 2e-6);

      return solution.cost;
    });
  }
}

// @todo(Project management, when we release our in-house LP solvers) Test consistency on all kinds of linear programs (unbounded, infeasible, others?)

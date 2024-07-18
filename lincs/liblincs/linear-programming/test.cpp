// Copyright 2023-2024 Vincent Jacques

#include <limits>
#include <random>

#include "alglib.hpp"
#include "glop.hpp"

#include "../vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


template<typename LinearProgram>
void test_linear_program() {
  const float infinity = std::numeric_limits<float>::infinity();

  LinearProgram lp;

  // Objective: minimize:
  //     F(x0, x1) = -0.1 * x0 - x1
  auto x0 = lp.create_variable();
  auto x1 = lp.create_variable();

  lp.mark_all_variables_created();

  lp.set_objective_coefficient(x0, -0.1);
  lp.set_objective_coefficient(x1, -1);

  // Constraints:
  //     -1 <= x0 <= +1
  {
    auto c = lp.create_constraint();
    c.set_bounds(-1, 1);
    c.set_coefficient(x0, 1);
  }

  //     -1 <= x1 <= +1
  {
    auto c = lp.create_constraint();
    c.set_bounds(-1, 1);
    c.set_coefficient(x1, 1);
  }

  //     x0 - x1 >= -1
  {
    auto c = lp.create_constraint();
    c.set_bounds(-1, +infinity);
    c.set_coefficient(x0, 1);
    c.set_coefficient(x1, -1);
  }

  //     x0 + x1 <= 1
  {
    auto c = lp.create_constraint();
    c.set_bounds(-infinity, 1);
    c.set_coefficient(x0, 1);
    c.set_coefficient(x1, 1);
  }

  // Expected solution:
  //     x0 = 0
  //     x1 = 1
  // with cost = -1
  auto solution = lp.solve();
  CHECK(std::abs(solution.assignments[x0] - 0) < 1e-6);
  CHECK(std::abs(solution.assignments[x1] - 1) < 1e-6);
  CHECK(std::abs(solution.cost - -1) < 1e-6);
}

TEST_CASE("GLOP linear program") {
  test_linear_program<lincs::GlopLinearProgram>();
}

TEST_CASE("Alglib linear program") {
  test_linear_program<lincs::AlglibLinearProgram>();
}

typedef std::tuple<
  lincs::GlopLinearProgram,
  lincs::AlglibLinearProgram
> LinearPrograms;

TEST_CASE("Linear program solvers consistency on programs with optimal solutions") {
  for (unsigned seed = 0; seed != 10'000; ++seed) {
    CAPTURE(seed);

    LinearPrograms linear_programs;

    const auto costs = std::apply(
      [seed](auto&... linear_program) {
        return std::make_tuple(([seed, &linear_program]() {
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
        })()...);
      },
      linear_programs
    );

    CHECK(std::abs(std::get<0>(costs) - std::get<1>(costs)) < 1e-6);
  }
}

// @todo(Project management, when we release our in-house LP solvers) Test consistency on all kinds of linear programs (unbounded, infeasible, others?)

// Copyright 2023 Vincent Jacques

#include <limits>

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
  auto solution = lp.solve();
  CHECK(std::abs(solution[x0] - 0) < 1e-6);
  CHECK(std::abs(solution[x1] - 1) < 1e-6);
}

TEST_CASE("GLOP linear program") {
  test_linear_program<lincs::GlopLinearProgram>();
}

TEST_CASE("Alglib linear program") {
  test_linear_program<lincs::AlglibLinearProgram>();
}

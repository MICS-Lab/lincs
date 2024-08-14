// Copyright 2023-2024 Vincent Jacques

#include <limits>
#include <random>

#include "alglib.hpp"
#include "custom-on-cpu.hpp"
#include "glop.hpp"

#include "../vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


float relative_difference(float a, float b) {
  if (a == 0 || b == 0) {
    return std::max(std::abs(a), std::abs(b));
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

const float infinity = std::numeric_limits<float>::infinity();

// @todo(Project management, when we release our in-house LP solvers) Add tests from https://www4.uwsp.edu/math/afelt/slptestset/download.html

TEST_CASE("Example 1.1 of https://webspace.maths.qmul.ac.uk/felix.fischer/teaching/opt/notes/notes.pdf") {
  test([](auto& linear_program){
    const auto x1 = linear_program.create_variable();
    const auto x2 = linear_program.create_variable();
    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x1, -1);
    linear_program.set_objective_coefficient(x2, -1);

    linear_program.create_constraint().set_coefficient(x1, 1).set_coefficient(x2, 2).set_bounds(-infinity, 6);
    linear_program.create_constraint().set_coefficient(x1, 1).set_coefficient(x2, -1).set_bounds(-infinity, 3);
    const auto solution = linear_program.solve();

    CHECK_NEAR(solution.assignments[x1], 4);
    CHECK_NEAR(solution.assignments[x2], 1);
    CHECK_NEAR(solution.cost, -5);

    return solution.cost;
  });
}

TEST_CASE("Simplex orthogonal to objective gradient, origin feasible but not optimal") {
  test([](auto& linear_program) {
    const auto x = linear_program.create_variable();
    const auto y = linear_program.create_variable();
    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x, -1);
    linear_program.set_objective_coefficient(y, -1);

    linear_program.create_constraint().set_coefficient(x, 1).set_coefficient(y, 1).set_bounds(-infinity, +1);

    const auto solution = linear_program.solve();

    // Can't check the assignments because they are not unique
    CHECK_NEAR(solution.cost, -1);

    return solution.cost;
  });
}

TEST_CASE("'Small house' linear program") {
  test([](auto& linear_program) {
    const auto x = linear_program.create_variable();
    const auto y = linear_program.create_variable();
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
    const auto solution = linear_program.solve();
    CHECK_NEAR(solution.assignments[x], 0);
    CHECK_NEAR(solution.assignments[y], 1);
    CHECK_NEAR(solution.cost, -1);

    return solution.cost;
  });
}

TEST_CASE("Octagon linear program - solution on octagon") {
  test([](auto& linear_program) {
    const auto x = linear_program.create_variable();
    const auto y = linear_program.create_variable();

    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x, -1);
    linear_program.set_objective_coefficient(y, -1);

    // Octagon centered on (0,0)
    linear_program.create_constraint().set_coefficient(x, 2).set_coefficient(y, 1).set_bounds(-infinity, 3);
    linear_program.create_constraint().set_coefficient(x, 1).set_coefficient(y, 2).set_bounds(-infinity, 3);
    linear_program.create_constraint().set_coefficient(x, 2).set_coefficient(y, -1).set_bounds(-infinity, 3);
    linear_program.create_constraint().set_coefficient(x, 1).set_coefficient(y, -2).set_bounds(-infinity, 3);
    linear_program.create_constraint().set_coefficient(x, -2).set_coefficient(y, 1).set_bounds(-infinity, 3);
    linear_program.create_constraint().set_coefficient(x, -1).set_coefficient(y, 2).set_bounds(-infinity, 3);
    linear_program.create_constraint().set_coefficient(x, -2).set_coefficient(y, -1).set_bounds(-infinity, 3);
    linear_program.create_constraint().set_coefficient(x, -1).set_coefficient(y, -2).set_bounds(-infinity, 3);

    const auto solution = linear_program.solve();
    CHECK_NEAR(solution.assignments[x], 1);
    CHECK_NEAR(solution.assignments[y], 1);
    CHECK_NEAR(solution.cost, -2);

    return solution.cost;
  });
}

TEST_CASE("Octagon linear program - solution on origin (Because of implicit positivity constraints on each variable!)") {
  test([](auto& linear_program) {
    const auto x = linear_program.create_variable();
    const auto y = linear_program.create_variable();

    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x, 1);
    linear_program.set_objective_coefficient(y, 1);

    // Octagon centered on (0,0)
    linear_program.create_constraint().set_coefficient(x, 2).set_coefficient(y, 1).set_bounds(-infinity, 3);
    linear_program.create_constraint().set_coefficient(x, 1).set_coefficient(y, 2).set_bounds(-infinity, 3);
    linear_program.create_constraint().set_coefficient(x, 2).set_coefficient(y, -1).set_bounds(-infinity, 3);
    linear_program.create_constraint().set_coefficient(x, 1).set_coefficient(y, -2).set_bounds(-infinity, 3);
    linear_program.create_constraint().set_coefficient(x, -2).set_coefficient(y, 1).set_bounds(-infinity, 3);
    linear_program.create_constraint().set_coefficient(x, -1).set_coefficient(y, 2).set_bounds(-infinity, 3);
    linear_program.create_constraint().set_coefficient(x, -2).set_coefficient(y, -1).set_bounds(-infinity, 3);
    linear_program.create_constraint().set_coefficient(x, -1).set_coefficient(y, -2).set_bounds(-infinity, 3);

    const auto solution = linear_program.solve();
    CHECK_NEAR(solution.assignments[x], 0);
    CHECK_NEAR(solution.assignments[y], 0);
    CHECK_NEAR(solution.cost, 0);

    return solution.cost;
  });
}

TEST_CASE("One unused variable") {
  test([](auto& linear_program) {
    const auto x0 = linear_program.create_variable();
    const auto x1 = linear_program.create_variable();
    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x0, -1);

    linear_program.create_constraint().set_coefficient(x0, 1).set_bounds(-infinity, 1);

    const auto solution = linear_program.solve();
    CHECK_NEAR(solution.assignments[x0], 1);
    CHECK_NEAR(solution.assignments[x1], 0);  // Not unique but all current solvers happen to return this value
    CHECK_NEAR(solution.cost, -1);

    return solution.cost;
  });
}

TEST_CASE("Two unused variables") {
  test([](auto& linear_program) {
    const auto x0 = linear_program.create_variable();
    const auto x1 = linear_program.create_variable();
    const auto x2 = linear_program.create_variable();
    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x2, -1);

    linear_program.create_constraint().set_coefficient(x2, 1).set_bounds(-infinity, 1);

    const auto solution = linear_program.solve();
    CHECK_NEAR(solution.assignments[x0], 0);  // Not unique but all current solvers happen to return this value
    CHECK_NEAR(solution.assignments[x1], 0);  // Idem
    CHECK_NEAR(solution.assignments[x2], 1);
    CHECK_NEAR(solution.cost, -1);

    return solution.cost;
  });
}

TEST_CASE("Equality constraint with origin feasible") {
  test([](auto& linear_program) {
    const auto x1 = linear_program.create_variable();
    const auto x2 = linear_program.create_variable();
    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x1, -1);
    linear_program.set_objective_coefficient(x2, -1);

    linear_program.create_constraint().set_coefficient(x1, 1).set_coefficient(x2, -2).set_bounds(0, 0);
    linear_program.create_constraint().set_coefficient(x1, 1).set_coefficient(x2, 1).set_bounds(-infinity, 3);

    const auto solution = linear_program.solve();
    CHECK_NEAR(solution.assignments[x1], 2);
    CHECK_NEAR(solution.assignments[x2], 1);
    CHECK_NEAR(solution.cost, -3);

    return solution.cost;
  });
}

TEST_CASE("Section 8 of https://webspace.maths.qmul.ac.uk/felix.fischer/teaching/opt/notes/notes.pdf") {
  test([](auto& linear_program) {
    const auto x1 = linear_program.create_variable();
    const auto x2 = linear_program.create_variable();
    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x1, 6);
    linear_program.set_objective_coefficient(x2, 3);

    linear_program.create_constraint().set_coefficient(x1, 1).set_coefficient(x2, 1).set_bounds(1, infinity);
    linear_program.create_constraint().set_coefficient(x1, 2).set_coefficient(x2, -1).set_bounds(1, infinity);
    linear_program.create_constraint().set_coefficient(x2, 3).set_bounds(-infinity, 2);

    const auto solution = linear_program.solve();

    CHECK_NEAR(solution.assignments[x1], 2./3);
    CHECK_NEAR(solution.assignments[x2], 1./3);
    CHECK_NEAR(solution.cost, 5);

    return solution.cost;
  });

  // Small variations because the example has several equal constants in different places, so we need to make sure we don't mix them up
  // @todo Restore these tests.
  // Are we hitting numerical instability?
  // For example for seed == 2, the phase 1 of the two-phase Simplex goes too far: after its finds a BFS with all
  // artificial variables at 0, it thinks it can still optimize the sum of the artificial variables, to a *negative* value,
  // which is inconsistent with the implicit non-negativity constraints on variables.
  // If, in 'find_entering_column', we define positivity as '> 1e-6' instead of '> 0', we get the expected result, but doing so
  // is a scary hack that hides the actual problam and might cause other silent issues.
  // We *could* interrupt phase 1 as soon as all artificial variables are non-basic. Is that less scary?
  // for (unsigned seed = 0; seed != 1'000; ++seed) {
  //   CAPTURE(seed);

  //   test([seed](auto& linear_program) {
  //     std::mt19937 mt(seed);
  //     std::uniform_real_distribution<float> make_small_variation(-0.01, 0.01);

  //     const auto x1 = linear_program.create_variable();
  //     const auto x2 = linear_program.create_variable();
  //     linear_program.mark_all_variables_created();

  //     linear_program.set_objective_coefficient(x1, 6 + make_small_variation(mt));
  //     linear_program.set_objective_coefficient(x2, 3 + make_small_variation(mt));

  //     linear_program.create_constraint().set_coefficient(x1, 1 + make_small_variation(mt)).set_coefficient(x2, 1 + make_small_variation(mt)).set_bounds(1 + make_small_variation(mt), infinity);
  //     linear_program.create_constraint().set_coefficient(x1, 2 + make_small_variation(mt)).set_coefficient(x2, -1 + make_small_variation(mt)).set_bounds(1 + make_small_variation(mt), infinity);
  //     linear_program.create_constraint().set_coefficient(x2, 3 + make_small_variation(mt)).set_bounds(-infinity, 2 + make_small_variation(mt));

  //     const auto solution = linear_program.solve();

  //     return solution.cost;
  //   });
  // }
}

TEST_CASE("Origin not feasible") {
  test([](auto& linear_program) {
    const auto x = linear_program.create_variable();
    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x, 1);

    linear_program.create_constraint().set_coefficient(x, 1).set_bounds(1, +infinity);

    const auto solution = linear_program.solve();

    CHECK_NEAR(solution.assignments[x], 1);
    CHECK_NEAR(solution.cost, 1);

    return solution.cost;
  });
}

TEST_CASE("Simplex orthogonal to objective gradient, origin not feasible - 1") {
  test([](auto& linear_program) {
    const auto x = linear_program.create_variable();
    const auto y = linear_program.create_variable();
    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x, 1);
    linear_program.set_objective_coefficient(y, 1);

    linear_program.create_constraint().set_coefficient(x, 1).set_coefficient(y, 1).set_bounds(1, +infinity);

    const auto solution = linear_program.solve();

    // Can't check the assignments because they are not unique
    CHECK_NEAR(solution.cost, 1);

    return solution.cost;
  });
}

TEST_CASE("Simplex orthogonal to objective gradient, origin not feasible - 2") {
  test([](auto& linear_program) {
    const auto x = linear_program.create_variable();
    const auto y = linear_program.create_variable();
    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x, 1);
    linear_program.set_objective_coefficient(y, 1);

    linear_program.create_constraint().set_coefficient(x, -1).set_coefficient(y, -1).set_bounds(-infinity, -1);

    const auto solution = linear_program.solve();

    // Can't check the assignments because they are not unique
    CHECK_NEAR(solution.cost, 1);

    return solution.cost;
  });
}

TEST_CASE("Triangle far from origin linear program - 1") {
  test([](auto& linear_program) {
    const auto x = linear_program.create_variable();
    const auto y = linear_program.create_variable();

    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x, 1);
    linear_program.set_objective_coefficient(y, 2);

    linear_program.create_constraint().set_coefficient(x, 1).set_coefficient(y, -1).set_bounds(1, infinity);
    linear_program.create_constraint().set_coefficient(x, 1).set_bounds(-infinity, 2);

    const auto solution = linear_program.solve();
    CHECK_NEAR(solution.assignments[x], 1);
    CHECK_NEAR(solution.assignments[y], 0);
    CHECK_NEAR(solution.cost, 1);

    return solution.cost;
  });
}

TEST_CASE("Triangle far from origin linear program - 2") {
  test([](auto& linear_program) {
    const auto x = linear_program.create_variable();
    const auto y = linear_program.create_variable();

    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x, -2);
    linear_program.set_objective_coefficient(y, -1);

    linear_program.create_constraint().set_coefficient(x, 1).set_coefficient(y, -1).set_bounds(1, infinity);
    linear_program.create_constraint().set_coefficient(x, 1).set_bounds(-infinity, 2);

    const auto solution = linear_program.solve();
    CHECK_NEAR(solution.assignments[x], 2);
    CHECK_NEAR(solution.assignments[y], 1);
    CHECK_NEAR(solution.cost, -5);

    return solution.cost;
  });
}

TEST_CASE("Triangle far from origin linear program - 3") {
  test([](auto& linear_program) {
    const auto x = linear_program.create_variable();
    const auto y = linear_program.create_variable();

    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x, -1);
    linear_program.set_objective_coefficient(y, 1);

    linear_program.create_constraint().set_coefficient(x, 1).set_coefficient(y, -1).set_bounds(1, infinity);
    linear_program.create_constraint().set_coefficient(x, 1).set_bounds(-infinity, 2);

    const auto solution = linear_program.solve();
    CHECK_NEAR(solution.assignments[x], 2);
    CHECK_NEAR(solution.assignments[y], 0);
    CHECK_NEAR(solution.cost, -2);

    return solution.cost;
  });
}

TEST_CASE("Wikipedia example 1") {
  // https://en.wikipedia.org/wiki/Simplex_algorithm#Example
  test([](auto& linear_program) {
    const auto x = linear_program.create_variable();
    const auto y = linear_program.create_variable();
    const auto z = linear_program.create_variable();

    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x, -2);
    linear_program.set_objective_coefficient(y, -3);
    linear_program.set_objective_coefficient(z, -4);

    linear_program.create_constraint().set_coefficient(x, 3).set_coefficient(y, 2).set_coefficient(z, 1).set_bounds(-infinity, 10);
    linear_program.create_constraint().set_coefficient(x, 2).set_coefficient(y, 5).set_coefficient(z, 3).set_bounds(-infinity, 15);

    const auto solution = linear_program.solve();
    CHECK_NEAR(solution.cost, -20);
    const float recomputed_cost = -2 * solution.assignments[x] -3 * solution.assignments[y] - 4 * solution.assignments[z];
    CHECK_NEAR(recomputed_cost, solution.cost);

    return solution.cost;
  });
}

TEST_CASE("Wikipedia example 2 - equality constraints, origin not feasible") {
  // https://en.wikipedia.org/wiki/Simplex_algorithm#Example_2
  test([](auto& linear_program) {
    const auto x = linear_program.create_variable();
    const auto y = linear_program.create_variable();
    const auto z = linear_program.create_variable();

    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x, -2);
    linear_program.set_objective_coefficient(y, -3);
    linear_program.set_objective_coefficient(z, -4);

    linear_program.create_constraint().set_coefficient(x, 3).set_coefficient(y, 2).set_coefficient(z, 1).set_bounds(10, 10);
    linear_program.create_constraint().set_coefficient(x, 2).set_coefficient(y, 5).set_coefficient(z, 3).set_bounds(15, 15);

    const auto solution = linear_program.solve();
    CHECK_NEAR(solution.cost, -130./7.);
    const float recomputed_cost = -2 * solution.assignments[x] -3 * solution.assignments[y] - 4 * solution.assignments[z];
    CHECK_NEAR(recomputed_cost, solution.cost);

    return solution.cost;
  });
}

TEST_CASE("Random linear programs with optimal solutions") {
  for (unsigned seed = 0; seed != 10'000; ++seed) {
    CAPTURE(seed);

    test([seed](auto& linear_program) {
      std::mt19937 mt(seed);

      std::uniform_int_distribution<unsigned> make_variables_count(2, 10);
      const unsigned variables_count = make_variables_count(mt);
      std::uniform_real_distribution<float> make_objective_coefficient(-3, 3);
      std::uniform_int_distribution<unsigned> make_constraints_count(0, 2 * variables_count);
      const unsigned constraints_count = make_constraints_count(mt);
      std::uniform_real_distribution<float> make_constraint_coefficient(-1, 1);

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

      const auto solution = linear_program.solve();

      float expected_cost = 0;
      for (unsigned i = 0; i != variables_count; ++i) {
        expected_cost += objective_coefficients[i] * solution.assignments[variables[i]];
      }
      CHECK_NEAR(solution.cost, expected_cost);

      return solution.cost;
    });
  }
}

// @todo(Project management, when we release our in-house LP solvers) Test consistency on all kinds of linear programs (unbounded, infeasible, others?)

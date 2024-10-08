// Copyright 2023-2024 Vincent Jacques

#include "testing.hpp"


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
    const auto solution = *linear_program.solve();

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

    const auto solution = *linear_program.solve();

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
    const auto solution = *linear_program.solve();
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

    const auto solution = *linear_program.solve();
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

    const auto solution = *linear_program.solve();
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

    const auto solution = *linear_program.solve();
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

    const auto solution = *linear_program.solve();
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

    const auto solution = *linear_program.solve();
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

    const auto solution = *linear_program.solve();

    CHECK_NEAR(solution.assignments[x1], 2./3);
    CHECK_NEAR(solution.assignments[x2], 1./3);
    CHECK_NEAR(solution.cost, 5);

    return solution.cost;
  });

  // Small variations because the example has several equal constants in different places, so we need to make sure we don't mix them up
  for (unsigned seed = 0; seed != 1'000; ++seed) {
    CAPTURE(seed);

    test([seed](auto& linear_program) {
      std::mt19937 mt(seed);
      std::uniform_real_distribution<float> make_small_variation(-0.01, 0.01);

      const auto x1 = linear_program.create_variable();
      const auto x2 = linear_program.create_variable();
      linear_program.mark_all_variables_created();

      linear_program.set_objective_coefficient(x1, 6 + make_small_variation(mt));
      linear_program.set_objective_coefficient(x2, 3 + make_small_variation(mt));

      linear_program.create_constraint().set_coefficient(x1, 1 + make_small_variation(mt)).set_coefficient(x2, 1 + make_small_variation(mt)).set_bounds(1 + make_small_variation(mt), infinity);
      linear_program.create_constraint().set_coefficient(x1, 2 + make_small_variation(mt)).set_coefficient(x2, -1 + make_small_variation(mt)).set_bounds(1 + make_small_variation(mt), infinity);
      linear_program.create_constraint().set_coefficient(x2, 3 + make_small_variation(mt)).set_bounds(-infinity, 2 + make_small_variation(mt));

      const auto solution = *linear_program.solve();

      return solution.cost;
    });
  }
}

TEST_CASE("Origin optimal") {
  test([](auto& linear_program) {
    const auto x = linear_program.create_variable();
    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x, 1);

    linear_program.create_constraint().set_coefficient(x, 1).set_bounds(-1, 1);

    const auto solution = *linear_program.solve();

    CHECK_NEAR(solution.assignments[x], 0);
    CHECK_NEAR(solution.cost, 0);

    return solution.cost;
  });
}

TEST_CASE("Origin not feasible") {
  test([](auto& linear_program) {
    const auto x = linear_program.create_variable();
    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x, 1);

    linear_program.create_constraint().set_coefficient(x, 1).set_bounds(1, +infinity);

    const auto solution = *linear_program.solve();

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

    const auto solution = *linear_program.solve();

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

    const auto solution = *linear_program.solve();

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

    const auto solution = *linear_program.solve();
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

    const auto solution = *linear_program.solve();
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

    const auto solution = *linear_program.solve();
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

    const auto solution = *linear_program.solve();
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

    const auto solution = *linear_program.solve();
    CHECK_NEAR(solution.cost, -130./7.);
    const float recomputed_cost = -2 * solution.assignments[x] -3 * solution.assignments[y] - 4 * solution.assignments[z];
    CHECK_NEAR(recomputed_cost, solution.cost);

    return solution.cost;
  });
}

TEST_CASE("Random linear programs with optimal solutions reached in one Simplex phase") {
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

      const auto solution = *linear_program.solve();
      CHECK(solution.cost != -infinity);
      CHECK(!std::isnan(solution.cost));

      float expected_cost = 0;
      for (unsigned i = 0; i != variables_count; ++i) {
        expected_cost += objective_coefficients[i] * solution.assignments[variables[i]];
      }
      CHECK_NEAR(solution.cost, expected_cost);

      return solution.cost;
    });
  }
}

TEST_CASE("Random linear programs requiring two Simplex phases") {
  for (unsigned seed = 0; seed != 10'000; ++seed) {
    CAPTURE(seed);

    test([seed](auto& linear_program) -> std::optional<float> {
      std::mt19937 mt(seed);

      std::uniform_int_distribution<unsigned> make_variables_count(2, 10);
      const unsigned variables_count = make_variables_count(mt);
      std::uniform_real_distribution<float> make_objective_coefficient(-3, 3);
      std::uniform_int_distribution<unsigned> make_constraints_count(1, 2 * variables_count);
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
        c.set_bounds(-infinity, 1);
        c.set_coefficient(v, 1);
      }

      for (unsigned i = 0; i != constraints_count; ++i) {
        auto c = linear_program.create_constraint();
        c.set_bounds(1, 10);
        // @todo(Project management, when we release our in-house LP solvers) Let some variables not appear in some constraints
        for (const auto& v : variables) {
          c.set_coefficient(v, make_constraint_coefficient(mt));
        }
      }

      const auto solution = linear_program.solve();
      if (solution) {
        float expected_cost = 0;
        for (unsigned i = 0; i != variables_count; ++i) {
          expected_cost += objective_coefficients[i] * solution->assignments[variables[i]];
        }
        CHECK_NEAR(solution->cost, expected_cost);
        return solution->cost;
      } else {
        return std::nullopt;
      }
    });
  }
}

TEST_CASE("Unbounded (single phase)") {
  test([](auto& linear_program) -> std::optional<float> {
    const auto x = linear_program.create_variable();
    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x, -1);

    linear_program.create_constraint().set_coefficient(x, 1).set_bounds(0, infinity);

    const auto solution = linear_program.solve();
    CHECK_FALSE(solution);
    return std::nullopt;
  });
}

TEST_CASE("Unbounded (two phases)") {
  test([](auto& linear_program) -> std::optional<float> {
    const auto x = linear_program.create_variable();
    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x, -1);

    linear_program.create_constraint().set_coefficient(x, 1).set_bounds(1, infinity);

    const auto solution = linear_program.solve();
    CHECK_FALSE(solution);
    return std::nullopt;
  });
}

TEST_CASE("Infeasible") {
  test([](auto& linear_program) -> std::optional<float> {
    const auto x = linear_program.create_variable();
    linear_program.mark_all_variables_created();

    linear_program.set_objective_coefficient(x, -1);

    linear_program.create_constraint().set_coefficient(x, 1).set_bounds(-2, -1);

    const auto solution = linear_program.solve();
    CHECK_FALSE(solution);
    return std::nullopt;
  });
}

TEST_CASE("More random linear programs with optimal solutions") {
  for (unsigned variables_count = 2; variables_count != 10; ++variables_count) {
    CAPTURE(variables_count);

    for (unsigned seed = 0; seed != 100; ++seed) {
      CAPTURE(seed);

      test([variables_count, seed](auto& linear_program) {
        std::vector<typename std::remove_reference_t<decltype(linear_program)>::variable_type> variables;
        for (unsigned i = 0; i != variables_count; ++i) {
          variables.push_back(linear_program.create_variable());
        }
        linear_program.mark_all_variables_created();
        for (const auto& v : variables) {
          linear_program.create_constraint().set_coefficient(v, 1).set_bounds(0, 1000);
        }

        std::mt19937 mt(seed);

        std::uniform_real_distribution<float> make_objective_coefficient(-3, 3);
        for (const auto& v : variables) {
          linear_program.set_objective_coefficient(v, make_objective_coefficient(mt));
        }

        std::uniform_int_distribution<unsigned> make_constraints_count(0, 2 * variables_count);
        std::uniform_int_distribution<unsigned> make_constraint_variables_count(2, variables_count);
        const unsigned constraints_count = make_constraints_count(mt);
        for (unsigned constraint_index = 0; constraint_index != constraints_count; ++constraint_index) {
          std::vector<unsigned> variable_indices(variables_count, 0);
          std::iota(variable_indices.begin(), variable_indices.end(), 0);
          std::shuffle(variable_indices.begin(), variable_indices.end(), mt);
          variable_indices.resize(make_constraint_variables_count(mt));

          auto constraint = linear_program.create_constraint();
          float value_at_one = 0;
          for (const auto& variable_index : variable_indices) {
            const float coefficient = make_objective_coefficient(mt);
            constraint.set_coefficient(variables[variable_index], coefficient);
            value_at_one += coefficient;
          }
          // Make sure the Simplex contains point (1, 1, ..., 1)
          constraint.set_bounds(value_at_one - 1, value_at_one + 1);
        }

        const auto solution = linear_program.solve();
        CHECK(solution);
        return solution->cost;
      });
    }
  }
}

// @todo(Project management, when we release our in-house LP solvers) Test consistency on all kinds of linear programs (unbounded, infeasible, others?)

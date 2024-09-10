// Copyright 2023-2024 Vincent Jacques

#include "testing.hpp"


TEST_CASE("Bug") {
  test([](auto& linear_program) -> std::optional<float> {
    {
      const auto x0 = linear_program.create_variable();
      const auto x1 = linear_program.create_variable();
      linear_program.mark_all_variables_created();
      linear_program.set_objective_coefficient(x0, 5.9987197);
      linear_program.set_objective_coefficient(x1, 2.9937017);
      { auto c = linear_program.create_constraint(); c.set_bounds(1.0009933, infinity); c.set_coefficient(x0, 0.9905185); c.set_coefficient(x1, 1.0086309); }
      { auto c = linear_program.create_constraint(); c.set_bounds(0.999695, infinity); c.set_coefficient(x0, 2.0089545); c.set_coefficient(x1, -1.0012935); }
      { auto c = linear_program.create_constraint(); c.set_bounds(-infinity, 1.9964107); c.set_coefficient(x1, 2.9984074); }
    }
    lincs::CustomOnCpuVerbose verbose;
    const auto solution = linear_program.solve();
    if (solution) {
      return solution->cost;
    } else {
      return std::nullopt;
    }
  });
}

// Copyright 2023-2024 Vincent Jacques

#include "testing.hpp"


TEST_CASE("Bug") {
  test([](auto& linear_program) -> std::optional<float> {
    {
      std::vector<typename std::remove_reference_t<decltype(linear_program)>::variable_type> v;
      v.push_back(linear_program.create_variable());
      v.push_back(linear_program.create_variable());
      v.push_back(linear_program.create_variable());
      linear_program.mark_all_variables_created();
      linear_program.set_objective_coefficient(v[0], 1);
      linear_program.set_objective_coefficient(v[1], -1);
      linear_program.set_objective_coefficient(v[2], -3);
      linear_program.create_constraint().set_bounds(0, infinity).set_coefficient(v[0], 1);
      linear_program.create_constraint().set_bounds(-1, 1).set_coefficient(v[0], 0.29).set_coefficient(v[1], 0.86).set_coefficient(v[2], -1.29);
      linear_program.create_constraint().set_bounds(-1, 1).set_coefficient(v[0], 0.08).set_coefficient(v[1], -2.05).set_coefficient(v[2], 1.96);
      linear_program.create_constraint().set_bounds(1.21, 3.21).set_coefficient(v[0], 1.74).set_coefficient(v[1], 0.54).set_coefficient(v[2], -0.07);
    }
    const auto solution = linear_program.solve();
    if (solution) {
      return solution->cost;
    } else {
      return std::nullopt;
    }
  });
}

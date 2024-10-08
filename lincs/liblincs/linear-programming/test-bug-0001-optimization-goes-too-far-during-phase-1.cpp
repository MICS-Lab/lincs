// Copyright 2023-2024 Vincent Jacques

#include "testing.hpp"


TEST_CASE("Bug") {
  test([](auto& linear_program) -> std::optional<float> {
    {
      std::vector<typename std::remove_reference_t<decltype(linear_program)>::variable_type> v;
      v.push_back(linear_program.create_variable());
      v.push_back(linear_program.create_variable());
      linear_program.mark_all_variables_created();
      linear_program.set_objective_coefficient(v[0], 5.9987197);
      linear_program.set_objective_coefficient(v[1], 2.9937017);
      linear_program.create_constraint().set_bounds(1.0009933, infinity).set_coefficient(v[0], 0.9905185).set_coefficient(v[1], 1.0086309);
      linear_program.create_constraint().set_bounds(0.999695, infinity).set_coefficient(v[0], 2.0089545).set_coefficient(v[1], -1.0012935);
      linear_program.create_constraint().set_bounds(-infinity, 1.9964107).set_coefficient(v[1], 2.9984074);
    }
    const auto solution = linear_program.solve();
    if (solution) {
      return solution->cost;
    } else {
      return std::nullopt;
    }
  });
}

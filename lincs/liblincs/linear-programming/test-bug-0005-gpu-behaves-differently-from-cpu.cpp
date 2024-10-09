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
      linear_program.set_objective_coefficient(v[0], -2.6169455);
      linear_program.set_objective_coefficient(v[1], -0.56562924);
      linear_program.set_objective_coefficient(v[2], -2.6307747);
      linear_program.create_constraint().set_bounds(-1, 1).set_coefficient(v[0], 1);
      linear_program.create_constraint().set_bounds(-1, 1).set_coefficient(v[1], 1);
      linear_program.create_constraint().set_bounds(-1, 1).set_coefficient(v[2], 1);
      linear_program.create_constraint().set_bounds(-1, 1).set_coefficient(v[0], 0.3803965).set_coefficient(v[1], -0.9824758).set_coefficient(v[2], 0.4877367);
      linear_program.create_constraint().set_bounds(-1, 1).set_coefficient(v[0], 0.57658184).set_coefficient(v[1], -0.8118078).set_coefficient(v[2], 0.014208794);
      linear_program.create_constraint().set_bounds(-1, 1).set_coefficient(v[0], 0.85147953).set_coefficient(v[1], -0.5650891).set_coefficient(v[2], 0.15271592);
    }
    const auto solution = linear_program.solve();
    if (solution) {
      return solution->cost;
    } else {
      return std::nullopt;
    }
  });
}

// Copyright 2024 Vincent Jacques

#include "custom-on-cpu.hpp"

#include <ortools/glop/lp_solver.h>
#undef CHECK
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_LT
#undef CHECK_GT
#undef CHECK_LE
#undef CHECK_GE
#undef LOG

namespace lincs {

CustomOnCpuLinearProgram::solution_type CustomOnCpuLinearProgram::solve() {
  operations_research::glop::LinearProgram glop_program;

  std::vector<operations_research::glop::ColIndex> glop_variables;
  for (unsigned variable = 0; variable < next_variable_index; ++variable) {
    glop_variables.push_back(glop_program.CreateNewVariable());
  }

  for (auto [variable, coefficient] : objective_coefficients) {
    glop_program.SetObjectiveCoefficient(glop_variables[variable], coefficient);
  }

  for (const auto& c : constraints) {
    auto index = glop_program.CreateNewConstraint();
    glop_program.SetConstraintBounds(index, std::get<0>(c), std::get<1>(c));
    for (auto [variable, coefficient] : std::get<2>(c)) {
      glop_program.SetCoefficient(index, glop_variables[variable], coefficient);
    }
  }

  operations_research::glop::LPSolver glop_solver;
  operations_research::glop::GlopParameters glop_parameters;
  glop_parameters.set_provide_strong_optimal_guarantee(true);
  glop_solver.SetParameters(glop_parameters);

  glop_program.CleanUp();
  #ifndef NDEBUG
  auto glop_status =
  #endif
  glop_solver.Solve(glop_program);
  assert(glop_status == operations_research::glop::ProblemStatus::OPTIMAL);
  const auto glop_assignments = glop_solver.variable_values();
  const float glop_cost = float(glop_solver.GetObjectiveValue());

  solution_type solution;
  solution.assignments.resize(next_variable_index);
  for (int variable = 0; variable < next_variable_index; ++variable) {
    solution.assignments[variable] = glop_assignments[glop_variables[variable]];
  }
  solution.cost = glop_cost;

  return solution;
}

}  // namespace lincs

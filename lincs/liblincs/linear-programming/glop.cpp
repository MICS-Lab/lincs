// Copyright 2023-2024 Vincent Jacques

#include "glop.hpp"

#include "../chrones.hpp"


namespace lincs {

std::optional<GlopLinearProgram::solution_type> GlopLinearProgram::solve() {
  CHRONE();

  operations_research::glop::LPSolver solver;
  operations_research::glop::GlopParameters parameters;
  parameters.set_provide_strong_optimal_guarantee(true);
  solver.SetParameters(parameters);

  program.CleanUp();
  auto status = solver.Solve(program);
  if (status == operations_research::glop::ProblemStatus::OPTIMAL) {
    return solution_type{solver.variable_values(), float(solver.GetObjectiveValue())};
  } else {
    return {};
  }
}

}  // namespace lincs

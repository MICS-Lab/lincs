// Copyright 2023-2024 Vincent Jacques

#include "glop.hpp"

#include "../chrones.hpp"


namespace lincs {

GlopLinearProgram::solution_type GlopLinearProgram::solve() {
  CHRONE();

  operations_research::glop::LPSolver solver;
  operations_research::glop::GlopParameters parameters;
  parameters.set_provide_strong_optimal_guarantee(true);
  solver.SetParameters(parameters);

  #ifndef NDEBUG
  auto status =
  #endif
  solver.Solve(program);
  assert(status == operations_research::glop::ProblemStatus::OPTIMAL);

  return solution_type{solver.variable_values(), float(solver.GetObjectiveValue())};
}

}  // namespace lincs

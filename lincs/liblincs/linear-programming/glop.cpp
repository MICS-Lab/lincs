// Copyright 2023-2024 Vincent Jacques

#include "glop.hpp"

#include "../chrones.hpp"


namespace lincs {

operations_research::glop::DenseRow GlopLinearProgram::solve() {
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
  return solver.variable_values();
}

}  // namespace lincs

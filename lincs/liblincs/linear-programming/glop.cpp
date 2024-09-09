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

  program.CleanUp();
  #ifndef NDEBUG
  auto status =
  #endif
  solver.Solve(program);
  if (status == operations_research::glop::ProblemStatus::OPTIMAL) {
    return solution_type{solver.variable_values(), float(solver.GetObjectiveValue())};
  } else if (status == operations_research::glop::ProblemStatus::INFEASIBLE_OR_UNBOUNDED) {
    // @todo Check it is indeed unbounded. See https://github.com/google/or-tools/blob/v8.2/ortools/lp_data/lp_types.h#L116
    return solution_type{solver.variable_values(), -std::numeric_limits<float>::infinity()};
  } else {
    std::cerr << "Unexpected GLOP status: " << status << std::endl;
    assert(false);
  }
}

}  // namespace lincs

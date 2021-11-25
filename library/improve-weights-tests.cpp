// Copyright 2021 Vincent Jacques

#include "improve-weights.hpp"

#include <gtest/gtest.h>

#include <ortools/glop/lp_solver.h>
#include <ortools/lp_data/lp_data.h>
#include <ortools/lp_data/lp_types.h>


TEST(GlopExploration, FromSample) {
  // Exploration test inspired from
  // https://github.com/google/or-tools/blob/stable/ortools/glop/samples/simple_glop_program.cc
  // and simplified

  operations_research::glop::LinearProgram lp;

  operations_research::glop::ColIndex col_x = lp.CreateNewVariable();
  lp.SetVariableBounds(col_x, -30.0, 6.0);
  operations_research::glop::ColIndex col_y = lp.CreateNewVariable();
  lp.SetVariableBounds(col_y, -3.0, 60.0);

  operations_research::glop::RowIndex row_r1 = lp.CreateNewConstraint();
  lp.SetConstraintBounds(row_r1, 3.0, 3.0);
  lp.SetCoefficient(row_r1, col_x, 1);
  lp.SetCoefficient(row_r1, col_y, 1);

  lp.SetObjectiveCoefficient(col_x, 2);
  lp.SetObjectiveCoefficient(col_y, 1);
  lp.SetMaximizationProblem(true);

  EXPECT_EQ(lp.num_variables(), 2);
  EXPECT_EQ(lp.num_constraints(), 1);
  EXPECT_EQ(
    lp.Dump(),
    "max: + 2 c0 + c1;\n"
    "r0: + c0 + c1 = 3;\n"
    "-30 <= c0 <= 6;\n"
    "-3 <= c1 <= 60;\n");

  operations_research::glop::LPSolver solver;
  operations_research::glop::GlopParameters parameters;
  parameters.set_provide_strong_optimal_guarantee(true);
  solver.SetParameters(parameters);

  EXPECT_EQ(solver.Solve(lp), operations_research::glop::ProblemStatus::OPTIMAL);

  EXPECT_NEAR(solver.GetObjectiveValue(), 9, 1e-6);

  const operations_research::glop::DenseRow& values = solver.variable_values();
  EXPECT_NEAR(values[col_x], 6, 1e-6);
  EXPECT_NEAR(values[col_y], -3, 1e-6);
}

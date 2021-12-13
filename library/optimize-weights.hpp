// Copyright 2021 Vincent Jacques

#ifndef OPTIMIZE_WEIGHTS_HPP_
#define OPTIMIZE_WEIGHTS_HPP_

#include <ortools/glop/lp_solver.h>

#include <vector>

#include "problem.hpp"


namespace ppl {

namespace glp = operations_research::glop;

struct LinearProgram {
  glp::LinearProgram program;

  std::vector<glp::ColIndex> weight_variables;
  std::vector<glp::ColIndex> x_variables;
  std::vector<glp::ColIndex> xp_variables;
  std::vector<glp::ColIndex> y_variables;
  std::vector<glp::ColIndex> yp_variables;

  // @todo Fix naming
  std::vector<glp::RowIndex> a_constraints;
  std::vector<glp::RowIndex> b_constraints;
};

/*
Implement 3.3.3 of https://tel.archives-ouvertes.fr/tel-01370555/document
*/
class WeightsOptimizer {
 public:
  explicit WeightsOptimizer(const Models<Host>&);

 public:
  void optimize_weights(Models<Host>*);

 private:
  std::vector<LinearProgram> _linear_programs;
  std::vector<glp::LPSolver> _solvers;
};

}  // namespace ppl

#endif  // OPTIMIZE_WEIGHTS_HPP_

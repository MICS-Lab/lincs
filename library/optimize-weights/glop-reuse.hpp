// Copyright 2021-2022 Vincent Jacques

#ifndef OPTIMIZE_WEIGHTS_GLOP_REUSE_HPP_
#define OPTIMIZE_WEIGHTS_GLOP_REUSE_HPP_

#include <ortools/glop/lp_solver.h>

#include <memory>
#include <vector>

#include "../optimize-weights.hpp"


namespace ppl {

/*
Implement 3.3.3 of https://tel.archives-ouvertes.fr/tel-01370555/document
using GLOP to solve the linear program, reusing the linear programs and solvers
to try and benefit from the reuse optimization in GLOP.
*/
class OptimizeWeightsUsingGlopAndReusingPrograms : public WeightsOptimizationStrategy {
 public:
  explicit OptimizeWeightsUsingGlopAndReusingPrograms(const Models<Host>&);

 public:
  void optimize_weights(std::shared_ptr<Models<Host>>) override;

 public:
  struct LinearProgram {
    operations_research::glop::LinearProgram program;

    std::vector<operations_research::glop::ColIndex> weight_variables;
    std::vector<operations_research::glop::ColIndex> x_variables;
    std::vector<operations_research::glop::ColIndex> xp_variables;
    std::vector<operations_research::glop::ColIndex> y_variables;
    std::vector<operations_research::glop::ColIndex> yp_variables;

    // @todo Fix naming
    std::vector<operations_research::glop::RowIndex> a_constraints;
    std::vector<operations_research::glop::RowIndex> b_constraints;
  };

 private:
  std::vector<LinearProgram> _linear_programs;
  std::vector<operations_research::glop::LPSolver> _solvers;
};

}  // namespace ppl

#endif  // OPTIMIZE_WEIGHTS_GLOP_REUSE_HPP_

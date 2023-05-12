// Copyright 2023 Vincent Jacques

#include "glop.hpp"

#include <ortools/glop/lp_solver.h>


namespace glp = operations_research::glop;

namespace lincs {

void OptimizeWeightsUsingGlop::optimize_weights() {
  #pragma omp parallel for
  for (unsigned model_index = 0; model_index != models.models_count; ++model_index) {
    optimize_model_weights(model_index);
  }
};

struct OptimizeWeightsUsingGlop::LinearProgram {
  std::shared_ptr<glp::LinearProgram> program;
  std::vector<glp::ColIndex> weight_variables;
  std::vector<glp::ColIndex> x_variables;
  std::vector<glp::ColIndex> xp_variables;
  std::vector<glp::ColIndex> y_variables;
  std::vector<glp::ColIndex> yp_variables;
};

auto OptimizeWeightsUsingGlop::solve_linear_program(std::shared_ptr<OptimizeWeightsUsingGlop::LinearProgram> lp) {
  operations_research::glop::LPSolver solver;
  operations_research::glop::GlopParameters parameters;
  parameters.set_provide_strong_optimal_guarantee(true);
  solver.SetParameters(parameters);

  auto status = solver.Solve(*lp->program);
  assert(status == operations_research::glop::ProblemStatus::OPTIMAL);
  auto values = solver.variable_values();

  return values;
}

void OptimizeWeightsUsingGlop::optimize_model_weights(unsigned model_index) {
  auto lp = make_internal_linear_program(1e-6, model_index);
  auto values = solve_linear_program(lp);

  for (unsigned criterion_index = 0; criterion_index != models.criteria_count; ++criterion_index) {
    models.weights[criterion_index][model_index] = values[lp->weight_variables[criterion_index]];
  }
}

std::shared_ptr<OptimizeWeightsUsingGlop::LinearProgram> OptimizeWeightsUsingGlop::make_internal_linear_program(
  const float epsilon,
  unsigned model_index
) {
  auto lp = std::make_shared<LinearProgram>();

  lp->program = std::make_shared<glp::LinearProgram>();
  lp->weight_variables.reserve(models.criteria_count);
  for (unsigned criterion_index = 0; criterion_index != models.criteria_count; ++criterion_index) {
    lp->weight_variables.push_back(lp->program->CreateNewVariable());
  }

  lp->x_variables.reserve(models.learning_alternatives_count);
  lp->xp_variables.reserve(models.learning_alternatives_count);
  lp->y_variables.reserve(models.learning_alternatives_count);
  lp->yp_variables.reserve(models.learning_alternatives_count);
  for (unsigned alternative_index = 0; alternative_index != models.learning_alternatives_count; ++alternative_index) {
    lp->x_variables.push_back(lp->program->CreateNewVariable());
    lp->xp_variables.push_back(lp->program->CreateNewVariable());
    lp->y_variables.push_back(lp->program->CreateNewVariable());
    lp->yp_variables.push_back(lp->program->CreateNewVariable());

    lp->program->SetObjectiveCoefficient(lp->xp_variables.back(), 1);
    lp->program->SetObjectiveCoefficient(lp->yp_variables.back(), 1);

    const unsigned category_index = models.learning_assignments[alternative_index];

    if (category_index != 0) {
      glp::RowIndex c = lp->program->CreateNewConstraint();
      lp->program->SetConstraintBounds(c, 1, 1);
      lp->program->SetCoefficient(c, lp->x_variables.back(), -1);
      lp->program->SetCoefficient(c, lp->xp_variables.back(), 1);
      for (unsigned criterion_index = 0; criterion_index != models.criteria_count; ++criterion_index) {
        const float alternative_value = models.learning_alternatives[criterion_index][alternative_index];
        const float profile_value = models.profiles[criterion_index][category_index - 1][model_index];
        if (alternative_value >= profile_value) {
          lp->program->SetCoefficient(c, lp->weight_variables[criterion_index], 1);
        }
      }
    }

    if (category_index != models.categories_count - 1) {
      glp::RowIndex c = lp->program->CreateNewConstraint();
      lp->program->SetConstraintBounds(c, 1 - epsilon, 1 - epsilon);
      lp->program->SetCoefficient(c, lp->y_variables.back(), 1);
      lp->program->SetCoefficient(c, lp->yp_variables.back(), -1);
      for (unsigned criterion_index = 0; criterion_index != models.criteria_count; ++criterion_index) {
        const float alternative_value = models.learning_alternatives[criterion_index][alternative_index];
        const float profile_value = models.profiles[criterion_index][category_index][model_index];
        if (alternative_value >= profile_value) {
          lp->program->SetCoefficient(c, lp->weight_variables[criterion_index], 1);
        }
      }
    }
  }

  return lp;
}

}  // namespace lincs

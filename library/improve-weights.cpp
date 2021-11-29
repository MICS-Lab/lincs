// Copyright 2021 Vincent Jacques

#include "improve-weights.hpp"

#include <ortools/glop/lp_solver.h>

#include <string>
#include <vector>

#include "stopwatch.hpp"


namespace ppl::improve_weights {

namespace glp = operations_research::glop;

struct LinearProgram {
  std::shared_ptr<glp::LinearProgram> program;
  std::vector<glp::ColIndex> weight_variables;
  std::vector<glp::ColIndex> x_variables;
  std::vector<glp::ColIndex> xp_variables;
  std::vector<glp::ColIndex> y_variables;
  std::vector<glp::ColIndex> yp_variables;
};

std::shared_ptr<LinearProgram> make_internal_linear_program(
    const float epsilon, const ModelsView& models, uint model_index) {
  auto lp = std::make_shared<LinearProgram>();

  lp->program = std::make_shared<glp::LinearProgram>();
  lp->weight_variables.reserve(models.domain.criteria_count);
  for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
    lp->weight_variables.push_back(lp->program->CreateNewVariable());
  }

  lp->x_variables.reserve(models.domain.learning_alternatives_count);
  lp->xp_variables.reserve(models.domain.learning_alternatives_count);
  lp->y_variables.reserve(models.domain.learning_alternatives_count);
  lp->yp_variables.reserve(models.domain.learning_alternatives_count);
  for (uint alt_index = 0; alt_index != models.domain.learning_alternatives_count; ++alt_index) {
    lp->x_variables.push_back(lp->program->CreateNewVariable());
    lp->xp_variables.push_back(lp->program->CreateNewVariable());
    lp->y_variables.push_back(lp->program->CreateNewVariable());
    lp->yp_variables.push_back(lp->program->CreateNewVariable());

    lp->program->SetObjectiveCoefficient(lp->xp_variables.back(), 1);
    lp->program->SetObjectiveCoefficient(lp->yp_variables.back(), 1);

    const uint category_index = models.domain.learning_assignments[alt_index];

    if (category_index != 0) {
      glp::RowIndex c = lp->program->CreateNewConstraint();
      lp->program->SetConstraintBounds(c, 1, 1);
      lp->program->SetCoefficient(c, lp->x_variables.back(), -1);
      lp->program->SetCoefficient(c, lp->xp_variables.back(), 1);
      for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
        const float alternative_value = models.domain.learning_alternatives[crit_index][alt_index];
        const float profile_value = models.profiles[crit_index][category_index - 1][model_index];
        if (alternative_value >= profile_value) {
          lp->program->SetCoefficient(c, lp->weight_variables[crit_index], 1);
        }
      }
    }

    if (category_index != models.domain.categories_count - 1) {
      glp::RowIndex c = lp->program->CreateNewConstraint();
      lp->program->SetConstraintBounds(c, 1 - epsilon, 1 - epsilon);
      lp->program->SetCoefficient(c, lp->y_variables.back(), 1);
      lp->program->SetCoefficient(c, lp->yp_variables.back(), -1);
      for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
        const float alternative_value = models.domain.learning_alternatives[crit_index][alt_index];
        const float profile_value = models.profiles[crit_index][category_index][model_index];
        if (alternative_value >= profile_value) {
          lp->program->SetCoefficient(c, lp->weight_variables[crit_index], 1);
        }
      }
    }
  }

  return lp;
}

std::shared_ptr<LinearProgram> make_verbose_linear_program(
    const float epsilon, const ModelsView& models, uint model_index) {
  auto lp = make_internal_linear_program(epsilon, models, model_index);

  assert(lp->weight_variables.size() == models.domain.criteria_count);
  for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
    lp->program->SetVariableName(lp->weight_variables[crit_index], "w_" + std::to_string(crit_index));
  }

  assert(lp->x_variables.size() == models.domain.learning_alternatives_count);
  assert(lp->xp_variables.size() == models.domain.learning_alternatives_count);
  assert(lp->y_variables.size() == models.domain.learning_alternatives_count);
  assert(lp->yp_variables.size() == models.domain.learning_alternatives_count);
  for (uint alt_index = 0; alt_index != models.domain.learning_alternatives_count; ++alt_index) {
    lp->program->SetVariableName(lp->x_variables[alt_index], "x_" + std::to_string(alt_index));
    lp->program->SetVariableName(lp->xp_variables[alt_index], "x'_" + std::to_string(alt_index));
    lp->program->SetVariableName(lp->y_variables[alt_index], "y_" + std::to_string(alt_index));
    lp->program->SetVariableName(lp->yp_variables[alt_index], "y'_" + std::to_string(alt_index));
  }

  return lp;
}

std::shared_ptr<glp::LinearProgram> make_verbose_linear_program(
    const float epsilon, const Models<Host>& models_, uint model_index) {
  return make_verbose_linear_program(epsilon, models_.get_view(), model_index)->program;
}

void improve_weights(const ModelsView& models) {
  // Embarassingly parallel
  for (uint model_index = 0; model_index != models.models_count; ++model_index) {
    auto lp = make_internal_linear_program(1e-6, models, model_index);

    operations_research::glop::LPSolver solver;
    operations_research::glop::GlopParameters parameters;
    parameters.set_provide_strong_optimal_guarantee(true);
    solver.SetParameters(parameters);

    auto status = solver.Solve(*lp->program);
    assert(status == operations_research::glop::ProblemStatus::OPTIMAL);
    auto values = solver.variable_values();

    for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
      models.weights[crit_index][model_index] = values[lp->weight_variables[crit_index]];
    }
  }
}

void improve_weights(Models<Host>* models) {
  STOPWATCH("improve_weights (Host)");

  improve_weights(models->get_view());
}

}  // namespace ppl::improve_weights

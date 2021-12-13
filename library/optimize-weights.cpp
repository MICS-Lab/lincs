// Copyright 2021 Vincent Jacques

#include "optimize-weights.hpp"

#include <memory>
#include <string>

#include "stopwatch.hpp"


namespace ppl {

void structure_linear_program(LinearProgram* lp, const float epsilon, const ModelsView& models) {
  lp->weight_variables.reserve(models.domain.criteria_count);
  for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
    lp->weight_variables.push_back(lp->program.CreateNewVariable());
  }

  lp->x_variables.reserve(models.domain.learning_alternatives_count);
  lp->xp_variables.reserve(models.domain.learning_alternatives_count);
  lp->y_variables.reserve(models.domain.learning_alternatives_count);
  lp->yp_variables.reserve(models.domain.learning_alternatives_count);
  for (uint alt_index = 0; alt_index != models.domain.learning_alternatives_count; ++alt_index) {
    lp->x_variables.push_back(lp->program.CreateNewVariable());
    lp->xp_variables.push_back(lp->program.CreateNewVariable());
    lp->y_variables.push_back(lp->program.CreateNewVariable());
    lp->yp_variables.push_back(lp->program.CreateNewVariable());

    lp->program.SetObjectiveCoefficient(lp->xp_variables.back(), 1);
    lp->program.SetObjectiveCoefficient(lp->yp_variables.back(), 1);

    const uint category_index = models.domain.learning_assignments[alt_index];

    if (category_index != 0) {
      glp::RowIndex c = lp->program.CreateNewConstraint();
      lp->a_constraints.push_back(c);
      lp->program.SetConstraintBounds(c, 1, 1);
      lp->program.SetCoefficient(c, lp->x_variables.back(), -1);
      lp->program.SetCoefficient(c, lp->xp_variables.back(), 1);
    }

    if (category_index != models.domain.categories_count - 1) {
      glp::RowIndex c = lp->program.CreateNewConstraint();
      lp->b_constraints.push_back(c);
      lp->program.SetConstraintBounds(c, 1 - epsilon, 1 - epsilon);
      lp->program.SetCoefficient(c, lp->y_variables.back(), 1);
      lp->program.SetCoefficient(c, lp->yp_variables.back(), -1);
    }
  }
}

void label_linear_program(LinearProgram* lp, const ModelsView& models) {
  assert(lp->weight_variables.size() == models.domain.criteria_count);
  for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
    lp->program.SetVariableName(lp->weight_variables[crit_index], "w_" + std::to_string(crit_index));
  }

  assert(lp->x_variables.size() == models.domain.learning_alternatives_count);
  assert(lp->xp_variables.size() == models.domain.learning_alternatives_count);
  assert(lp->y_variables.size() == models.domain.learning_alternatives_count);
  assert(lp->yp_variables.size() == models.domain.learning_alternatives_count);
  for (uint alt_index = 0; alt_index != models.domain.learning_alternatives_count; ++alt_index) {
    lp->program.SetVariableName(lp->x_variables[alt_index], "x_" + std::to_string(alt_index));
    lp->program.SetVariableName(lp->xp_variables[alt_index], "x'_" + std::to_string(alt_index));
    lp->program.SetVariableName(lp->y_variables[alt_index], "y_" + std::to_string(alt_index));
    lp->program.SetVariableName(lp->yp_variables[alt_index], "y'_" + std::to_string(alt_index));
  }
}

void update_linear_program(LinearProgram* lp, const ModelsView& models, const uint model_index) {
  assert(lp->weight_variables.size() == models.domain.criteria_count);
  assert(lp->x_variables.size() == models.domain.learning_alternatives_count);
  assert(lp->xp_variables.size() == models.domain.learning_alternatives_count);
  assert(lp->y_variables.size() == models.domain.learning_alternatives_count);
  assert(lp->yp_variables.size() == models.domain.learning_alternatives_count);

  uint a_index = 0;
  uint b_index = 0;

  for (uint alt_index = 0; alt_index != models.domain.learning_alternatives_count; ++alt_index) {
    const uint category_index = models.domain.learning_assignments[alt_index];

    if (category_index != 0) {
      glp::RowIndex c = lp->a_constraints[a_index++];
      for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
        const float alternative_value = models.domain.learning_alternatives[crit_index][alt_index];
        const float profile_value = models.profiles[crit_index][category_index - 1][model_index];
        if (alternative_value >= profile_value) {
          lp->program.SetCoefficient(c, lp->weight_variables[crit_index], 1);
        } else {
          lp->program.SetCoefficient(c, lp->weight_variables[crit_index], 0);
        }
      }
    }

    if (category_index != models.domain.categories_count - 1) {
      glp::RowIndex c = lp->b_constraints[b_index++];
      for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
        const float alternative_value = models.domain.learning_alternatives[crit_index][alt_index];
        const float profile_value = models.profiles[crit_index][category_index][model_index];
        if (alternative_value >= profile_value) {
          lp->program.SetCoefficient(c, lp->weight_variables[crit_index], 1);
        } else {
          lp->program.SetCoefficient(c, lp->weight_variables[crit_index], 0);
        }
      }
    }
  }
}

std::shared_ptr<glp::LinearProgram> make_verbose_linear_program(
    const float epsilon, const Models<Host>& models, uint model_index) {
  auto models_view = models.get_view();

  LinearProgram lp;
  structure_linear_program(&lp, epsilon, models_view);
  label_linear_program(&lp, models_view);
  update_linear_program(&lp, models_view, model_index);

  auto r = std::make_shared<glp::LinearProgram>();
  r->PopulateFromLinearProgram(lp.program);
  return r;
}

WeightsOptimizer::WeightsOptimizer(const Models<Host>& models) :
    _linear_programs(models.get_view().models_count),
    _solvers(models.get_view().models_count) {
  #pragma omp parallel for
  for (auto& lp : _linear_programs) {
    structure_linear_program(&lp, 1e-6, models.get_view());
  }

  glp::GlopParameters parameters;
  parameters.set_provide_strong_optimal_guarantee(true);
  #pragma omp parallel for
  for (auto& solver : _solvers) {
    solver.SetParameters(parameters);
  }
}

void WeightsOptimizer::optimize_weights(Models<Host>* models) {
  STOPWATCH("WeightOptimize::optimize_weights (Host)");

  auto models_view = models->get_view();

  #pragma omp parallel for
  for (uint model_index = 0; model_index != models_view.models_count; ++model_index) {
    LinearProgram& lp = _linear_programs[model_index];
    update_linear_program(&lp, models_view, model_index);
    lp.program.CleanUp();

    auto status = _solvers[model_index].Solve(lp.program);
    assert(status == glp::ProblemStatus::OPTIMAL);
    auto values = _solvers[model_index].variable_values();

    for (uint crit_index = 0; crit_index != models_view.domain.criteria_count; ++crit_index) {
      models_view.weights[crit_index][model_index] = values[lp.weight_variables[crit_index]];
    }
  }
}

}  // namespace ppl

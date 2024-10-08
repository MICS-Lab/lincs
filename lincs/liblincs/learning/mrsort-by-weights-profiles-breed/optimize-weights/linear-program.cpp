// Copyright 2023-2024 Vincent Jacques

#include "linear-program.hpp"

#include "../../../chrones.hpp"
#include "../../../linear-programming/alglib.hpp"
#include "../../../linear-programming/glop.hpp"


namespace lincs {

template<typename LinearProgram>
void OptimizeWeightsUsingLinearProgram<LinearProgram>::optimize_weights(
  const unsigned model_indexes_begin,
  const unsigned model_indexes_end
) {
  CHRONE();

  const int model_indexes_end_ = model_indexes_end;

  #pragma omp parallel for
  for (int model_indexes_index = model_indexes_begin; model_indexes_index < model_indexes_end_; ++model_indexes_index) {
    const unsigned model_index = models_being_learned.model_indexes[model_indexes_index];
    optimize_model_weights(model_index);
  }
}

template<typename LinearProgram>
void OptimizeWeightsUsingLinearProgram<LinearProgram>::optimize_model_weights(unsigned model_index) {
  const float epsilon = 1e-6;
  LinearProgram program;

  std::vector<typename LinearProgram::variable_type> weight_variables;  // Indexed by [criterion_index]
  std::vector<typename LinearProgram::variable_type> x_variables;  // [alternative_index]
  std::vector<typename LinearProgram::variable_type> xp_variables;  // [alternative_index]
  std::vector<typename LinearProgram::variable_type> y_variables;  // [alternative_index]
  std::vector<typename LinearProgram::variable_type> yp_variables;  // [alternative_index]

  weight_variables.reserve(preprocessed_learning_set.criteria_count);
  for (unsigned criterion_index = 0; criterion_index != preprocessed_learning_set.criteria_count; ++criterion_index) {
    weight_variables.push_back(program.create_variable());
  }

  x_variables.reserve(preprocessed_learning_set.alternatives_count);
  xp_variables.reserve(preprocessed_learning_set.alternatives_count);
  y_variables.reserve(preprocessed_learning_set.alternatives_count);
  yp_variables.reserve(preprocessed_learning_set.alternatives_count);
  for (unsigned alternative_index = 0; alternative_index != preprocessed_learning_set.alternatives_count; ++alternative_index) {
    x_variables.push_back(program.create_variable());
    xp_variables.push_back(program.create_variable());
    y_variables.push_back(program.create_variable());
    yp_variables.push_back(program.create_variable());
  }

  program.mark_all_variables_created();

  for (unsigned alternative_index = 0; alternative_index != preprocessed_learning_set.alternatives_count; ++alternative_index) {
    program.set_objective_coefficient(xp_variables[alternative_index], 1);
    program.set_objective_coefficient(yp_variables[alternative_index], 1);

    const unsigned category_index = preprocessed_learning_set.assignments[alternative_index];

    if (category_index != 0) {  // Except bottom category
      const unsigned boundary_index = category_index - 1;  // Profile below category
      auto c = program.create_constraint();
      c.set_bounds(1, 1);
      c.set_coefficient(x_variables[alternative_index], -1);
      c.set_coefficient(xp_variables[alternative_index], 1);
      for (unsigned criterion_index = 0; criterion_index != preprocessed_learning_set.criteria_count; ++criterion_index) {
        if (LearnMrsortByWeightsProfilesBreed::is_accepted(preprocessed_learning_set, models_being_learned, model_index, boundary_index, criterion_index, alternative_index)) {
          c.set_coefficient(weight_variables[criterion_index], 1);
        }
      }
    }

    if (category_index != preprocessed_learning_set.categories_count - 1) {  // Except top category
      const unsigned boundary_index = category_index;  // Profile above category
      auto c = program.create_constraint();
      c.set_bounds(1 - epsilon, 1 - epsilon);
      c.set_coefficient(y_variables[alternative_index], 1);
      c.set_coefficient(yp_variables[alternative_index], -1);
      for (unsigned criterion_index = 0; criterion_index != preprocessed_learning_set.criteria_count; ++criterion_index) {
        if (LearnMrsortByWeightsProfilesBreed::is_accepted(preprocessed_learning_set, models_being_learned, model_index, boundary_index, criterion_index, alternative_index)) {
          c.set_coefficient(weight_variables[criterion_index], 1);
        }
      }
    }
  }

  auto values = program.solve()->assignments;

  for (unsigned criterion_index = 0; criterion_index != preprocessed_learning_set.criteria_count; ++criterion_index) {
    models_being_learned.weights[model_index][criterion_index] = values[weight_variables[criterion_index]];
  }
}

template class OptimizeWeightsUsingLinearProgram<GlopLinearProgram>;
template class OptimizeWeightsUsingLinearProgram<AlglibLinearProgram>;

}  // namespace lincs

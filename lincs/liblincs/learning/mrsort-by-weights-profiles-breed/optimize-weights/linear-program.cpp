// Copyright 2023 Vincent Jacques

#include "linear-program.hpp"

#include "../../../linear-programming/glop.hpp"
#include "../../../linear-programming/alglib.hpp"

namespace lincs {

template<typename LinearProgram>
void OptimizeWeightsUsingLinearProgram<LinearProgram>::optimize_weights() {
  #pragma omp parallel for
  for (unsigned model_index = 0; model_index != learning_data.models_count; ++model_index) {
    optimize_model_weights(model_index);
  }
}

template<typename LinearProgram>
void OptimizeWeightsUsingLinearProgram<LinearProgram>::optimize_model_weights(unsigned model_index) {
  const float epsilon = 1e-6;
  LinearProgram program;

  std::vector<typename LinearProgram::variable_type> weight_variables;
  std::vector<typename LinearProgram::variable_type> x_variables;
  std::vector<typename LinearProgram::variable_type> xp_variables;
  std::vector<typename LinearProgram::variable_type> y_variables;
  std::vector<typename LinearProgram::variable_type> yp_variables;

  weight_variables.reserve(learning_data.criteria_count);
  for (unsigned criterion_index = 0; criterion_index != learning_data.criteria_count; ++criterion_index) {
    weight_variables.push_back(program.create_variable());
  }

  x_variables.reserve(learning_data.learning_alternatives_count);
  xp_variables.reserve(learning_data.learning_alternatives_count);
  y_variables.reserve(learning_data.learning_alternatives_count);
  yp_variables.reserve(learning_data.learning_alternatives_count);
  for (unsigned alternative_index = 0; alternative_index != learning_data.learning_alternatives_count; ++alternative_index) {
    x_variables.push_back(program.create_variable());
    xp_variables.push_back(program.create_variable());
    y_variables.push_back(program.create_variable());
    yp_variables.push_back(program.create_variable());
  }

  program.mark_all_variables_created();

  for (unsigned alternative_index = 0; alternative_index != learning_data.learning_alternatives_count; ++alternative_index) {
    program.set_objective_coefficient(xp_variables[alternative_index], 1);
    program.set_objective_coefficient(yp_variables[alternative_index], 1);

    const unsigned category_index = learning_data.learning_assignments[alternative_index];

    if (category_index != 0) {
      auto c = program.create_constraint();
      c.set_bounds(1, 1);
      c.set_coefficient(x_variables[alternative_index], -1);
      c.set_coefficient(xp_variables[alternative_index], 1);
      for (unsigned criterion_index = 0; criterion_index != learning_data.criteria_count; ++criterion_index) {
        const float alternative_value = learning_data.learning_alternatives[criterion_index][alternative_index];
        const float profile_value = learning_data.profiles[criterion_index][category_index - 1][model_index];
        if (alternative_value >= profile_value) {
          c.set_coefficient(weight_variables[criterion_index], 1);
        }
      }
    }

    if (category_index != learning_data.categories_count - 1) {
      auto c = program.create_constraint();
      c.set_bounds(1 - epsilon, 1 - epsilon);
      c.set_coefficient(y_variables[alternative_index], 1);
      c.set_coefficient(yp_variables[alternative_index], -1);
      for (unsigned criterion_index = 0; criterion_index != learning_data.criteria_count; ++criterion_index) {
        const float alternative_value = learning_data.learning_alternatives[criterion_index][alternative_index];
        const float profile_value = learning_data.profiles[criterion_index][category_index][model_index];
        if (alternative_value >= profile_value) {
          c.set_coefficient(weight_variables[criterion_index], 1);
        }
      }
    }
  }

  auto values = program.solve();

  for (unsigned criterion_index = 0; criterion_index != learning_data.criteria_count; ++criterion_index) {
    learning_data.weights[criterion_index][model_index] = values[weight_variables[criterion_index]];
  }
}

template class OptimizeWeightsUsingLinearProgram<GlopLinearProgram>;
template class OptimizeWeightsUsingLinearProgram<AlglibLinearProgram>;

}  // namespace lincs

// Copyright 2024 Vincent Jacques

#ifndef LINCS__LINEAR_PROGRAMMING__CUSTOM_ON_CPU_HPP
#define LINCS__LINEAR_PROGRAMMING__CUSTOM_ON_CPU_HPP

#include <limits>
#include <map>
#include <vector>


namespace lincs {

class CustomOnCpuLinearProgram {
 public:
  typedef unsigned variable_type;
  variable_type create_variable() {
    return next_variable_index++;
  }

  void mark_all_variables_created() {}

  void set_objective_coefficient(variable_type variable, float coefficient) {
    objective_coefficients[variable] = coefficient;
  }

  struct Constraint {
    Constraint(CustomOnCpuLinearProgram* program, unsigned index) : program(program), index(index) {}

    Constraint& set_bounds(float lower_bound, float upper_bound) {
      std::get<0>(program->constraints[index]) = lower_bound;
      std::get<1>(program->constraints[index]) = upper_bound;
      return *this;
    }

    Constraint& set_coefficient(variable_type variable, float coefficient) {
      std::get<2>(program->constraints[index])[variable] = coefficient;
      return *this;
    }

   private:
    CustomOnCpuLinearProgram* const program;
    const unsigned index;
  };

  Constraint create_constraint() {
    constraints.emplace_back();
    return {this, constraints.size() - 1};
  }

  struct solution_type {
    std::vector<float> assignments;
    float cost;
  };
  solution_type solve();

 private:
  variable_type next_variable_index = 0;
  std::map<variable_type, float> objective_coefficients;
  std::vector<std::tuple<float, float, std::map<variable_type, float>>> constraints;
};

}  // namespace lincs

#endif  // LINCS__LINEAR_PROGRAMMING__CUSTOM_ON_CPU_HPP

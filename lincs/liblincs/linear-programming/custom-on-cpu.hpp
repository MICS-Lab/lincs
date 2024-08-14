// Copyright 2024 Vincent Jacques

#ifndef LINCS__LINEAR_PROGRAMMING__CUSTOM_ON_CPU_HPP
#define LINCS__LINEAR_PROGRAMMING__CUSTOM_ON_CPU_HPP

#include <limits>
#include <map>
#include <vector>


namespace lincs {

struct CustomOnCpuVerbose {
  CustomOnCpuVerbose();
  ~CustomOnCpuVerbose();
};

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

  struct ConstraintFacade {
    ConstraintFacade(CustomOnCpuLinearProgram* program, unsigned index) : program(program), index(index) {}

    ConstraintFacade& set_bounds(float lower_bound, float upper_bound) {
      program->constraints[index].lower_bound = lower_bound;
      program->constraints[index].upper_bound = upper_bound;
      return *this;
    }

    ConstraintFacade& set_coefficient(variable_type variable, float coefficient) {
      program->constraints[index].coefficients[variable] = coefficient;
      return *this;
    }

   private:
    CustomOnCpuLinearProgram* const program;
    const unsigned index;
  };

  ConstraintFacade create_constraint() {
    constraints.emplace_back();
    return {this, unsigned(constraints.size() - 1)};
  }

  struct solution_type {
    std::vector<float> assignments;
    float cost;
  };
  solution_type solve();

 public:
  unsigned variables_count() const { return next_variable_index; }

  const std::map<variable_type, float>& get_objective_coefficients() const { return objective_coefficients; }

  struct Constraint {
    float lower_bound;
    float upper_bound;
    std::map<variable_type, float> coefficients;
  };
  const std::vector<Constraint>& get_constraints() const { return constraints; }

 private:
  variable_type next_variable_index = 0;
  std::map<variable_type, float> objective_coefficients;
  std::vector<Constraint> constraints;
};

}  // namespace lincs

#endif  // LINCS__LINEAR_PROGRAMMING__CUSTOM_ON_CPU_HPP

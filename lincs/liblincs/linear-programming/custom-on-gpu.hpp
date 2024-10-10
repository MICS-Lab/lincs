// Copyright 2024 Vincent Jacques

#ifndef LINCS__LINEAR_PROGRAMMING__CUSTOM_ON_GPU_HPP
#define LINCS__LINEAR_PROGRAMMING__CUSTOM_ON_GPU_HPP

#include <limits>
#include <map>
#include <optional>
#include <vector>

#ifndef NDEBUG
#include "custom-on-cpu.hpp"
#endif


namespace lincs {

class InHouseSimplexOnGpuLinearProgram {
 public:
  typedef unsigned variable_type;
  variable_type create_variable() {
    #ifndef NDEBUG
    on_cpu_variables.push_back(on_cpu_program.create_variable());
    #endif
    return next_variable_index++;
  }

  void mark_all_variables_created() {
    #ifndef NDEBUG
    on_cpu_program.mark_all_variables_created();
    #endif
  }

  void set_objective_coefficient(variable_type variable, float coefficient) {
    objective_coefficients[variable] = coefficient;
    #ifndef NDEBUG
    on_cpu_program.set_objective_coefficient(on_cpu_variables[variable], coefficient);
    #endif
  }

  struct ConstraintFacade {
    ConstraintFacade(InHouseSimplexOnGpuLinearProgram* program, unsigned index) :
      program(program),
      index(index)
      #ifndef NDEBUG
      , on_cpu_constraint(program->on_cpu_program.create_constraint())
      #endif
    {}

    ConstraintFacade& set_bounds(float lower_bound, float upper_bound) {
      program->constraints[index].lower_bound = lower_bound;
      program->constraints[index].upper_bound = upper_bound;
      #ifndef NDEBUG
      on_cpu_constraint.set_bounds(lower_bound, upper_bound);
      #endif
      return *this;
    }

    ConstraintFacade& set_coefficient(variable_type variable, float coefficient) {
      program->constraints[index].coefficients[variable] = coefficient;
      #ifndef NDEBUG
      on_cpu_constraint.set_coefficient(program->on_cpu_variables[variable], coefficient);
      #endif
      return *this;
    }

   private:
    InHouseSimplexOnGpuLinearProgram* const program;
    const unsigned index;
    #ifndef NDEBUG
    InHouseSimplexOnCpuLinearProgram::ConstraintFacade on_cpu_constraint;
    #endif
  };

  ConstraintFacade create_constraint() {
    constraints.emplace_back();
    return {this, unsigned(constraints.size() - 1)};
  }

  struct solution_type {
    std::vector<float> assignments;
    float cost;
  };
  std::optional<solution_type> solve();

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
#ifndef NDEBUG
  InHouseSimplexOnCpuLinearProgram on_cpu_program;
  std::vector<InHouseSimplexOnCpuLinearProgram::variable_type> on_cpu_variables;
#endif
};

}  // namespace lincs

#endif  // LINCS__LINEAR_PROGRAMMING__CUSTOM_ON_GPU_HPP

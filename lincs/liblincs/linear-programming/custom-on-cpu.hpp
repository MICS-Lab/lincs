// Copyright 2024 Vincent Jacques

#ifndef LINCS__LINEAR_PROGRAMMING__CUSTOM_ON_CPU_HPP
#define LINCS__LINEAR_PROGRAMMING__CUSTOM_ON_CPU_HPP

#include <limits>
#include <map>
#include <optional>
#include <vector>

#ifndef NDEBUG
#include "glop.hpp"
#endif


namespace lincs {

#ifndef NDEBUG

struct InHouseSimplexOnCpuVerbose {
  InHouseSimplexOnCpuVerbose(int = 2);
  ~InHouseSimplexOnCpuVerbose();
};

#endif

class InHouseSimplexOnCpuLinearProgram {
 public:
  typedef unsigned variable_type;
  variable_type create_variable() {
    #ifndef NDEBUG
    glop_variables.push_back(glop_program.create_variable());
    #endif
    return next_variable_index++;
  }

  void mark_all_variables_created() {
    #ifndef NDEBUG
    glop_program.mark_all_variables_created();
    #endif
  }

  void set_objective_coefficient(variable_type variable, float coefficient) {
    objective_coefficients[variable] = coefficient;
    #ifndef NDEBUG
    glop_program.set_objective_coefficient(glop_variables[variable], coefficient);
    #endif
  }

  struct ConstraintFacade {
    ConstraintFacade(InHouseSimplexOnCpuLinearProgram* program, unsigned index) :
      program(program),
      index(index)
      #ifndef NDEBUG
      , glop_constraint(program->glop_program.create_constraint())
      #endif
    {}

    ConstraintFacade& set_bounds(float lower_bound, float upper_bound) {
      program->constraints[index].lower_bound = lower_bound;
      program->constraints[index].upper_bound = upper_bound;
      #ifndef NDEBUG
      glop_constraint.set_bounds(lower_bound, upper_bound);
      #endif
      return *this;
    }

    ConstraintFacade& set_coefficient(variable_type variable, float coefficient) {
      program->constraints[index].coefficients[variable] = coefficient;
      #ifndef NDEBUG
      glop_constraint.set_coefficient(program->glop_variables[variable], coefficient);
      #endif
      return *this;
    }

   private:
    InHouseSimplexOnCpuLinearProgram* const program;
    const unsigned index;
    #ifndef NDEBUG
    GlopLinearProgram::Constraint glop_constraint;
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
  GlopLinearProgram glop_program;
  std::vector<GlopLinearProgram::variable_type> glop_variables;
#endif
};

}  // namespace lincs

#endif  // LINCS__LINEAR_PROGRAMMING__CUSTOM_ON_CPU_HPP

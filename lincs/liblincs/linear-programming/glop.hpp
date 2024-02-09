// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__LINEAR_PROGRAMMING__GLOP_HPP
#define LINCS__LINEAR_PROGRAMMING__GLOP_HPP

#include <ortools/glop/lp_solver.h>
#undef CHECK
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_LT
#undef CHECK_GT
#undef CHECK_LE
#undef CHECK_GE
#undef LOG


namespace lincs {

class GlopLinearProgram {
 public:
  typedef operations_research::glop::ColIndex variable_type;
  variable_type create_variable() {
    return program.CreateNewVariable();
  }

  void mark_all_variables_created() {}

  void set_objective_coefficient(variable_type variable, float coefficient) {
    program.SetObjectiveCoefficient(variable, coefficient);
  }

  struct Constraint {
    Constraint(operations_research::glop::LinearProgram& program_) : program(program_), index(program_.CreateNewConstraint()) {}

    void set_bounds(float lower_bound, float upper_bound) {
      program.SetConstraintBounds(index, lower_bound, upper_bound);
    }

    void set_coefficient(variable_type variable, float coefficient) {
      program.SetCoefficient(index, variable, coefficient);
    }

   private:
    operations_research::glop::LinearProgram& program;
    operations_research::glop::RowIndex index;
  };

  Constraint create_constraint() {
    return Constraint(program);
  }

  operations_research::glop::DenseRow solve();

 private:
  operations_research::glop::LinearProgram program;
};

}  // namespace lincs

#endif  // LINCS__LINEAR_PROGRAMMING__GLOP_HPP

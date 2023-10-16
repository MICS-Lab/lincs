// Copyright 2023 Vincent Jacques

#ifndef LINCS__LINEAR_PROGRAMMING__ALGLIB_HPP
#define LINCS__LINEAR_PROGRAMMING__ALGLIB_HPP

#include <cassert>
#include <map>

#include "../vendored/alglib/optimization.h"


namespace lincs {

class AlglibLinearProgram {
 public:
  AlglibLinearProgram() : next_variable_index(0), objective_coefficients(), state() {}

  typedef unsigned variable_type;
  variable_type create_variable() {
    assert(objective_coefficients.empty());
    return next_variable_index++;
  }

  void mark_all_variables_created() {
    assert(objective_coefficients.empty());
    alglib::minlpcreate(next_variable_index, state);
    objective_coefficients.resize(next_variable_index);

    // Relax Alglib's default constraints to match GLOP's
    const float infinity = std::numeric_limits<float>::infinity();
    alglib::minlpsetbcall(state, 0, +infinity);
  }

  void set_objective_coefficient(variable_type variable, float coefficient) {
    assert(variable < objective_coefficients.size());
    objective_coefficients[variable] = coefficient;
  }

  struct Constraint {
    Constraint(alglib::minlpstate& state_) : state(state_) {}
    ~Constraint() {
      const unsigned nnz = coefficients.size();
      // @todo(Performance, later) Try using 'alglib::minlpsetbci' when 'nnz == 1'
      // Maybe Alglib can be faster with box constraints than with general linear constraints?
      alglib::integer_1d_array idxa;
      idxa.setlength(nnz);
      alglib::real_1d_array vala;
      vala.setlength(nnz);
      unsigned i = 0;
      for (auto& [variable, coefficient] : coefficients) {
        idxa[i] = variable;
        vala[i] = coefficient;
        ++i;
      }
      alglib::minlpaddlc2(state, idxa, vala, nnz, lower_bound, upper_bound);
    }

    void set_bounds(float lower_bound_, float upper_bound_) {
      lower_bound = lower_bound_;
      upper_bound = upper_bound_;
    }

    void set_coefficient(variable_type variable, float coefficient) {
      coefficients[variable] = coefficient;
    }

   private:
    alglib::minlpstate& state;
    float lower_bound;
    float upper_bound;
    std::map<variable_type, float> coefficients;
  };

  Constraint create_constraint() {
    return Constraint(state);
  }

  alglib::real_1d_array solve();

 private:
  variable_type next_variable_index;
  std::vector<double> objective_coefficients;
  alglib::minlpstate state;
};

}  // namespace lincs

#endif  // LINCS__LINEAR_PROGRAMMING__ALGLIB_HPP

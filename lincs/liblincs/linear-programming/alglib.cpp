// Copyright 2023-2024 Vincent Jacques

#include "alglib.hpp"

#include "../chrones.hpp"


namespace lincs {

AlglibLinearProgram::solution_type AlglibLinearProgram::solve() {
  CHRONE();

  alglib::real_1d_array c;
  c.setcontent(objective_coefficients.size(), objective_coefficients.data());
  alglib::minlpsetcost(state, c);

  alglib::minlpoptimize(state);
  alglib::real_1d_array x;
  alglib::minlpreport rep;
  alglib::minlpresults(state, x, rep);
  assert(rep.terminationtype >= 1 && rep.terminationtype <= 4);

  return AlglibLinearProgram::solution_type{x, float(rep.f)};
}

}  // namespace lincs

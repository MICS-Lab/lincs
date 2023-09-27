// Copyright 2023 Vincent Jacques

#include "alglib.hpp"

#include "../chrones.hpp"


namespace lincs {

alglib::real_1d_array AlglibLinearProgram::solve() {
  CHRONE();

  alglib::real_1d_array c;
  c.setcontent(objective_coefficients.size(), objective_coefficients.data());
  alglib::minlpsetcost(state, c);

  alglib::minlpoptimize(state);
  alglib::real_1d_array x;
  alglib::minlpreport rep;
  alglib::minlpresults(state, x, rep);

  return x;
}

}  // namespace lincs

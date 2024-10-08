// Copyright 2023-2024 Vincent Jacques

#include "alglib.hpp"

#include "../chrones.hpp"


namespace lincs {

std::optional<AlglibLinearProgram::solution_type> AlglibLinearProgram::solve() {
  CHRONE();

  alglib::real_1d_array c;
  c.setcontent(objective_coefficients.size(), objective_coefficients.data());
  alglib::minlpsetcost(state, c);

  alglib::minlpoptimize(state);
  alglib::real_1d_array x;
  alglib::minlpreport rep;
  alglib::minlpresults(state, x, rep);
  if (rep.terminationtype >= 1 && rep.terminationtype <= 4) {
    return solution_type{x, float(rep.f)};
  } else {
    return {};
  }
}

}  // namespace lincs

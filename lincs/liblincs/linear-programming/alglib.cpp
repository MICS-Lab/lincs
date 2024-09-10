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
  if (rep.terminationtype >= 1 && rep.terminationtype <= 4) {
    return AlglibLinearProgram::solution_type{x, float(rep.f)};
  } else if (rep.terminationtype == -2) {
    // Doc https://www.alglib.net/translator/man/manual.cpp.html#struct_minlpreport says
    //     -4    LP problem is primal unbounded
    // but in actual tests, we've seen -2 instead of -4.
    return AlglibLinearProgram::solution_type{x, -std::numeric_limits<float>::infinity()};
  } else if (rep.terminationtype == -3) {
    return AlglibLinearProgram::solution_type{x, std::numeric_limits<float>::quiet_NaN()};
  } else {
    std::cerr << "Unexpected AlgLib termination type: " << rep.terminationtype << std::endl;
    assert(false);
  }
}

}  // namespace lincs

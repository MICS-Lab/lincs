// Copyright 2023 Vincent Jacques

#include "minisat.hpp"

#include <doctest.h>  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


template<typename SatProblem>
void test_sat_problem() {
  SatProblem problem;

  auto x1 = problem.create_variable();
  auto x2 = problem.create_variable();
  auto x3 = problem.create_variable();

  problem.mark_all_variables_created();

  problem.add_clause({x1, -x3});
  problem.add_clause({x2, x3, -x1});

  auto solution = problem.solve();

  CHECK((solution[x1] || !solution[x3]));
  CHECK((solution[x2] || solution[x3] || !solution[x1]));
}

TEST_CASE("Minisat SAT problem") {
  test_sat_problem<lincs::MinisatSatProblem>();
}

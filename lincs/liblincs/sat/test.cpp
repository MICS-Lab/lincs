// Copyright 2023 Vincent Jacques

#include "eval-max-sat.hpp"
#include "minisat.hpp"

#include "../vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


template<typename SatProblem>
void test_sat_problem() {
  {
    SatProblem problem;

    auto x1 = problem.create_variable();
    auto x2 = problem.create_variable();
    auto x3 = problem.create_variable();

    problem.mark_all_variables_created();

    problem.add_clause({x1, -x3});
    problem.add_clause({x2, x3, -x1});

    std::optional<std::vector<bool>> solution = problem.solve();

    CHECK(((*solution)[x1] || !(*solution)[x3]));
    CHECK(((*solution)[x2] || (*solution)[x3] || !(*solution)[x1]));
  }
  {
    SatProblem problem;

    auto x1 = problem.create_variable();

    problem.mark_all_variables_created();

    problem.add_clause({x1});
    problem.add_clause({-x1});

    std::optional<std::vector<bool>> solution = problem.solve();

    CHECK(!solution);
  }
}

template<typename MaxSatProblem>
void test_max_sat_problem() {
  MaxSatProblem problem;

  auto x1 = problem.create_variable();
  auto x2 = problem.create_variable();
  auto x3 = problem.create_variable();

  problem.add_clause({-x1, -x2, -x3});  // a

  problem.add_weighted_clause({x1, x2}, 1);  // b
  problem.add_weighted_clause({x3}, 1);  // c
  problem.add_weighted_clause({x1, -x3}, 1);  // d
  problem.add_weighted_clause({x2, -x3}, 1);  // e

  // x1 x2 x3  a  b c d e  score
  // 0  0  0   1  0 0 1 1  2
  // 0  0  1   1  0 1 0 0  1
  // 0  1  0   1  1 0 1 1  3
  // 0  1  1   1  1 1 0 1  3
  // 1  0  0   1  1 0 1 1  3
  // 1  0  1   1  1 1 1 0  3
  // 1  1  0   1  1 0 1 1  3
  // 1  1  1   0  1 1 1 1  4

  auto solution = problem.solve();

  CHECK((!(*solution)[x1] || !(*solution)[x2] || !(*solution)[x3]));

  int score = 0;
  if ((*solution)[x1] || (*solution)[x2]) ++score;
  if ((*solution)[x3]) ++score;
  if ((*solution)[x1] || !(*solution)[x3]) ++score;
  if ((*solution)[x2] || !(*solution)[x3]) ++score;
  CHECK(score == 3);
}

// Some SAT solvers:
// - minisat
// - glucose
// - cadical
// - lingeling
// - cryptominisat
// - picosat
// - riss
// - kissat
// - minisatp

// Some max-SAT solvers:
// - EvalMaxSAT
// - open-wbo
// - maxhs
// - cryptominisat
// - maxwalksat
// - maxsatz

TEST_CASE("Minisat SAT problem") {
  test_sat_problem<lincs::MinisatSatProblem>();
}

TEST_CASE("EvalMaxSAT SAT problem") {
  test_sat_problem<lincs::EvalmaxsatMaxSatProblem>();
}

TEST_CASE("EvalMaxSAT max-SAT problem") {
  test_max_sat_problem<lincs::EvalmaxsatMaxSatProblem>();
}

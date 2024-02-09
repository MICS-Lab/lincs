// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__SAT__EVALMAXSAT_HPP
#define LINCS__SAT__EVALMAXSAT_HPP

#include <optional>
#include <vector>

#pragma GCC diagnostic ignored "-Wclass-memaccess"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "../vendored/eval-max-sat/EvalMaxSAT.h"
#pragma GCC diagnostic pop  // No associated push => restore command-line options
#undef LOG


namespace lincs {

class EvalmaxsatMaxSatProblem {
 public:
  EvalmaxsatMaxSatProblem(
    unsigned nb_minimize_threads = 0,
    unsigned timeout_fast_minimize = 60,  // Seconds. Documented as "Magic number" in EvalMaxSAT source code.
    unsigned coef_minimize_time = 2  // Documented as "Magic number" in EvalMaxSAT source code.
  ) : solver(nb_minimize_threads) {
    solver.setTimeOutFast(timeout_fast_minimize);
    solver.setCoefMinimize(coef_minimize_time);
  }

 public:
  typedef int variable_type;
  variable_type create_variable() {
    int v = solver.newVar();
    variables.push_back(v);
    return v;
  }

  void mark_all_variables_created() {}

 public:
  void add_clause(std::vector<variable_type> clause) {
    solver.addClause(clause);
  }

  typedef unsigned weight_type;
  void add_weighted_clause(std::vector<variable_type> clause, weight_type weight) {
    solver.addWeightedClause(clause, weight);
  }

  std::optional<std::vector<bool>> solve();

 private:
  EvalMaxSAT solver;
  std::vector<int> variables;
};

}  // namespace lincs

#endif  // LINCS__SAT__EVALMAXSAT_HPP

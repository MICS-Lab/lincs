// Copyright 2023 Vincent Jacques

#ifndef LINCS__SAT__EVALMAXSAT_HPP
#define LINCS__SAT__EVALMAXSAT_HPP

#include <optional>
#include <vector>

#pragma GCC diagnostic ignored "-Wclass-memaccess"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "../vendored/eval-max-sat/EvalMaxSAT.h"
#pragma GCC diagnostic pop  // No associated push => restore command-line options
#undef LOG


namespace lincs {

class EvalmaxsatSatProblem {
 public:
  EvalmaxsatSatProblem() : solver(0) {}

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

  std::optional<std::vector<bool>> solve() {
    if (solver.solve()) {
      std::vector<bool> solution(variables.back() + 1);
      for (const int v : variables) {
        solution[v] = solver.getValue(v);
      }
      return solution;
    } else {
      return std::nullopt;
    }
  }

 private:
  EvalMaxSAT solver;
  std::vector<int> variables;
};

}  // namespace lincs

#endif  // LINCS__SAT__EVALMAXSAT_HPP

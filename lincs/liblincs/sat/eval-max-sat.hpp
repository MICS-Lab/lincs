// Copyright 2023 Vincent Jacques

#ifndef LINCS__SAT__EVALMAXSAT_HPP
#define LINCS__SAT__EVALMAXSAT_HPP

#include <vector>

#include "../vendored/eval-max-sat/EvalMaxSAT.h"
#undef LOG


namespace lincs {

class EvalmaxsatSatProblem {
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

  typedef int weight_type;
  void add_weighted_clause(std::vector<variable_type> clause, weight_type weight) {
    solver.addWeightedClause(clause, weight);
  }

  auto solve() {
    solver.solve();

    std::vector<bool> solution(variables.back() + 1);
    for (const int v : variables) {
      solution[v] = solver.getValue(v);
    }

    return solution;
  }

 private:
  EvalMaxSAT solver;
  std::vector<int> variables;
};

}  // namespace lincs

#endif  // LINCS__SAT__EVALMAXSAT_HPP
